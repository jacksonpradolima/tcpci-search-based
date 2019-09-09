import argparse
import multiprocessing
import numpy as np
import os
import random
import time

import scenarios

from deap import base
from deap import creator
from deap import tools
from algorithm import eaSimple
from utils import sort_update_actions
from fitness import evaluate_prioritization

from evaluation import NAPFDMetric

DEFAULT_EXPERIMENT_DIR = 'results/optimal_approximate/'
DEFAULT_SCHED_TIME_RATIO = 0.5

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Empirical parameters
CXPB, MUTPB, POP, NGEN = 0.8, 0.1, 100, 100

max_cpus = multiprocessing.cpu_count()
max_cpus = max_cpus - 1 if max_cpus > 1 else max_cpus
pool = multiprocessing.Pool(processes=max_cpus)


def genetic_algorithm_tcp(actions, metric):
    """
    Genetic Algorithm to find the best NAPFD for a given build
    :param actions: Build information
    :param metric: Evaluation Metric
    :return: Best individual (NAPFD) found
    """
    IND_SIZE = len(actions)

    # We create a permutation problem where we try to find the best priority (indexes)
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_prioritization, actions=actions, metric=metric)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)

    # Multiprocessing
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(1)

    # Statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats = tools.MultiStatistics(fitness=stats_fit)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the algorithm
    eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats, halloffame=hof, verbose=False)

    # Return the best individual
    return hof[0]


def run_optimal(dataset, repo_path, output_dir, sched_time_ratio):
    scenario_provider = scenarios.IndustrialDatasetScenarioProvider(f"{repo_path}/{dataset}/features-engineered.csv",
                                                                    sched_time_ratio)

    print(f"Calculating optimal values for '{scenario_provider.name}'")
    print(f"Saving in {output_dir}{dataset}.csv")

    metric = NAPFDMetric('Verdict') if dataset in ['iofrol', 'paintcontrol', 'gsdtsr'] else NAPFDMetric()

    with open(f"{output_dir}{dataset}.csv", "w") as f:
        # Header
        f.write("experiment;step;policy;reward_function;prioritization_time;detected;missed;tests_ran;tests_not_ran;" +
                "ttf;time_reduction;fitness;avg_precision\n")
        f.flush()

        start = time.time()

        for (t, vsc) in enumerate(scenario_provider, start=1):
            metric.update_available_time(vsc.get_available_time())
            actions = vsc.get_testcases()

            # Compute time
            start_exp = time.time()

            # Run GA to find the best NAPFD in current commit
            ind = genetic_algorithm_tcp(actions, metric)

            end_exp = time.time()

            metric.evaluate(sort_update_actions(np.array(ind) + 1, actions))

            print(f"Commit: {t} - Fitness: {metric.fitness} - Duration: {end_exp - start_exp}")

            time_reduction = scenario_provider.total_build_duration - metric.ttf_duration

            f.write(
                f"{1};{t};GA;optimal_approx;{end_exp - start_exp};"
                f"{metric.detected_failures};{metric.undetected_failures};{len(metric.scheduled_testcases)};"
                f"{len(metric.unscheduled_testcases)};{metric.ttf};{time_reduction};"
                f"{metric.fitness};{metric.avg_precision}\n")

        end = time.time()
        print(f"Time expend to run the experiments: {end - start}")


def main_test():
    dataset_dir = "/mnt/NAS/japlima/mab-datasets"
    dataset = 'deeplearning4j@deeplearning4j'

    if not os.path.exists(DEFAULT_EXPERIMENT_DIR):
        os.makedirs(DEFAULT_EXPERIMENT_DIR)

    run_optimal(dataset, dataset_dir, DEFAULT_EXPERIMENT_DIR, DEFAULT_SCHED_TIME_RATIO)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Optimal')

    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--datasets', nargs='+', default=[], required=True,
                    help='Datasets to analyse. Ex: \'deeplearning4j@deeplearning4j\'')

    ap.add_argument('--sched_time_ratio', type=int, default=DEFAULT_SCHED_TIME_RATIO)
    ap.add_argument('-o', '--output_dir', default=DEFAULT_EXPERIMENT_DIR)

    args = ap.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for dataset in args.datasets:
        run_optimal(dataset, args.dataset_dir, args.output_dir, args.sched_time_ratio)
