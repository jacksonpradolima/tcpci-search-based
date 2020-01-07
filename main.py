import argparse
import logging
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
from reward import RNFailReward, RRankReward, TimeRankReward
from utils import sort_update_actions
from fitness import evaluate_prioritization

from evaluation import NAPFDMetric

__author__ = "Jackson Antonio do Prado Lima"
__email__ = "jacksonpradolima@gmail.com"
__license__ = "MIT"
__version__ = "1.0"

# Empirical parameters
CXPB, MUTPB, POP, NGEN = 0.8, 0.01, 100, 100

DEFAULT_EXPERIMENT_DIR = 'results/optimal_approximated/'
DEFAULT_SCHED_TIME_RATIO = 0.5

### DEAP CONFIGURATION ###
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def get_pool():
    max_cpus = multiprocessing.cpu_count()
    max_cpus = max_cpus - 1 if max_cpus > 1 else max_cpus

    return multiprocessing.Pool(processes=max_cpus)


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

    # Attribute generator
    toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

    # Structure initializers
    toolbox.register("individual", tools.initIterate,
                     creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operator registering
    toolbox.register("evaluate", evaluate_prioritization,
                     actions=actions, metric=metric)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)

    # Multiprocessing
    pool = get_pool()
    toolbox.register("map", pool.map)

    # init population
    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(1)

    # Run the algorithm
    eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, halloffame=hof, verbose=False)

    pool.close()

    return hof[0]


def run_optimal(dataset, repo_path, output_dir, sched_time_ratio):
    logging.debug(f"Running for {dataset}")

    metric = NAPFDVerdictMetric() if dataset in [
        'iofrol', 'paintcontrol', 'gsdtsr', 'lexis'] else NAPFDMetric()

    reward_functions = [RNFailReward(), RRankReward(), TimeRankReward()]

    all_data_file = "experiment;step;policy;reward_function;sched_time;sched_time_duration;prioritization_time;detected;missed;tests_ran;tests_not_ran;" \
        + "ttf;time_reduction;fitness;cost;rewards;avg_precision\n"

    start = time.time()

    # 30 independent executions
    for i in range(1, 31):
        scenario_provider = scenarios.IndustrialDatasetScenarioProvider(f"{repo_path}/{dataset}/features-engineered.csv",
                                                                        sched_time_ratio)

        for (t, vsc) in enumerate(scenario_provider, start=1):
            metric.update_available_time(vsc.get_available_time())
            actions = vsc.get_testcases()

            # Compute time
            start_exp = time.time()

            ind = [0]
            if (len(actions) > 1):
                # Run GA to find the best NAPFD in current commit
                ind = genetic_algorithm_tcp(actions, metric)

            end_exp = time.time()

            last_prioritization = sort_update_actions(
                np.array(ind) + 1, actions)

            metric.evaluate(last_prioritization)

            # Get the Test Case names
            last_prioritization = [tc['Name'] for tc in last_prioritization]

            time_reduction = scenario_provider.total_build_duration - metric.ttf_duration

            for reward_function in reward_functions:
                last_reward = reward_function.evaluate(
                    metric, last_prioritization)

                all_data_file += f"{i};{t};GA;{reward_function.get_name()};" \
                    + f"{scenario_provider.avail_time_ratio};{vsc.get_available_time()};{end_exp - start_exp};" \
                    + f"{metric.detected_failures};{metric.undetected_failures};{len(metric.scheduled_testcases)};" \
                    + f"{len(metric.unscheduled_testcases)};{metric.ttf};{time_reduction};" \
                    + f"{metric.fitness};{metric.cost};{np.mean(last_reward)};{metric.avg_precision}\n"

            logging.debug(f"Exp {i} - Ep {t} - Policy GA - NAPFD/APFDc: {metric.fitness:.4f}/{metric.cost:.4f}")

    end = time.time()
    logging.debug(f"Time expend to run the experiments: {end - start}")

    print(f"Saving in {output_dir}/{dataset}.csv")
    with open(f"{output_dir}/{dataset}.csv", "w") as f:
        f.write(all_data_file)


def main_test():
    dataset_dir = "data"
    # dataset = 'alibaba@fastjson'
    dataset = 'deeplearning4j@deeplearning4j'

    output_dir = os.path.join(
        DEFAULT_EXPERIMENT_DIR, f"time_ratio_{int(DEFAULT_SCHED_TIME_RATIO*100)}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_optimal(dataset, dataset_dir, output_dir,
                DEFAULT_SCHED_TIME_RATIO)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # main_test()
    ap = argparse.ArgumentParser(description='Optimal')

    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--datasets', nargs='+', default=[], required=True,
                    help='Datasets to analyse. Ex: \'deeplearning4j@deeplearning4j\'')
    ap.add_argument('--sched_time_ratio', nargs='+',
                    default=[], help='Schedule Time Ratio')
    ap.add_argument('-o', '--output_dir', default=DEFAULT_EXPERIMENT_DIR)

    args = ap.parse_args()

    time_ratio = [float(t) for t in args.sched_time_ratio]

    for tr in time_ratio:
        output_dir = os.path.join(DEFAULT_EXPERIMENT_DIR, f"time_ratio_{int(tr*100)}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for dataset in args.datasets:
            run_optimal(dataset, args.dataset_dir, output_dir, tr)
