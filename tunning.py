import argparse
import logging
import multiprocessing
import numpy as np
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

__author__ = "Jackson Antonio do Prado Lima"
__email__ = "jacksonpradolima@gmail.com"
__license__ = "MIT"
__version__ = "1.0"

# Empirical parameters
CXPB, MUTPB, POP, NGEN = 0.8, 0.01, 150, 200

DEFAULT_SCHED_TIME_RATIO = 0.5

### DEAP CONFIGURATION ###
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def get_pool():
    # max_cpus = multiprocessing.cpu_count()
    # max_cpus = max_cpus - 1 if max_cpus > 1 else max_cpus
    max_cpus = 2

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
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operator registering
    toolbox.register("evaluate", evaluate_prioritization, actions=actions, metric=metric)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)

    # Multiprocessing
    pool = get_pool()
    toolbox.register("map", pool.map)

    ## init population
    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(1)

    # Run the algorithm
    eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, halloffame=hof, verbose=False)

    pool.close()

    return hof[0]


def run_optimal(dataset, repo_path, datfile, sched_time_ratio=DEFAULT_SCHED_TIME_RATIO):
    scenario_provider = scenarios.IndustrialDatasetScenarioProvider(f"{repo_path}/{dataset}/features-engineered.csv",
                                                                    sched_time_ratio)

    logging.debug(f"Running for {dataset}")
    metric = NAPFDMetric('Verdict') if dataset in ['iofrol', 'paintcontrol', 'gsdtsr'] else NAPFDMetric()

    i = 1
    mean_fitness = 0
    start = time.time()

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

        metric.evaluate(sort_update_actions(np.array(ind)+1, actions))
        i += 1
        mean_fitness += metric.fitness


        logging.debug(f"commit: {t} - fitness: {metric.fitness} - duration: {end_exp - start_exp}")

    end = time.time()
    logging.debug(f"Time expend to run the experiments: {end - start}")
    logging.debug(mean_fitness / i)

    with open(datfile, 'w') as f:
        f.write(str(mean_fitness / i))


def main_test():
    dataset_dir = "/mnt/NAS/japlima/mab-datasets"
    dataset = 'alibaba@fastjson'
    # dataset = 'deeplearning4j@deeplearning4j'

    run_optimal(dataset, dataset_dir, "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Configuration for Genetic Algorithm applied on " \
                                             + "Test Case Prioritization in Continuous Integration environments")

    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    ap.add_argument('--pop', dest='pop', type=int, required=True, help='Population size', default=POP)
    ap.add_argument('--cros', dest='cros', type=float, required=True, help='Crossover probability', default=CXPB)
    ap.add_argument('--mut', dest='mut', type=float, required=True, help='Mutation probability',default=MUTPB)
    ap.add_argument('--datfile', dest='datfile', type=str, required=True,
                    help='File where it will be save the score (result)')

    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--dataset', required=True, help='Datasets to analyse. Ex: \'deeplearning4j@deeplearning4j\'')
    ap.add_argument('--sched_time_ratio', type=int, default=DEFAULT_SCHED_TIME_RATIO)

    args = ap.parse_args()

    CXPB, MUTPB, POP = args.cros, args.mut, args.pop

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    run_optimal(args.dataset, args.dataset_dir, args.datfile, args.sched_time_ratio)
