from src import scenarios

from src.evaluation import NAPFDMetric
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
# from jmetal.operator import BinaryTournamentSelection, SwapMutation, PMXCrossover
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from src.jmetal.problem import TCPCI

if __name__ == "__main__":
    metric = NAPFDMetric()

    repo_path = "/mnt/NAS/japlima/mab-datasets"
    dataset = 'deeplearning4j@deeplearning4j'

    scenario_provider = scenarios.IndustrialDatasetScenarioProvider(f"{repo_path}/{dataset}/features-engineered.csv")

    for (t, vsc) in enumerate(scenario_provider, start=1):
        available_time = vsc.get_available_time()
        metric.update_available_time(available_time)

        test_cases = vsc.get_testcases()
        IND_SIZE = len(test_cases)

        problem = TCPCI(metric=metric, test_cases=test_cases, number_of_variables=IND_SIZE)

        algorithm = GeneticAlgorithm(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            # mutation=SwapMutation(0.1),
            # crossover=PMXCrossover(0.9),
            # selection=BinaryTournamentSelection(),
            termination_criterion=StoppingByEvaluations(max=500000)
        )

        algorithm.observable.register(observer=PrintObjectivesObserver(1000))

        algorithm.run()
        result = algorithm.get_result()

        print(f"Commit: {t} - Fitness: {result.objectives[0]}")
        # print('Algorithm: ' + algorithm.get_name())
        # print('Problem: ' + problem.get_name())
        # print('Solution: ' + str(result.variables[0]))
        # print('Fitness:  ' + str(result.objectives[0]))
        # print('Computing time: ' + str(algorithm.total_computing_time))

