import random

from jmetal.core.problem import PermutationProblem
from jmetal.core.solution import PermutationSolution
from utils import sort_update_actions


class TCPCI(PermutationProblem):

    def __init__(self, metric, test_cases, number_of_variables: int = 10):
        super(TCPCI, self).__init__()

        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['NAPFD']
        self.metric = metric
        self.test_cases = test_cases

        self.lower_bound = [0 for _ in range(number_of_variables)]
        self.upper_bound = [number_of_variables for _ in range(number_of_variables)]

        PermutationSolution.lower_bound = self.lower_bound
        PermutationSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        self.metric.evaluate(sort_update_actions(solution.variables, self.test_cases))
        solution.objectives[0] = self.metric.fitness*-1.0

        return solution

    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                           number_of_objectives=self.number_of_objectives)
        new_solution.variables = random.sample(range(1, self.number_of_variables+1), self.number_of_variables)

        return new_solution

    def get_name(self) -> str:
        return 'TCPCI'
