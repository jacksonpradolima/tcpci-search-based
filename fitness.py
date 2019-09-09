import numpy as np
from utils import sort_update_actions


def evaluate_prioritization(individual, actions, metric):
    """
    Evaluate an individual generated in seach-based algorithm
    :param individual: Prioritization order
    :param actions: Build information
    :param metric: Evaluation Metric
    :return:
    """
    metric.evaluate(sort_update_actions(np.array(individual) + 1, actions))

    return metric.fitness,


__all__ = ['evaluate_prioritization']
