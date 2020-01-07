from typing import List

import numpy as np

from evaluation import EvaluationMetric


class Reward(object):
    """
    A reward function is used by the agent in the observe method
    """

    def evaluate(self, reward: EvaluationMetric, last_prioritization: List[str]):
        """
        The reward function evaluate a bandit result and return a reward
        :param reward:
        :param last_prioritization:
        :return:
        """
        return None


class FailCountReward(Reward):
    """
    This reward function all test cases, both scheduled and unscheduled,
    receive the number of failed test cases in the schedule as a reward.
    It is a basic, but intuitive reward function directly rewarding
    the RL agent on the goal of maximizing the number of failed test cases.
    The reward function acknowledges the prioritized test suite in total,
    including positive feedback on low priorities for test cases regarded as unimportant.
    This risks encouraging low priorities for test cases which would have failed if executed,
    and could encourage undesired behavior,
    but at the same time it strengthens the influence all priorities in the test suite have.
    """

    def __str__(self):
        return 'Failure Count Reward'

    def evaluate(self, reward: EvaluationMetric, last_prioritization: List[str]):
        return [float(reward.detected_failures)] * len(last_prioritization)


class TCFailReward(Reward):
    """
    This reward function returns the test case's verdict as each test case's individual reward.
    Scheduling failing test cases is intended and therefore reinforced.
    If a test case passed, no specific reward is given as including it
    neither improved nor reduced the schedule's quality according to available information.
    Still, the order of test cases is not explicitly included in the reward.
    It is implicitly included by encouraging the agent to focus on failing test cases and prioritizing them higher.
    """

    def __str__(self):
        return 'Test Case Failure Reward'

    def get_name(self):
        return 'tcfail'

    def evaluate(self, reward: EvaluationMetric, last_prioritization: List[str]):
        total = reward.detected_failures

        if total == 0:
            return [0.0] * len(last_prioritization)

        rank_idx = np.array(reward.detection_ranks) - 1
        no_scheduled = len(reward.scheduled_testcases)

        rewards = np.zeros(no_scheduled)
        rewards[rank_idx] = 1

        ordered_rewards = []

        for tc in last_prioritization:
            if tc in reward.scheduled_testcases:
                idx = reward.scheduled_testcases.index(tc)
                ordered_rewards.append(rewards[idx])
            else:
                ordered_rewards.append(0.0)  # Unscheduled test case

        return ordered_rewards


class TimeRankReward(Reward):
    """
    This reward function explicitly includes the order of test cases and rewards each test case
    based on its rank in the test schedule and whether it failed.
    As a good schedule executes failing test cases early,
    every passed test case reduces the schedule's quality if it precedes a failing test case.
    Each test cases is rewarded by the total number of failed test cases,
    for failed test cases it is the same as reward function 'TCFailReward'.
    For passed test cases, the reward is further decreased by the number of failed test cases ranked
    after the passed test case to penalize scheduling passing test cases early.
    """

    def __str__(self):
        return 'Time-ranked Reward'

    def get_name(self):
        return 'timerank'

    def evaluate(self, reward: EvaluationMetric, last_prioritization: List[str]):
        # number of test cases which failed
        detected_failures = len(reward.detection_ranks)

        if detected_failures == 0:
            return [0.0] * len(last_prioritization)

        rank_idx = np.array(reward.detection_ranks) - 1
        no_scheduled = len(reward.scheduled_testcases)

        rewards = np.zeros(no_scheduled)
        rewards[rank_idx] = 1
        rewards = np.cumsum(rewards)  # Rewards for passed testcases
        rewards[rank_idx] = detected_failures  # Rewards for failed testcases

        ordered_rewards = []

        for tc in last_prioritization:
            if tc in reward.scheduled_testcases:
                idx = reward.scheduled_testcases.index(tc)  # Slow call
                ordered_rewards.append(rewards[idx])
            else:
                ordered_rewards.append(0.0)  # Unscheduled test case
        return ordered_rewards


class RNFailReward(Reward):
    """
    Reward Based on Failures (RNFail)

    This reward function is based on the number of failures associated with test cases t' in T':
    1 if t' failed; 0 otherwise
    """

    def __str__(self):
        return 'Reward Based on Failures'

    def get_name(self):
        return 'RNFail'

    def evaluate(self, reward: EvaluationMetric, last_prioritization: List[str]):
        total = reward.detected_failures

        if total == 0:
            return [0.0] * len(last_prioritization)

        rank_idx = np.array(reward.detection_ranks) - 1
        no_scheduled = len(reward.scheduled_testcases)

        rewards = np.zeros(no_scheduled)
        rewards[rank_idx] = 1

        ordered_rewards = []

        for tc in last_prioritization:
            if tc in reward.scheduled_testcases:
                idx = reward.scheduled_testcases.index(tc)
                ordered_rewards.append(rewards[idx])
            else:
                ordered_rewards.append(0.0)  # Unscheduled test case

        return ordered_rewards


class RRankReward(Reward):
    """
    Reward Based on Rank (RRank)

    This reward function is based on the rank of t' in T'. The idea is to evaluate whether failed test cases,
    with a greater number of failures, ar ranked in the first positions in T'. To this end,
    a test case t that does not fail and precedes failed test cases are penalized by their early scheduling.
    In this way, RRank has two components, the first one RRankPos is based on the position of t' in the rank
    regarding the failed test cases, and the second one RRankFailures is based on the number of failures of t'.
    """

    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        return 'Reward Based on Rank'

    def get_name(self):
        return 'RRank'

    def RRankPos(self, no_scheduled, rank_idx, detected_failures):
        """

        :param no_scheduled: Number of test cases executed
        :param rank_idx: Test cases indexes which failed
        :param detected_failures: Number of test cases which failed
        :return:
        """
        rewards = np.zeros(no_scheduled)
        rewards[rank_idx] = 1
        rewards = np.cumsum(rewards)  # Rewards for passed testcases
        rewards[rank_idx] = detected_failures  # Rewards for failed testcases

        # Normalize
        rewards = rewards / detected_failures

        return rewards

    def RRankFailures(self, no_scheduled, rank_idx, failures):
        """

        :param no_scheduled: Number of test cases executed
        :param rank_idx: Test cases indexes which failed
        :param failures: Number of failures revealed by each test case
        :return:
        """
        rewards = np.zeros(no_scheduled)
        rewards[rank_idx] = failures

        # Normalize
        rewards = rewards / sum(failures)

        return rewards

    def evaluate(self, reward: EvaluationMetric, last_prioritization: List[str]):
        # number of test cases which failed
        detected_failures = len(reward.detection_ranks)

        if detected_failures == 0:
            return [0.0] * len(last_prioritization)

        rank_idx = np.array(reward.detection_ranks) - 1
        no_scheduled = len(reward.scheduled_testcases)

        rrankpos = self.RRankPos(no_scheduled, rank_idx, detected_failures)
        rrankfailures = self.RRankFailures(
            no_scheduled, rank_idx, reward.detection_ranks_failures)

        rewards = self.alpha * rrankpos + self.beta * rrankfailures

        ordered_rewards = []

        for tc in last_prioritization:
            if tc in reward.scheduled_testcases:
                idx = reward.scheduled_testcases.index(tc)  # Slow call
                ordered_rewards.append(rewards[idx])
            else:
                ordered_rewards.append(0.0)  # Unscheduled test case
        return ordered_rewards


class SimpleDiscreteReward(Reward):
    def __str__(self):
        return 'Simple Discrete Reward'

    def evaluate(self, reward: EvaluationMetric, last_prioritization=None):
        if reward.undetected_failures > 0:
            scenario_reward = -1.0
        elif reward.detected_failures > 0:
            scenario_reward = 1.0
        else:
            scenario_reward = 0.0

        return scenario_reward


class SimpleContinuousReward(Reward):
    def __str__(self):
        return 'Simple Continuous Reward'

    def evaluate(self, reward: EvaluationMetric, last_prioritization=None):
        failure_count = reward.detected_failures + reward.undetected_failures

        if reward.undetected_failures > 0:
            scenario_reward = -1.0  # - (3*result[1]/failure_count)
        elif reward.detected_failures > 0:
            scenario_reward = 1.0 + (3.0 - 3.0 * reward.ttf)
        else:
            scenario_reward = 0.0

        return scenario_reward * len(last_prioritization)


class NAPFDReward(Reward):
    def __str__(self):
        return 'NAPFD Reward'

    def evaluate(self, reward: EvaluationMetric, last_prioritization=None):
        total_failures = reward.detected_failures + reward.undetected_failures
        scaling_factor = 1.0

        result = 0.0

        if total_failures == 0:
            result = 0.0
        elif reward.detected_failures == 0:
            result = -1 * scaling_factor
        else:
            # Apply NAPFD
            result = reward.fitness * scaling_factor

        return [result] * len(last_prioritization)


class ShiftedNAPFDReward(Reward):
    def __str__(self):
        return 'Shifted NAPFD Reward'

    def evaluate(self, reward: EvaluationMetric, last_prioritization=None):
        total_failures = reward.detected_failures + reward.undetected_failures
        scaling_factor = 1.0

        result = 0.0
        if total_failures == 0:
            result = 0.0
        elif reward.detected_failures == 0:
            result = -1.0 * scaling_factor
        elif reward.fitness < 0.3:
            result = reward.fitness - 0.3
        else:
            # Apply NAPFD
            result = reward.fitness * scaling_factor

        return [result] * len(last_prioritization)


class BinaryPositiveDetectionReward(Reward):
    def __str__(self):
        return 'Binary Positive Detection Reward'

    def evaluate(self, reward: EvaluationMetric, last_prioritization=None):
        rew = 1 if reward.detected_failures > 0 else 0
        return [float(rew)] * len(last_prioritization)
