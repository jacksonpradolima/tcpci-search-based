class EvaluationMetric(object):
    """
    Evaluation Metric
    """

    def __init__(self, column_key='NumErrors'):
        self.available_time = 0
        self.reset()
        self.column_key = column_key

    def update_available_time(self, available_time):
        self.available_time = available_time

    def reset(self):
        self.scheduled_testcases = []
        self.unscheduled_testcases = []
        self.detection_ranks = []
        self.detection_ranks_time = []
        self.detection_ranks_failures = []
        # Time to Fail (rank value)
        self.ttf = 0
        self.ttf_duration = 0
        # APFD or NAPFD value
        self.fitness = 0
        self.detected_failures = 0
        self.undetected_failures = 0
        self.recall = 0
        self.avg_precision = 0

    def evaluate(self, test_suite):
        raise NotImplementedError('This method must be override')

class NAPFDMetric(EvaluationMetric):
    """
    Normalized Average Percentage of Faults Detected (NAPFD) Metric based on NumErrors
    """

    def __init__(self, column_key='NumErrors'):
        super().__init__(column_key)

    def __str__(self):
        return 'NAPFD'

    def evaluate(self, test_suite):
        super().reset()

        rank_counter = 1
        total_failure_count = 0
        scheduled_time = 0
        self.detected_failures = 0

        # Build prefix sum of durations to find cut off point
        for row in test_suite:
            total_failure_count += row[self.column_key]
            if scheduled_time + row['Duration'] <= self.available_time:
                # Time spent to fail
                if len(self.detection_ranks_time) == 0:
                    self.ttf_duration += row['Duration']

                # If the Verdict is "Failed"
                if row[self.column_key] > 0:
                    self.detected_failures += row[self.column_key]
                    self.detection_ranks.append(rank_counter)

                    # Individual information
                    self.detection_ranks_failures.append(row[self.column_key])
                    self.detection_ranks_time.append(row['Duration'])

                scheduled_time += row['Duration']
                self.scheduled_testcases.append(row['Name'])
                rank_counter += 1
            else:
                self.unscheduled_testcases.append(row['Name'])
                self.undetected_failures += row[self.column_key]

        if total_failure_count > 0:
            # Time to Fail (rank value)
            self.ttf = self.detection_ranks[0] if self.detection_ranks else 0
            self.avg_precision = 123
            self.recall = self.detected_failures / total_failure_count
            # NAPFD
            p = self.recall if self.undetected_failures > 0 else 1
            no_testcases = len(test_suite)
            self.fitness = p - sum(self.detection_ranks) / (total_failure_count * no_testcases) + p / (
                        2 * no_testcases)
        else:
            # Time to Fail (rank value)
            self.ttf = -1
            # NAPFD
            self.fitness = 1
            self.recall = 1
            self.avg_precision = 1