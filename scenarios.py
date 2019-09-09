import json
import os

import pandas as pd


class VirtualScenario(object):
    def __init__(self, available_time, testcases, build_id, total_build_duration):
        self.available_time = available_time
        self.testcases = testcases
        self.build_id = build_id
        self.total_build_duration = total_build_duration
        self.reset()

    def reset(self):
        # Reset the priorities
        for row in self.get_testcases():
            row['CalcPrio'] = 0

    def get_available_time(self):
        return self.available_time

    def get_testcases(self):
        return self.testcases

    def get_testcases_names(self):
        return [row['Name'] for row in self.get_testcases()]

    def get_scenario_metadata(self):
        """
        Get scenario metadata
        :return:
        """
        execTimes, durations = zip(*[(tc['LastRun'], tc['Duration']) for tc in self.testcases()])

        metadata = {
            'availAgents': 1,
            'totalTime': self.available_time,
            'minExecTime': min(execTimes),
            'maxExecTime': max(execTimes),
            'scheduleDate': self.schedule_date,
            'minDuration': min(durations),
            'maxDuration': max(durations)
        }

        return metadata


class IndustrialDatasetScenarioProvider():
    """
    Scenario provider to process CSV files for experimental evaluation.
    Required columns are `self.tc_fieldnames`
    """

    def __init__(self, tcfile, sched_time_ratio=0.5):
        self.name = os.path.split(os.path.dirname(tcfile))[1]

        self.tcdf = pd.read_csv(tcfile, sep=';', parse_dates=['LastRun'])

        if self.name not in ['iofrol', 'paintcontrol', 'gsdtsr']:
            self.tcdf['LastResults'] = self.tcdf['LastResults'].apply(json.loads)
        self.tcdf["Duration"] = self.tcdf["Duration"].apply(lambda x: float(x.replace(',', '')) if type(x) == str else x)

        self.build = 0
        self.max_builds = max(self.tcdf.BuildId)
        self.scenario = None
        self.avail_time_ratio = sched_time_ratio
        self.total_build_duration = 0

        # ColName | Description
        # Name | Unique numeric identifier of the test case
        # Duration | Approximated runtime of the test case
        # CalcPrio | Priority of the test case, calculated by the prioritization algorithm(output column, initially 0)
        # LastRun | Previous last execution of the test case as date - time - string(Format: `YYYY - MM - DD HH: ii`)
        # NumRan | Test runs
        # NumErrors | Test errors revealed
        # LastResults | List of previous test results(Failed: 1, Passed: 0), ordered by ascending age

        if self.name in ['iofrol', 'paintcontrol', 'gsdtsr']:
            self.tc_fieldnames = ['Name', 'Duration', 'CalcPrio', 'LastRun', 'Verdict', 'LastResults']
        else:
            self.tc_fieldnames = ['Name', 'Duration', 'CalcPrio', 'LastRun', 'NumRan', 'NumErrors', 'Verdict',
                              'LastResults']

    def __str__(self):
        return self.name

    def last_build(self, build):
        self.build = build

    def get(self):
        """
        This function is called when the __next__ function is called.
        In this function the data is "separated" by builds. Each next build is returned.
        :return:
        """
        self.build += 1

        # Stop when reaches the max build
        if self.build > self.max_builds:
            self.scenario = None
            return None

        # Select the data for the current build
        builddf = self.tcdf.loc[self.tcdf.BuildId == self.build]

        # Convert the solutions to a list of dict
        seltc = builddf[self.tc_fieldnames].to_dict(orient='record')

        self.total_build_duration = builddf['Duration'].sum()
        total_time = self.total_build_duration * self.avail_time_ratio

        # This test set is a "scenario" that must be evaluated.
        self.scenario = VirtualScenario(testcases=seltc,
                                        available_time=total_time,
                                        build_id=self.build,
                                        total_build_duration = self.total_build_duration)

        return self.scenario

    # Generator functions
    def __iter__(self):
        return self

    def __next__(self):
        sc = self.get()

        if sc is None:
            raise StopIteration()

        return sc
