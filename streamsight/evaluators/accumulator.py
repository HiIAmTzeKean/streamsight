import logging
from collections import defaultdict
from logging import warn
from typing import Optional

import pandas as pd
from deprecation import deprecated

from streamsight.evaluators.util import MetricLevelEnum
from streamsight.metrics import Metric

logger = logging.getLogger(__name__)

class MetricAccumulator():
    def __init__(self):
        self.acc: defaultdict[str, dict[str, Metric]] = defaultdict(dict)

    def __getitem__(self, key):
        return self.acc[key]

    def add(self, metric: Metric, algorithm_name: str) -> None:
        """Add a metric to the accumulator
        
        Takes a :class:`Metric` object and adds it under the algorithm name. If
        the specified metric already exists for the algorithm, it will be
        overwritten with the new metric.

        :param metric: Metric to store
        :type metric: Metric
        :param algorithm_name: Name of the algorithm
        :type algorithm_name: str
        """
        if metric.identifier in self.acc[algorithm_name]:
            warn(
                f"Metric {metric.identifier} already exists for algorithm {algorithm_name}. Overwriting...")

        logger.debug(
            f"Metric {metric.identifier} created for algorithm {algorithm_name}")

        self.acc[algorithm_name][metric.identifier] = metric

    @property
    def user_level_metrics(self):
        results = defaultdict()
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                metric = self.acc[algo_name][metric_identifier]
                results[(algo_name, f"t={metric.timestamp_limit}", metric.name)] = metric.micro_result
        return results

    @property
    def window_level_metrics(self) -> defaultdict:
        results = defaultdict(dict)
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                metric = self.acc[algo_name][metric_identifier]
                results[(algo_name, f"t={metric.timestamp_limit}", metric.name)]["score"] = metric.macro_result
                results[(algo_name, f"t={metric.timestamp_limit}", metric.name)]["num_user"] = metric.num_users
        return results

    @deprecated(details="Use `window_level_metrics` instead")
    @property
    def metrics(self) -> defaultdict:
        results = defaultdict(dict)
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                metric = self.acc[algo_name][metric_identifier]
                results[(algo_name, f"t={metric.timestamp_limit}", metric.name)]["score"] = metric.macro_result
                results[(algo_name, f"t={metric.timestamp_limit}", metric.name)]["num_user"] = metric.num_users
        return results

    def df_user_level_metric(self) -> pd.DataFrame:
        """User metric across all timestamps
        
        Computation of metrics evaluated on the user level

        :return: _description_
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(self.user_level_metrics, orient="index").explode(["user_id","score"])
        df = df.rename_axis(["Algorithm", "Timestamp", "Metric"])
        return df

    def df_window_level_metric(self) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(self.window_level_metrics, orient="index").explode(["score","num_user"])
        df = df.rename_axis(["Algorithm", "Timestamp", "Metric"])
        df.rename(columns={"score":"window_score"}, inplace=True)
        return df

    def df_macro_level_metric(self) -> pd.DataFrame:
        """Macro metric across all timestamps

        :return: _description_
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(self.window_level_metrics, orient="index").explode(["score","num_user"])
        df = df.rename_axis(["Algorithm", "Timestamp", "Metric"])
        result = df.groupby(["Algorithm", "Metric"]).mean()["score"].to_frame()
        result["num_window"] = df.groupby(["Algorithm", "Metric"]).count()["score"]
        result = result.rename(columns={"score": "macro_score"})
        return result

    def df_micro_level_metric(self) -> pd.DataFrame:
        """Micro metric across all timestamps

        :return: _description_
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(self.user_level_metrics, orient="index").explode(["user_id","score"])
        df = df.rename_axis(["Algorithm", "Timestamp", "Metric"])
        result =  df.groupby(["Algorithm", "Metric"]).mean()["score"].to_frame()
        result["num_user"] = df.groupby(["Algorithm", "Metric"]).count()["score"]
        result = result.rename(columns={"score": "micro_score"})
        return result

    def df_metric(
        self,
        filter_timestamp: Optional[int] = None,
        filter_algo: Optional[str] = None,
        level: MetricLevelEnum = MetricLevelEnum.MACRO,
    ) -> pd.DataFrame:
        """Dataframe representation of the metric

        Returns a dataframe representation of the metric. The dataframe can be
        filtered based on the algorithm name and the timestamp.

        :param filter_timestamp: Timestamp value to filter on, defaults to None
        :type filter_timestamp: Optional[int], optional
        :param filter_algo: Algorithm name to filter on, defaults to None
        :type filter_algo: Optional[str], optional
        :param level: Level of the metric to compute, defaults to MetricLevelEnum.MACRO
        :type level: MetricLevelEnum, optional
        :return: Dataframe representation of the metric
        :rtype: pd.DataFrame
        """
        if level == MetricLevelEnum.MACRO:
            df = self.df_macro_level_metric()
        elif level == MetricLevelEnum.MICRO:
            df = self.df_micro_level_metric()
        elif level == MetricLevelEnum.WINDOW:
            df = self.df_window_level_metric()
        elif level == MetricLevelEnum.USER:
            df = self.df_user_level_metric()
        else:
            raise ValueError("Invalid level specified")

        if filter_algo:
            df = df.filter(like=filter_algo, axis=0)
        if filter_timestamp:
            df = df.filter(like=f"t={filter_timestamp}", axis=0)
        return df
