#!/usr/bin/env python
# coding=utf-8

"""MetricsEvaluator engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger
from ..spark_serializer import SparkSerializer

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['MetricsEvaluator']


logger = get_logger('metrics_evaluator')


class MetricsEvaluator(SparkSerializer, EngineBaseTraining):

    def __init__(self, **kwargs):
        super(MetricsEvaluator, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        # Model Evaluation

        from pyspark.ml.evaluation import MulticlassClassificationEvaluator

        import findspark

        findspark.init()


        predictions = self.marvin_model.transform(self.marvin_dataset['test'])

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                      metricName="accuracy")
        self.marvin_metrics = evaluator.evaluate(predictions)

