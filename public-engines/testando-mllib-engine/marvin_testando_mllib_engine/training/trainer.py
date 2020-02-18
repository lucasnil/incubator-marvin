#!/usr/bin/env python
# coding=utf-8

"""Trainer engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger
from ..spark_serializer import SparkSerializer

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['Trainer']


logger = get_logger('trainer')


class Trainer(SparkSerializer, EngineBaseTraining):

    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        # Model Training
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import NaiveBayes

        import findspark

        findspark.init()


        nb = NaiveBayes()

        pipeline = Pipeline().setStages([nb])

        self.marvin_model = pipeline.fit(self.marvin_dataset['train'])

