#!/usr/bin/env python
# coding=utf-8

"""AcquisitorAndCleaner engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger
from ..spark_serializer import SparkSerializer

from marvin_python_toolbox.engine_base import EngineBaseDataHandler

__all__ = ['AcquisitorAndCleaner']


logger = get_logger('acquisitor_and_cleaner')


class AcquisitorAndCleaner(SparkSerializer, EngineBaseDataHandler):

    def __init__(self, **kwargs):
        super(AcquisitorAndCleaner, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        # Data Acquisitor
        from marvin_python_toolbox.common.data import MarvinData
        from pyspark.sql import SparkSession
        import findspark
        import tempfile
        import numpy as np
        findspark.init()


        # Building SparkSession
        spark = SparkSession.builder.master("local[*]").getOrCreate()

        sc = spark.sparkContext


        file_path = MarvinData.download_file(url="https://s3.amazonaws.com/marvin-engines-data/Iris.csv")

        iris_df = spark.read.csv(file_path, header="true")

        self.marvin_initial_dataset = iris_df.drop("Id")

