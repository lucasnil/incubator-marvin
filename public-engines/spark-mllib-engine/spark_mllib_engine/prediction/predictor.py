#!/usr/bin/env python
# coding=utf-8

"""Predictor engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger
from ..spark_serializer import SparkSerializer

from marvin_python_toolbox.engine_base import EngineBasePrediction

__all__ = ['Predictor']


logger = get_logger('predictor')


class Predictor(SparkSerializer, EngineBasePrediction):

    def __init__(self, **kwargs):
        super(Predictor, self).__init__(**kwargs)

    def execute(self, input_message, params, **kwargs):
        # Predictor

        from pyspark.sql import SparkSession
        from pyspark.sql.types import FloatType
        from pyspark.sql.types import StructType, StructField
        from pyspark.ml.feature import VectorAssembler

        import findspark

        findspark.init()


        # Building SparkSession
        spark = SparkSession.builder \
            .master("local") \
            .appName("Spark MLlib") \
            .config("spark.executor.memory", "1gb") \
            .getOrCreate()

        sc = spark.sparkContext


        field = [StructField("SepalLengthCm", FloatType(), True), StructField("SepalWidthCm", FloatType(), True),
                 StructField("PetalLengthCm", FloatType(), True), StructField("PetalWidthCm", FloatType(), True)]

        input_schema = StructType(field)

        input_message = [input_message]

        input_message = spark.createDataFrame(input_message, schema=input_schema)

        colunas = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

        for coluna in colunas:
            input_message = input_message.withColumn(coluna, input_message[coluna].cast(FloatType()))

        assembler = VectorAssembler(inputCols=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], outputCol="features")
        input_message = assembler.transform(input_message)


        final_prediction = self.marvin_model.transform(input_message)

        final_prediction = final_prediction.select('prediction').collect()[0][0]

        return final_prediction
