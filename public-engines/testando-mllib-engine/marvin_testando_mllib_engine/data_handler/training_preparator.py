#!/usr/bin/env python
# coding=utf-8

"""TrainingPreparator engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger
from ..spark_serializer import SparkSerializer

from marvin_python_toolbox.engine_base import EngineBaseDataHandler

__all__ = ['TrainingPreparator']


logger = get_logger('training_preparator')


class TrainingPreparator(SparkSerializer,EngineBaseDataHandler):

    def __init__(self, **kwargs):
        super(TrainingPreparator, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        # Training Preparator

        from pyspark.sql.types import DoubleType
        from pyspark.ml.feature import StringIndexer
        from pyspark.ml.feature import VectorAssembler
        from pyspark.mllib.regression import LabeledPoint

        l_atributos = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

        dataset = self.marvin_initial_dataset


        # Chaniging atribute types to double
        for coluna in l_atributos:
            dataset = dataset.withColumn(coluna, dataset[coluna].cast(DoubleType()))


        # Maping column "Species" to a numerical value in a new collumn named "label"
        label_indexer = StringIndexer().setInputCol("Species").setOutputCol("label")

        dataset = label_indexer.fit(dataset).transform(dataset)


        # Concatenating all features into a single vector and naming the resulting column as "features"
        assembler = VectorAssembler(inputCols=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], outputCol="features")
        dataset = assembler.transform(dataset)


        (train, test) = dataset.randomSplit([0.7, 0.3])

        self.marvin_dataset = {'train': train, 'test': test}

