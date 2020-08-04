import os
import shutil
import findspark
import pickle
import pandas as pd
import tempfile
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession


findspark.init()

from pyspark.sql import SparkSession

# Building SparkSession
spark = SparkSession.builder \
   .master("local") \
   .appName("Spark MLlib") \
   .config("spark.executor.memory", "1gb") \
   .getOrCreate()

sc = spark.sparkContext

spark.conf.set("spark.sql.execution.arrow.enabled", "true")

class SparkSerializer(object):

    def _serializer_load(self, object_file_path):
        if object_file_path.split(os.sep)[-1] == 'model':
            model = PipelineModel.load(object_file_path)
            return model
        elif object_file_path.split(os.sep)[-1] == 'initialdataset':
            pickleRdd = sc.pickleFile(object_file_path).collect()
            df = spark.createDataFrame(pickleRdd)
            return df
        elif object_file_path.split(os.sep)[-1] == 'dataset':
            with open(object_file_path, 'rb') as handle:
                data_dict = pickle.load(handle)
            data_dict['train'] = spark.createDataFrame(data_dict['train'])
            data_dict['test'] = spark.createDataFrame(data_dict['test'])
            return data_dict
        

    def _serializer_dump(self, obj, object_file_path):
        if object_file_path.split(os.sep)[-1] == 'model':
            if os.path.isdir(object_file_path):
                shutil.rmtree(object_file_path)
            obj.save(object_file_path)
        elif object_file_path.split(os.sep)[-1] == 'initialdataset':
            if os.path.isdir(object_file_path):
                shutil.rmtree(object_file_path)
            obj.rdd.saveAsPickleFile(object_file_path)
        elif object_file_path.split(os.sep)[-1] == 'dataset':
            if os.path.isdir(object_file_path):
                shutil.rmtree(object_file_path)
            obj['train'] = obj['train'].select("*").toPandas()
            obj['test'] = obj['test'].select("*").toPandas()
            with open(object_file_path, 'wb') as handle:
                pickle.dump(obj, handle)
        
