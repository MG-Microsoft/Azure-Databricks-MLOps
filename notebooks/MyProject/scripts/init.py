# Databricks notebook source
import mlflow
import mlflow.spark
import mlflow.azureml
from pyspark.sql.functions import lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from azureml.core import Workspace
from azureml.mlflow import get_portal_url
from azureml.core.authentication import ServicePrincipalAuthentication
import mlflow.azureml

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# COMMAND ----------

def service_principal_auth(tenant_id, sp_id, sp_secret):
  return ServicePrincipalAuthentication(
      tenant_id=tenant_id,
      service_principal_id=sp_id,
      service_principal_password=sp_secret)

# COMMAND ----------

def load_data(size):
  random.seed(42)
  data = [[random.uniform(1, 50000), random.uniform(-10,10)] for i in range(size)]
  columns = ['mile','noise']
  trainData, testData = (spark
        .createDataFrame(data)
        .toDF(*columns)
        .selectExpr("mile", "(mile * 1.60934) * (1 + noise /100) as km")
        .randomSplit([.8, .2], seed=42)
       )
  return trainData, testData

# COMMAND ----------

def load_tf_dataset(trainDF, testDF):
  #trainDF, testDF = load_data(size)
  # Prepare data
  train = trainDF.toPandas()
  train_x,train_y = train, train.pop('km')
  test = testDF.toPandas()
  test_x,test_y = test,test.pop('km')

  # Prepare the training dataset
  train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(128)

  # Prepare the validation dataset
  test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
  test_dataset = test_dataset.batch(128)
  return train_dataset, test_dataset

# COMMAND ----------

def evaluation_plot(model, valDF):
  # Show plot
  valData = valDF.limit(1000)
  predictions = model.transform(valData).select("mile", "km", "prediction").toPandas()

  plt.scatter(predictions["mile"], predictions["km"], color='g')
  plt.scatter(predictions["mile"],predictions["prediction"], color='r')
  #plt.show()
  plt.savefig("eval.png")

# COMMAND ----------

def loss_rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 