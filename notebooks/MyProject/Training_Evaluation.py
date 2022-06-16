# Databricks notebook source
# MAGIC %md # Overview
# MAGIC The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of MLflow Models. It provides model lineage (which MLflow Experiment and Run produced the model), model versioning, stage transitions, annotations, and deployment management.
# MAGIC 
# MAGIC In this notebook, you use each of the MLflow Model Registry's components to develop and manage a production machine learning application. This notebook covers the following topics:
# MAGIC 
# MAGIC - Track and log models with MLflow
# MAGIC - Register models with the Model Registry
# MAGIC - Describe models and make model version stage transitions
# MAGIC - Integrate registered models with production applications
# MAGIC - Search and discover models in the Model Registry
# MAGIC - Archive and delete models
# MAGIC 
# MAGIC ## Requirements
# MAGIC - A cluster running Databricks Runtime 6.4 ML or above. Note that if your cluster is running Databricks Runtime 6.4 ML, you must upgrade the installed version of MLflow to 1.7.0. You can install this version from PyPI. See ([AWS](https://docs.databricks.com/libraries/cluster-libraries.html#cluster-installed-library)|[Azure](https://docs.microsoft.com/azure/databricks/libraries/cluster-libraries#cluster-installed-library)) for instructions. 

# COMMAND ----------

# MAGIC %md # Machine learning application: Forecasting wind power
# MAGIC 
# MAGIC In this notebook, you use the MLflow Model Registry to build a machine learning application that forecasts the daily power output of a [wind farm](https://en.wikipedia.org/wiki/Wind_farm). Wind farm power output depends on weather conditions: generally, more energy is produced at higher wind speeds. Accordingly, the machine learning models used in the notebook predict power output based on weather forecasts with three features: `wind direction`, `wind speed`, and `air temperature`.

# COMMAND ----------

# MAGIC %md *This notebook uses altered data from the [National WIND Toolkit dataset](https://www.nrel.gov/grid/wind-toolkit.html) provided by NREL, which is publicly available and cited as follows:*
# MAGIC 
# MAGIC *Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. Overview and Meteorological Validation of the Wind Integration National Dataset Toolkit (Technical Report, NREL/TP-5000-61740). Golden, CO: National Renewable Energy Laboratory.*
# MAGIC 
# MAGIC *Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. "The Wind Integration National Dataset (WIND) Toolkit." Applied Energy 151: 355366.*
# MAGIC 
# MAGIC *Lieberman-Cribbin, W., C. Draxl, and A. Clifton. 2014. Guide to Using the WIND Toolkit Validation Code (Technical Report, NREL/TP-5000-62595). Golden, CO: National Renewable Energy Laboratory.*
# MAGIC 
# MAGIC *King, J., A. Clifton, and B.M. Hodge. 2014. Validation of Power Output for the WIND Toolkit (Technical Report, NREL/TP-5D00-61714). Golden, CO: National Renewable Energy Laboratory.*

# COMMAND ----------

# MAGIC %md ## Load the dataset
# MAGIC 
# MAGIC The following cells load a dataset containing weather data and power output information for a wind farm in the United States. The dataset contains `wind direction`, `wind speed`, and `air temperature` features sampled every eight hours (once at `00:00`, once at `08:00`, and once at `16:00`), as well as daily aggregate power output (`power`), over several years.

# COMMAND ----------

# I am changing the training code for a better version

# COMMAND ----------

import pandas as pd
wind_farm_data = pd.read_csv("https://github.com/dbczumar/model-registry-demo-notebook/raw/master/dataset/windfarm_data.csv", index_col=0)

def get_training_data():
  training_data = pd.DataFrame(wind_farm_data["2014-01-01":"2018-01-01"])
  X = training_data.drop(columns="power")
  y = training_data["power"]
  return X, y

def get_validation_data():
  validation_data = pd.DataFrame(wind_farm_data["2018-01-01":"2019-01-01"])
  X = validation_data.drop(columns="power")
  y = validation_data["power"]
  return X, y

def get_weather_and_forecast():
  format_date = lambda pd_date : pd_date.date().strftime("%Y-%m-%d")
  today = pd.Timestamp('today').normalize()
  week_ago = today - pd.Timedelta(days=5)
  week_later = today + pd.Timedelta(days=5)
  
  past_power_output = pd.DataFrame(wind_farm_data)[format_date(week_ago):format_date(today)]
  weather_and_forecast = pd.DataFrame(wind_farm_data)[format_date(week_ago):format_date(week_later)]
  if len(weather_and_forecast) < 10:
    past_power_output = pd.DataFrame(wind_farm_data).iloc[-10:-5]
    weather_and_forecast = pd.DataFrame(wind_farm_data).iloc[-10:]

  return weather_and_forecast.drop(columns="power"), past_power_output["power"]

# COMMAND ----------

# MAGIC %md Display a sample of the data for reference.

# COMMAND ----------

wind_farm_data["2019-01-01":"2019-01-14"]

# COMMAND ----------

# MAGIC %md # Train a power forecasting model and track it with MLflow
# MAGIC 
# MAGIC The following cells train a neural network to predict power output based on the weather features in the dataset. MLflow is used to track the model's hyperparameters, performance metrics, source code, and artifacts.

# COMMAND ----------

# MAGIC %md Define a power forecasting model using TensorFlow Keras.

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# COMMAND ----------

def train_keras_model(X, y):
  
  model = Sequential()
  model.add(Dense(100, input_shape=(X_train.shape[-1],), activation="relu", name="hidden_layer"))
  model.add(Dense(1))
  model.compile(loss="mse", optimizer="adam")

  model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=.2)
  return model

# COMMAND ----------

# MAGIC %md Train the model and use MLflow to track its parameters, metrics, artifacts, and source code.

# COMMAND ----------

# MAGIC %md
# MAGIC This code set the mflow to model registery workspace (Run the using databricks CLI in Bash and make sure Databricks cli is installed) - you have to do this for all of your Dev, staging and prod workspaces

# COMMAND ----------

databricks configure --token
enter host (with worksapce id start with ?O)
enter token of model dev workspace
databricks secrets create-scope --scope modelregistery
databricks secrets put --scope modelregistery --key modelregistery-token --string-value dapi5d4a1a907559461e73117957709bfbb6-2
databricks secrets put --scope modelregistery --key modelregistery-workspace-id --string-value 8074051404611178
databricks secrets put --scope modelregistery --key modelregistery-host --string-value https://adb-8074051404611178.18.azuredatabricks.net/

# COMMAND ----------

import mlflow

registry_uri = f'databricks://modelregistery:modelregistery'
mlflow.set_registry_uri(registry_uri)


# COMMAND ----------

import mlflow
import mlflow.keras
import mlflow.tensorflow

X_train, y_train = get_training_data()

with mlflow.start_run():
  # Automatically capture the model's parameters, metrics, artifacts,
  # and source code with the `autolog()` function
  mlflow.tensorflow.autolog()
  
  train_keras_model(X_train, y_train)
  run_id = mlflow.active_run().info.run_id

# COMMAND ----------

run_id

# COMMAND ----------

# MAGIC %md # Register the model with the MLflow Model Registry API
# MAGIC 
# MAGIC Now that a forecasting model has been trained and tracked with MLflow, the next step is to register it with the MLflow Model Registry. You can register and manage models using the MLflow UI or the MLflow API .
# MAGIC 
# MAGIC The following cells use the API to register your forecasting model, add rich model descriptions, and perform stage transitions. See the documentation for the UI workflow.

# COMMAND ----------

model_name = "power-forecasting-model" # Replace this with the name of your registered model, if necessary.

# COMMAND ----------

# MAGIC %md ### Create a new registered model using the API
# MAGIC 
# MAGIC The following cells use the `mlflow.register_model()` function to create a new registered model whose name begins with the string `power-forecasting-model`. This also creates a new model version (for example, `Version 1` of `power-forecasting-model`).

# COMMAND ----------

import mlflow

# The default path where the MLflow autologging function stores the model
artifact_path = "model"
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md After creating a model version, it may take a short period of time to become ready. Certain operations, such as model stage transitions, require the model to be in the `READY` state. Other operations, such as adding a description or fetching model details, can be performed before the model version is ready (for example, while it is in the `PENDING_REGISTRATION` state).
# MAGIC 
# MAGIC The following cell uses the `MlflowClient.get_model_version()` function to wait until the model is ready.

# COMMAND ----------

import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

def wait_until_ready(model_name, model_version):
  client = MlflowClient()
  for _ in range(10):
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)
  
wait_until_ready(model_details.name, model_details.version)
