# Databricks notebook source
# MAGIC %md
# MAGIC # Centralized Feature Store example
# MAGIC 
# MAGIC In this notebook, you create a feature table in a remote Feature Store workspace (workspace B). Then, you use the remote feature table to train a model and register the model to a Model Registry in a different remote workspace (workspace C).
# MAGIC 
# MAGIC ## Notebook setup
# MAGIC 1. In the workspace where the feature table will be created (workspace B), create an access token.
# MAGIC 2. In the current workspace, create secrets and store the access token and the remote workspace information. The easiest way is to use the Databricks CLI, but you can also use the Secrets REST API.
# MAGIC   First, Here Databricks CLI is used in Azure Cloud Shell of Azure portal.
# MAGIC   * `pip install databricks-cli`
# MAGIC   * `databricks configure --token`
# MAGIC   *  Databricks Host (should begin with https://) :`https://adb-<workspace-id>.<random-number>.azuredatabricks.net`
# MAGIC   *  Token:`<Databricks personal access token>`
# MAGIC   
# MAGIC   Then the scope and keys are created in this Databricks environment .
# MAGIC   
# MAGIC   1. Create a secret scope: `databricks secrets create-scope --scope <scope>`.
# MAGIC   2. Pick a unique name for the target workspace, which we'll refer to as `<prefix>`. Then create three secrets:
# MAGIC     * `databricks secrets put --scope <scope> --key <prefix>-host`. Enter the hostname of the feature store workspace (workspace B).
# MAGIC     * `databricks secrets put --scope <scope> --key <prefix>-token`. Enter the access token from the feature store workspace.
# MAGIC     * `databricks secrets put --scope <scope> --key <prefix>-workspace-id`. Enter the workspace ID for the feature store workspace which can be found in the URL of any page in the workspace.
# MAGIC 
# MAGIC **Before you run this notebook, enter the secret scope and key prefix corresponding to the remote feature store workspace (workspace B) in the notebook parameter fields above.**

# COMMAND ----------

dbutils.widgets.text('feature_store_secret_scope', '')
dbutils.widgets.text('feature_store_secret_key_prefix', '')

dbutils.widgets.text('model_registry_secret_scope', '')
dbutils.widgets.text('model_registry_secret_key_prefix', '')

fs_scope = str(dbutils.widgets.get('feature_store_secret_scope'))
fs_key = str(dbutils.widgets.get('feature_store_secret_key_prefix'))

mr_scope = str(dbutils.widgets.get('model_registry_secret_scope'))
mr_key = str(dbutils.widgets.get('model_registry_secret_key_prefix'))

feature_store_uri = f'databricks://{fs_scope}:{fs_key}' if fs_scope and fs_key else None
model_registry_uri = f'databricks://{mr_scope}:{mr_key}' if mr_scope and mr_key else None

# COMMAND ----------

feature_store_uri

# COMMAND ----------

model_registry_uri

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Table setup
# MAGIC In this step, you create the database for the feature table and create a data frame, `features_df`, that will be used to create the remote feature table.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS feature_store_multi_workspace;

# COMMAND ----------

from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    FloatType,
)

feature_table_name = "feature_store_multi_workspace.feature_table"

feature_table_schema = StructType(
    [
        StructField("user_id", IntegerType(), False),
        StructField("user_feature", FloatType(), True),
    ]
)

features_df = spark.createDataFrame(
    [
        (123, 100.2),
        (456, 12.4),
    ],
    feature_table_schema,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a remote feature table 
# MAGIC The API call to create a remote feature table depends on the version of Databricks Runtime for ML on your cluster.  
# MAGIC - With Databricks Runtime 10.2 ML or above, use `FeatureStoreClient.create_table`.  
# MAGIC - With Databricks Runtime 10.1 ML or below, use `FeatureStoreClient.create_feature_table`.  
# MAGIC 
# MAGIC In this step, you create the feature table in the remote workspace (Workspace B).

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient(feature_store_uri=feature_store_uri, model_registry_uri=model_registry_uri)

# COMMAND ----------

# Use this command with Databricks Runtime 10.2 ML or above
fs.create_table(
    feature_table_name,
    primary_keys="user_id",
    df=features_df,
    description="Sample feature table",
)

# COMMAND ----------

# To run this notebook with Databricks Runtime 10.1 ML or below, uncomment and run this cell

#fs.create_feature_table(
#    feature_table_name,
#    "user_id",
#    features_df=features_df,
#    description="Sample feature table",
#)

# COMMAND ----------

# MAGIC %md
# MAGIC You should be able to see the new feature table in the feature store workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read from a remote feature table to train a model

# COMMAND ----------

import mlflow

class SampleModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return model_input.sum(axis=1, skipna=False)

# COMMAND ----------

record_table_schema = StructType(
    [
        StructField("id", IntegerType(), False),
        StructField("income", IntegerType(), False),
    ]
)

record_table = spark.createDataFrame(
    [
        (123, 10000),
        (456, 20000),
        (789, 30000),
    ],
    record_table_schema,
)

# COMMAND ----------

from databricks.feature_store import FeatureLookup

feature_lookups = [
    FeatureLookup(
        table_name=feature_table_name,
        feature_names="user_feature",
        lookup_key="id",
    ),
]

# COMMAND ----------

training_set = fs.create_training_set(
    record_table,
    feature_lookups=feature_lookups,
    exclude_columns=["id"],
    label="income",
)

# Load the TrainingSet. load_df() returns a dataframe that can be passed into sklearn for training a model
training_df = training_set.load_df()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model with a remote Model Registry (workspace C)

# COMMAND ----------

with mlflow.start_run() as new_run:
  fs.log_model(
      SampleModel(),
      artifact_path="model",
      flavor=mlflow.pyfunc,
      training_set=training_set,
      registered_model_name="multi_workspace_fs_model",
  )

# COMMAND ----------

# MAGIC %md
# MAGIC At this point, you should be able to see the new model version in the remote model registry workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use model in remote Model Registry for batch inference

# COMMAND ----------

# Get the model URI
model_uri = f"models:/multi_workspace_fs_model/1"

# Call score_batch to get the predictions from the model
with_predictions = fs.score_batch(model_uri, record_table.drop("income"))

# COMMAND ----------

with_predictions.head(5)