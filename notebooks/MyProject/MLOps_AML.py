
# Databricks notebook source
# MAGIC %md
# MAGIC # Problem Statement - KM Predictor
# MAGIC 
# MAGIC ### In this problem, we create a machine learning model to predict KM based on miles. 
# MAGIC 
# MAGIC **Input:** Mile (float) 
# MAGIC 
# MAGIC **Output:** KM (float)
# MAGIC 
# MAGIC 
# MAGIC Data source: Random sampled numbers between (1, 50000) with random noise.
# MAGIC 
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/tp_ml.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' alt="Traditional Programming vs Machine Learning" width='1000px'>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/KM_achitecture.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' alt="MLOps Architecture" width='600px'>

# COMMAND ----------

# MAGIC %run "./scripts/init"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Step 1 : Model Training
# MAGIC 
# MAGIC **Algorithms**
# MAGIC - Linear Regression
# MAGIC - 2 hidden layer NN
# MAGIC 
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/mlflow_train.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' width='600px'>

# COMMAND ----------

#Prepare data
trainDF, testDF = load_data(5000)
display(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline : Linear Regression

# COMMAND ----------

from azureml.core import Workspace
#test5
subscription_id='f3f672c1-6cfc-4f72-92ae-2b1ab1c0cf69'
tenant_id = '72f988bf-86f1-41af-91ab-2d7cd011db47'
service_principal_clientid = dbutils.secrets.get(scope = "secret", key ="service-principal-client-id") # Service Principal ID
service_principal_secret = dbutils.secrets.get(scope = "secret", key ="service-principal-secret") # Service Principal Secret

# Connect to Azure Machine Learning space
azureml_workspace = Workspace(
       subscription_id=subscription_id,
       resource_group='sparkdemorg',
       workspace_name='sparkml',
       auth=service_principal_auth(tenant_id, service_principal_clientid , service_principal_secret))

experiment_name = "mlflow-demo-new" 
mlflow.set_tracking_uri(azureml_workspace.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)

# COMMAND ----------

with mlflow.start_run(run_name="linear-model") as run:

  mlflow.log_param("trainDataSize", trainDF.count())
  
  # training
  featureCols = [col for col in trainDF.columns if col != 'km']
  vecAssembler = VectorAssembler(inputCols=featureCols, outputCol="features")
  lr = LinearRegression(featuresCol="features", labelCol="km")
  stages = [vecAssembler, lr]
  pipeline = Pipeline(stages=stages)
  model = pipeline.fit(trainDF)
  
  # Log model
  mlflow.spark.log_model(model, "linear", input_example=trainDF.limit(5).toPandas()) 
  
  # Evaulation
  regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="km", metricName="rmse")
  rmse = regressionEvaluator.evaluate(model.transform(testDF))
  mlflow.log_metric("rmse", rmse)
  
  # Plot
  evaluation_plot(model, testDF)
  mlflow.log_artifact("eval.png")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Build a TensorFlow FCN model

# COMMAND ----------

# enable MLflow autolog
mlflow.tensorflow.autolog()

# define model
model = Sequential()
model.add(Dense(8, input_shape=(1,)))
model.add(Dense(24,activation='relu'))
model.add(Dense(1))
# compile the model
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

# prepare data
train_dataset, test_dataset = load_tf_dataset(trainDF, testDF)

# fit the model
history = model.fit(train_dataset, validation_data=test_dataset, epochs=500, verbose=0)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Step 2 : Model Selection
# MAGIC 
# MAGIC In this step, we are selecting the best trained model by **RMSE** recorded in MLflow tracking and register it in **Azure Machine Learning service**
# MAGIC 
# MAGIC 
# MAGIC ### 2.1 Select best model
# MAGIC 
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/mlflow_model_selection.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' width='600px'>

# COMMAND ----------

from mlflow.tracking import MlflowClient

# Create an experiment with a name that is unique and case sensitive.
client = MlflowClient()

experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
run_list = client.search_runs(experiment_id)

end_timestamp = 0
rmse = 9999999
for r in run_list:
  run_info = r.to_dictionary()
  if r.info.status == 'FINISHED' and r.info.end_time > end_timestamp and rmse >= r.data.metrics['rmse']:
    last_run_id = r.info.run_id
    # for errors of not getting rmse, wait a minute then retry. Sometime there is a latency
    rmse = r.data.metrics['rmse']

model_uri = f"runs:/{last_run_id}/model"
print(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Register Model (optional)
# MAGIC 
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/mlflow_model_registry.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' width='600px'>

# COMMAND ----------

# Register the model only
# result = mlflow.register_model(model_uri=model_uri, name='km-predictor-model')

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3 : Model Deployment
# MAGIC 
# MAGIC In this step, model registry will trigger an automatic deployment to production. A published Azure Machine Learning pipeline will be triggered to create a web service using recent registered model.
# MAGIC 
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/mlflow_production.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' width='600px'>

# COMMAND ----------

from mlflow.deployments import get_deploy_client
import json
  
# prepare deploy json file
# Data to be written
deploy_config ={
    "computeType": "aci",
    "containerResourceRequirements": {"cpu": 2, "memoryInGB": 12},
}
# Serializing json 
json_object = json.dumps(deploy_config)
  
# Writing to sample.json
with open("deployment_config.json", "w") as outfile:
    outfile.write(json_object)
    
# set the deployment config
deployment_config_path = "deployment_config.json"
test_config = {'deploy-config-file': deployment_config_path}

# set the tracking uri as the deployment client
client = get_deploy_client(azureml_workspace.get_mlflow_tracking_uri())

# define the model path and the name is the service name
# the model gets registered automatically and a name is autogenerated using the "name" parameter below 
service = client.create_deployment(model_uri=model_uri, config=test_config, name="km-predictor-aci")

# COMMAND ----------

from mlflow.deployments import get_deploy_client
import json
  
# prepare deploy json file
# Data to be written
deploy_config ={
    "computeType": "aks",
    "computeTargetName": "cpu-inference"
}
# Serializing json 
json_object = json.dumps(deploy_config)
  
# Writing to sample.json
with open("deployment_config.json", "w") as outfile:
    outfile.write(json_object)
    
# set the deployment config
deployment_config_path = "deployment_config.json"
test_config = {'deploy-config-file': deployment_config_path}

# set the tracking uri as the deployment client
client = get_deploy_client(azureml_workspace.get_mlflow_tracking_uri())

# define the model path and the name is the service name
# the model gets registered automatically and a name is autogenerated using the "name" parameter below 
service = client.create_deployment(model_uri=model_uri, config=test_config, name="km-predictor-aks")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Deploy register model into Staging or Production environment (CI/CD)
# MAGIC 
# MAGIC **Azure Pipeline (aka Azure DevOps) can monitor Azure Machine Learning Model Registry and trigger a release pipeline once a new model is registered**
# MAGIC 
# MAGIC Below is an example of using Azure Pipeline Classic mode:
# MAGIC 
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/ReleasePipeline.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' width='600px'>
# MAGIC 
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/DevOpsPublishedPipeline.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' width='600px'>
# MAGIC 
# MAGIC **To create a release pipeline Azure Machine Learning, we will need to connect to Azure Machine Learning workspace**

# COMMAND ----------

from azureml.core.webservice import AciWebservice
svc=None
for s in AciWebservice.list(azureml_workspace):
  if s.name=='km-predictor-aks-svc':
    svc = s

if svc!=None:
  print(svc.scoring_uri)
  # Prepare the data as json for calling the service
  X = '{ "data": [[10],[11]]}'
  X = bytes(X, encoding = 'utf8')
  print(X)
  print("Predict:", svc.run(input_data=X))

# COMMAND ----------

if svc!=None:
  print(svc.scoring_uri)
  # Prepare the data as json for calling the service
  for i in range(5000):
    X = '{ "data": [[' + str(i) +']]}'
    X = bytes(X, encoding = 'utf8')
    print("Data:", i," | Predict:", svc.run(input_data=X))

# COMMAND ----------



# COMMAND ----------

