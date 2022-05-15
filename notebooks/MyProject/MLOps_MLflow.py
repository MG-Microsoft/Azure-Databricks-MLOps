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

# MAGIC %run "./scripts/init"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Step 1 : Model Training
# MAGIC 
# MAGIC **Algorithms**
# MAGIC - Linear Regression
# MAGIC - 2 hidden layer NN

# COMMAND ----------

experiment_name = "/Shared/mlflow-demo" 
mlflow.set_experiment(experiment_name)

# COMMAND ----------

#Prepare data
trainDF, testDF = load_data(5000)
display(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline : Linear Regression

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
model.add(Dense(16))
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
    rmse = r.data.metrics['rmse']

model_uri = f"runs:/{last_run_id}/model"
print(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Register Model 

# COMMAND ----------

result = mlflow.register_model(model_uri=model_uri, name='km-predictor-model')

# COMMAND ----------

result.version

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3 : Model Deployment
# MAGIC 
# MAGIC In this step, model registry will trigger an automatic deployment to production. A published Azure Machine Learning pipeline will be triggered to create a web service using recent registered model.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Review Model Registry
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/ADB_Model_Serving.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' width='1000px'>
# MAGIC 
# MAGIC 
# MAGIC ### (Preview) Batch Inference
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/ADB_Model_Serving_2.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' width='1000px'>
# MAGIC 
# MAGIC ### (Preview) Real-time Inference
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/ADB_Model_Serving_3.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' width='1000px'>

# COMMAND ----------

 