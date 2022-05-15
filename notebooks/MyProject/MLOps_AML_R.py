# Databricks notebook source
# MAGIC %md
# MAGIC # Problem Statement - R model deployment
# MAGIC 
# MAGIC ### In this notebook, we will build a sample R model and deploy as REST endpoint in AML
# MAGIC 
# MAGIC **Input:** Fahrenheit (float) 
# MAGIC 
# MAGIC **Output:** Celsius (float)
# MAGIC 
# MAGIC 
# MAGIC Data source: Random sampled numbers between (1, 50000) with random noise.
# MAGIC 
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/tp_ml.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' alt="Traditional Programming vs Machine Learning" width='1000px'>

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/KM_achitecture.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' alt="MLOps Architecture" width='600px'>

# COMMAND ----------

# MAGIC %run "./scripts/packages"

# COMMAND ----------

# MAGIC %run "./scripts/init"

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1 : Model Training
# MAGIC 
# MAGIC - Generating random data (5000) to be loaded as part of the scoring, you can either bring your own model created in ADB or a random sample Data is created for demo purposes

# COMMAND ----------

# MAGIC %r
# MAGIC RandomNum <- runif(5000, 1, 5000)
# MAGIC save(RandomNum,file="/tmp/class.rds")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline : Connect to AML

# COMMAND ----------

from azureml.core import Workspace
subscription_id='7c1d967f-37f1-4047-bef7-05af9aa80fe2'
tenant_id = '72f988bf-86f1-41af-91ab-2d7cd011db47'
service_principal_clientid = dbutils.secrets.get(scope = "key-vault-secrets", key ="clientId") # Service Principal ID
service_principal_secret = dbutils.secrets.get(scope = "key-vault-secrets", key ="clientSecret") # Service Principal Secret

# Connect to Azure Machine Learning space
azureml_workspace = Workspace(
       subscription_id=subscription_id,
       resource_group='demo-rg-01',
       workspace_name='aml-workspace-01',
       auth=service_principal_auth(tenant_id, service_principal_clientid , service_principal_secret))

experiment_name = "mlflow-R-demo" 
mlflow.set_tracking_uri(azureml_workspace.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample scoring function to convert F to C
# MAGIC 
# MAGIC - This function can be used to load any reference file class.rds or re-use your existing R scoring logic

# COMMAND ----------

# MAGIC %%writefile score.R
# MAGIC fahrenheit_to_celsius <- function(temp_F) {
# MAGIC   temp_C <- (temp_F - 32) * 5 / 9
# MAGIC   return(temp_C)
# MAGIC }

# COMMAND ----------

import os
os.makedirs('deployment', exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2 : Model Deployment
# MAGIC 
# MAGIC In this step, model registry will trigger an automatic deployment to production. A published Azure Machine Learning pipeline will be triggered to create a web service using recent registered model.
# MAGIC 
# MAGIC <img src='https://onebigdatabag.blob.core.windows.net/sparkdemo/mlflow_production.jpg?sp=r&st=2022-01-20T02:45:00Z&se=2042-01-20T10:45:00Z&spr=https&sv=2020-08-04&sr=c&sig=jjz2%2BIgQu%2Bh9gBkx1mRMpbQtDQQBiGHsrzSqgmo6QUk%3D' width='600px'>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy register model into Staging or Production environment (CI/CD)
# MAGIC 
# MAGIC **Azure Pipeline (aka Azure DevOps) can monitor Azure Machine Learning Model Registry and trigger a release pipeline once a new model is registered**
# MAGIC 
# MAGIC **To create a release pipeline Azure Machine Learning, we will need to connect to Azure Machine Learning workspace**

# COMMAND ----------

# MAGIC %md
# MAGIC - AML scoring expects python file and below is example of how you can use R function from Python using rpy2 package ( https://rpy2.github.io/doc/latest/html/introduction.html# )

# COMMAND ----------

# MAGIC %%writefile score.py
# MAGIC import readline
# MAGIC import joblib
# MAGIC import os
# MAGIC import json
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import rpy2
# MAGIC import rpy2.robjects as ro
# MAGIC from rpy2.robjects.packages import importr
# MAGIC from rpy2.robjects import pandas2ri
# MAGIC from rpy2.robjects.conversion import localconverter
# MAGIC from inference_schema.schema_decorators import input_schema, output_schema
# MAGIC from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
# MAGIC from azureml.core.model import Model
# MAGIC 
# MAGIC def init():
# MAGIC     global r_model_path,r_entry_script, score_function_r
# MAGIC     # The AZUREML_MODEL_DIR environment variable indicates
# MAGIC     # a directory containing the model file you registered.
# MAGIC     r_model_path = Model.get_model_path(model_name='my-r-model')    
# MAGIC     r_entry_script = Model.get_model_path(model_name='my-r-script')
# MAGIC     
# MAGIC     # Defining the R script and loading the instance in Python
# MAGIC     r = ro.r
# MAGIC     r['source'](r_entry_script)
# MAGIC     # Loading the function we have defined in R.
# MAGIC     score_function_r = ro.globalenv['fahrenheit_to_celsius']
# MAGIC 
# MAGIC def run(raw_data):
# MAGIC     pd_df = pd.DataFrame(json.loads(raw_data)['data'])
# MAGIC     with localconverter(ro.default_converter + pandas2ri.converter):
# MAGIC         r_from_pd_df = ro.conversion.py2rpy(pd_df)
# MAGIC 
# MAGIC     # r_from_pd_df
# MAGIC     df_result_r = score_function_r(r_from_pd_df)
# MAGIC 
# MAGIC     with localconverter(ro.default_converter + pandas2ri.converter):
# MAGIC         pd_from_r_df = ro.conversion.rpy2py(df_result_r)
# MAGIC 
# MAGIC     return pd_from_r_df.to_json()

# COMMAND ----------

# MAGIC %md
# MAGIC - A custom docker image is created to build, compile R and install any required R libraries. NOTE: Intial build can take upto 20 min

# COMMAND ----------

# MAGIC %%writefile deployment/Dockerfile
# MAGIC FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04
# MAGIC RUN mkdir /usr/share/man/man1 && apt-get update && apt-get install --no-install-recommends -y openjdk-8-jdk
# MAGIC ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
# MAGIC ENV DEBIAN_FRONTEND=noninteractive
# MAGIC RUN apt-get update \
# MAGIC   && apt-get install --yes \
# MAGIC     libssl-dev \
# MAGIC     libfuse-dev \
# MAGIC     python3 python3-pip \
# MAGIC     wget \
# MAGIC     openjdk-8-jdk \
# MAGIC     build-essential gfortran libreadline-dev libxml2-dev libcurl4-openssl-dev libpcre2-dev libbz2-dev liblzma-dev \
# MAGIC   && apt-get clean \
# MAGIC   && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# MAGIC # Conda Environment
# MAGIC ENV MINICONDA_VERSION py37_4.9.2
# MAGIC ENV PATH /opt/miniconda/bin:$PATH
# MAGIC RUN wget -qO /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
# MAGIC     bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
# MAGIC     conda clean -ay && \
# MAGIC     rm -rf /opt/miniconda/pkgs && \
# MAGIC     rm /tmp/miniconda.sh && \
# MAGIC     find / -type d -name __pycache__ | xargs rm -rf
# MAGIC RUN wget -qO R-4.0.4.tar.gz https://cran.r-project.org/src/base/R-4/R-4.0.4.tar.gz && \
# MAGIC     tar -xvf R-4.0.4.tar.gz && \
# MAGIC     cd R-4.0.4 && \
# MAGIC     ./configure --enable-R-shlib --with-x=no --without-recommended-packages && \
# MAGIC     make -j4 && make install && rm -rf /R-4.0.4.tar.gz
# MAGIC ENV LD_LIBRARY_PATH="/usr/local/lib/R/lib:$LD_LIBRARY_PATH"
# MAGIC RUN ldconfig
# MAGIC RUN R -e "install.packages(c('dplyr'), repos = 'https://cloud.r-project.org/')"
# MAGIC RUN R -e "install.packages(c('conflicted'), repos = 'https://cloud.r-project.org/')"

# COMMAND ----------

# MAGIC %md
# MAGIC - Python dependency file to read input data, pass to R function and respond back in json

# COMMAND ----------

# MAGIC %%writefile deployment/conda_dependencies.yml
# MAGIC channels:
# MAGIC - conda-forge
# MAGIC dependencies:
# MAGIC - python=3.7
# MAGIC - pip:
# MAGIC   - azureml-core==1.34.0
# MAGIC   - azureml-defaults==1.34.0
# MAGIC   - azureml-telemetry==1.34.0
# MAGIC   - azureml-train-restclients-hyperdrive==1.34.0
# MAGIC   - azureml-train-core==1.34.0
# MAGIC   - azureml-monitoring
# MAGIC   - joblib
# MAGIC   - pandas
# MAGIC   - tzlocal
# MAGIC   - rpy2==3.4.5
# MAGIC name: azureml_4dd1b7149337b0438e0e64ae5a60fd4e

# COMMAND ----------

# MAGIC %md
# MAGIC - Deployment

# COMMAND ----------

from azureml.core import Image, Workspace, Webservice, Model, Environment
from azureml.core.webservice import AksWebservice
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import InferenceConfig
from azureml.core.run import Run
from azureml.core.compute import ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies

# get run context
run = Run.get_context()
ws = azureml_workspace


# Choose a name for your AKS cluster
aks_name = 'cpu-inference'    
aks_target = ComputeTarget(workspace=ws, name=aks_name)

# Create a customer docker container with R
r_env = Environment("custom")
r_env.docker.enabled = True
r_env.docker.base_image = None
r_env.inferencing_stack_version='latest'
r_env.docker.base_dockerfile = "./deployment/Dockerfile"
r_env.python.conda_dependencies = CondaDependencies(conda_dependencies_file_path='./deployment/conda_dependencies.yml')

inference_config = InferenceConfig(entry_script="score.py", environment=r_env)
aks_config = AksWebservice.deploy_configuration(collect_model_data=True, enable_app_insights=True)

# Sample r model or reference file
azureml_r_model = Model.register(workspace=ws,
                       model_name='my-r-model',      # Name of the registered model in your workspace.
                       model_path='/tmp/class.rds',  # Local file to upload and register as a model.
                       description='R Model RDS',
                       tags={'area': 'azureml', 'type': 'databricks notebook', 'language': 'R'})
# R entry script 
azureml_r_entry_script = Model.register(workspace=ws,
                       model_name='my-r-script', # Name of the registered model in your workspace.
                       model_path='./score.R',   # Local file to upload and register as a model.
                       description='R model scoring scrpt',
                       tags={'area': 'azureml', 'type': 'databricks notebook', 'language': 'R'})

# Deploy webservice
print("Deploying web service")
aks_service = Model.deploy(workspace=ws,
                           name="r-model-f-to-c",
                           models=[azureml_r_model,azureml_r_entry_script],
                           inference_config=inference_config,
                           deployment_config=aks_config,
                           deployment_target=aks_target,
                          overwrite=True)

aks_service.wait_for_deployment(show_output = False)


# COMMAND ----------

from azureml.core.webservice import AksWebservice
svc=None
for s in AksWebservice.list(azureml_workspace):
  if s.name=='r-model-f-to-c':
    svc = s

if svc!=None:
  print(svc.scoring_uri)
  # Prepare the data as json for calling the service
  X = '{ "data": [[32],[100]]}'
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