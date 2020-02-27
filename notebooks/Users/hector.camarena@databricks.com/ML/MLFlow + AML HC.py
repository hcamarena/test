# Databricks notebook source
# NO NEED TO USE IF INSTALLED ON CLUSTER (PREFERED)
# dbutils.library.installPyPI("mlflow", extras="extras")
# dbutils.library.installPyPI("azureml-sdk")
# dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import mlflow.azureml
import azureml
from azureml.core import Workspace

mlflow.set_experiment("/Users/hector.camarena@databricks.com/ML/HC_Experiment/")

# COMMAND ----------

# MAGIC %md Import Delta Data

# COMMAND ----------

deltaDf = spark.sql('SELECT * FROM delta.`/mnt/mcgen2/delta/OG/processedSensors` WHERE nameWell = "EnergisticsWell2016-A"')

# COMMAND ----------

pivotDf = (deltaDf
            .filter(col('SensorName').isin('SPT_1S', 'SPT_2S', 'SPPA', 'SWOB', 'HKLD'))
            .groupBy('name', 'nameWell', 'nameWellbore', 'TIME')
            .pivot('SensorName')
            .agg(first('SensorValue'))
            .orderBy('Time')
            .select(
                col('nameWell'),
                col('TIME').cast('timestamp').alias('TIME'),
                col('HKLD').cast('float').alias('HKLD'),
                col('SPPA').cast('float').alias('SPPA'),
                col('SPT_1S').cast('float').alias('SPT_1S'),
                col('SPT_2S').cast('float').alias('SPT_2S'),
                col('SWOB').cast('float').alias('SWOB')
                
              )
          )
display(pivotDf)

# COMMAND ----------

dataDf = pivotDf.dropna(how = "any")
display(dataDf)

# COMMAND ----------

# MAGIC %md Convert to Pandas and Split into Test and Train

# COMMAND ----------

### Randomly split data into training and test sets. set seed for reproducibility
data = dataDf.toPandas()

x = data.drop(["nameWell", "TIME", "SWOB"], axis=1)
y = data[["SWOB"]]

train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.20, random_state=30)

# COMMAND ----------

# MAGIC %md MLFlow - Train Linear Regression

# COMMAND ----------

with mlflow.start_run():
  
  # Define and log the parameters
  modelType = "LinReg"
  mlflow.log_param("modelType", modelType)
  
  # Fit, train, and score the model
  model = LinearRegression()
  model.fit(train_x, train_y)
  preds = model.predict(test_x)

  # Log the output paramters and metrics
  mlflow.log_param('coef', model.coef_)
  mlflow.log_param('intercept', model.intercept_)
  mlflow.log_metric('MSE', mean_squared_error(test_y, preds))
  mlflow.log_metric('r2', r2_score(test_y, preds))
   
  # Log the model
  mlflow.sklearn.log_model(model, "LinRegModel")
  
print("Done!")

# COMMAND ----------

# MAGIC %md MLFlow - Train Lasso

# COMMAND ----------

with mlflow.start_run():
  
  # Define and log the parameters
  modelType = "Lasso"
  mlflow.log_param("modelType", modelType)
  
  # Fit, train, and score the model
  model = Lasso()
  model.fit(train_x, train_y)
  preds = model.predict(test_x)

  # Log the output paramters and metrics
  mlflow.log_param('coef', model.coef_)
  mlflow.log_param('intercept', model.intercept_)
  mlflow.log_metric('MSE', mean_squared_error(test_y, preds))
  mlflow.log_metric('r2', r2_score(test_y, preds))
   
  # Log the model
  mlflow.sklearn.log_model(model, "Lasso")
  
print("Done!")

# COMMAND ----------

# MAGIC %md MLFlow - Train Random Forest

# COMMAND ----------

with mlflow.start_run():
  
  # Define and log the parameters
  modelType = "RandFor"
  max_depth = 10
  mlflow.log_param("modelType", modelType)
  mlflow.log_param("maxDepth", max_depth)
  
  # Fit, train, and score the model
  model = RandomForestRegressor(max_depth = max_depth)
  model.fit(train_x, train_y)
  preds = model.predict(test_x)

  # Log the output paramters and metrics
  mlflow.log_metric('MSE', mean_squared_error(test_y, preds))
  mlflow.log_metric('r2', r2_score(test_y, preds))
   
  # Log the model
  mlflow.sklearn.log_model(model, "RandFor")

print("Done!")

# COMMAND ----------

# MAGIC %md Deploy Best Model as Web Service in Azure Machine Learning

# COMMAND ----------

import azureml
from azureml.core import Workspace

workspace_name = "rmh-mls"
workspace_location = "eastus2"
resource_group = "rmh-rg"
subscription_id = "3f2e4d32-8e8d-46d6-82bc-5bb8d962328b"

workspace = Workspace.create(name = workspace_name,
                             location = workspace_location,
                             resource_group = resource_group,
                             subscription_id = subscription_id,
                             exist_ok=True)

# COMMAND ----------

myModel = "dbfs:/user/hector.camarena@databricks.com/ml/experiment/34f7580a10fc46eb930059beda1da722/artifacts/RandFor"

model_image, azure_model = mlflow.azureml.build_image(model_uri=myModel, 
                                                      workspace=workspace,
                                                      model_name="hc-randfor00",
                                                      image_name="hc-randfor00-image",
                                                      description="HC SKLearn Random Forest model for OG",
                                                      synchronous=False)

model_image.wait_for_creation(show_output=True)

# COMMAND ----------

from azureml.core.webservice import AciWebservice, Webservice

webservice_name = "hc-randfor00-service"
webservice_deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
webservice = Webservice.deploy_from_image(name=webservice_name, image=model_image, deployment_config=webservice_deployment_config, workspace=workspace)
webservice.wait_for_deployment(show_output=True) 

# COMMAND ----------

# MAGIC %md Get Model Web Service URI

# COMMAND ----------

uri = webservice.scoring_uri
print(uri)

# COMMAND ----------

# MAGIC %md Predict Using Web Service URI

# COMMAND ----------

sample_json = {
    "columns": [
        "HKLD",
        "SPPA",
        "SPT_1S",
        "SPT_2S"
    ],
    "data": [
        [65.7845, 16613.676, 90.69767,	100.329124]
    ]
}

print(sample_json)

# COMMAND ----------

import requests
import json

def service_query(input_data):
  response = requests.post(
              url=uri, data=json.dumps(input_data),
              headers={"Content-type": "application/json"})
  prediction = response.text
  print(prediction)
  return prediction

# COMMAND ----------

service_query(sample_json)

# COMMAND ----------

