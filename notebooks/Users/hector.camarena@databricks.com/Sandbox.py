# Databricks notebook source
# MAGIC %fs ls /user/hector.camarena@databricks.com/

# COMMAND ----------

hello world

# COMMAND ----------

deltaDf = spark.sql('SELECT * FROM delta.`/mnt/mcgen2/delta/OG/processedSensors` WHERE nameWell = "EnergisticsWell2016-A"')
display(deltaDf)

# COMMAND ----------

# MAGIC %sh ls -ltr

# COMMAND ----------

# MAGIC %fs ls /mnt/mcgen2/delta/OG/processedSensors/nameWell=EnergisticsWell2016-A/

# COMMAND ----------

pip install azureml-sdk==1.0.76


# COMMAND ----------

dbutils.library.installPyPI("mlflow", extras="extras")
dbutils.library.installPyPI("azureml-sdk")


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

# COMMAND ----------

