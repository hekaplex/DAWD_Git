# Databricks notebook source
# MAGIC %md
# MAGIC # 3.0 - Model Preparation
# MAGIC In this notebook, you will prepare a machine learning model using a banking dataset. You will train a classification model and deploy it using Databricks.
# MAGIC
# MAGIC **Steps:**
# MAGIC - **Step 1:** Prepare Model
# MAGIC > - **Step 1.1:** Load Dataset
# MAGIC > - **Step 1.2:** Train / New Requests Split
# MAGIC > - **Step 1.3:** Fit a Classification Model
# MAGIC - **Step 2:** Deploying the Model

# COMMAND ----------

# MAGIC %run
# MAGIC ../Includes/Classroom-Setup-3.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Prepare Model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1.1: Load Dataset
# MAGIC - Load the CSV file into a Spark DataFrame.
# MAGIC - Rename columns to remove invalid characters.
# MAGIC - Convert to Pandas DataFrame and display it.

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import lit, col
from pyspark.sql.types import DoubleType

# Load the Delta table into a Spark DataFrame
dataset_path = f"{DA.paths.working_dir}/loan-dataset"
loan_df = spark.read.format("delta").load(dataset_path)

# Convert to Pandas DataFrame
loan_pd_df = loan_df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 1.2: Train / New Requests Split
# MAGIC - Split the data into training and request sets.
# MAGIC - Convert to Pandas DataFrames and prepare features and labels.

# COMMAND ----------

# Split the data into train and request sets
train_df, request_df = loan_df.randomSplit(weights=[0.6, 0.4], seed=42)

# Convert to Pandas DataFrames
train_pd_df = train_df.toPandas()
request_pd_df = request_df.toPandas()
target_col = "Personal_Loan"
ID = "ID"
X_train = train_df.drop(target_col, ID)
y_train = train_df.select(target_col)
X_request = request_df.drop(target_col)
y_request = request_df.select(target_col)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 1.3: Fit a Classification Model
# MAGIC - Fit a Random Forest model.
# MAGIC - Register the model with Unity Catalog.

# COMMAND ----------

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
from sklearn.exceptions import DataConversionWarning
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=DataConversionWarning, module="sklearn")

# Register models in UC
mlflow.set_registry_uri("databricks-uc")
mlflow.sklearn.autolog(log_input_examples=True)

X_train_pd = X_train.toPandas()
y_train_pd = y_train.toPandas()

# Define model parameters
rf_params = {
    "n_estimators": 100,
    "random_state": 42
}
rfc = RandomForestClassifier(**rf_params)
rfc_mdl = rfc.fit(X_train_pd, y_train_pd)

model_name = f"{catalog_name}.{schema_name}.loan_model"
signature = infer_signature(X_train, y_train)
model_info = mlflow.sklearn.log_model(
    sk_model=rfc_mdl, 
    artifact_path="model-artifacts",
    signature=signature,
    registered_model_name=model_name
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Deploying the Model
# MAGIC
# MAGIC You can deploy your model by executing the provided Python code in your notebook. This method offers flexibility and control, allowing for programmatically managing various aspects of the deployment.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointTag
import mlflow

w = WorkspaceClient()

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()
served_model_name = model_name.split('.')[-1]

endpoint_config_dict = {
    "served_models": [
        {
            "model_name": model_name,
            "scale_to_zero_enabled": True,
            "model_version": "1",
            "workload_size": "Small",
        },
    ]
}
endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

endpoint_name = f"{DA.username}_MLas04_M03_Lab"
endpoint_name = endpoint_name.replace(".", "-")
endpoint_name = endpoint_name.replace("@", "-")
dbutils.jobs.taskValues.set(key="endpoint_name", value=endpoint_name)
print(f"Endpoint name: {endpoint_name}")

try:
    w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=endpoint_config,
        tags=[EndpointTag.from_dict({"key": "db_academy", "value": "Loan_monitor_model"})]
    )
    print(f"Creating endpoint {endpoint_name} with models {model_name} versions 1")
except Exception as e:
    if "already exists" in e.args[0]:
        print(f"Endpoint with name {endpoint_name} already exists")
    else:
        raise(e)
# Display the endpoint URL
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{endpoint_name}">Model Serving Endpoint')
