# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC - To run this notebook, you need to use one of the following Databricks runtime(s): `14.3.x-cpu-ml-scala2.12`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-2.1

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions**
# MAGIC
# MAGIC Throughout this lab, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 1: Load Cleaned Data
# MAGIC Load the cleaned data from the Delta table and prepare it for training.

# COMMAND ----------

import pandas as pd

# Read dataset from the feature store table
table_name = f"{DA.catalog_name}.{DA.schema_name}.bank_loan"
feature_data_pd = spark.table(table_name).toPandas()

# Display the DataFrame and print the columns
display(feature_data_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split the Data
# MAGIC Split the dataset into training and testing sets.

# COMMAND ----------

from sklearn.model_selection import train_test_split

print(f"We have {feature_data_pd.shape[0]} records in our source dataset")

# Specify the target column
target_col = "Personal_Loan"
X_all = feature_data_pd.drop(labels=target_col, axis=1)
y_all = feature_data_pd[target_col]

# Test / train split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.8, random_state=42)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 2: Train and Track the Model with MLflow
# MAGIC Train a RandomForestClassifier model, evaluate its performance, and log the model along with its parameters and metrics using MLflow.
# MAGIC
# MAGIC

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
import pandas as pd

# Configure MLflow client to access models in Unity Catalog
mlflow.set_registry_uri('databricks-uc')

# Fetch Model information
client = mlflow.tracking.MlflowClient()
# Use 3-level namespace for model name
model_name = f"{DA.catalog_name}.{DA.schema_name}.loan-prediction" 

# Helper function to get the latest model version
def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions(f"name = '{model_name}'")
    return max([model_version_info.version for model_version_info in model_version_infos])

# Set the path for MLflow experiment
mlflow.set_experiment(f"/Users/{DA.username}/Lab-2.1-Model-traning-with-MLflow")

# Turn off autologging
mlflow.sklearn.autolog(disable=True)

# Define model parameters
rf_params = {
    "n_estimators": 100,
    "random_state": 42
}

# Start an MLflow run
with mlflow.start_run(run_name="Model Training Lab") as run:
    # Log the dataset as artifacts
    feature_data_pd.to_csv("/tmp/feature_data.csv", index=False)
    X_train.to_csv("/tmp/X_train.csv", index=False)
    X_test.to_csv("/tmp/X_test.csv", index=False)
    
    mlflow.log_artifact("/tmp/feature_data.csv", artifact_path="feature_data")
    mlflow.log_artifact("/tmp/X_train.csv", artifact_path="training_data")
    mlflow.log_artifact("/tmp/X_test.csv", artifact_path="test_data")

    # Log our parameters
    mlflow.log_params(rf_params)

    # Fit the model
    rf = RandomForestClassifier(**rf_params)
    rf_mdl = rf.fit(X_train, y_train)

    # Define model signature
    signature = infer_signature(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=rf_mdl,
        artifact_path="model-artifacts",
        signature=signature,
        registered_model_name=model_name  # Provide a valid name for the registered model
    )

    # Evaluate on the training set
    y_train_pred = rf_mdl.predict(X_train)
    mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_train_pred))
    mlflow.log_metric("train_precision", precision_score(y_train, y_train_pred))
    mlflow.log_metric("train_recall", recall_score(y_train, y_train_pred))
    mlflow.log_metric("train_f1", f1_score(y_train, y_train_pred))

    # Evaluate on the test set
    y_test_pred = rf_mdl.predict(X_test)
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
    mlflow.log_metric("test_precision", precision_score(y_test, y_test_pred))
    mlflow.log_metric("test_recall", recall_score(y_test, y_test_pred))
    mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred))

    # Set model alias
    latest_model_version = get_latest_model_version(model_name)
    client.set_registered_model_alias(model_name, "champion", latest_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Deploy the Trained Model
# MAGIC Deploy the trained model to a Databricks model serving endpoint.

# COMMAND ----------

# Import necessary libraries
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointTag

# Initialize Databricks Workspace client
w = WorkspaceClient()

# Set the MLflow registry URI to Databricks Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Initialize the MLflow client
client = mlflow.MlflowClient()

# Define the model name
model_name = f"{DA.catalog_name}.{DA.schema_name}.loan-prediction"

# Get the champion version of the model
model_version_champion = client.get_model_version_by_alias(name=model_name, alias="champion").version
print(f"Champion model version: {model_version_champion}")

# Define the endpoint configuration
endpoint_config_dict = {
    "served_models": [
        {
            "model_name": model_name,
            "model_version": model_version_champion,
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
             
        }
    ]
}

# Create the endpoint configuration input from the dictionary
endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

# Construct the endpoint name
endpoint_name = f"{DA.username}_ML_AS_04_Demo2"
endpoint_name = endpoint_name.replace(".", "-")
endpoint_name = endpoint_name.replace("@", "-")
# Create the endpoint_name key/value pair to be passed on in the job configuration
dbutils.jobs.taskValues.set(key="endpoint_name", value=endpoint_name)
print(f"Endpoint name: {endpoint_name}")

# Attempt to create or update the serving endpoint
try:
    w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=endpoint_config,
        tags=[EndpointTag.from_dict({"key": "db_academy", "value": "Demo2_jobs_model"})]
    )
    print(f"Creating endpoint {endpoint_name} with models {model_name} versions {model_version_champion}")
except Exception as e:
    if "already exists" in e.args[0]:
        print(f"Endpoint with name {endpoint_name} already exists")
    else:
        raise(e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up Classroom
# MAGIC 🚨 **Do not run Clean up**
# MAGIC
# MAGIC You would be using the created endpoint in the next lab, so avoid running the clean up.

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this demo, you learned how to train a machine learning model using a RandomForestClassifier on a bank loan dataset, track the model using MLflow, and deploy the trained model to a Databricks model serving endpoint. This end-to-end process from data preparation to model deployment is crucial for operationalizing machine learning workflows in a real-world setting.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
