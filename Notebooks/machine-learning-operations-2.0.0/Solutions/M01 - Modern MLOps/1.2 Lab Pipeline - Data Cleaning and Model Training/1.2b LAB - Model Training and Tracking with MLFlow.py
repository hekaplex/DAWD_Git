# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB - Model Training and Tracking using MLflow
# MAGIC In this lab, you will train a machine learning model on cleaned data and track it using MLflow.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC _This lab contains the following tasks:_
# MAGIC 1. Load and split the cleaned dataset.
# MAGIC 2. Train a RandomForestClassifier model.
# MAGIC 3. Evaluate the model's performance.
# MAGIC 4. Save the best model for deployment.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC - To run this notebook, you need to use one of the following Databricks runtime(s): `14.3.x-cpu-ml-scala2.12`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:
# MAGIC

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup-1.3

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
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Data Preparation
# MAGIC This task involves loading the cleaned dataset, handling missing values, and splitting the data for training and testing.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.1 Load the Cleaned Data
# MAGIC Load the cleaned data from the Delta table.

# COMMAND ----------

import pandas as pd
# Read dataset from the feature store table
table_name = f"{DA.catalog_name}.{DA.schema_name}.telco_cleaned_table"
feature_data_pd = spark.table(table_name).toPandas()

# Drop the 'customerID' column
feature_data_pd = feature_data_pd.drop(columns=['customerID'])

# Add unique_id column
feature_data_pd['unique_id'] = range(len(feature_data_pd))

# Convert all columns in the DataFrame to the 'double' data type
feature_data_pd = feature_data_pd.astype("double")

# Display the DataFrame and print the columns
display(feature_data_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.2 Handle Missing Values
# MAGIC Use an imputer to fill in the missing values.

# COMMAND ----------

from sklearn.impute import SimpleImputer

# Handle missing values
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(feature_data_pd)
feature_data_pd = pd.DataFrame(imputed_data, columns=feature_data_pd.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.3 Split the Data
# MAGIC Split the data into training and testing sets.

# COMMAND ----------

from sklearn.model_selection import train_test_split

print(f"We have {feature_data_pd.shape[0]} records in our source dataset")

# split target variable into its own dataset
target_col = "Churn"
X_all = feature_data_pd.drop(labels=target_col, axis=1)
y_all = feature_data_pd[target_col]

# test / train split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.8, random_state=42)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Fit and Log the Model
# MAGIC Train a RandomForestClassifier model, evaluate its performance, and log the model along with its parameters and metrics using MLflow.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure MLflow client to access models in Unity Catalog
mlflow.set_registry_uri('databricks-uc')

# Fetch Model information
client = mlflow.tracking.MlflowClient()
model_name = f"{DA.catalog_name}.{DA.schema_name}.churn-prediction" 

# Helper function to get the latest model version
def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions(f"name = '{model_name}'")
    return max([model_version_info.version for model_version_info in model_version_infos])

# Set the path for MLflow experiment
mlflow.set_experiment(f"/Users/{DA.username}/Lab-1.2b-Model-traning-with-MLflow")

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
    client.set_registered_model_alias(model_name, "Baseline", latest_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Visualize Model Performance
# MAGIC Generate and log confusion matrix and feature importance plots.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.1 Confusion Matrix
# MAGIC Generate and log a confusion matrix to evaluate the model's performance.

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Computing the confusion matrix
cm = confusion_matrix(y_test, y_test_pred, labels=[1, 0])

# Creating a figure object and axes for the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))

# Plotting the confusion matrix using the created axes
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
disp.plot(cmap=plt.cm.Blues, ax=ax)

# Setting the title of the plot
ax.set_title('Confusion Matrix')

# Now 'fig' can be used with MLFlow's log_figure function
client.log_figure(run.info.run_id, figure=fig, artifact_file="confusion_matrix.png")

# Showing the plot here for demonstration
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.2 Feature Importance Plots
# MAGIC Generate and log a feature importance plot to understand the significance of each feature.

# COMMAND ----------

import numpy as np

# Retrieving feature importances
feature_importances = rf_mdl.feature_importances_
feature_names = X_train.columns.to_list()

# Plotting the feature importances
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(feature_names))
ax.bar(y_pos, feature_importances, align='center', alpha=0.7)
ax.set_xticks(y_pos)
ax.set_xticklabels(feature_names, rotation=45)
ax.set_ylabel('Importance')
ax.set_title('Feature Importances in  RandomForest Classifier')

# log to mlflow
client.log_figure(run.info.run_id, figure=fig, artifact_file="feature_importances.png")

# display here
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Search for the Best Run

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4.1 Find the best run in the AutoML experiment
# MAGIC Execute AutoML to find the best model.

# COMMAND ----------

from databricks import automl
from datetime import datetime
# Start an AutoML classification run
automl_run = automl.classify(
    dataset = spark.table("telco_cleaned_table"),
    target_col = "Churn",
    timeout_minutes = 5
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4.2 Get Experiment ID and Search for Runs by Experiment
# MAGIC Get the experiment ID and search for the runs by experiment.

# COMMAND ----------

import mlflow

# Get the experiment path by experiment ID
exp_path = mlflow.get_experiment(automl_run.experiment.experiment_id).name
# Find the most recent experiment in the AutoML folder
filter_string=f'name LIKE "{exp_path}"'
automl_experiment_id = mlflow.search_experiments(
  filter_string=filter_string,
  max_results=1,
  order_by=["last_update_time DESC"])[0].experiment_id

from mlflow.entities import ViewType

# Find the best run ...
automl_runs_pd = mlflow.search_runs(
  experiment_ids=[automl_experiment_id],
  filter_string=f"attributes.status = 'FINISHED'",
  run_view_type=ViewType.ACTIVE_ONLY,
  order_by=["metrics.val_f1_score DESC"]
)
# Display the best run
automl_runs_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5: Save the Best Model
# MAGIC Save the model if it meets the accuracy threshold.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 5.1 Register the Best Model
# MAGIC Register the best model from the AutoML experiment.

# COMMAND ----------

from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# Get the best run ID from AutoML
best_run_id = automl_runs_pd.at[0, 'run_id']
model_uri = f"runs:/{best_run_id}/model"
model_name = "best_automl_model"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 5.2 Save and Log the Best Model
# MAGIC Save and log the best model to the MLflow model registry.

# COMMAND ----------

# Register the model in MLflow
model_details = mlflow.register_model(model_uri, model_name)

# Assuming model_details is a ModelVersion object
best_model_name = f"{DA.catalog_name}.{DA.schema_name}.best_automl_model" 

# Helper function to get the latest model version
def get_latest_model_version(best_model_name):
    """Helper function to get latest model version"""
    best_model_version_infos = client.search_model_versions(f"name = '{best_model_name}'")
    return max([model_version_info.version for model_version_info in best_model_version_infos])

# Set the path for MLflow experiment
mlflow.set_experiment(f"/Users/{DA.username}/Lab-1.2b-Best_Model-traning-with-MLflow")

# Turn off autologging
mlflow.sklearn.autolog(disable=True)

# Start an MLflow run
with mlflow.start_run(run_name="Best Model Training Lab") as run:
    # Define model signature
    signature = infer_signature(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=rf_mdl,
        artifact_path="model-artifacts",
        signature=signature,
        registered_model_name=best_model_name  # Provide a valid name for the registered model
    )

    # Set model alias
    latest_best_model_version = get_latest_model_version(best_model_name)
    client.set_registered_model_alias(best_model_name, "Baseline", latest_best_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up Classroom
# MAGIC 🚨 **Do not run Clean up**
# MAGIC
# MAGIC You will be using the logged model in the next module, so avoid running the clean up.

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this lab, you learned how to train a machine learning model on cleaned data and track it using MLflow. You also evaluated the model's performance and saved the best model for deployment.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
