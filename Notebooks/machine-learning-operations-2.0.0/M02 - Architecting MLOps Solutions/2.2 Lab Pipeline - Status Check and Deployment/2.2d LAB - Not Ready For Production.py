# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #LAB - Not Ready for Production
# MAGIC In this lab, you will check the status of a machine learning model in the MLflow registry to determine if it is ready for production deployment.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC _By the end of this lab, you will be able to:_
# MAGIC
# MAGIC - Fetch and display detailed metadata for all versions of a model.
# MAGIC - Determine if the latest version of the model is ready or Not for production based on its alias.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Requirements
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC To run this notebook, you need to use one of the following Databricks runtime(s):**`14.3.x-cpu-ml-scala2.12`**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup-1.3

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions**
# MAGIC Throughout this lab, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Step 1: Set Up MLflow and Display Model Info
# MAGIC Set up MLflow, and define a function to display detailed metadata for all versions of a model and check if the latest version is ready for production.

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

mlflow.set_registry_uri('databricks-uc')

def display_model_info(model_name):
    client = MlflowClient()
    alias = "Champion"  # Define the 'alias' variable
    
    try:
        # Fetch all versions of the model
        model_versions = client.search_model_versions(f"name='{model_name}'")
        if not model_versions:
            print("No versions found for this model.")
            return
        
        # Display detailed metadata for all model versions
        for version_info in model_versions:
            version = version_info.version
            creation_time = datetime.fromtimestamp(version_info.creation_timestamp / 1000.0)
            last_updated_time = datetime.fromtimestamp(version_info.last_updated_timestamp / 1000.0)
            description = version_info.description
            current_stage = version_info.current_stage
            user_id = version_info.user_id

            print(f"Model Name: {model_name}")
            print(f"Version: {version}")
            print(f"Creation Time: {creation_time}")
            print(f"Last Updated Time: {last_updated_time}")
            print(f"Description: {description}")
            print(f"Current Stage: {current_stage}")
            print(f"Created By: {user_id}")
            print("-" * 40)

        try:
            # Fetch the model version associated with the alias 'Champion'
            alias_model_version_info = client.get_model_version_by_alias(model_name, alias)
            alias_model_version = alias_model_version_info.version
            print(f"Alias 'Champion' Model Version: {alias_model_version}")
        except mlflow.exceptions.MlflowException:
            print(f"No alias version found for alias '{alias}'")
            alias_model_version = None

        # Find the latest model version based on the version number
        latest_version_info = max(model_versions, key=lambda x: int(x.version))
        latest_version = latest_version_info.version

        # Check if the latest version has the alias 'Champion'
        if alias_model_version and alias_model_version == latest_version:
            dbutils.jobs.taskValues.set(key="model_status", value="ready_for_production")
            print("The most recent version of the model has been evaluated as ready for production.")
            return "ready_for_production"
        else:
            print("The most recent version of the model has not been evaluated as ready for production. The model version must have an alias of 'Champion' to be deployed via this pipeline.")
            return "not_ready_for_production"

    except mlflow.exceptions.MlflowException as e:
        # Handle any Mlflow exceptions and return "not_ready_for_production"
        print(f"Error occurred: {e}")
        return "not_ready_for_production"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Check Model Status
# MAGIC Use the defined function to check the status of the model and determine if it is ready for production.

# COMMAND ----------

# Example usage
dbutils.jobs.taskValues.set(key="model_status", value="not_ready_for_production")
model_name = "churn-prediction"
print(f"Original Model Name: {model_name}")
model_name = f"{DA.catalog_name}.{DA.schema_name}.{model_name}" # Use 3-level namespace
print(f"Full Model Name: {model_name}")
model_status = display_model_info(model_name)
print(f"Model status: {model_status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up Classroom
# MAGIC 🚨 **Do not run clean up**
# MAGIC
# MAGIC You would be using the output in the next notebook, so avoid running the clean up.

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this lab, you learned how to check the status of a machine learning model in the MLflow registry. You fetched and displayed detailed metadata for all versions of a model and determined if the latest version is ready for production based on its alias. This step is crucial for ensuring that only the most recent and validated model versions are deployed in a production environment.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
