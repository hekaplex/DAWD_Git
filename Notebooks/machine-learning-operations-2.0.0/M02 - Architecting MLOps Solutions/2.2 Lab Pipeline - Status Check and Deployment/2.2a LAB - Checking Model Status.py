# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #LAB - Checking Model Status
# MAGIC In this lab, you will learn how to check the status of a machine learning model in the Databricks Model Registry. You will fetch the model version, check if it is ready for production, and set appropriate task values.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC _By the end of this lab, you will be able to:_
# MAGIC
# MAGIC - Fetch model versions from the Databricks Model Registry.
# MAGIC - Check if the latest model version is ready for production.
# MAGIC - Set task values based on the model's readiness status.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC To run this notebook, you need to use one of the following Databricks runtime(s): **`14.3.x-cpu-ml-scala2.12`**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

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
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

#model_name = dbutils.widgets.get("model_name")
#print(model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task: Check Model Status
# MAGIC In this task, you will check the status of a machine learning model to determine if it is ready for production. This involves fetching the latest model version and comparing it to the version tagged as "champion".
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC - Import Libraries and Set Up MLflow Client:
# MAGIC
# MAGIC > - Import necessary libraries.
# MAGIC > - Set up the MLflow client to access the model registry.
# MAGIC
# MAGIC - Define Function to Check Model Status:
# MAGIC
# MAGIC > - Create a function `check_model_status` to fetch model versions and check their status.
# MAGIC
# MAGIC - Fetch and Print Model Status:
# MAGIC
# MAGIC > - Use the defined function to fetch the model status.
# MAGIC > - Print the model status.

# COMMAND ----------

import mlflow

# Set MLflow registry URI to Databricks Unity Catalog
mlflow.set_registry_uri('databricks-uc')

# Define a function to check the model status
def check_model_status(model_name):
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    alias = "champion"  # Define the 'alias' variable
    
    try:
        # Fetch all versions of the model
        print(f"name='{model_name}'")
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        # Find the latest model version based on the version number
        latest_version_info = max(model_versions, key=lambda x: int(x.version))
        latest_version = latest_version_info.version
        print(f"Latest version: {latest_version}")
        
        # Fetch the model version associated with the alias 'Champion'
        alias_model_version_info = client.get_model_version_by_alias(model_name, alias)
        alias_model_version = alias_model_version_info.version
        print(f"Alias model version: {alias_model_version}")
        
        # Check if the latest version has the alias 'Champion'
        if alias_model_version == latest_version:
            # If the latest version has the alias 'Champion', return "ready_for_production"
            dbutils.jobs.taskValues.set(key = "model_status", value = "ready_for_production")
            return "ready_for_production"
        else:
            # If the latest version does not have the alias 'Champion', return "not_ready_for_production"
            return "not_ready_for_production"

    except mlflow.exceptions.MlflowException as e:
        # Handle any Mlflow exceptions and return "not_ready_for_production"
        print(f"Error occurred: {e}")
        return "not_ready_for_production"

# Example usage
dbutils.jobs.taskValues.set(key = "model_status", value = "not_ready_for_production")
#model_name = dbutils.widgets.get("model_name")
model_name = "churn-prediction"
print(model_name)
model_name = f"{DA.catalog_name}.{DA.schema_name}.{model_name}" # Use 3-level namespace
print(model_name)
model_status = check_model_status(model_name)
print(f"Model status: {model_status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up Classroom
# MAGIC 🚨 **Do not run Clean up**
# MAGIC
# MAGIC You would be using the output in the next notebook, so avoid running the clean up.

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this lab, you have learned how to check the status of a machine learning model in the Databricks Model Registry. By fetching the latest model version and comparing it with the version tagged as "champion," you determined whether the model is ready for production. This process ensures that only validated models are deployed, enhancing the reliability and performance of machine learning workflows.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
