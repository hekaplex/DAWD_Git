# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB - Model Deployment 
# MAGIC
# MAGIC In this lab, you will learn how to deploy the best machine learning model from the MLflow registry to a Databricks model serving endpoint. This will involve loading the model, configuring the endpoint, and deploying it.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC _By the end of this lab, you will be able to:_
# MAGIC
# MAGIC - Load the best model from the MLflow registry.
# MAGIC - Configure and create a Databricks model serving endpoint.
# MAGIC - Deploy the model to the endpoint.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC - To run this notebook, you need to use one of the following Databricks runtime(s): **`14.3.x-cpu-ml-scala2.12`**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:
# MAGIC

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()

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
# MAGIC
# MAGIC ## Task 1: Load the Best Model from MLflow Registry
# MAGIC Load the best model that was saved in the previous lab from the MLflow registry.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC > - Import necessary libraries.
# MAGIC > - Initialize the Databricks Workspace client.
# MAGIC > - Set the MLflow registry URI to Databricks Unity Catalog.
# MAGIC > - Retrieve the best model version from the MLflow registry.

# COMMAND ----------

# Import necessary libraries
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointTag

# Initialize Databricks Workspace client
w = WorkspaceClient()

# Set the MLflow registry URI to Databricks Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Initialize the MLflow client
client = mlflow.MlflowClient()

# Retrieve the model name
model_name = "churn-prediction"

# Construct the 3-level namespace for the model
model_name = f"{DA.catalog_name}.{DA.schema_name}.{model_name}"

# Get the champion version of the model
model_version_champion = client.get_model_version_by_alias(name=model_name, alias="champion").version
print(f"Champion model version: {model_version_champion}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configure and Create the Serving Endpoint
# MAGIC Configure the serving endpoint with the model details and create the endpoint.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC > - Define the endpoint configuration.
# MAGIC > - Create the endpoint configuration input.
# MAGIC > - Construct the endpoint name.
# MAGIC > - Create or update the serving endpoint.

# COMMAND ----------

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
endpoint_name = f"{DA.username}_ML_AS_04_Lab2"
endpoint_name = endpoint_name.replace(".", "-")
endpoint_name = endpoint_name.replace("@", "-")
# Create the endpoint_name key/value pair to be passed on in the job configuration
dbutils.jobs.taskValues.set(key = "endpoint_name", value = endpoint_name)
print(f"Endpoint name: {endpoint_name}")

# Attempt to create or update the serving endpoint
try:
    w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=endpoint_config,
        tags=[EndpointTag.from_dict({"key": "db_academy", "value": "lab1_jobs_model"})]
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
# MAGIC 🚨 **Do not run clean up**
# MAGIC
# MAGIC You would be using the output in the next notebook, so avoid running the clean up.

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this lab, you learned how to deploy the best machine learning model from the MLflow registry to a Databricks model serving endpoint. By following the steps to load the model, configure the endpoint, and deploy it, you ensured that the model is ready for serving predictions in a production environment. This process is essential for operationalizing machine learning workflows and making models accessible for batch inference.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
