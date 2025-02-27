# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Demo - Working with Asset Bundles
# MAGIC
# MAGIC Databricks Asset Bundles are an excellent way to develop complex projects in your own development environment and deploy them to your Databricks workspace. You can use common CI/CD practices to keep track of the history of your workflow.
# MAGIC
# MAGIC In this demo, we will show you how to use a Databricks Asset Bundle to start, deploy, run, and destroy a simple workflow job.
# MAGIC
# MAGIC Normally, the process of working with asset bundles would be done in a terminal on your local computer or through a CI/CD workflow configured on your repository system (e.g., GitHub). In this demo, we will use shell scripts that run on the driver of the current cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-1.1

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Authentication
# MAGIC
# MAGIC Usually, you would have to set up authentication for the CLI. But in this training environment, that's already taken care of if you ran through the accompanying 
# MAGIC **'Generate Tokens'** notebook. 
# MAGIC If you did, credentials will already be loaded into the **`DATABRICKS_HOST`** and **`DATABRICKS_TOKEN`** environment variables. 
# MAGIC
# MAGIC #####*If you did not, run through it now then restart this notebook.*

# COMMAND ----------

DA.get_credentials()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install CLI
# MAGIC
# MAGIC Install the Databricks CLI using the following cell. Note that this procedure removes any existing version that may already be installed, and installs the newest version of the [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/index.html). A legacy version exists that is distributed through **`pip`**, however we recommend following the procedure here to install the newer one.

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -f $(which databricks); curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/v0.211.0/install.sh | sh

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating a Databricks Asset Bundle from a Template
# MAGIC
# MAGIC There are a handful of templates available that allow you to quickly create an asset bundle. We are going to use a python-based bundle that will generate a Databricks Workflow Job.
# MAGIC
# MAGIC In your day-to-day work, you can start with a template and make changes to it as needed, or you can create [your own templates](https://docs.databricks.com/en/dev-tools/bundles/templates.html).
# MAGIC
# MAGIC The command below invokes the Databricks CLI and uses the `init` command from the `bundle` command group to create and initialize a bundle from the `default-python` template.

# COMMAND ----------

# MAGIC %sh 
# MAGIC databricks bundle init default-python

# COMMAND ----------

# MAGIC %md
# MAGIC ## The `my_project` Directory
# MAGIC
# MAGIC As you can see from the output of the cell above, our project has been created in the `my_project` directory. Note that this is in the same directory as this notebook. You may need to refresh this page in your browser.
# MAGIC
# MAGIC Open the folder and note all the sub-directories and files contained within.
# MAGIC
# MAGIC We are going to make a few changes before we deploy this bundle. We are going to change the job configuration to use the same cluster we are currently using, instead of a Jobs Cluster. This is not recommended practice, but we are doing it just for the sake of this course.
# MAGIC
# MAGIC Run the cell below to make these changes.

# COMMAND ----------

DA.update_bundle()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Databricks Asset Bundle
# MAGIC
# MAGIC In the command below, we first `cd` into the `my_project` directory (since `my_project` is the root of the asset bundle). We then run `validate`. This command allows us to ensure our files are syntactically correct.

# COMMAND ----------

# MAGIC %sh
# MAGIC cd my_project
# MAGIC databricks bundle validate

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploying Databricks Asset Bundles
# MAGIC
# MAGIC We can deploy the bundle to our workspace by running the simple command below.
# MAGIC
# MAGIC After the deployment is complete, we will have a new workflow job in our workspace.

# COMMAND ----------

# MAGIC %sh
# MAGIC cd my_project
# MAGIC databricks bundle deploy

# COMMAND ----------

# MAGIC %md
# MAGIC ## View the Created Job
# MAGIC
# MAGIC 1. Click **`Workflows`** in the left sidebar menu, and search for `[dev {DA.username}] my_project_job`.
# MAGIC 2. Click the name of the job.
# MAGIC 3. Click the **`Tasks`** tab.
# MAGIC
# MAGIC Note that there are two tasks in the job. Also, note that the job is connected to a Databricks Asset Bundle and that we should not edit the job directly in the workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Job
# MAGIC
# MAGIC We could run the job manually within the workspace, but we will run the job using `run` from the `bundle` command group:

# COMMAND ----------

# MAGIC %sh
# MAGIC cd my_project
# MAGIC databricks bundle run my_project_job

# COMMAND ----------

# MAGIC %sh
# MAGIC job_id=$(databricks jobs list | awk '/my_project_job/ {print $1; exit}')
# MAGIC echo "Job ID: $job_id"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting Up Monitoring and Notifications
# MAGIC
# MAGIC
# MAGIC To add monitoring and notifications, we can use the Databricks API to configure job notifications programmatically. Below is an example of how to do this using Python:
# MAGIC
# MAGIC

# COMMAND ----------

# Use the Token class to get the token
token_obj = Token()
token = token_obj.token

# COMMAND ----------

import requests
import json

# Replace these with your own values
databricks_instance = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
api_token = token.strip("{}").strip("'")
job_id = "##################" #update this from the output of cell 23

# Define the notification settings
notification_settings = {
    "settings": {
        "email_notifications": {
            "on_failure": [f"{DA.username}"],
            "on_start": [],
            "on_success": []
        }
    }
}

# Make the API request to update job settings
response = requests.post(
    f"{databricks_instance}/api/2.0/jobs/update",
    headers={"Authorization": f"Bearer {api_token}"},
    json={
        "job_id": job_id,
        "new_settings": notification_settings["settings"]
    }
)

# Check the response
if response.status_code == 200:
    print("Notification settings updated successfully.")
else:
    print(f"Failed to update notification settings: {response.text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modifying and Redeploying the Bundle
# MAGIC
# MAGIC To emphasize job exploration and modification, let’s modify a part of the job and redeploy it. For instance, we can update the Python script in the bundle to include additional logging or a simple transformation.
# MAGIC
# MAGIC 1. Open the `my_project` directory and locate the Python script used in the job.  (my_project/src/my_project/main.py)
# MAGIC 2. Make necessary modifications (e.g., add logging).

# COMMAND ----------

# MAGIC %md
# MAGIC ###Redeploy the Updated Bundle
# MAGIC Redeploy the updated bundle to apply the changes.

# COMMAND ----------

# MAGIC %sh
# MAGIC cd my_project
# MAGIC databricks bundle deploy

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Modified Job
# MAGIC
# MAGIC Run the modified job to ensure the changes have been applied.
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC cd my_project
# MAGIC databricks bundle run my_project_job

# COMMAND ----------

# MAGIC %md
# MAGIC ## Destroying the Bundle Deployment
# MAGIC
# MAGIC Notice that we can complete all of these tasks without ever being in a workspace, as long as we have a token configured.
# MAGIC
# MAGIC Our last step is to destroy the deployed infrastructure:

# COMMAND ----------

# MAGIC %sh
# MAGIC cd my_project
# MAGIC databricks bundle destroy --auto-approve

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Up
# MAGIC After completing the demo, it's essential to remove any resources generated during its execution. Run the following cell to remove all assets specifically created during this demonstration:

# COMMAND ----------

DA.cleanup(validate_datasets=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion
# MAGIC
# MAGIC In this demo, we focused on building, modifying, and exploring jobs using Databricks Asset Bundles within an MLOps context. We ensured the steps align with best practices for operationalizing machine learning models, including job execution, CI/CD integration, and monitoring. We also demonstrated how to modify and redeploy jobs to adapt to changing requirements.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
