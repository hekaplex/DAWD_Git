# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab- Deploying Models with the Databricks CLI
# MAGIC
# MAGIC In this Lab, you will update the alias of a previously created model to **"Champion"**, signifying its readiness for deployment. Utilizing the **Databricks CLI**, you will construct and initiate a **Workflow Job**. This job will deploy the latest model version marked **"Champion"** if it meets the production-ready criteria, followed by executing a **batch inference** process.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC _In this lab, you will complete the following tasks:_
# MAGIC - **Task 1:** Identify and update a model's alias to **"Champion"**.
# MAGIC - **Task 2:** Configure and use the Databricks CLI to manage jobs.
# MAGIC - **Task 3:** Create and run a workflow job for model deployment and Batch Inferencing.
# MAGIC - **Task 4:** Monitor and explore the executing workflow job.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-02Lab

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
# MAGIC
# MAGIC ### Install CLI
# MAGIC
# MAGIC Install the Databricks CLI using the following cell. Note that this procedure removes any existing version that may already be installed, and installs the newest version of the [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/index.html). A legacy version exists that is distributed through **`pip`**, however we recommend following the procedure here to install the newer one.

# COMMAND ----------

# MAGIC %sh rm -f $(which databricks); curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/v0.211.0/install.sh | sh

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notebook Path Setup Continued
# MAGIC
# MAGIC This code cell performs the following setup tasks:
# MAGIC - Retrieves the current Databricks **cluster ID** and displays it.
# MAGIC - Identifies the path of the currently running notebook.
# MAGIC - Constructs **paths to related notebooks** for checking model status, deploying the model, performing model inference, and handling cases where the model is not ready for production. These paths are printed to confirm their accuracy.
# MAGIC

# COMMAND ----------

# Retrieve the current cluster ID
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
print(cluster_id)

# Get the current notebook path
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# Base path for related notebooks
base_path = notebook_path.rsplit('/', 1)[0] + "/2.2 Lab Pipeline - Status Check and Deployment"

# Paths for specific process notebooks
check_status_notebook_path = f"{base_path}/2.2a LAB - Checking Model Status"
print(check_status_notebook_path)

deploy_notebook_path = f"{base_path}/2.2b LAB - Model Deployment"
print(deploy_notebook_path)

inference_notebook_path = f"{base_path}/2.2c LAB - Batch Inferencing"
print(inference_notebook_path)

notready_notebook_path = f"{base_path}/2.2d LAB - Not Ready For Production"
print(notready_notebook_path)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task 1: Identify Model as Ready for Deployment
# MAGIC 1. Navigate to Models in the left sidebar.
# MAGIC 1. Apply the filter for models Owned by Me.
# MAGIC 1. Locate and select the model named churn-prediction.
# MAGIC     - **Note**: If the model isn't listed, verify that your first Job Run is complete.
# MAGIC 1. Review the information provided when your model is registered in the Unity Catalog's Model Registry.
# MAGIC 1. Click the pencil icon next to the alias "baseline" and change it to "champion". Save alias.
# MAGIC
# MAGIC **_This step identifies your model as ready for deployment._**
# MAGIC
# MAGIC
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/Model_Deploy_Alias_Change.png" alt="Alias Change" width="700"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Configuration of a Workflow Job
# MAGIC
# MAGIC This code cell constructs a JSON configuration string for a Databricks workflow job. The configuration specifies a series of tasks designed to handle the deployment and inference stages for a model called **"churn-prediction"**. Here are the key components and their functions:
# MAGIC
# MAGIC - **General Settings**:
# MAGIC   - The job runs sequentially with a **maximum of one concurrent run** and is named using the current user's username with the suffix `-deploy-workflow-job`.
# MAGIC   - It includes **email notifications** on failure, configured to alert the user.
# MAGIC
# MAGIC - **Tasks Defined**:
# MAGIC   - **Check Status**: Verifies if the model is ready for production by checking its status.
# MAGIC   - **Production Ready**: Proceeds if the model status is 'ready for production'.
# MAGIC   - **Deploy**: Deploys the model if the previous task confirms readiness.
# MAGIC   - **Batch Inference**: Executes a batch inference process following successful deployment.
# MAGIC   - **Not Ready**: Handles scenarios where the model is not ready for production.
# MAGIC
# MAGIC - **Conditional Execution**:
# MAGIC   - Each task, except for the initial status check, includes conditions that depend on the success of the preceding tasks.
# MAGIC   - If conditions are met, the next task in the workflow is triggered, otherwise, alternative actions are specified.
# MAGIC
# MAGIC - **File Operations**:
# MAGIC   - The constructed JSON string is written to a file named `workflow-job-lab.json` in write mode, ensuring that the entire workflow configuration is saved externally for deployment purposes.
# MAGIC
# MAGIC This setup is essential for automating model deployment workflows in a controlled and predictable manner, allowing for efficient scaling and maintenance of machine learning models.
# MAGIC

# COMMAND ----------

workflow_config = f"""
{{
  "email_notifications": {{
    "on_failure": [
      "{DA.username}"
    ]
  }},
  "format": "MULTI_TASK",
  "max_concurrent_runs": 1,
  "name": "{DA.username}-deploy-workflow-job",
  "notification_settings": {{
    "alert_on_last_attempt": <FILL_IN>,
    "no_alert_for_canceled_runs": <FILL_IN>,
    "no_alert_for_skipped_runs": <FILL_IN>
  }},
  "tasks": [
    {{
      "existing_cluster_id": "{cluster_id}",
      "notebook_task": {{
        "notebook_path": "<FILL_IN>",
        "source": "WORKSPACE"
      }},
      "run_if": "<FILL_IN>",
      "task_key": "check_status"
    }},
    {{
      "condition_task": {{
        "left": "{{{{<FILL_IN>}}}}",
        "op": "EQUAL_TO",
        "right": "<FILL_IN>"
      }},
      "depends_on": [
        {{
          "task_key": "check_status"
        }}
      ],
      "email_notifications": {{}},
      "notification_settings": {{
        "alert_on_last_attempt": <FILL_IN>,
        "no_alert_for_canceled_runs": <FILL_IN>,
        "no_alert_for_skipped_runs": <FILL_IN>
      }},
      "run_if": "<FILL_IN>",
      "task_key": "Production_Ready",
      "timeout_seconds": 0,
      "webhook_notifications": {{}}
    }},
    {{
      "depends_on": [
        {{
          "outcome": "<FILL_IN>",
          "task_key": "Production_Ready"
        }}
      ],
      "email_notifications": {{}},
      "existing_cluster_id": "{cluster_id}",
      "notebook_task": {{
        "notebook_path": "<FILL_IN>",
        "source": "WORKSPACE"
      }},
      "notification_settings": {{
        "alert_on_last_attempt": <FILL_IN>,
        "no_alert_for_canceled_runs": <FILL_IN>,
        "no_alert_for_skipped_runs": <FILL_IN>
      }},
      "run_if": "<FILL_IN>",
      "task_key": "Deploy",
      "timeout_seconds": 0,
      "webhook_notifications": {{}}
    }},
    {{
      "depends_on": [
        {{
          "task_key": "Deploy"
        }}
      ],
      "email_notifications": {{}},
      "existing_cluster_id": "{cluster_id}",
      "notebook_task": {{
        "notebook_path": "<FILL_IN>",
        "source": "WORKSPACE"
      }},
      "notification_settings": {{
        "alert_on_last_attempt": <FILL_IN>,
        "no_alert_for_canceled_runs": <FILL_IN>,
        "no_alert_for_skipped_runs": <FILL_IN>
      }},
      "run_if": "<FILL_IN>",
      "task_key": "Batch_Inference",
      "timeout_seconds": 0,
      "webhook_notifications": {{}}
    }},
    {{
      "depends_on": [
        {{
          "outcome": "<FILL_IN>",
          "task_key": "Production_Ready"
        }}
      ],
      "email_notifications": {{}},
      "existing_cluster_id": "{cluster_id}",
      "notebook_task": {{
        "notebook_path": "<FILL_IN>",
        "source": "WORKSPACE"
      }},
      "notification_settings": {{
        "alert_on_last_attempt": <FILL_IN>,
        "no_alert_for_canceled_runs": <FILL_IN>,
        "no_alert_for_skipped_runs": <FILL_IN>
      }},
      "run_if": "<FILL_IN>",
      "task_key": "Not_Ready",
      "timeout_seconds": 0,
      "webhook_notifications": {{}}
    }}
  ],
  "queue": {{
    "enabled": true
  }},
  "run_as": {{
    "user_name": "{DA.username}"
  }}
}}
"""

with open('workflow-job-lab.json', 'w') as file:
    file.write(workflow_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Creating and Running a Workflow Job
# MAGIC
# MAGIC This section details the process of creating and executing a workflow job using the Databricks CLI:
# MAGIC
# MAGIC 1. **Creating the Job**:
# MAGIC    - The job is created by passing the JSON configuration file `workflow-job-lab.json` to the `databricks jobs create` command. This command returns a JSON object containing details of the created job, including the `job_id`.
# MAGIC
# MAGIC 2. **Extracting the Job ID**:
# MAGIC    - The `job_id` is extracted from the JSON output using a combination of `grep` and `awk`. The `grep` command isolates the line containing `job_id`, and `awk` is used to select the second field (the actual ID value), which is then stripped of extra characters using `tr`.
# MAGIC
# MAGIC 3. **Running the Job**:
# MAGIC    - With the `job_id` extracted, the job is initiated using `databricks jobs run-now`. This command triggers the execution of the workflow defined in the job configuration file.
# MAGIC    
# MAGIC This process automates the deployment of tasks defined in the Databricks environment, ensuring that the model deployment and associated tasks are handled efficiently.
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC # Create the job and capture the output
# MAGIC output=$(<FILL_IN>)
# MAGIC echo <FILL_IN>
# MAGIC
# MAGIC # Extract the job_id from the output
# MAGIC job_id=$(<FILL_IN>)
# MAGIC echo "Extracted job_id: <FILL_IN>"
# MAGIC
# MAGIC # Run the job using the extracted job_id
# MAGIC databricks jobs <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 4:  Monitoring and Exploring the Executing Workflow Job
# MAGIC
# MAGIC To effectively manage and gain insights from your executing workflow job in the Databricks environment, follow these steps:
# MAGIC
# MAGIC 1. **Access the Jobs Console**:
# MAGIC    - From the Databricks sidebar, navigate to the **Jobs** section, which lists all configured jobs.
# MAGIC
# MAGIC 2. **Find and View the Job**:
# MAGIC    - Use the `job_id` (``) or `{DA.username}-deploy-workflow-job` to locate your job. Click on the job name to access its details page.
# MAGIC
# MAGIC 3. **Explore Tasks**:
# MAGIC    - The job's details page displays its current status (e.g., *running*, *success*, *failure*). Click on a **task tab** to view the created workflow.  Click on each tasks to see the details.
# MAGIC
# MAGIC 4. **Explore Run Outputs**:
# MAGIC    - Go back to the **Run Tab** click on the Run.  Investigate the output logs, metrics, etc. of tasks for debugging information or to verify successful execution.
# MAGIC
# MAGIC `

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson.

# COMMAND ----------

DA.cleanup(validate_datasets=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this lab, you successfully set up and executed a workflow job using the Databricks CLI. You configured the job to check the model status, deploy the model, perform batch inference, and handle scenarios where the model is not ready for production.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
