# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Demo - Model Testing with the Databricks CLI
# MAGIC
# MAGIC This Demo guides you through creating a job workflow using the Databricks CLI. The workflow will include tasks for model tracking with MLflow and testing the model. We will configure the workflow using a JSON configuration file and submit it via Databricks CLI.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this lab, you will be able to:
# MAGIC - **Install the Databricks CLI** and **configure authentication** for a Databricks workspace, building upon credentials set in a previous notebook.
# MAGIC - **Execute the `help` command** within the Databricks CLI to explore available functionalities.
# MAGIC - **Manage and execute Databricks Job commands** directly from the notebook.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC - To run this notebook, you need to use one of the following Databricks runtime(s): **`14.3.x-cpu-ml-scala2.12`**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-02

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
# MAGIC This code cell performs the following setup tasks:
# MAGIC
# MAGIC - Retrieves the current Databricks cluster ID and displays it.
# MAGIC - Identifies the path of the currently running notebook.
# MAGIC - Constructs paths to related notebooks for Training and deploying the model, Performance Testing,  Model Prediction Analysis, and printing the Summary report of the Model testing. These paths are printed to confirm their accuracy.

# COMMAND ----------

# Retrieve the current cluster ID
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
print(cluster_id)

# Get the current notebook path
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# Base path for related notebooks
base_path = notebook_path.rsplit('/', 1)[0] + "/2.1 Demo Pipeline - Testing Model Performance"

# Paths for specific process notebooks
Deploy_model_notebook_path = f"{base_path}/2.1a Train and Deploy Model"
print(Deploy_model_notebook_path)
Parallel_Task_1_notebook_path = f"{base_path}/2.1b Parallel Task 1 - Performance Test" 
print(Parallel_Task_1_notebook_path)
Parallel_Task_2_notebook_path = f"{base_path}/2.1b Parallel Task 2 - Model Prediction Analysis" 
print(Parallel_Task_2_notebook_path)
Summary_report_path = f"{base_path}/2.1c Model Summary Peport"
print(Summary_report_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using the CLI
# MAGIC The cell below illustrates simple CLI usage. This displays a list of the contents of the /Users workspace directory.

# COMMAND ----------

# MAGIC %sh databricks workspace ls /Users

# COMMAND ----------

# MAGIC %md
# MAGIC The CLI provides access to a broad swath of Databricks functionality. Use the **`help`** command to access the online help.

# COMMAND ----------

# MAGIC %sh databricks help

# COMMAND ----------

# MAGIC %sh databricks jobs list

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration of a Workflow Job
# MAGIC
# MAGIC This code cell constructs a JSON configuration string for a Databricks workflow job. The configuration specifies a series of tasks designed to handle the deployment and inference stages for a model called "churn-prediction". Here are the key components and their functions:
# MAGIC
# MAGIC **General Settings:**
# MAGIC > - **Job Naming**: The job is named using the current user's username with the suffix `-testing-workflow-job`.
# MAGIC > - **Concurrent Runs**: The job runs sequentially with a maximum of one concurrent run.
# MAGIC > - **Email Notifications**: Configured to alert the user on failure.
# MAGIC > - **Notification Settings**: Ensures alerts are not skipped for canceled or skipped runs.
# MAGIC > - **Timeout**: No timeout is set for the job.
# MAGIC > - **Queue Enabled**: Ensures tasks are queued if the job is already running.
# MAGIC
# MAGIC **Tasks Defined:**
# MAGIC > 1. **Deploy Trained Model**: Deploys the trained model.
# MAGIC > 2. **Model Performance Test**: Tests the model's performance.
# MAGIC > 3. **Model Prediction Analysis**: Analyzes the model's predictions.
# MAGIC > 4. **Summary Report**: Generates a summary report based on the model performance and prediction analysis.
# MAGIC
# MAGIC **Conditional Execution:**
# MAGIC - Each task (except the initial deployment) includes conditions that depend on the success of the preceding tasks. If conditions are met, the next task in the workflow is triggered.
# MAGIC
# MAGIC **File Operations:**
# MAGIC > - The constructed JSON string is written to a file named `workflow-job-demo.json` in write mode, ensuring that the entire workflow configuration is saved externally for deployment purposes.
# MAGIC
# MAGIC This setup is essential for automating model deployment workflows in a controlled and predictable manner, allowing for efficient scaling and maintenance of machine learning models.

# COMMAND ----------

# Define the configuration for the workflow
workflow_config_testing = f"""
{{
  "name": "{DA.username}-testing-workflow-job",
  "email_notifications": {{
    "on_failure": [
      "{DA.username}"
    ],
    "no_alert_for_skipped_runs": false
  }},
  "webhook_notifications": {{}},
  "notification_settings": {{
    "no_alert_for_skipped_runs": false,
    "no_alert_for_canceled_runs": false
  }},
  "timeout_seconds": 0,
  "max_concurrent_runs": 1,
  "tasks": [
    {{
      "task_key": "Deploy_best_model",
      "run_if": "ALL_SUCCESS",
      "notebook_task": {{
        "notebook_path": "{Deploy_model_notebook_path}",
        "source": "WORKSPACE"
      }},
      "existing_cluster_id": "{cluster_id}",
      "timeout_seconds": 0,
      "email_notifications": {{}},
      "notification_settings": {{
        "no_alert_for_skipped_runs": false,
        "no_alert_for_canceled_runs": false,
        "alert_on_last_attempt": false
      }},
      "webhook_notifications": {{}}
    }},
    {{
      "task_key": "Model_Performance_Test",
      "depends_on": [
        {{
          "task_key": "Deploy_best_model"
        }}
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {{
        "notebook_path": "{Parallel_Task_1_notebook_path}",
        "source": "WORKSPACE"
      }},
      "existing_cluster_id": "{cluster_id}",
      "timeout_seconds": 0,
      "email_notifications": {{}},
      "notification_settings": {{
        "no_alert_for_skipped_runs": false,
        "no_alert_for_canceled_runs": false,
        "alert_on_last_attempt": false
      }},
      "webhook_notifications": {{}}
    }},
    {{
      "task_key": "Model_Prediction_Analysis",
      "depends_on": [
        {{
          "task_key": "Deploy_best_model"
        }}
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {{
        "notebook_path": "{Parallel_Task_2_notebook_path}",
        "source": "WORKSPACE"
      }},
      "existing_cluster_id": "{cluster_id}",
      "timeout_seconds": 0,
      "email_notifications": {{}},
      "notification_settings": {{
        "no_alert_for_skipped_runs": false,
        "no_alert_for_canceled_runs": false,
        "alert_on_last_attempt": false
      }},
      "webhook_notifications": {{}}
    }},
    {{
      "task_key": "summary_report",
      "depends_on": [
        {{
          "task_key": "Model_Prediction_Analysis"
        }},
        {{
          "task_key": "Model_Performance_Test"
        }}
      ],
      "run_if": "ALL_SUCCESS",
      "notebook_task": {{
        "notebook_path": "{Summary_report_path}",
        "source": "WORKSPACE"
      }},
      "existing_cluster_id": "{cluster_id}",
      "timeout_seconds": 0,
      "email_notifications": {{}},
      "notification_settings": {{
        "no_alert_for_skipped_runs": false,
        "no_alert_for_canceled_runs": false,
        "alert_on_last_attempt": false
      }},
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

# Write the configuration to a JSON file
with open('workflow-job-demo.json', 'w') as file:
    file.write(workflow_config_testing)

print("Workflow configuration file 'workflow-job-demo.json' created successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating and Running a Workflow Job
# MAGIC
# MAGIC This section details the process of creating and executing a workflow job using the Databricks CLI:
# MAGIC
# MAGIC 1. **Creating the Job**:
# MAGIC    - The job is created by passing the JSON configuration file `workflow-job-demo.json` to the `databricks jobs create` command. This command returns a JSON object containing details of the created job, including the `job_id`.
# MAGIC
# MAGIC 2. **Extracting the Job ID**:
# MAGIC    - The `job_id` is extracted from the JSON output using a combination of `grep` and `awk`. The `grep` command isolates the line containing `job_id`, and `awk` is used to select the second field (the actual ID value), which is then stripped of extra characters using `tr`.
# MAGIC
# MAGIC 3. **Running the Job**:
# MAGIC    - With the `job_id` extracted, the job is initiated using `databricks jobs run-now`. This command triggers the execution of the workflow defined in the job configuration file.
# MAGIC    
# MAGIC This process automates the deployment of tasks defined in the Databricks environment, ensuring that the model deployment and associated tasks are handled efficiently.

# COMMAND ----------

# MAGIC %sh
# MAGIC # Create the job and capture the output
# MAGIC output=$(databricks jobs create --json @workflow-job-demo.json)
# MAGIC echo $output
# MAGIC
# MAGIC # Extract the job_id from the output
# MAGIC job_id=$(echo $output | grep -o '"job_id":[0-9]*' | awk -F':' '{print $2}')
# MAGIC echo "Extracted job_id: $job_id"
# MAGIC
# MAGIC # Run the job using the extracted job_id
# MAGIC databricks jobs run-now $job_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring and Exploring the Executing Workflow Job
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
# MAGIC    - The job's details page displays its current status (e.g., *running*, *success*, *failure*). Click on a **task tab** to view the created workflow.  Click on each task to see the details.
# MAGIC
# MAGIC 4. **Explore Run Outputs**:
# MAGIC    - Go back to the **Run Tab** click on the Run.  Investigate the output logs, metrics, etc. of tasks for debugging information or to verify successful execution.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson.

# COMMAND ----------

DA.cleanup(validate_datasets=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this demo, we have successfully set up and executed a workflow job using the Databricks CLI. We configured the job to deploy a model, perform performance tests, analyze predictions, and generate a summary report. This process demonstrated how to automate and streamline model deployment and testing in Databricks, leveraging the CLI for efficient job management.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
