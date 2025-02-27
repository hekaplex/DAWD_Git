# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB - Creating and Managing Workflow Jobs using UI
# MAGIC
# MAGIC In this lab, you will learn how to set up a workflow to deploy a machine learning model with manual triggers and email notifications using the Databricks UI. This will involve creating tasks in the Workflow Jobs, configuring job dependencies, and enabling email notifications.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC _In this lab, you will complete the following tasks:_
# MAGIC
# MAGIC - Create and configure a Workflow job with multiple tasks using the UI.
# MAGIC - Enable email notifications for job status updates.
# MAGIC - Manually trigger the deployment workflow.
# MAGIC - Monitor the job run to ensure successful execution.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC - To run this notebook, you need to use one of the following Databricks runtime(s): `14.3.x-cpu-ml-scala2.12`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Create a Databricks Job
# MAGIC
# MAGIC 1. **Navigate to the Workflows**:
# MAGIC    - In your Databricks workspace, click on the **Workflows** icon in the left sidebar.
# MAGIC    
# MAGIC 2. **Create a New Job**:
# MAGIC    - Click on the **Create Job** button on the top right.
# MAGIC    - Name the job, for example, "ML Model Training Workflow".

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Add Tasks to the Job:

# COMMAND ----------

# MAGIC %md
# MAGIC > - ### Task 1: Data Cleaning and Feature Engineering
# MAGIC > 
# MAGIC > 1. **Add First Task**:
# MAGIC >    - Name the task: `Data_Cleaning_and_Feature_Engineering`.
# MAGIC >    - Set **Type** to `Notebook`.
# MAGIC >    - Set **Source** to `Workspace`.
# MAGIC >    - Set **Path** to the notebook path: `$/1.2 Lab Pipeline - Data Cleaning and Model Training/1.2a LAB - Data Cleaning and Feature Engineering`.
# MAGIC >    - Choose an appropriate cluster for this task.
# MAGIC >    - Click **Create Task**.
# MAGIC > 
# MAGIC > ![Task 1 Configuration](https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/Data_Cleaning_and_Feature_Engineering_Task%2B1.png)
# MAGIC >

# COMMAND ----------

# MAGIC %md
# MAGIC > - ### Task 2: Model Training
# MAGIC > 
# MAGIC > 2. **Add Second Task**:
# MAGIC >    - Click on **Add Task --> Notebook**.
# MAGIC >    - Name the task: `Model_Training`.
# MAGIC >    - Set **Type** to `Notebook`.
# MAGIC >    - Set **Source** to `Workspace`.
# MAGIC >    - Set **Path** to the notebook path: `$/1.2 Lab Pipeline - Data Cleaning and Model Training/1.2b LAB - Model Training`.
# MAGIC >    - Choose the same cluster as the first task.
# MAGIC >    - Set **Depends on** to `Data_Cleaning_and_Feature_Engineering`.
# MAGIC >    - Click **Create Task**.
# MAGIC > 
# MAGIC > ![Task 2 Configuration](https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/Model_Training_Task_2%2B.png)
# MAGIC >

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Enable Email Notifications
# MAGIC
# MAGIC 1. **Enable Notifications**:
# MAGIC    - For task 2, click on **Add** under **Notifications**.
# MAGIC    - Add your email to receive notifications on job status updates.
# MAGIC
# MAGIC ![Email Notification Configuration](https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/Task+Notification.png)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Manually Trigger the Job Run
# MAGIC
# MAGIC 1. **Run the Job**:
# MAGIC    - Click on **Run Now** in the top right corner to manually trigger the job. ![Run Job](https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/Manually_trigger.png)
# MAGIC
# MAGIC **Optional:** You can also set a scheduled trigger from the Schedules & Triggers option as shown below:
# MAGIC
# MAGIC ![Scheduled Trigger](https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/Schedules+%26+Triggers.png)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5: Monitor the Job Run
# MAGIC
# MAGIC 1. **Navigate to the Runs Tab**:
# MAGIC    - Go to the **Runs** tab to view current and past job executions.
# MAGIC    
# MAGIC 2. **View Running Jobs**:
# MAGIC    - Identify the job with a **Running** status.
# MAGIC    - Click on the **Start Time** of the run to access detailed information.
# MAGIC    
# MAGIC 3. **Observe Task Execution**:
# MAGIC    - Select the **Task** square to observe the execution of individual cells and their outputs.
# MAGIC    - Continue to explore until the run is fully completed. *It should take about 9-10 minutes*.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you have learned how to create and configure a Workflow job with multiple tasks using the Databricks UI. You also enabled email notifications for job status updates and manually triggered the deployment workflow. By monitoring the job run, you ensured successful execution of the tasks. This process helps in automating machine learning workflows, ensuring that data processing and model training are executed seamlessly.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
