# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Introduction
# MAGIC
# MAGIC In the next few notebooks, we will use the Databricks CLI to run code from a notebook, in addition to using the UI. Since we are in a learning environment, we will save a credentials file right here in the workspace. In a production environment, follow your organization's security policies for storing credentials.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC - To run this notebook, you need to use one of the following Databricks runtime(s): `14.3.x-cpu-ml-scala2.12`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC Before starting the Notebooks, run the provided classroom setup script. This script will define configuration variables necessary for the future demos and labs. Execute the following cell:
# MAGIC

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-0

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
# MAGIC ## Create a Landing Pad for the Credentials
# MAGIC A token is just like a username and password, so you should treat it with the same level of security as your own credentials. If you ever suspect a token has leaked, delete it immediately.
# MAGIC
# MAGIC For the purpose of this training, we will create a landing pad in this notebook to record and store the credentials within the workspace. When using credentials in production, follow the security practices of your organization.
# MAGIC
# MAGIC Run the following cell to create two text fields which you will populate in the next section:
# MAGIC
# MAGIC - **Host:** The URL of the target workspace, which will form the base for all REST API endpoints. The framework will populate this value automatically using the current workspace, but this value can be overridden if desired.
# MAGIC - **Token:** A bearer token to authenticate with the target workspace.

# COMMAND ----------

DA.get_credentials()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Credentials
# MAGIC
# MAGIC Create an authorization token for use with the Databricks CLI and API. If you are unable to create a token, please reach out to your workspace admin.
# MAGIC
# MAGIC **Steps to Generate a Token:**
# MAGIC
# MAGIC 1. Click on your username in the top bar and select **User Settings** from the drop-down menu.
# MAGIC 1. Click **User &gt; Developer**, then click **Access tokens &gt; Manage**.
# MAGIC 1. Click **Generate new token**.
# MAGIC 1. Specify the following:
# MAGIC    * A comment describing the purpose of the token (for example, *CLI Demo*).
# MAGIC    * The lifetime of the token; estimate the number of days you anticipate needing to complete this module.
# MAGIC 1. Click **Generate**.
# MAGIC 1. Copy the displayed token to the clipboard. You will not be able to view the token again; if you lose it, you will need to delete it and create a new one.
# MAGIC 1. Paste the token into the **Token** field above.
# MAGIC 1. If you are targeting a workspace other than the current one, paste it into the **Host** field. Otherwise, leave this value as-is.
# MAGIC 1. Click **Done**.
# MAGIC
# MAGIC In response to these inputs, these values will be recorded as follows:
# MAGIC * In the environment variables **`DATABRICKS_HOST`** and **`DATABRICKS_TOKEN`** so that they can be used for [authentication](https://docs.databricks.com/en/dev-tools/auth/index.html) by the Databricks CLI, APIs, and SDK that we use in subsequent notebooks
# MAGIC * Since environment variables are limited in scope to the current execution context, the values are persisted to a [file in your workspace](https://docs.databricks.com/en/files/workspace.html#) for use by subsequent notebooks

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
