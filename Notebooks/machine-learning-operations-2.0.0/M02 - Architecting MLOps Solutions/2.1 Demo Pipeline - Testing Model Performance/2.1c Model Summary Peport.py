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

# MAGIC %pip install --upgrade 'mlflow-skinny[databricks]'
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-2.1c

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
# MAGIC ## Task: Load and Display Evaluation Metrics
# MAGIC Load and display the contents of performance metrics, classification reports, and other evaluation results.

# COMMAND ----------

import json

# Define file paths
classification_report_path = "classification_report.txt"
performance_metrics_path = "performance_metrics.json"
results_path = "results.json"

# Function to read and print text file
def read_and_print_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            print(f"Contents of {file_path}:\n{content}\n")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Function to read and print JSON file
def read_and_print_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = json.load(file)
            print(f"Contents of {file_path}:\n{json.dumps(content, indent=4)}\n")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Read and print the contents of the files

read_and_print_json_file(performance_metrics_path)
read_and_print_text_file(classification_report_path)
read_and_print_json_file(results_path)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC 🚨 **Do not run Clean up**
# MAGIC
# MAGIC You would be using the output in the next notebook, so avoid running the clean up.

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this notebook, you consolidated the results from various evaluation metrics into a summary report. You loaded and displayed performance metrics, classification reports, and other evaluation results. This comprehensive report helps in understanding the overall performance of the deployed model and is crucial for making informed decisions about further model improvements or deployments.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
