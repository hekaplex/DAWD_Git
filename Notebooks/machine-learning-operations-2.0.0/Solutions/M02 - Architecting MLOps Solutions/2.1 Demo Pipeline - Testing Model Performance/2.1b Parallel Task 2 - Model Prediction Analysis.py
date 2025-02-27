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

import time

# Wait for 1 minute (60 seconds)
time.sleep(60)

# COMMAND ----------

# MAGIC %pip install --upgrade 'mlflow-skinny[databricks]'
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-2.1b

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
# MAGIC ## Task 1: Load Test Data
# MAGIC Load the test data from the Delta table and prepare it for the prediction analysis.

# COMMAND ----------

import pandas as pd

# Load test data
test_data_path = f"{DA.catalog_name}.{DA.schema_name}.bank_loan"
test_data = spark.table(test_data_path).toPandas()

# Split data into features and target
X_test = test_data.drop(columns=['Personal_Loan'])
y_test = test_data['Personal_Loan']

# Display the DataFrame
display(test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Perform Model Predictions and Analyze Results
# MAGIC Send the test data to the model's endpoint, receive predictions, and calculate various evaluation metrics.

# COMMAND ----------

# Use the Token class to get the token
token_obj = Token()
token = token_obj.token

# COMMAND ----------

# Import necessary libraries
import time
import requests
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Define the endpoint URL and authentication token
endpoint_name = f"{DA.username}_ML_AS_04_Demo2"
endpoint_name = endpoint_name.replace(".", "-").replace("@", "-")
# Retrieve the workspace host using spark configuration
try:
    workspace_host = spark.conf.get('spark.databricks.workspaceUrl')
    if not workspace_host:
        raise ValueError("Could not determine the workspace host.")
except Exception as e:
    raise Exception(f"Failed to retrieve the workspace host: {str(e)}")
endpoint_url = f"https://{workspace_host}/serving-endpoints/{endpoint_name}/invocations"
token = token.strip("{}").strip("'")

# Define the headers for the request
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Convert test data to JSON format
data_json = json.dumps({"dataframe_split": X_test.to_dict(orient="split")})

# Send the request to the endpoint
response = requests.post(endpoint_url, headers=headers, data=data_json)

# Check if the request was successful
if response.status_code != 200:
    raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

# Parse the predictions from the response
predictions = response.json()["predictions"]

# Ensure predictions are in the correct format
if isinstance(predictions, list):
    predictions = [pred[0] if isinstance(pred, list) else pred for pred in predictions]

# Calculate and display confusion matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Display classification report
report = classification_report(y_test, predictions)
print("Classification Report:\n", report)

# Save the classification report to a text file
classification_report_path = "classification_report.txt"
with open(classification_report_path, 'w') as f:
    f.write(report)

# Save the confusion matrix and classification report to JSON
results_path = 'results.json'
results = {
    "confusion_matrix": cm.tolist()
}

with open(results_path, 'w') as f:
    json.dump(results, f)

print(f"Classification report saved to: {classification_report_path}")
print(f"Results JSON saved to: {results_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up Classroom
# MAGIC 🚨 **Do not run Clean up**
# MAGIC
# MAGIC You would be using the generated output in the next lab, so avoid running the clean up.

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC
# MAGIC In this demo, you learned how to analyze the performance of a deployed machine learning model by evaluating its predictions. You sent test data to the model's endpoint, received predictions, and calculated various evaluation metrics, including a confusion matrix and classification report. This process is essential for understanding how well your model performs in a production environment and identifying areas for improvement.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
