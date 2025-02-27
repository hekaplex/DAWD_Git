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
print(f"token:             {Token}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Load Test Data
# MAGIC Load the test data from the Delta table and prepare it for the performance test.

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
# MAGIC ## Task 2: Performance Test
# MAGIC Perform a performance test on the deployed model using the endpoint.

# COMMAND ----------

# Use the Token class to get the token
token_obj = Token()
token = token_obj.token

# COMMAND ----------

import time
import requests
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# Measure prediction time for performance test
start_time = time.time()

# Send the request to the endpoint
response = requests.post(endpoint_url, headers=headers, data=data_json)
end_time = time.time()

# Initialize error counter
error_count = 0

# Check if the request was successful
if response.status_code != 200:
    error_count += 1
    raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

# Parse the predictions from the response
predictions = response.json()["predictions"]

# Ensure predictions are in the correct format
if isinstance(predictions, list):
    predictions = [pred[0] if isinstance(pred, list) else pred for pred in predictions]

# Calculate performance metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
execution_time = end_time - start_time

# Calculate error rate
total_requests = 1  # Since we only make one request here
error_rate = (error_count / total_requests) * 100

# Save performance metrics to JSON file
performance_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "execution_time": execution_time,
    "error_rate": error_rate
}
performance_metrics_path = "performance_metrics.json"
with open(performance_metrics_path, 'w') as file:
    json.dump(performance_metrics, file, indent=4)

# Log performance metrics
print(f"Performance Test Results:\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}\nExecution Time: {execution_time:.2f} seconds\nError Rate: {error_rate:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Conduct a Detailed Performance Test
# MAGIC Conduct a more detailed performance test by sending multiple requests and measuring the latency and success rate.

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Define the number of requests for the performance test
num_requests = 100

# Function to send a request to the model endpoint
def send_request(data):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    start_time = time.time()
    response = requests.post(endpoint_url, headers=headers, json=data)
    end_time = time.time()
    latency = end_time - start_time
    success = response.status_code == 200
    return latency, success, response

# Generate sample input data
input_data = {"dataframe_split": X_test.sample(1).to_dict(orient="split")}

# Perform the test using multiple threads
def perform_test(num_requests, input_data):
    latencies = []
    successes = []
    responses = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(send_request, input_data) for _ in range(num_requests)]
        for future in futures:
            latency, success, response = future.result()
            latencies.append(latency)
            successes.append(success)
            responses.append(response)
    
    return latencies, successes, responses

# Run the performance test
latencies, successes, responses = perform_test(num_requests, input_data)

# Calculate performance metrics
average_latency = np.mean(latencies)
throughput = num_requests / sum(latencies)
error_rate = (1 - np.mean(successes)) * 100

# Display performance metrics
print(f"Average Latency: {average_latency:.4f} seconds")
print(f"Throughput: {throughput:.2f} requests/second")
print(f"Error Rate: {error_rate:.2f}%")

# Append performance metrics to JSON file
performance_metrics = {
    "average_latency": average_latency,
    "throughput": throughput,
    "error_rate": error_rate
}

try:
    with open(performance_metrics_path, 'r') as file:
        existing_data = json.load(file)
        if isinstance(existing_data, list):
            existing_data.append(performance_metrics)
        else:
            existing_data = [existing_data, performance_metrics]
except FileNotFoundError:
    existing_data = [performance_metrics]

with open(performance_metrics_path, 'w') as file:
    json.dump(existing_data, file, indent=4)

print(f"Performance metrics appended to: {performance_metrics_path}")

# Display the first failed response for debugging
for response in responses:
    if not response.ok:
        print(f"Failed Response: {response.text}")
        break

# Visualize latency distribution
plt.hist(latencies, bins=30, edgecolor='black')
plt.xlabel('Latency (seconds)')
plt.ylabel('Frequency')
plt.title('Latency Distribution')
plt.show()

# Save results to a DataFrame and export to CSV
results_df = pd.DataFrame({
    'Latency': latencies,
    'Success': successes
})
results_df.to_csv('/dbfs/tmp/performance_test_results.csv', index=False)

# Display the DataFrame
results_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up Classroom
# MAGIC 🚨 **Do not run Clean up**
# MAGIC
# MAGIC You would be using the generated output in the next lab, so avoid running the clean up.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this demo, you learned how to perform a performance test on a deployed machine learning model. You measured various performance metrics such as accuracy, precision, recall, F1 score, latency, and throughput. You also visualized the latency distribution and saved the results for further analysis. This process is crucial for ensuring the reliability and efficiency of machine learning models in production environments.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
