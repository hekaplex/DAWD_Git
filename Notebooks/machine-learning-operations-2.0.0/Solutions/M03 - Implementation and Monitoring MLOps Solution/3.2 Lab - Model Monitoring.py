# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab - Model Monitoring
# MAGIC In this notebook, you will monitor the performance of a deployed machine learning model using Databricks. You will enable an inference table, send batched requests, and set up comprehensive monitoring.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC _In this lab, you will need to complete the following tasks:_
# MAGIC - **Task 1:** Enable Inference Table in Serving Endpoint using UI
# MAGIC - **Task 2:** Save the Training Data as Reference for Drift
# MAGIC - **Task 3:** Sending Batched Requests to the Model Endpoint
# MAGIC - **Task 4:** Processing and Monitoring Inference Data
# MAGIC > - **4.1:** Processing Inference Table Data  
# MAGIC > - **4.2:** Monitoring the Inference Table 
# MAGIC > - **4.3:** Analyzing Processed Requests
# MAGIC - **Task 5:** Persisting Processed Model Logs
# MAGIC - **Task 6:** Setting Up and Monitoring Inference Data
# MAGIC > - **6.1:** Creating an Inference Monitor with Databricks Lakehouse Monitoring
# MAGIC > - **6.2:** Inspect and Monitor Metrics Tables
# MAGIC
# MAGIC
# MAGIC 📝 **Your task:** Complete the **`<FILL_IN>`** sections in the code blocks and follow the other steps as instructed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC - To run this notebook, you need to use one of the following Databricks runtime(s): **`14.3.x-cpu-ml-scala2.12`**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the Lab, run the provided classroom setup script. This script will define configuration variables necessary for the Lab. Execute the following cell:
# MAGIC
# MAGIC 🚨 **_Please wait for the classroom setup to run, as it may take around 20 minutes to execute and create a serving endpoint that you will be using for the Lab._**

# COMMAND ----------

# MAGIC %pip install -q databricks-sdk --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-3.Lab

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {catalog_name}")
print(f"Schema Name:       {schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load Data
# MAGIC Load the banking dataset into a Pandas DataFrame and prepare it for training and requesting.

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import lit, col
from pyspark.sql.types import DoubleType

# Load the Delta table into a Spark DataFrame
dataset_path = loan_pd_df

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Train / New Requests Split
# MAGIC Split the data into training and request sets.
# MAGIC
# MAGIC

# COMMAND ----------

# Split the data into train and request sets
train_df, request_df = loan_df.randomSplit(weights=[0.6, 0.4], seed=42)

# Convert to Pandas DataFrames
train_pd_df = train_df.toPandas()
request_pd_df = request_df.toPandas()
target_col = "Personal_Loan"
ID = "ID"

X_request = request_df.drop(target_col)
y_request = request_df.select(target_col)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Define Model Name and Display the Model Serving Endpoint
# MAGIC - Set the model name for registration in the Databricks Model Registry.
# MAGIC
# MAGIC - Display the Model Serving Endpoint URL for easy access.

# COMMAND ----------

model_name = f"{catalog_name}.{schema_name}.loan_model"
# Display the endpoint URL
endpoint_name = f"{DA.username}_MLas04_M03_Lab"
endpoint_name = endpoint_name.replace(".", "-")
endpoint_name = endpoint_name.replace("@", "-")
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{endpoint_name}">Model Serving Endpoint')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Enable Inference Table in Serving Endpoint using UI 
# MAGIC Configure an inference table through the Databricks Model Serving UI to log incoming requests for monitoring purposes.
# MAGIC
# MAGIC **Steps 1.1:**
# MAGIC 1. **Access Model Serving Endpoint:** Click the link printed above in the output of **Cell 15**.
# MAGIC 2. **Edit Endpoint:** Click the **Edit endpoint** button.
# MAGIC 3. **Configure Inference Tables:** 
# MAGIC    - Expand the **Inference tables** section.
# MAGIC    - Check the **Enable inference tables** box.
# MAGIC    - Enter the catalog, schema, and table information for the inference table:
# MAGIC      - **Catalog Name:** `{catalog_name}`
# MAGIC      - **Schema Name:** `{schema_name}`
# MAGIC      - **Table Name Prefix:** `<Your Table Name>` (e.g., loan)
# MAGIC After enabling the inference table and entering the required information, return to this notebook to proceed with the next steps.

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 1.2: Verify Endpoint Updation**
# MAGIC
# MAGIC After creating the inference table, it is essential to verify that the endpoint is ready to use. Follow the instructions below to ensure the endpoint is correctly created and updated:
# MAGIC
# MAGIC 1. Wait for a couple of minutes to allow the endpoint to initialize.
# MAGIC 2. Run the provided Python code to verify the status of the endpoint.
# MAGIC

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import json

w = WorkspaceClient()

endpoint = w.serving_endpoints.wait_get_serving_endpoint_not_updating(endpoint_name)
assert endpoint.state.config_update.value == "NOT_UPDATING" and endpoint.state.ready.value == "READY", "Endpoint not ready or failed"
# Print to check that the endpoint was created
print('Endpoint was created and is ready.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Save the Training Data as Reference for Drift
# MAGIC  
# MAGIC Save the training data as a Delta table to serve as a reference for detecting data drift.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. **Convert Data:** Convert the Pandas DataFrame to a Spark DataFrame.
# MAGIC 2. **Save Data:** Save the Spark DataFrame as a Delta table.
# MAGIC 3. **Read and Update Data:** Read the Delta table and update the data types to match the required schema.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import lit, col
from pyspark.sql.types import DoubleType
from pyspark.sql import DataFrame, functions as F, types as T

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(train_pd_df).withColumn('model_id', lit(0)).withColumn("labeled_data", col("Personal_Loan").cast(DoubleType()))

(spark_df
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema", True)
  .option("delta.enableChangeDataFeed", "true")
  .saveAsTable(f"{catalog_name}.{schema_name}.baseline_features"))

# Read the existing table into a DataFrame
baseline_features_df = spark.table(f"{catalog_name}.{schema_name}.baseline_features")

# Cast the labeled_data and CCAvg columns to INT
baseline_features_df = baseline_features_df.withColumn('labeled_data', F.col('labeled_data').cast(T.IntegerType()))
baseline_features_df = baseline_features_df.withColumn('CCAvg', F.col('CCAvg').cast(T.IntegerType()))

# Overwrite the existing table with the updated DataFrame
(baseline_features_df
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema", "true")
  .saveAsTable(f"{catalog_name}.{schema_name}.baseline_features"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Sending Batched Requests to the Model Endpoint
# MAGIC In this task, you will simulate and send a series of batched requests to the model endpoint to evaluate its performance and response under different conditions.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. **Simulate Requests:** Generate a series of requests using pre-saved data.
# MAGIC 2. **Send Requests:** Send batched requests to the model endpoint and capture the responses.
# MAGIC
# MAGIC

# COMMAND ----------

import time

# Wait for 10 seconds to prevent error and needing to rerun
time.sleep(10)

# Convert PySpark DataFrame to Pandas DataFrame
requests = X_request.orderBy('ID').toPandas().reset_index()

batch_size = 10
number_of_batches = len(requests) // batch_size

for batch_num in range(number_of_batches):
    batch = payload(requests[requests.index // batch_size == batch_num].to_dict(orient='split'))
    query_response = w.serving_endpoints.query(name=endpoint_name, dataframe_split=batch)
    print(f"Batch {batch_num + 1} response:")
    # Print the query responses 
    print(query_response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Processing and Monitoring Inference Data 
# MAGIC This task involves processing the logged data, monitoring the inference table, and analyzing the processed requests to ensure continuous availability and accuracy of model performance data.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4.1: Processing Inference Table Data
# MAGIC In this task, you will extract and analyze the data logged in the inference table to prepare it for monitoring.
# MAGIC
# MAGIC **Steps:**
# MAGIC   1. **Define Functions:** Define helper functions for JSON conversion.
# MAGIC   2. **Process Requests:** Process the raw requests and unpack JSON payloads.
# MAGIC   3. **Convert Data:** Convert timestamps and explode batched requests into individual rows.
# MAGIC

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F, types as T
import json
import pandas as pd

"""
Conversion helper functions.
"""
def convert_to_record_json(json_str: str) -> str:
    """
    Converts records from the four accepted JSON formats for Databricks
    Model Serving endpoints into a common, record-oriented
    DataFrame format which can be parsed by the PySpark function from_json.
    
    :param json_str: The JSON string containing the request or response payload.
    :return: A JSON string containing the converted payload in record-oriented format.
    """
    try:
        request = json.loads(json_str)
    except json.JSONDecodeError:
        return json_str
    output = []
    if isinstance(request, dict):
        obj_keys = set(request.keys())
        if "dataframe_records" in obj_keys:
            # Record-oriented DataFrame
            output.extend(request["dataframe_records"])
        elif "dataframe_split" in obj_keys:
            # Split-oriented DataFrame
            dataframe_split = request["dataframe_split"]
            output.extend([dict(zip(dataframe_split["columns"], values)) for values in dataframe_split["data"]])
        elif "instances" in obj_keys:
            # TF serving instances
            output.extend(request["instances"])
        elif "inputs" in obj_keys:
            # TF serving inputs
            output.extend([dict(zip(request["inputs"], values)) for values in zip(*request["inputs"].values())])
        elif "predictions" in obj_keys:
            # Predictions
            output.extend([{'predictions': prediction} for prediction in request["predictions"]])
        return json.dumps(output)
    else:
        # Unsupported format, pass through
        return json_str


@F.pandas_udf(T.StringType())
def json_consolidation_udf(json_strs: pd.Series) -> pd.Series:
    """A UDF to apply the JSON conversion function to every request/response."""
    return json_strs.apply(convert_to_record_json)

# COMMAND ----------

def process_requests(requests_raw: DataFrame) -> DataFrame:
    """
    Takes a stream of raw requests and processes them by:
        - Unpacking JSON payloads for requests and responses
        - Exploding batched requests into individual rows
        - Converting Unix epoch millisecond timestamps to be Spark TimestampType
        
    :param requests_raw: DataFrame containing raw requests. Assumed to contain the following columns:
                            - `request`
                            - `response`
                            - `timestamp_ms`
    :param request_fields: List of StructFields representing the request schema
    :param response_field: A StructField representing the response schema
    :return: A DataFrame containing processed requests
    """
    # Convert the timestamp milliseconds to TimestampType for downstream processing.
    # Here we are doctoring the timestamps to simulate more time passing between the thousands of requests that we just made. 
    current_ts = spark.sql("select cast(timestamp(current_date()) as int)").collect()[0][0]
    requests_timestamped = (requests_raw 
        #.withColumn('timestamp', (F.col("timestamp_ms") / 1000).cast(T.TimestampType())) # This line is for un-doctored timestamps
        .withColumn('timestamp', ((F.col("timestamp_ms") / 1000 - current_ts)*50 + current_ts).cast(T.TimestampType())) # This line is for doctored timestamps (it stretches time)
        .drop("timestamp_ms"))

    # Consolidate and unpack JSON.
    requests_unpacked = requests_timestamped \
        .withColumn("request", json_consolidation_udf(F.col("request"))) \
        .withColumn('request', F.from_json(F.col("request"), F.schema_of_json('[{"ID": 1.0,"Age": 40.0, "Experience": 10.0, "Income": 84.0, "ZIP_Code": 9302.0, "Family": 3.0, "CCAvg": 2.0, "Education": 2.0, "Mortgage": 0.0, "Securities_Account": 0.0, "CD_Account": 0.0, "Online": 1.0, "CreditCard": 1.0}]'))) \
        .withColumn("response", F.from_json(F.col("response"), F.schema_of_json('{"predictions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}')))  \
        .withColumn('response', F.col("response").predictions)

    # Explode batched requests into individual rows.
    DB_PREFIX = "__db"
    requests_exploded = requests_unpacked \
        .withColumn(f"{DB_PREFIX}_request_response", F.arrays_zip(F.col("request"), F.col("response"))) \
        .withColumn(f"{DB_PREFIX}_request_response", F.explode(F.col(f"{DB_PREFIX}_request_response"))) \
        .select(F.col("*"), F.col(f"{DB_PREFIX}_request_response.request.*"), F.col(f"{DB_PREFIX}_request_response.response").alias("Personal_Loan")) \
        .drop(f"{DB_PREFIX}_request_response", "request", "response") \
        .withColumn('model_id', F.lit(0))

    return requests_exploded

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4.2: Monitoring the Inference Table
# MAGIC In this task, you will monitor the inference table to ensure that data is populating correctly as the model starts receiving requests.
# MAGIC
# MAGIC **Steps:**
# MAGIC   1. **Monitor Table:** Monitor the inference table to ensure data is populating correctly.
# MAGIC   2. **Check Table:** Implement a loop to wait and check for the table population, retrying until the data appears.
# MAGIC

# COMMAND ----------

import time
from datetime import datetime, timedelta

# Set the start time and the duration for which the loop should run (20 minutes)
start_time = datetime.now()
duration = timedelta(minutes=20)

# Loop until the inference table populates and is found or the duration has elapsed
while datetime.now() - start_time < duration:
    try:
        # Attempt to read the table
        inference_df = spark.read.table(f"{catalog_name}.{schema_name}.loan_payload")
        
        # Check if the table is not empty
        if inference_df.count() > 0:
            # If successful and the table is not empty, display the DataFrame and break the loop
            display(inference_df)
            break
        else:
            # If the table is empty, wait for 10 seconds before trying again
            print("Table is empty, trying again in 10 seconds.")
            time.sleep(10)
    except Exception as e:
        # If the table is not found, wait for 10 seconds before trying again
        print("Table not found, trying again in 10 seconds.")
        time.sleep(10)

# If the loop completes without breaking, it means the table was not found within the duration
else:
    print("Failed to find the table within 20 minutes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4.3: Analyzing Processed Requests
# MAGIC
# MAGIC After processing and unpacking the logged data from the inference table, the next step is to analyze the requests that were successfully answered by the model, filtering and joining with additional label data for comprehensive analysis.
# MAGIC
# MAGIC **Steps:**
# MAGIC   1. **Process Data:** Filter and analyze the requests that were successfully answered by the model.
# MAGIC   2. **Join Data:** Join with additional label data for comprehensive analysis.
# MAGIC

# COMMAND ----------

# Process the inference data
model_logs_df = process_requests(inference_df.where("status_code = 200")) # Let's ignore bad requests

# Ensure the ID column is added during processing
display(model_logs_df)

# COMMAND ----------

# Convert Pandas DataFrame to Spark DataFrame
loan_spark_df = spark.createDataFrame(loan_pd_df)

# Rename 'Personal_Loan' to 'labeled_data' in the loan_spark_df
label_spark_df = loan_spark_df.withColumnRenamed('Personal_Loan', 'labeled_data')

# Join with model_logs_df
model_logs_df_labeled = model_logs_df.join(
    label_spark_df.select("ID", "labeled_data"), 
    on="ID", 
    how="left"
)

# Display the joined DataFrame
display(model_logs_df_labeled)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5: Persisting Processed Model Logs
# MAGIC
# MAGIC In this task, you will save the enriched model logs to ensure long-term availability for ongoing monitoring and analysis.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. **Convert Data Types:** Convert all columns to appropriate data types.
# MAGIC 2. **Save Logs:** Save the processed logs to a designated storage for long-term use.
# MAGIC 3. **Enable CDF:** Enable Change Data Feed (CDF) to facilitate efficient incremental processing of new data.
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F, types as T

# Convert all columns to IntegerType except those of type MAP<STRING, STRING>
for col_name, col_type in model_logs_df_labeled.dtypes:
    if col_type != 'map<string,string>':
        model_logs_df_labeled = model_logs_df_labeled.withColumn(
            col_name, 
            F.col(col_name).cast(T.IntegerType())
        )

# Overwrite the existing table with the updated DataFrame
model_logs_df_labeled.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{catalog_name}.{schema_name}.model_logs")

# COMMAND ----------

# MAGIC %md
# MAGIC For efficient execution, enable CDF (Change Data Feeed) so monitoring can incrementally process the data.

# COMMAND ----------

spark.sql(f'ALTER TABLE {catalog_name}.{schema_name}.model_logs SET TBLPROPERTIES (delta.enableChangeDataFeed = true)')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6: Setting Up and Monitoring Inference Data
# MAGIC
# MAGIC This task includes setting up the monitoring of inference data and ensuring its continuous availability for analysis and monitoring.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 6.1: Creating an Inference Monitor with Databricks Lakehouse Monitoring
# MAGIC Set up monitoring to continuously track model performance and detect any anomalies or drift in real-time.
# MAGIC **Steps:**
# MAGIC   1. **Configure Monitor:** Configure and initiate the monitoring of model logs.
# MAGIC   2. **Create Monitor:** Create an inference monitor and validate its creation.
# MAGIC   3. **Verify Metrics:** Verify that metrics tables are created and populated.
# MAGIC

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorInfoStatus, MonitorRefreshInfoState, MonitorMetric

w = WorkspaceClient()
table_name = f'{catalog_name}.{schema_name}.model_logs'
baseline_table_name = f"{catalog_name}.{schema_name}.baseline_features"

# ML problem type, either "classification" or "regression"
PROBLEM_TYPE = MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION

# Window sizes to analyze data over
GRANULARITIES = ["5 minutes"]

# Directory to store generated dashboard
ASSETS_DIR = f"/Workspace/Users/{DA.username}/databricks_lakehouse_monitoring/model_logs"

# Optional parameters
SLICING_EXPRS = ["age<20", "age>70", "CreditCard='1'", "Income<20", "Income>120"]   # Expressions to slice data with
print(f"Creating monitor for model_logs")

info = w.quality_monitors.create(
  table_name=table_name,
  inference_log=MonitorInferenceLog(
    timestamp_col='timestamp',
    granularities=GRANULARITIES,
    model_id_col='model_id', # Model version number 
    prediction_col='Personal_Loan',
    problem_type=PROBLEM_TYPE,
    label_col='labeled_data' # Optional
  ),
  baseline_table_name=baseline_table_name,
  slicing_exprs=SLICING_EXPRS,
  output_schema_name=f"{catalog_name}.{schema_name}",
  assets_dir=ASSETS_DIR
)

# COMMAND ----------

import time

# Wait for monitor to be created
while info.status == MonitorInfoStatus.MONITOR_STATUS_PENDING:
  info = w.quality_monitors.get(table_name=table_name)
  time.sleep(10)

assert info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE, "Error creating monitor"
# A metric refresh will automatically be triggered on creation
refreshes = w.quality_monitors.list_refreshes(table_name=table_name).refreshes
assert(len(refreshes) > 0)

run_info = refreshes[0]
while run_info.state in (MonitorRefreshInfoState.PENDING, MonitorRefreshInfoState.RUNNING):
  run_info = w.quality_monitors.get_refresh(table_name=table_name, refresh_id=run_info.refresh_id)
  time.sleep(30)

assert run_info.state == MonitorRefreshInfoState.SUCCESS, "Monitor refresh failed"

w.quality_monitors.get(table_name=table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 6.2: Inspect and Monitor Metrics Tables
# MAGIC
# MAGIC In this task, you will learn how to inspect and monitor the metrics tables generated by the Databricks quality monitoring tools. These tables provide valuable insights into the performance and behavior of your models, including summary statistics and data drift detection. Additionally, you will learn how to create alerts based on these metrics to proactively manage and respond to any anomalies or changes in your data.
# MAGIC
# MAGIC You will perform the following steps:
# MAGIC 1. **Inspect the Metrics Tables Using UI**: Locate and review the profile and drift metrics tables created by the monitoring process. These tables are saved in your default database and provide detailed metrics and visualizations.
# MAGIC 2. **Monitor Alerts**: Learn how to create alerts from the monitor dashboard to keep track of important metrics and be notified of any significant changes or anomalies in your data.

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 6.2.1: Inspect the Metrics Tables Using UI**
# MAGIC
# MAGIC
# MAGIC By default, the metrics tables are saved in the default database.
# MAGIC
# MAGIC The `create_monitor` call created two new tables: the profile metrics table and the drift metrics table.
# MAGIC
# MAGIC - **Profile Metrics Table**: This table records summary statistics for each column in the monitored table.
# MAGIC - **Drift Metrics Table**: This table records metrics that compare current values in the monitored table to baseline values, identifying potential drift.
# MAGIC
# MAGIC These tables use the same name as the primary table to be monitored, with the suffixes `_profile_metrics` and `_drift_metrics`.
# MAGIC
# MAGIC > **Instructions:**
# MAGIC > 1. Go to the Table where the monitor is created: `(Table name=f'{catalog_name}.{schema_name}.model_logs')`.
# MAGIC > 2. Check the output tables:
# MAGIC >    - Locate the table with the suffix `_profile_metrics` to view summary statistics for each column.
# MAGIC >    - Locate the table with the suffix `_drift_metrics` to view metrics that compare current values to baseline values.
# MAGIC > 3. View the dashboard associated with these metrics tables.
# MAGIC > 4. Explore the different metrics and visualizations created. The dashboard provides insights into data distribution, potential data drift, and other key metrics.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 6.2.2: Monitor Alerts**
# MAGIC
# MAGIC
# MAGIC You can also quickly create an alert from the monitor dashboard as follows:
# MAGIC
# MAGIC 1. On the dashboard, find the chart for which you want to create an alert.
# MAGIC 2. Click the Kebab menu (three vertical dots) in the upper-right corner of the chart and select **View query**. The SQL editor opens.
# MAGIC 3. In the SQL editor, click the Kebab menu above the editor window and select **Create alert**. The New alert dialog opens in a new tab.
# MAGIC 4. Configure the alert according to your requirements and click **Create alert**.
# MAGIC
# MAGIC **Note**: If the query uses parameters, then the alert is based on the default values for these parameters. You should confirm that the default values reflect the intent of the alert.
# MAGIC
# MAGIC > **Additional Instructions:**
# MAGIC > 1. You can edit the query in the SQL editor and rerun it to adjust the data and metrics as needed.
# MAGIC > 2. Set alerts to monitor any updates or specific conditions in the data.
# MAGIC > 3. Regularly check the dashboard and alert notifications to ensure timely responses to data changes or issues.

# COMMAND ----------

# MAGIC %md
# MAGIC # Explore the dashboard!

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
# MAGIC In this lab, you successfully deployed a machine learning model and set up a monitoring framework to track its performance. You sent batched requests to the model endpoint and monitored the responses to detect any anomalies or drift. Additionally, you explored how to use Databricks Lakehouse Monitoring to continuously track and alert on model performance metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
