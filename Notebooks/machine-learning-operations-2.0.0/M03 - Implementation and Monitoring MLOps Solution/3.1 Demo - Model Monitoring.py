# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Demo - Model Monitoring
# MAGIC
# MAGIC In this Demo, we will show you how to deploy a machine learning model and monitor its performance using Databricks. We will use a diabetes dataset to train a model, deploy it, and then send batched requests to evaluate the model's performance. We will also set up comprehensive monitoring using Databricks' built-in features.  Additionally, you will handle drift detection and trigger retraining and redeployment when necessary.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this Demo, you will be able to:
# MAGIC - Train and deploy a machine learning model.
# MAGIC - Send batched requests to the deployed model endpoint.
# MAGIC - Monitor the model's performance and detect anomalies or drift.
# MAGIC - Handle drift detection and trigger retraining and redeployment when necessary.
# MAGIC - Set up and utilize Databricks Lakehouse Monitoring to continuously track and alert on model performance metrics.

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

# MAGIC %pip install "databricks-sdk>=0.28.0"
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-3.1

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
# MAGIC # Prepare Model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load dataset
# MAGIC
# MAGIC In this section, we load the dataset for diabetes classification. Since the dataset is small and we want to go straight to training a classic model, we load it directly with Pandas.

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import lit, col
from pyspark.sql.types import DoubleType

# Load the Delta table into a Spark DataFrame
dataset_path = f"{DA.paths.working_dir}/diabetes-dataset"
diabetes_df = spark.read.format("delta").load(dataset_path)

# Convert to Pandas DataFrame
diabetes_df_pd = diabetes_df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train / New Requests Split
# MAGIC
# MAGIC In a normal model training situation, we would split the data between a **training** and a **test** set in order to evaluate the performance of that model. 
# MAGIC Here, we are concerned about what happens once a model is already trained and deployed in production, and new requests that are not part of our existing dataset arrive. 
# MAGIC
# MAGIC To simulate this situation, we are simply going to split our existing data between a **training** set, and a **new requests** set that we will send to the endpoint once the model is deployed. 
# MAGIC
# MAGIC In order to simulate a drift in the model features, we will use the `Age` feature of our data to split these sets.

# COMMAND ----------

train_df = diabetes_df_pd[diabetes_df_pd['Age'] <= 9]
request_df = diabetes_df_pd[diabetes_df_pd['Age'] > 9]

target_col = "Diabetes_binary" 
X_train = train_df.drop(labels=[target_col, 'id'], axis=1)
y_train = train_df[target_col]
X_request = request_df.drop(labels=target_col, axis=1)
y_request = request_df[target_col]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit a Classification Model
# MAGIC
# MAGIC Let's go ahead and fit a Decision Tree model and register it with Unity Catalog.

# COMMAND ----------

import mlflow
from math import sqrt
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
# register models in UC
mlflow.set_registry_uri("databricks-uc")
mlflow.sklearn.autolog(log_input_examples=True)

dtc = DecisionTreeClassifier()
dtc_mdl = dtc.fit(X_train, y_train)

model_name = f"{catalog_name}.{schema_name}.diabetes_model"
signature = infer_signature(X_train, y_train)
mlflow.sklearn.log_model(
    sk_model = dtc_mdl, 
    artifact_path="model-artifacts",
    signature=signature,
    registered_model_name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploying the Model
# MAGIC
# MAGIC Once your model is trained and ready for production, the next step is deployment. You can deploy your model using one of the following two methods depending on your workflow preference:

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1: Using the Notebook
# MAGIC For those who prefer coding, you can deploy your model by executing the provided Python code in your notebook. This method offers flexibility and control, allowing for programmatically managing various aspects of the deployment.
# MAGIC
# MAGIC **🚨Warning:** This process may take around 20 minutes to complete. If an error occurs indicating that the endpoint has not been created within 20 minutes, please check in the Serving section. It may take a few more minutes for the endpoint to become fully ready.
# MAGIC

# COMMAND ----------

# Deploying the Model
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointTag
import mlflow

w = WorkspaceClient()

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()
served_model_name = model_name.split('.')[-1]

# Configure the endpoint
endpoint_config_dict = {
    "served_models": [
        {
            "model_name": model_name,
            "model_version": "1",
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
        },
    ],
    "auto_capture_config":{
        "catalog_name": catalog_name,
        "schema_name": schema_name,
        "table_name_prefix": "diabetes_binary"
    }
}

endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

endpoint_name = f"{DA.username}_ML_AS04_Demo_3b"
endpoint_name = endpoint_name.replace(".", "-")
endpoint_name = endpoint_name.replace("@", "-")

dbutils.jobs.taskValues.set(key="endpoint_name", value=endpoint_name)
print(f"Endpoint name: {endpoint_name}")

try:
    w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=endpoint_config,
        tags=[EndpointTag.from_dict({"key": "db_academy", "value": "Demo3_monitor_model"})]
    )
    print(f"Creating endpoint {endpoint_name} with models {model_name} versions 1")
except Exception as e:
    if "already exists" in e.args[0]:
        print(f"Endpoint with name {endpoint_name} already exists")
    else:
        raise(e)

# Display the endpoint URL
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{endpoint_name}">Model Serving Endpoint')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2: Using the UI
# MAGIC If you prefer a graphical interface, Databricks provides a user-friendly UI that guides you through the deployment process step-by-step. This method is useful for those who prefer not to engage with code directly.
# MAGIC
# MAGIC Both methods will set up your model in the serving environment where it can begin handling inference requests. Detailed instructions and code snippets for the notebook method, as well as step-by-step guidelines for the UI method, are provided below.
# MAGIC

# COMMAND ----------

print(f'Endpoint name: {endpoint_name}')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Create a serving endpoint with the UI
# MAGIC
# MAGIC To provision a serving endpoint via UI, follow the steps below.
# MAGIC
# MAGIC 1. In the left sidebar, click **Serving**.
# MAGIC
# MAGIC 2. To create a new serving endpoint, click **Create serving endpoint**.   
# MAGIC   
# MAGIC     a. In the **Name** field, type a name for the endpoint provided above.  
# MAGIC   
# MAGIC     b. Click in the **Entity** field. A dialog appears. Select **Unity catalog model**, and then select the catalog, schema, and model from the drop-down menus.  
# MAGIC   
# MAGIC     c. In the **Version** drop-down menu, select the version of the model to use.  
# MAGIC   
# MAGIC     d. Click **Confirm**.  
# MAGIC   
# MAGIC     e. In the **Compute Scale-out** drop-down, select **Small**. 
# MAGIC
# MAGIC     f. Check **Scale to zero**
# MAGIC
# MAGIC     g. Scroll to the bottom of the settings and Expand the **Inference tables** section.
# MAGIC
# MAGIC     h. Check the **Enable inference tables** box.
# MAGIC
# MAGIC     i. Enter the catalog, schema, and table information for the inference table:
# MAGIC     - **Catalog Name:** `{catalog_name}`
# MAGIC     - **Schema Name:** `{schema_name}`
# MAGIC     - **Table Name Prefix:** `<Your Table Name>` `(e.g.: diabetes_binary)`
# MAGIC     
# MAGIC     j. Click **Create**. The endpoint page opens and the endpoint creation process starts.  
# MAGIC  
# MAGIC   <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/inference_df.png"  width="=50%"> 
# MAGIC   
# MAGIC See the Databricks documentation for details ([AWS](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html#ui-workflow)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints#--ui-workflow)).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verify Endpoint Creation
# MAGIC
# MAGIC After a couple of minutes, the endpoint should be ready to use.
# MAGIC Let's verify that the endpoint is created and ready.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import json

w = WorkspaceClient()

endpoint = w.serving_endpoints.wait_get_serving_endpoint_not_updating(endpoint_name)

assert endpoint.state.config_update.value == "NOT_UPDATING" and endpoint.state.ready.value == "READY" , "Endpoint not ready or failed"

print('Endpoint was created and is ready.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the Training Data as Reference for Drift
# MAGIC
# MAGIC We can save the training dataset that was used to train the model. This can be used later during monitoring to provide a reference to determine if a drift has happened between training and the new incoming requests.

# COMMAND ----------

from pyspark.sql.functions import lit, col
from pyspark.sql.types import DoubleType

spark_df = spark.createDataFrame(train_df).withColumn('model_id', lit(0)).withColumn("labeled_data", col("Diabetes_binary").cast(DoubleType()))

(spark_df
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema",True)
  .option("delta.enableChangeDataFeed", "true")
  .saveAsTable(f"{catalog_name}.{schema_name}.baseline_features")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sending Batched Requests to the Model Endpoint
# MAGIC
# MAGIC To evaluate the model's performance and response under varying conditions, we will simulate a series of requests using pre-saved data. This process helps us monitor how the model handles inputs over time and under different load scenarios.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Steps for Sending Requests:
# MAGIC 1. **Initial Pause**: Start with a brief pause of 10 seconds. This prevents potential errors that might occur due to immediate loading after setup, ensuring that the endpoint is fully ready to receive requests.
# MAGIC 2. **Prepare Data**: Sort the requests by their IDs to maintain a consistent order of processing. This ordering can be crucial for testing models under controlled conditions to simulate real-world scenarios.
# MAGIC 3. **Batching the Requests**: Divide the sorted data into smaller batches. In this example, we use a batch size of 10 to mimic a realistic scenario where requests come in small groups rather than all at once.
# MAGIC 4. **Sending Requests**: Loop through each batch and send them sequentially to the endpoint. This methodical approach allows us to closely monitor the model’s predictions for each batch and adjust parameters if needed.
# MAGIC 5. **Processing Responses**: For each batch sent, capture the response from the model. This data is essential for subsequent analysis to evaluate model accuracy and performance under simulated conditions.
# MAGIC
# MAGIC This structured approach to sending batched requests helps in understanding the model's behavior in a production-like environment, ensuring it performs well under various data flows and conditions.
# MAGIC

# COMMAND ----------


# Wait for 10 seconds to prevent error and needing to rerun
time.sleep(10)

requests = X_request.sort_values(by=['id']).reset_index()

batch_size = 10
number_of_batches = len(requests) // batch_size

for batch_num in range(number_of_batches):
    batch = payload(requests[requests.index//batch_size == batch_num].to_dict(orient='split'))
    query_response = w.serving_endpoints.query(name=endpoint_name, dataframe_split=batch)
    print(f"Batch {batch_num + 1} response:")
    print(query_response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Processing Inference Table Data
# MAGIC
# MAGIC After deploying the model and starting to receive requests, the next step is to extract and analyze the data logged in the inference table. The inference table logs detailed data on each request and response, but the format is optimized for storage rather than ease of analysis. We will convert this data into a more analyzable format using several steps:
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Steps to Process Inference Table Data:
# MAGIC 1. **Timestamp Conversion**: Convert timestamp data from milliseconds to a more readable timestamp format. This adjustment helps in analyzing the data according to the time events occurred.
# MAGIC 2. **Unpacking JSON**: The requests and responses in the inference table are stored in JSON format. We use custom functions to unpack these JSON strings into a structured DataFrame format, which allows for easier manipulation and analysis.
# MAGIC 3. **Exploding Batched Requests**: If the model receives batched requests, these are exploded into individual records to simplify analysis. This process ensures that each entry in a batch request is treated as a separate data point for accuracy in monitoring.
# MAGIC 4. **Schema Transformation**: We transform the schema of the extracted data to align with analytical needs, facilitating easier data interpretation and analysis.
# MAGIC
# MAGIC This transformation is crucial for monitoring model performance and understanding how the model interacts with incoming data. For more detailed examples and helper functions, see the [Databricks documentation on model serving](https://docs.databricks.com/en/machine-learning/model-serving/inference-tables.html).

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F, types as T

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
        .withColumn('request', F.from_json(F.col("request"), F.schema_of_json('[{"HighBP": 1.0, "HighChol": 0.0, "CholCheck": 1.0, "BMI": 26.0, "Smoker": 0.0, "Stroke": 0.0, "HeartDiseaseorAttack": 0.0, "PhysActivity": 1.0, "Fruits": 0.0, "Veggies": 1.0, "HvyAlcoholConsump": 0.0, "AnyHealthcare": 1.0, "NoDocbcCost": 0.0, "GenHlth": 3.0, "MentHlth": 5.0, "PhysHlth": 30.0, "DiffWalk": 0.0, "Sex": 1.0, "Age": 4.0, "Education": 6.0, "Income": 8.0,"id": 1}]'))) \
        .withColumn("response", F.from_json(F.col("response"), F.schema_of_json('{"predictions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}')))  \
        .withColumn('response', F.col("response").predictions)
        #.withColumn("response", json_consolidation_udf(F.col("response"))) \

    # Explode batched requests into individual rows.
    DB_PREFIX = "__db"
    requests_exploded = requests_unpacked \
        .withColumn(f"{DB_PREFIX}_request_response", F.arrays_zip(F.col("request"), F.col("response"))) \
        .withColumn(f"{DB_PREFIX}_request_response", F.explode(F.col(f"{DB_PREFIX}_request_response"))) \
        .select(F.col("*"), F.col(f"{DB_PREFIX}_request_response.request.*"), F.col(f"{DB_PREFIX}_request_response.response").alias("Diabetes_binary")) \
        .drop(f"{DB_PREFIX}_request_response", "request", "response") \
        .withColumn('model_id', F.lit(0))

    return requests_exploded

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring the Inference Table
# MAGIC
# MAGIC Once your model starts receiving requests, the data will begin populating in the inference table. This process typically starts within a few minutes after deployment.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Steps to Wait for the Table to Populate:
# MAGIC 1. **Initial Wait**: The script starts by waiting for the inference table to populate. It checks every 10 seconds to see if the table is not empty.
# MAGIC 2. **Reading the Table**: It attempts to read the table using Spark's `read.table` function, specifying the table name constructed from the catalog, schema, and endpoint payload details.
# MAGIC 3. **Handling No Data**: If the table is empty, the script pauses for 10 seconds before trying again. This is repeated until the table is found or the specified duration (20 minutes) elapses.
# MAGIC 4. **Error Handling**: If the table is not found or cannot be accessed, the script will print an error message and retry after 10 seconds.
# MAGIC 5. **Completion**: If the loop completes without breaking (i.e., the table remains not found or continuously empty for 20 minutes), it prints a message indicating failure to find the table.
# MAGIC
# MAGIC **Note:** When the endpoint receives batched requests, sometimes the request is down-sampled and not all requests may appear in the inference table. The fraction of records from a batch that gets logged is indicated by the `sampling_fraction` column. This value is between 0 and 1, where 1 represents that 100% of incoming requests were included.
# MAGIC

# COMMAND ----------

# Waiting for the inference table to start to populate.  Usually takes about 5 minutes.

import time
from datetime import datetime, timedelta

# Set the start time and the duration for which the loop should run (20 minutes)
start_time = datetime.now()
duration = timedelta(minutes=20)

# Loop until the inference table populates and is found or the duration has elapsed
while datetime.now() - start_time < duration:
    try:
        # Attempt to read the table
        inference_df = spark.read.table(f"{catalog_name}.{schema_name}.diabetes_binary_payload")
        
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
# MAGIC ## Analyzing Processed Requests
# MAGIC
# MAGIC After processing and unpacking the logged data from the inference table, the next step is to analyze the requests that were successfully answered by the model. This involves filtering and merging additional data for comprehensive analysis.
# MAGIC
# MAGIC ### Steps for Analyzing Processed Requests:
# MAGIC 1. **Filtering Requests**: Initially, we filter out the requests to only include those with a successful status code (200). This ensures that we are analyzing only the requests where the model was able to generate predictions.
# MAGIC 2. **Displaying Logs**: The filtered logs are then displayed, providing a clear view of the processed requests and their outcomes.
# MAGIC 3. **Merging Data**: We also merge these logs with additional label data that categorizes the results. This step is crucial for evaluating the model's performance against known outcomes.
# MAGIC 4. **Final Display**: The merged DataFrame is displayed, showing the complete information of the requests along with their corresponding labels. This provides a full picture of how the model is performing in real-world scenarios.
# MAGIC
# MAGIC This detailed view helps in understanding the effectiveness of the model and in making necessary adjustments based on real-world data feedback.
# MAGIC

# COMMAND ----------

model_logs_df = process_requests(inference_df.where("status_code = 200")) # Let's ignore bad requests
model_logs_df.display()

# COMMAND ----------

label_pd_df = diabetes_df_pd.rename(columns={'Diabetes_binary': 'labeled_data'})
label_pd_df = spark.createDataFrame(label_pd_df)
model_logs_df_labeled = model_logs_df.join(
    label_pd_df.select("id", "labeled_data"), 
    on=["id"], 
    how="left"
).drop("id")
display(model_logs_df_labeled)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persisting Processed Model Logs
# MAGIC
# MAGIC After analyzing the model's responses and merging them with relevant labels, the final step involves saving these enriched logs for long-term monitoring and analysis. This step is critical for maintaining a historical record of model performance and for enabling further analytical studies.
# MAGIC
# MAGIC ### Steps for Saving Model Logs:
# MAGIC 1. **Preparing the Data**: The model logs that have been processed and labeled are now ready to be stored in a more permanent storage solution.
# MAGIC 2. **Setting the Storage Mode**: We use the `append` mode to save the DataFrame. This mode ensures that new entries are added to the existing dataset without overwriting, allowing us to accumulate a comprehensive log over time.
# MAGIC 3. **Saving the DataFrame**: The logs are saved to a table in the Databricks catalog under the specified catalog name and schema. This organizational structure helps in managing the data effectively and facilitates easy access for future queries.
# MAGIC 4. **Confirming Save Operation**: The operation concludes with the logs being successfully appended to the designated table, confirming the completion of data persistence.
# MAGIC
# MAGIC By systematically saving these logs, we establish a robust foundation for ongoing monitoring of the model's performance, enabling proactive management and optimization of machine learning operations.
# MAGIC

# COMMAND ----------

model_logs_df_labeled.write.mode("append").saveAsTable(f'{catalog_name}.{schema_name}.model_logs')

# COMMAND ----------

# MAGIC %md
# MAGIC For efficient execution, enable CDF so monitoring can incrementally process the data.

# COMMAND ----------

spark.sql(f'ALTER TABLE {catalog_name}.{schema_name}.model_logs SET TBLPROPERTIES (delta.enableChangeDataFeed = true)')

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## Creating an Inference Monitor with Databricks Lakehouse Monitoring
# MAGIC
# MAGIC Once your model logs are saved and structured for analysis, the next essential step is setting up monitoring to continuously track model performance and detect any anomalies or drift in real-time. You can set up an inference monitor using Databricks Lakehouse Monitoring through two approaches both methods will enable you to monitor your model's performance efficiently, ensuring that any necessary adjustments or retraining can be handled promptly.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1: Using the Notebook
# MAGIC For those who are comfortable with scripting and want more control over the monitoring setup, you can use the provided notebook commands to configure and initiate the monitoring of your model logs. This method allows you to automate and customize the monitoring according to specific needs and thresholds.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorInfoStatus, MonitorRefreshInfoState, MonitorMetric

w = WorkspaceClient()
Table_name = f'{catalog_name}.{schema_name}.model_logs'
Baseline_table_name = f"{catalog_name}.{schema_name}.baseline_features"

# COMMAND ----------

help(w.quality_monitors.create)

# COMMAND ----------

# ML problem type, either "classification" or "regression"
PROBLEM_TYPE = MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION

# Window sizes to analyze data over
GRANULARITIES = ["5 minutes"]   

# Directory to store generated dashboard
ASSETS_DIR = f"/Workspace/Users/{DA.username}/databricks_lakehouse_monitoring/model_logs"

# Optional parameters
SLICING_EXPRS = ["age<2", "age>10", "sex='1'", "HighChol='1'"]   # Expressions to slice data with

# COMMAND ----------

print(f"Creating monitor for model_logs")

info = w.quality_monitors.create(
  table_name=Table_name,
  inference_log=MonitorInferenceLog(
    timestamp_col='timestamp',
    granularities=GRANULARITIES,
    model_id_col='model_id', # Model version number 
    prediction_col='Diabetes_binary',
    problem_type=PROBLEM_TYPE,
    label_col='labeled_data' # Optional
  ),
  baseline_table_name=Baseline_table_name,
  slicing_exprs=SLICING_EXPRS,
  output_schema_name=f"{catalog_name}.{schema_name}",
  assets_dir=ASSETS_DIR
)

# COMMAND ----------

import time

# Wait for monitor to be created
while info.status ==  MonitorInfoStatus.MONITOR_STATUS_PENDING:
  info = w.quality_monitors.get(table_name=Table_name)
  time.sleep(10)

assert info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE, "Error creating monitor"

# COMMAND ----------

# A metric refresh will automatically be triggered on creation
refreshes = w.quality_monitors.list_refreshes(table_name=Table_name).refreshes
assert(len(refreshes) > 0)

run_info = refreshes[0]
while run_info.state in (MonitorRefreshInfoState.PENDING, MonitorRefreshInfoState.RUNNING):
  run_info = w.quality_monitors.get_refresh(table_name=Table_name, refresh_id=run_info.refresh_id)
  time.sleep(30)

assert run_info.state == MonitorRefreshInfoState.SUCCESS, "Monitor refresh failed"

# COMMAND ----------

# MAGIC %md
# MAGIC Click the highlighted Dashboard link in the cell output to open the dashboard. You can also navigate to the dashboard from the Catalog Explorer UI.

# COMMAND ----------

w.quality_monitors.get(table_name=Table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect the Metrics Tables
# MAGIC
# MAGIC By default, the metrics tables are saved in the default database.  
# MAGIC
# MAGIC The `create_monitor` call created two new tables: the profile metrics table and the drift metrics table. 
# MAGIC
# MAGIC These two tables record the outputs of analysis jobs. The tables use the same name as the primary table to be monitored, with the suffixes `_profile_metrics` and `_drift_metrics`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Orientation to the Profile Metrics Table
# MAGIC
# MAGIC The profile metrics table has the suffix `_profile_metrics`. For a list of statistics that are shown in the table, see the documentation ([AWS](https://docs.databricks.com/lakehouse-monitoring/monitor-output.html#profile-metrics-table)|[Azure](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/monitor-output#profile-metrics-table)).
# MAGIC
# MAGIC - For every column in the primary table, the profile table shows summary statistics for the baseline table and for the primary table. The column `log_type` shows `INPUT` to indicate statistics for the primary table, and `BASELINE` to indicate statistics for the baseline table. The column from the primary table is identified in the column `column_name`.
# MAGIC - For `TimeSeries` type analysis, the `granularity` column shows the granularity corresponding to the row. For baseline table statistics, the `granularity` column shows `null`.
# MAGIC - The table shows statistics for each value of each slice key in each time window, and for the table as whole. Statistics for the table as a whole are indicated by `slice_key` = `slice_value` = `null`.
# MAGIC - In the primary table, the `window` column shows the time window corresponding to that row. For baseline table statistics, the `window` column shows `null`.  
# MAGIC - Some statistics are calculated based on the table as a whole, not on a single column. In the column `column_name`, these statistics are identified by `:table`.

# COMMAND ----------

# Display profile metrics table
profile_table = f"{catalog_name}.{schema_name}.model_logs_profile_metrics"
profile_df = spark.sql(f"SELECT * FROM {profile_table}")
display(profile_df.orderBy(F.rand()).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Orientation to the Drift Metrics Table
# MAGIC
# MAGIC The drift metrics table has the suffix `_drift_metrics`. For a list of statistics that are shown in the table, see the documentation ([AWS](https://docs.databricks.com/lakehouse-monitoring/monitor-output.html#drift-metrics-table)|[Azure](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/monitor-output#drift-metrics-table)).
# MAGIC
# MAGIC - For every column in the primary table, the drift table shows a set of metrics that compare the current values in the table to the values at the time of the previous analysis run and to the baseline table. The column `drift_type` shows `BASELINE` to indicate drift relative to the baseline table, and `CONSECUTIVE` to indicate drift relative to a previous time window. As in the profile table, the column from the primary table is identified in the column `column_name`.
# MAGIC   - At this point, because this is the first run of this monitor, there is no previous window to compare to. So there are no rows where `drift_type` is `CONSECUTIVE`. 
# MAGIC - For `TimeSeries` type analysis, the `granularity` column shows the granularity corresponding to that row.
# MAGIC - The table shows statistics for each value of each slice key in each time window, and for the table as whole. Statistics for the table as a whole are indicated by `slice_key` = `slice_value` = `null`.
# MAGIC - The `window` column shows the the time window corresponding to that row. The `window_cmp` column shows the comparison window. If the comparison is to the baseline table, `window_cmp` is `null`.  
# MAGIC - Some statistics are calculated based on the table as a whole, not on a single column. In the column `column_name`, these statistics are identified by `:table`.

# COMMAND ----------

# Display the drift metrics table
drift_table = f"{catalog_name}.{schema_name}.model_logs_drift_metrics"
display(spark.sql(f"SELECT * FROM {drift_table} ORDER BY RAND() LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Look at fairness and bias metrics
# MAGIC Fairness and bias metrics are calculated for boolean type slices that were defined. The group defined by `slice_value=true` is considered the protected group ([AWS](https://docs.databricks.com/en/lakehouse-monitoring/fairness-bias.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/lakehouse-monitoring/fairness-bias)).

# COMMAND ----------

fb_cols = ["window", "model_id", "slice_key", "slice_value", "predictive_parity", "predictive_equality", "equal_opportunity", "statistical_parity"]
fb_metrics_df = profile_df.select(fb_cols).filter(f"column_name = ':table' AND slice_value = 'true'")
display(fb_metrics_df.orderBy(F.rand()).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2: Using the UI
# MAGIC If you prefer a graphical interface for setup, Databricks offers a user-friendly UI that guides you step-by-step to create and configure the inference monitor. This method is straightforward and ideal for those who wish to quickly set up monitoring without writing code.

# COMMAND ----------

print(f'Your catalog name is: {catalog_name}')
print(f'Your schema name is: {schema_name}')
print(f'Your baseline table name is: {catalog_name}.{schema_name}.baseline_features')

# COMMAND ----------

# MAGIC %md
# MAGIC To add a monitor on the log table, 
# MAGIC
# MAGIC 1. Open the **Catalog** menu from the left menu bar.
# MAGIC
# MAGIC 1. Select the table **model_logs** within your catalog and schema. 
# MAGIC
# MAGIC 1. Click on the **Quality** tab then on the **Get started** button.
# MAGIC
# MAGIC 1. As **Profile type** select **Inference profile**.
# MAGIC
# MAGIC 1. As **Problem type** select **classification**.
# MAGIC
# MAGIC 1. As the **Prediction column** select **Diabetes_binary**.
# MAGIC
# MAGIC 1. As the **Label column** select **labeled_data**
# MAGIC
# MAGIC 1. As **Metric granularities** select **5 minutes**, **1 hour**, and **1 day**. We will use the doctored timestamps to simulate requests that have been received over a large period of time. 
# MAGIC
# MAGIC 1. As **Timestamp column** select **timestamp**.
# MAGIC
# MAGIC 1. As **Model ID column** select **model_id**.
# MAGIC
# MAGIC 1. In Advanced Options --> **Unity catalog baseline table name (optional)** enter **Your baseline table name** from above
# MAGIC
# MAGIC 1. Click the **Create** button.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Handling Drift Detection and Triggering Retraining `(Additional)`
# MAGIC
# MAGIC Handle drift detection by analyzing the drift metrics, and trigger retraining and redeployment if drift is detected.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC - **Analyze Metrics:** Load and analyze the drift metrics table to detect any drift in data or model performance.
# MAGIC - **Trigger Retraining:** Retrain the model if drift is detected.
# MAGIC - **Trigger Redeployment:** Redeploy the model if retraining is triggered.
# MAGIC - **Set Up Alerts:** Set up SQL alerts to notify the team of significant changes or anomalies in the drift metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Analyze Drift Metrics
# MAGIC Analyze the drift metrics to determine if there is significant drift. Here is a sample analysis approach:

# COMMAND ----------

import pandas as pd
import json

# Load the drift metrics data from the Delta table
drift_table = f"{catalog_name}.{schema_name}.model_logs_drift_metrics"
drift_metrics_df = spark.read.table(drift_table)

# Convert to Pandas DataFrame
data = drift_metrics_df.toPandas()

# Convert Timestamp objects to strings in 'window' and 'window_cmp'
def convert_timestamp_to_string(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, pd.Timestamp):
                d[k] = v.isoformat()
            elif isinstance(v, dict):
                d[k] = convert_timestamp_to_string(v)
    return d

data['window'] = data['window'].apply(convert_timestamp_to_string)
data['window_cmp'] = data['window_cmp'].apply(convert_timestamp_to_string)

# Ensure JSON fields are strings
data['window'] = data['window'].apply(json.dumps)
data['window_cmp'] = data['window_cmp'].apply(json.dumps)
data['ks_test'] = data['ks_test'].apply(json.dumps) if data['ks_test'].notna().all() else None
data['chi_squared_test'] = data['chi_squared_test'].apply(json.dumps) if data['chi_squared_test'].notna().all() else None

# Convert the JSON string in 'window' and 'window_cmp' to dictionaries
for index, row in data.iterrows():
    row['window'] = json.loads(row['window'])
    row['window_cmp'] = json.loads(row['window_cmp'])
    row['ks_test'] = json.loads(row['ks_test']) if row['ks_test'] else None
    row['chi_squared_test'] = json.loads(row['chi_squared_test']) if row['chi_squared_test'] else None

# Analyze the drift metrics
drift_thresholds = {
    "js_distance": 0.7,
    "ks_statistic": 0.4,
    "ks_pvalue": 0.05,
    "wasserstein_distance": 0.7,
    "population_stability_index": 0.7,
    "chi_squared_statistic": 0.4,
    "chi_squared_pvalue": 0.05,
    "tv_distance": 0.7,
    "l_infinity_distance": 0.7
}

def check_drift(row):
    if row['js_distance'] is not None and row['js_distance'] > drift_thresholds['js_distance']:
        return True
    ks_test = row['ks_test']
    if ks_test and ks_test['statistic'] > drift_thresholds['ks_statistic'] and ks_test['pvalue'] < drift_thresholds['ks_pvalue']:
        return True
    if row['wasserstein_distance'] is not None and row['wasserstein_distance'] > drift_thresholds['wasserstein_distance']:
        return True
    if row['population_stability_index'] is not None and row['population_stability_index'] > drift_thresholds['population_stability_index']:
        return True
    chi_squared_test = row['chi_squared_test']
    if chi_squared_test and chi_squared_test['statistic'] > drift_thresholds['chi_squared_statistic'] and chi_squared_test['pvalue'] < drift_thresholds['chi_squared_pvalue']:
        return True
    if row['tv_distance'] is not None and row['tv_distance'] > drift_thresholds['tv_distance']:
        return True
    if row['l_infinity_distance'] is not None and row['l_infinity_distance'] > drift_thresholds['l_infinity_distance']:
        return True
    return False

data['drift_detected'] = data.apply(check_drift, axis=1)

# Display rows with drift detected
drifted_rows = data[data['drift_detected']]
no_drifted_rows = data[~data['drift_detected']]

print("Rows with drift detected:")
print(drifted_rows)

print("\nRows with no drift detected:")
print(no_drifted_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Trigger Retraining and Redeployment
# MAGIC If drift is detected, retrain the model and redeploy it using the same steps from the initial model preparation notebook.
# MAGIC
# MAGIC - **Step 1: Retrain and Log the Model**
# MAGIC
# MAGIC > - Retrain the model using DecisionTreeClassifier and log the new version with MLflow.

# COMMAND ----------

# Retrain and redeploy the model if drift is detected
if not drifted_rows.empty:
    print("Drift detected. Retraining and redeploying the model...")

    # Prepare data for retraining
    dtc = DecisionTreeClassifier()
    dtc_mdl = dtc.fit(X_train, y_train)

    model_name = model_name
    
    # Retrain the model
    dtc_mdl = dtc.fit(X_train, y_train)
    signature = infer_signature(X_train, y_train)
    
    mlflow.sklearn.log_model(
        sk_model=dtc_mdl, 
        artifact_path="model-artifacts",
        signature=signature,
        registered_model_name=model_name
    )

# COMMAND ----------

# MAGIC %md
# MAGIC - **Step 2: Update the Endpoint Configuration**
# MAGIC >   Manually update the endpoint configuration through the Databricks UI to use the newly created version of the model.
# MAGIC > 
# MAGIC >   1. **Access the Model Serving Endpoint:**
# MAGIC >     - Navigate to the Model Serving endpoint page in the Databricks workspace.
# MAGIC > 
# MAGIC >   2. **Edit the Endpoint:**
# MAGIC >     - Click the **Edit endpoint** button.
# MAGIC > 
# MAGIC >   3. **Update the Served Model Version:**
# MAGIC >     - In the **Served entities** section, locate the model that is currently being served.
# MAGIC >     - Update the model version to the newly created version.
# MAGIC > 
# MAGIC >   4. **Save Changes:**
# MAGIC >     - Click the **Update** button to save the changes and redeploy the model with the new version.
# MAGIC
# MAGIC These steps ensure that you have retrained the model programmatically and updated the serving endpoint configuration manually to use the new model version.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Monitor Alerts
# MAGIC You can also quickly create an alert from the monitor `dashboard` as follows:
# MAGIC
# MAGIC - On the `dashboard`, find the chart for which you want to create an `alert`.
# MAGIC - Click the Kebab menu in the upper-right corner of the chart and select `View query`. The SQL editor opens.
# MAGIC - In the SQL editor, click the Kebab menu above the editor window and select `Create alert`. The New alert dialog opens in a new tab.
# MAGIC - Configure the alert and click `Create alert`.
# MAGIC
# MAGIC Note that if the query uses parameters, then the alert is based on the default values for these parameters. You should confirm that the default values reflect the intent of the alert.

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
# MAGIC In this demo, we successfully deployed a machine learning model and set up a monitoring framework to track its performance. We sent batched requests to the model endpoint and monitored the responses to detect any anomalies or drift. Additionally, we explored how to use Databricks Lakehouse Monitoring to continuously track and alert on model performance metrics. Furthermore, we handled drift detection and demonstrated how to trigger retraining and redeployment of the model to maintain optimal performance. This comprehensive approach ensures that our machine learning models remain reliable and effective over time.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
