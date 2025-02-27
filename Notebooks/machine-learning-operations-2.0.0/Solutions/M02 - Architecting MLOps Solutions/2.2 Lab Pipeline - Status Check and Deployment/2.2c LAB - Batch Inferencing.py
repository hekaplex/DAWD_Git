# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB - Batch Inference
# MAGIC
# MAGIC In this lab, you will perform batch inference using a deployed machine learning model. This involves loading new data, preparing it for inference, and performing the inference using the model endpoint.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC _By the end of this lab, you will be able to:_
# MAGIC
# MAGIC - Load new data for inference.
# MAGIC - Prepare data for batch inference.
# MAGIC - Perform inference and test the endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC - To run this notebook, you need to use one of the following Databricks runtime(s): `14.3.x-cpu-ml-scala2.12`
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:
# MAGIC

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup-1.3

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions**
# MAGIC
# MAGIC Throughout this lab, we'll refer to the object DA. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load New Data for Inference
# MAGIC Load the new data for inference from the cleaned feature table.

# COMMAND ----------

import pandas as pd
from sklearn.impute import SimpleImputer

# Load new data for inference
new_data_path = f"{DA.catalog_name}.{DA.schema_name}.telco_cleaned_table"
new_data = spark.table(new_data_path).toPandas()
new_data = new_data.drop(columns=['customerID'])

# Add unique_id column
new_data['unique_id'] = range(len(new_data))

# Handle missing values
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(new_data)
new_data = pd.DataFrame(imputed_data, columns=new_data.columns)

# Convert all columns in the DataFrame to the 'double' data type
for column in new_data.columns:
    new_data[column] = new_data[column].astype("double")

display(new_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Prepare Data for Batch Inference
# MAGIC Prepare the new data for batch inference by selecting relevant features and converting the data back to a Spark DataFrame.
# MAGIC

# COMMAND ----------

# Features to use
primary_key = "unique_id"
response = "Churn"
features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'num_optional_services'
]

# Convert the DataFrame back to Spark DataFrame
new_data_spark = spark.createDataFrame(new_data)

# Split with 80 percent of the data in train_df and 20 percent of the data in test_df
train_df, test_df = new_data_spark.randomSplit([.8, .2], seed=42)

# Separate features and ground-truth
features_df = train_df.select(primary_key, *features)
response_df = train_df.select(primary_key, response)

# Review the features dataset
display(features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Perform Inference and Test the Endpoint
# MAGIC Perform inference on a test sample and display the results.

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient
import pandas as pd

# Initialize Databricks Workspace client
w = WorkspaceClient()

# Convert the Spark DataFrame to a list of dictionaries
features_records = features_df.toPandas().to_dict(orient='records')
endpoint_name = f"{DA.username}_ML_AS_04_Lab2"
endpoint_name = endpoint_name.replace(".", "-")
endpoint_name = endpoint_name.replace("@", "-")
# Perform batch scoring
query_response = w.serving_endpoints.query(
    name=endpoint_name, 
    dataframe_records=features_records
)

# Convert the predictions to a DataFrame
predictions_df = pd.DataFrame(query_response.predictions, columns=['prediction'])

# Convert features_df and response_df to pandas DataFrames
features_pd_df = features_df.toPandas()
response_pd_df = response_df.toPandas()

# Merge the features, response, and predictions
results_df = features_pd_df.copy()
results_df['Churn'] = response_pd_df.set_index('unique_id')['Churn']
results_df['prediction'] = predictions_df['prediction']

# Print inference results
print("Inference results:")

# Display the combined DataFrame
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up Classroom
# MAGIC 🚨 **Do not run clean up**
# MAGIC
# MAGIC You would be using the output in the next notebook, so avoid running the clean up.

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this lab, you learned how to perform batch inference using a deployed machine learning model. You loaded new data, prepared it for inference, and performed the inference using the model endpoint. This process is crucial for applying trained models to new data in a real-world setting, enabling automated and scalable predictions.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
