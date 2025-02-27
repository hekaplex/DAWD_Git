# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB - Data Cleaning and Feature Engineering
# MAGIC
# MAGIC In this lab, you will clean the telco customer churn data and perform feature engineering. This is the first step in the MLOps workflow, preparing the data for model training.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC _This lab contains the following tasks:_
# MAGIC - **Task 1:** Load and explore the raw dataset
# MAGIC - **Task 2:** Clean the dataset and handle missing values
# MAGIC - **Task 3:** Engineer features for model training
# MAGIC - **Task 4:** Create and load a feature table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC To run this notebook, you need to use one of the following Databricks runtime(s): 14.3.x-cpu-ml-scala2.12
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup-1.2

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
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
# MAGIC ## Task 1: Load and Explore the Raw Dataset
# MAGIC In this step, you will load the raw telco customer churn dataset and display its contents. This will help you understand the structure and content of the data.
# MAGIC
# MAGIC **Instructions:**
# MAGIC > - Load the raw telco customer churn dataset from a CSV file.
# MAGIC > - Display the dataset to understand its structure and content.

# COMMAND ----------

# Define the path to the dataset using an f-string for dynamic path construction
dataset_path = f"{DA.paths.datasets}/telco/telco-customer-churn-missing.csv"

# Read the dataset into a Spark DataFrame. The 'csv' format and 'true' header option are specified
telcoDF = spark.read.format("csv").option("header", "true").load(dataset_path)

# Display the DataFrame
display(telcoDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Define data cleaning and feature engineering logic
# MAGIC In this step, you will define a function to clean the data and perform feature engineering. The function will:
# MAGIC - Convert specific columns to appropriate data types.
# MAGIC - Handle missing values by filling them with suitable default values.
# MAGIC - Create a new feature that counts the number of optional services a customer has subscribed to.

# COMMAND ----------

# MAGIC %md
# MAGIC **Instructions:**
# MAGIC > - Define the data cleaning and feature engineering function.

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

def clean_churn_features(dataDF: DataFrame) -> DataFrame:
    """
    Cleans and prepares features for the churn dataset.
    """
    # Convert the Spark DataFrame to a pandas on Spark DataFrame
    data_psdf = dataDF.pandas_api()

    # Convert 'SeniorCitizen' from numeric to categorical string values
    data_psdf['SeniorCitizen'] = data_psdf['SeniorCitizen'].map({1: 'Yes', 0: 'No'})

    # Convert 'TotalCharges' to double type
    data_psdf = data_psdf.astype({'TotalCharges': 'double', 'SeniorCitizen': 'string'})

    # Fill missing values with 0.0 for specified columns
    data_psdf = data_psdf.fillna({'tenure': 0.0, 'MonthlyCharges': 0.0, 'TotalCharges': 0.0})

    # Define a function to count the number of optional services
    def sum_optional_services(df):
        # List of columns representing optional services
        cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        # Sum the count of 'Yes' values in the optional service columns
        return sum(map(lambda c: (df[c] == "Yes"), cols))

    # Columns to be converted from 'Yes'/'No' to 1/0
    binary_columns = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity",
                      "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                      "PaperlessBilling", "Churn"]
    
    # Columns to be converted from categorical strings to numeric values
    String_to_float = ["gender", "InternetService", "Contract", "PaymentMethod"]

    # Mapping categorical string values to numeric values for specified columns
    for col in String_to_float:
        data_psdf[col] = data_psdf[col].map({"Male": 1, "Female": 0, "Fiber optic": 1, "None": 0, "No": 0, "DSL": 2, 
                                             "Month-to-month": 1, "One year": 2, "Two year": 3, "Credit card (automatic)": 1, 
                                             "None": 0, "Mailed check": 2, "Bank transfer (automatic)": 3, "Electronic check": 4})

    # Mapping binary columns from 'Yes'/'No' to 1/0
    for col in binary_columns:
        data_psdf[col] = data_psdf[col].map({"Yes": 1, "No": 0})
   
    # Calculate the number of optional services for each record
    data_psdf["num_optional_services"] = sum_optional_services(data_psdf)
    
    # Convert the pandas on Spark DataFrame back to a Spark DataFrame and return it
    return data_psdf.to_spark()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2.2: Clean the data
# MAGIC
# MAGIC In this step, you will apply the data cleaning and feature engineering function to the raw dataset. 
# MAGIC
# MAGIC This will produce a cleaned dataset with new features that are ready for model training.
# MAGIC
# MAGIC > - Apply the data cleaning and feature engineering function to the raw dataset.
# MAGIC > - Display the cleaned dataset to verify the changes.
# MAGIC

# COMMAND ----------

# Apply the data cleaning function
cleaned_data = clean_churn_features(telcoDF)

# Display the cleaned data
display(cleaned_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Save the cleaned and feature-engineered data
# MAGIC
# MAGIC In this step, you will save the cleaned and feature-engineered data as a Delta table. This table will be used in subsequent labs for model training and evaluation.
# MAGIC
# MAGIC **Instructions:**
# MAGIC > - Save the cleaned and feature-engineered data as a Delta table.
# MAGIC > - Print the path where the cleaned data is saved.
# MAGIC

# COMMAND ----------

# Specify the table name
cleaned_data_path = f"/Users/{DA.username}/cleaned_telco_data"

# Save the cleaned_data DataFrame as a Delta table
cleaned_data.write.mode("overwrite").format("delta").save(cleaned_data_path)

# Print a success message
print(f"Cleaned data saved to: {cleaned_data_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 4: Create and Load Feature Table
# MAGIC In this final step, you will create a feature table from the cleaned data and load it for further analysis.
# MAGIC
# MAGIC **Instructions:**
# MAGIC > - Create a feature table from the cleaned data.
# MAGIC > - Load the feature table and display its contents.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

# Initialize the FeatureEngineeringClient
fe = FeatureEngineeringClient()

# Load the data into a DataFrame
df = spark.read.format("delta").load(cleaned_data_path)

# Define the name of the feature table using dynamic path construction
table_name = f"{DA.catalog_name}.{DA.schema_name}.telco_cleaned_table"

# Create a feature table from the dataset
fe.create_table(
    name=table_name,
    primary_keys=["customerID"],
    df=df,
    description="Telco customer features",
    tags={"source": "bronze", "format": "delta"}
)

# Retrieve the feature table by name
ft = fe.get_table(name=table_name)

# Print the description of the feature table
print(f"Feature Table description: {ft.description}")

# Display the data from the feature table
display(fe.read_table(name=table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up Classroom
# MAGIC 🚨 **Do not run Clean up**
# MAGIC
# MAGIC You would be using the created table in the next lab, so avoid running the clean up.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this lab, you learned how to load, clean, and perform feature engineering on a raw dataset. These prepared features will be used in subsequent labs for model training and evaluation.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
