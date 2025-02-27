# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
#DA.reset_lesson()                                   # Reset the lesson to a clean state

# COMMAND ----------

import uuid
import re

def generate_catalog_name():
    raw_name = f'demo_{uuid.uuid4()}_model_monitor'
    clean_name = re.sub(r'[\d-]', '', raw_name)
    return clean_name

catalog_name = generate_catalog_name()
schema_name = f'ws_{spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")}'

# COMMAND ----------

def create_features_table(self):
    from pyspark.sql.functions import monotonically_increasing_id, col
    #catalog_name = f'{DA.catalog_name}_Monitoring'
    #schema_name = f'ws_{spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")}'
    # define active catalog and schema
    
    #spark.sql(f"CREATE CATALOG {catalog_name}")
    #spark.sql(f"USE CATALOG {catalog_name}")
    # Read the dataset
    dataset_path = f"{DA.paths.datasets}/cdc-diabetes/diabetes_binary_5050split_BRFSS2015.csv"
    diabetes_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')
    
    # Convert all columns to double type
    for column in diabetes_df.columns:
        diabetes_df = diabetes_df.withColumn(column, col(column).cast("double"))
    
    # Add a unique ID column
    diabetes_df = diabetes_df.withColumn("id", monotonically_increasing_id())
    
    diabetes_df.write.format("delta").mode("overwrite").save(f"{DA.paths.working_dir}/diabetes-dataset")

DBAcademyHelper.monkey_patch(create_features_table)

class payload():
    def __init__(self, data):
        self.data = data
    def as_dict(self):
        return self.data

# COMMAND ----------

import time
import re
import io
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.functions import col, pandas_udf, transform, size, element_at


def unpack_requests(requests_raw: DataFrame, 
                    input_request_json_path: str, 
                    input_json_path_type: str, 
                    output_request_json_path: str, 
                    output_json_path_type: str,
                    keep_last_question_only: False) -> DataFrame:
    # Rename the date column and convert the timestamp milliseconds to TimestampType for downstream processing.
    requests_timestamped = (requests_raw
        .withColumn("timestamp", (col("timestamp_ms") / 1000))
        .drop("timestamp_ms"))

    # Convert the model name and version columns into a model identifier column.
    requests_identified = requests_timestamped.withColumn(
        "model_id",
        F.concat(
            col("request_metadata").getItem("model_name"),
            F.lit("_"),
            col("request_metadata").getItem("model_version")
        )
    )

    # Filter out the non-successful requests.
    requests_success = requests_identified.filter(col("status_code") == "200")

    # Unpack JSON.
    requests_unpacked = (requests_success
        .withColumn("request", F.from_json(F.expr(f"request:{input_request_json_path}"), input_json_path_type))
        .withColumn("response", F.from_json(F.expr(f"response:{output_request_json_path}"), output_json_path_type)))
    
    if keep_last_question_only:
        requests_unpacked = requests_unpacked.withColumn("request", F.array(F.element_at(F.col("request"), -1)))

    # Explode batched requests into individual rows.
    requests_exploded = (requests_unpacked
        .withColumn("__db_request_response", F.explode(F.arrays_zip(col("request").alias("input"), col("response").alias("output"))))
        .selectExpr("* except(__db_request_response, request, response, request_metadata)", "__db_request_response.*")
        )

    return requests_exploded

# COMMAND ----------

#DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
#DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.create_features_table()  
spark.sql(f"CREATE CATALOG {catalog_name}")
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"CREATE SCHEMA {schema_name}") 
spark.sql(f"USE SCHEMA {schema_name}")  

DA.conclude_setup()                                 # Finalizes the state and prints the config for the student
