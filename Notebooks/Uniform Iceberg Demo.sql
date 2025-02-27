-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Expose Iceberg - Uniform
-- MAGIC ## Read Unity Catalog Tables in Snowflake

-- COMMAND ----------

-- MAGIC %md
-- MAGIC [How to Read Unity Catalog Tables in Snowflake, in 4 Easy Steps](https://www.databricks.com/blog/read-unity-catalog-tables-in-snowflake)
-- MAGIC
-- MAGIC There are 4 steps to creating a REST catalog integration in Snowflake:
-- MAGIC
-- MAGIC - Enable UniForm on a Delta Lake table in Databricks to make it accessible through the Iceberg REST Catalog
-- MAGIC - Register Unity Catalog in Snowflake as your catalog
-- MAGIC - Register an S3 Bucket in Snowflake so it recognizes the source data
-- MAGIC - Create an Iceberg table in Snowflake so you can query your data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 1. With UC as delta catalog 
-- MAGIC 2. With UC + iceberg-rest endpoint

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## With Unity Catalog 
-- MAGIC
-- MAGIC - UC required
-- MAGIC - Enable column mapping
-- MAGIC - minReaderVersion >= 2 and minWriterVersion >= 7. 
-- MAGIC - DBR > 14.3LTS
-- MAGIC - Disable DV

-- COMMAND ----------

create catalog if not exists main;
create schema if not exists main.default;

-- COMMAND ----------

create or replace table  main.default.uc_table_name (id INT, name STRING)
tblproperties (
  'delta.columnMapping.mode' = 'name',
  'delta.enableIcebergCompatV2' = 'true',
  'delta.universalFormat.enabledFormats' = 'iceberg'
);

-- COMMAND ----------

ALTER TABLE main.default.uc_table_name
SET TBLPROPERTIES (
  'delta.columnMapping.mode' = 'name',
  'delta.enableIcebergCompatV2' = 'true',
  'delta.universalFormat.enabledFormats' = 'iceberg'
);

-- COMMAND ----------

-- MAGIC %python
-- MAGIC for i in range(1, 20):
-- MAGIC     spark.sql(f"INSERT INTO main.default.uc_table_name VALUES ({i}, 'Name_{i}')")
-- MAGIC

-- COMMAND ----------

describe detail main.default.uc_table_name

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dbutils.fs.ls("s3://databricks-e2demofieldengwest/b169b504-4c54-49f2-bc3a-adf4b128f36d/tables/3f3a5dc1-a1c7-40c1-93bd-25f40562265e/_delta_log")
-- MAGIC

-- COMMAND ----------

describe extended main.default.uc_table_name

-- COMMAND ----------

-- s3://databricks-e2demofieldengwest/b169b504-4c54-49f2-bc3a-adf4b128f36d/tables/3f3a5dc1-a1c7-40c1-93bd-25f40562265e/metadata/00058-1229fe63-b58b-4a73-9543-9320e404a62c.metadata.json

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dbutils.fs.ls("s3://databricks-e2demofieldengwest/b169b504-4c54-49f2-bc3a-adf4b128f36d/tables/3f3a5dc1-a1c7-40c1-93bd-25f40562265e/metadata/")

-- COMMAND ----------

-- MAGIC %python
-- MAGIC spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")

-- COMMAND ----------

VACUUM main.default.uc_table_name RETAIN 0 HOURS FULL

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Iceberg metadata generation is done asynchronously. You can manually trigger Iceberg metadata conversion
-- MAGIC

-- COMMAND ----------

MSCK REPAIR TABLE main.default.uc_table_name SYNC METADATA

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Hive table

-- COMMAND ----------

describe extended delta.`s3://databricks-e2demofieldengwest/b169b504-4c54-49f2-bc3a-adf4b128f36d/tables/3f3a5dc1-a1c7-40c1-93bd-25f40562265e/`

-- COMMAND ----------

-- MAGIC %python
-- MAGIC for i in range(1, 3):
-- MAGIC     spark.sql(f"INSERT INTO main.default.uc_table_name VALUES ({i}, 'Name_{i}')")
