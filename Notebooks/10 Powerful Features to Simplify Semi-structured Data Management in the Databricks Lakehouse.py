# Databricks notebook source
# MAGIC %md
# MAGIC #10 Powerful Features to Simplify Semi-structured Data Management in the Databricks Lakehouse
# MAGIC ## - Ingest and query complex JSON data like a pro with Delta Lake and SQL
# MAGIC Ingesting and querying complex JSON files with semi-structured data can be hard but Auto Loader and Delta Lake make it easy. JSON data is very flexible, which makes it powerful, but with this flexibility come issues that can make it difficult to ingest and query such as:
# MAGIC
# MAGIC * It’s tedious and fragile to have to define a schema of the JSON that you plan to ingest.
# MAGIC * The schema can change over time and you need to be able to handle those changes automatically.
# MAGIC * Computers don’t always pick the correct schema for your data and you need a way to hint at the correct format.
# MAGIC * Often data engineers have no control of upstream data sources generating the semi-structured data. Column name may be upper or lower case but denotes the same column, data type sometimes changes and you may not want to completely rewrite the already ingested data in delta lake.
# MAGIC * You may not want to do the upfront work of flattening out JSON files and extracting every single column and doing so may make the data very hard to use.
# MAGIC
# MAGIC In this notebook, we will show you what features make working with JSON at scale simple. Below is an Incremental ETL architecture, the left-hand side represents continuous and scheduled ingest and we will discuss how to do both with Auto Loader. After the JSON is ingested into a bronze Delta Lake table, we will discuss the features that make it easy to query complex and semi-structured data types common in JSON data. We will use sales order data to demonstrate how to easily ingest JSON. The nested JSON sales order data sets get complex quickly. In your industry, data may be different or even more complex. Whether your data looks like this or not, the problems above stay the same.
# MAGIC
# MAGIC
# MAGIC <img src="https://databricks.com/wp-content/uploads/2021/07/get-start-delta-blog-img-1.png" width=1000>
# MAGIC
# MAGIC ##NOTE: Databricks Runtime 9.1 and above is needed to run this notebook
# MAGIC ##NOTE: Run each Cmd one at a time

# COMMAND ----------

# DBTITLE 1,Set up object storage locations
username = spark.sql("select current_user()").collect()[0][0]
userPrefix = username.split("@")[0].replace(".", "")
# base object storage location path derived from the user's name
basePath = "/tmp/" + username + "/autoloaderDemo"
# various object storage locations used in the demo
landingZoneLocation = basePath + "/landingZone"
schemaLocation = basePath + "/schemaStore"
bronzeTableLocation = basePath + "/datastore/bronzeTbl" 
bronzeCheckPointLocation = basePath + "/datastore/bronzeCheckpoint"

spark.conf.set("c.bronzeTablePath", "dbfs:" + bronzeTableLocation)

# COMMAND ----------

# DBTITLE 1,Break up the data into chunks
# Ths sales order retail data is broken in chunks so that the parts can be used to write out to the landing zone so that Auto Loader can consume them at the correct time
dfBase = spark.read.json("/databricks-datasets/retail-org/sales_orders/")
dfIngest1 = dfBase.where(dfBase.customer_id <= 10000000)
dfIngest2 = dfBase.where(dfBase.customer_id.between(10000000, 20000000))
dfIngest3 = dfBase.where(dfBase.customer_id >= 20000000)

# COMMAND ----------

# DBTITLE 1,Write 1st data file to landing zone
# Uncomment next 2 rows if you would like to rerun from the beginning
spark.sql("DROP TABLE IF EXISTS autoloaderBronzeTable")
dbutils.fs.rm(basePath, True)

dfIngest1.write.json(landingZoneLocation)

# COMMAND ----------

# DBTITLE 1,Continuous ingest using Auto Loader with schema inference and hints
# We use the inferColumnTypes option for schema inference and the schemaHints option for the hints
df = spark.readStream.format("cloudFiles") \
  .option("cloudFiles.schemaLocation", schemaLocation) \
  .option("cloudFiles.format", "json") \
  .option("cloudFiles.inferColumnTypes", "true") \
  .option("cloudFiles.schemaEvolutionMode", "addNewColumns") \
  .option("cloudFiles.schemaHints", "clicked_items string, ordered_products.element.promotion_info string") \
  .load(landingZoneLocation)

# We use the mergeSchema option so that the schema can change over time and the checkpointLocation is used to store the state of the stream
df.writeStream \
  .format("delta") \
  .option("mergeSchema", "true") \
  .option("checkpointLocation", bronzeCheckPointLocation) \
  .start(bronzeTableLocation)

# COMMAND ----------

# DBTITLE 1,Display the data written to the Delta Table
# Sleeps for 10 seconds so that there is enough time to set up the stream and write to the table before displaying
import time
time.sleep(10) 
df = spark.read.format("delta").load(bronzeTableLocation)
display(df)

# COMMAND ----------

# DBTITLE 1,Write 2nd file to landing zone and add a nested column - as you can see, its stops the stream above
import pyspark.sql.functions as f

# add a new nested column "fulfillment_days" to the 2nd dataframe
dfIngest2 = dfIngest2.withColumn(
  "fulfillment_days", f.struct(\
  f.round(f.rand(), 2).alias("picking"), \
  f.round((f.rand() * 2), 2).cast("string").alias("packing"), \
  f.struct(f.lit("air").alias("type"),f.round((f.rand() * 5), 2).alias("days")).alias("shipping")))
  
# append the 2nd JSON file to the landin zone
# since the new data frame has a new column, the schema evolves which stops the query and can be started up right away
dfIngest2.write.mode("append").json(landingZoneLocation)

# COMMAND ----------

# DBTITLE 1,Scheduled ingest that turns itself off using Auto Loader with schema evolution and a new hint
# Added a new hint for the new column fulfillment_days
# As you can see from the dataframe below, the new column is accounted for in the stream
df = spark.readStream.format("cloudFiles") \
  .option("cloudFiles.schemaLocation", schemaLocation) \
  .option("cloudFiles.format", "json") \
  .option("cloudFiles.inferColumnTypes", "true") \
  .option("cloudFiles.schemaEvolutionMode", "addNewColumns") \
  .option("cloudFiles.schemaHints", "clicked_items string, ordered_products.element.promotion_info string, fulfillment_days string") \
  .load(landingZoneLocation)

# the trigger, "once=True" is what turns this version from a continuous stream to a scheduled one that turns itself off when it is finished
df.writeStream \
  .format("delta") \
  .trigger(once=True) \
  .option("mergeSchema", "true") \
  .option("checkpointLocation", bronzeCheckPointLocation) \
  .start(bronzeTableLocation)

# COMMAND ----------

# DBTITLE 1,Now that the data is in a delta table use SQL to manipulate the complex json structure
# MAGIC %sql
# MAGIC CREATE TABLE autoloaderBronzeTable
# MAGIC LOCATION '${c.bronzeTablePath}';

# COMMAND ----------

# DBTITLE 1,Semi-structured query - Extract top level and nested data. Also, casting those values
# MAGIC %sql
# MAGIC SELECT fulfillment_days, fulfillment_days:picking, 
# MAGIC   fulfillment_days:packing::double, fulfillment_days:shipping.days
# MAGIC FROM autoloaderBronzeTable
# MAGIC WHERE fulfillment_days IS NOT NULL

# COMMAND ----------

# DBTITLE 1,Semi-structured query - extract all and specific data from arrays. Also casting array values
# MAGIC %sql
# MAGIC SELECT *, reduce(all_click_count_array, 0, (acc, value) -> acc + value) as sum
# MAGIC FROM (
# MAGIC   SELECT order_number, clicked_items:[*][1] as all_click_counts, 
# MAGIC     from_json(clicked_items:[*][1], 'ARRAY<STRING>')::ARRAY<INT> as all_click_count_array
# MAGIC   FROM autoloaderBronzeTable
# MAGIC )

# COMMAND ----------

# DBTITLE 1,Complex aggregation in SQL with nested columns using Explode and dot notation
# MAGIC %sql
# MAGIC SELECT order_date, ordered_products_explode.name  as product_name, 
# MAGIC   SUM(ordered_products_explode.qty) as qantity
# MAGIC FROM (
# MAGIC   SELECT DATE(from_unixtime(order_datetime)) as order_date, 
# MAGIC     EXPLODE(ordered_products) as ordered_products_explode
# MAGIC   FROM autoloaderBronzeTable
# MAGIC   WHERE DATE(from_unixtime(order_datetime)) is not null
# MAGIC   )
# MAGIC GROUP BY order_date, ordered_products_explode.name
# MAGIC ORDER BY order_date, ordered_products_explode.name
