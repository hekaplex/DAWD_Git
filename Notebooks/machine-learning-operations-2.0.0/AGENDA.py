# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Machine Learning Operations
# MAGIC
# MAGIC This course will guide participants through a comprehensive exploration of machine learning model operations, focusing on MLOps and model lifecycle management. The initial segment covers essential MLOps components and best practices, providing participants with a strong foundation for effectively operationalizing machine learning models. In the latter part of the course, we will delve into the basics of the model lifecycle, demonstrating how to navigate it seamlessly using the Model Registry in conjunction with the Unity Catalog for efficient model management. By the course's conclusion, participants will have gained practical insights and a well-rounded understanding of MLOps principles, equipped with the skills needed to navigate the intricate landscape of machine learning model operations.
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Course Agenda
# MAGIC | Module &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Lessons &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|
# MAGIC | **[Modern MLOps]($./M01 - Modern MLOps)**    | **Lecture -** Defining MLOps <br> **Lecture -** MLOps on Databricks </br> [**Demo:** Working with Asset Bundles]($./M01 - Modern MLOps/1.1 Demo - Working with Asset Bundles) </br> [**Lab:** Creating and Managing Workflow Jobs using UI]($./M01 - Modern MLOps/1.2 LAB - Create an ML Workflow Job Using UI) | 
# MAGIC | **[Architecting MLOps Solutions]($./M02 - Architecting MLOps Solutions)** | **Lecture -** Opinionated MLOps Principles </br> **Lecture -** Recommended MLOps Architectures </br> [**Demo:** Model Testing Job with the Databricks CLI]($./M02 - Architecting MLOps Solutions/2.1 Demo - Model Testing with the Databricks CLI) </br> [**Lab:** Deploying Models with Jobs and the Databricks CLI]($./M02 - Architecting MLOps Solutions/2.2 LAB - Create Deployment and Inference Workflow Job Using the CLI) | 
# MAGIC | **[Implementation and Monitoring MLOps Solution]($./M03 - Implementation and Monitoring MLOps Solution)** | **Lecture -** Implementation of MLOps Stacks </br> **Lecture -** Type of Model Monitoring **Lecture -** Monitoring in Machine Learning </br> [**Demo:** Lakehouse Monitoring Dashboard]($./M03 - Implementation and Monitoring MLOps Solution/3.1 Demo - Model Monitoring) </br> [**Lab:** Model Monitoring]($./M03 - Implementation and Monitoring MLOps Solution/3.2 Lab - Model Monitoring)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run demo and lab notebooks, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
