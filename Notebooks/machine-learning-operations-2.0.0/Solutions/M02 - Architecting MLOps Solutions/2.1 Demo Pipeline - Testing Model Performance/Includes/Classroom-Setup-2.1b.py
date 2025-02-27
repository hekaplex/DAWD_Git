# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

#%run ./Classroom-Setup-0

# COMMAND ----------

class Token:
    def __init__(self, config_path='./../var/credentials.cfg'):
        self.config_path = config_path
        self.token = self.get_credentials()

    def get_credentials(self):
        import configparser
        import os

        # Check if the file exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"The configuration file was not found at {self.config_path}")

        config = configparser.ConfigParser()
        config.read(self.config_path)

        if 'DEFAULT' not in config:
            raise KeyError("The section 'DEFAULT' was not found in the configuration file.")

        if 'db_token' not in config['DEFAULT']:
            raise KeyError("The key 'db_token' was not found in the 'DEFAULT' section of the configuration file.")

        token = config['DEFAULT']['db_token']
        
        # Print the token for debugging purposes
        print(f"db_token: {token}")

        return token

# COMMAND ----------

# Use the Token class to get the token
token_obj = Token()
token = token_obj.token

# COMMAND ----------

def create_features_table(self):
    from pyspark.sql.functions import col

    # Define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")

    # Read the dataset
    dataset_path = f"{DA.paths.datasets}/banking/loan-clean.csv"
    loan_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

    # Select columns of interest and replace spaces with underscores
    loan_df = loan_df.selectExpr(
        "ID",
        "Age",
        "Experience",
        "Income",
        "`ZIP Code` as ZIP_Code",
        "Family",
        "CCAvg",
        "Education",
        "Mortgage",
        "`Personal Loan` as Personal_Loan",
        "`Securities Account` as Securities_Account",
        "`CD Account` as CD_Account",
        "Online",
        "CreditCard"
    )

    # Save df as delta table using Delta API
    loan_df.write.format("delta").mode("overwrite").saveAsTable("bank_loan")

# Monkey patch the method into DBAcademyHelper
DBAcademyHelper.monkey_patch(create_features_table)

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
#DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.create_features_table()     
#DA.get_credentials()                  
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student
