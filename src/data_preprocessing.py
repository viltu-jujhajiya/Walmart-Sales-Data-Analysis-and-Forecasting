import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, month, dayofweek, dayofmonth, year, sum, weekofyear, first, lag, avg
from pyspark.sql.window import Window

def setup_spark(config_spark):
    try:
        os.environ["SPARK_LOCAL_IP"] = config_spark["spark_local_ip"]
        spark = SparkSession.builder \
                .appName(config_spark["app_name"]) \
                .master(config_spark["master"]) \
                .getOrCreate()
        
        return spark

    except Exception as e:
        print(f"Exception in setup_spark: {e}")

def load_data(spark, config_datapath):
    try:
        features_path = config_datapath["features"]
        stores_path = config_datapath["stores"]
        train_path = config_datapath["train"]

        features = spark.read.csv(features_path,
                                inferSchema=True,
                                header=True
                                )

        stores = spark.read.csv(stores_path,
                                inferSchema=True,
                                header=True)

        train = spark.read.csv(train_path,
                            inferSchema=True,
                            header=True)

        return features, stores, train

    except Exception as e:
        print(f"Exception in load_data: {e}")


def build_dataset(features, stores, train):
    try:
        dataset = features.join(stores, on="Store", how="inner")
        dataset = dataset.join(train, on=["Date", "Store", "isHoliday"], how="inner")
        dataset = dataset.orderBy(col("Date").asc())

        return dataset

    except Exception as e:
        print(f"Exception in build_dataset: {e}")


def clean_data(dataset):
    try:
        '''Removing irrelevant features'''
        dataset = dataset.drop(col("MarkDown1"), col("MarkDown2"), col("MarkDown3"), col("MarkDown4"), col("MarkDown5"))

        '''Replacing "NA" with NULL value'''
        for c in dataset.columns:
            if dict(dataset.dtypes)[c] == "string":
                dataset = dataset.withColumn(c, when(col(c) == "NA", None).otherwise(col(c)))
        
        '''Droping null values'''
        dataset = dataset.dropna()

        return dataset

    except Exception as e:
        print(f"Exception in clean_data: {e}")


def feature_engineering(dataset):
    try:
        '''Adding department-wise weekly sales for a store.'''
        dataset = (
            dataset.groupBy("Store", "Date")
            .agg(
                sum("Weekly_Sales").alias("Weekly_Sales"),
                first("IsHoliday").alias("IsHoliday"),
                first("Temperature").alias("Temperature"),
                first("Fuel_Price").alias("Fuel_Price"),
                first("CPI").alias("CPI"),
                first("Unemployment").alias("Unemployment"),
                first("Type").alias("Type"),
                first("Size").alias("Size"),
            )
        )

        '''Change 'CPI' and 'Unemployment' data types from string to double'''
        dataset = dataset.withColumn("CPI", col("CPI").cast("double"))
        dataset = dataset.withColumn("Unemployment", col("Unemployment").cast("double"))

        '''Creating date-wise features'''
        dataset = dataset.withColumn("DayOfWeek", dayofweek("Date"))
        dataset = dataset.withColumn("Month", month("Date"))
        dataset = dataset.withColumn("Year", year("Date"))
        dataset = dataset.withColumn("WeekOfYear", weekofyear("Date"))
        dataset = dataset.withColumn("WeekOfMonth",((dayofmonth("Date") - 1) / 7).cast("int") + 1)

        '''Adding 'lag' features'''
        window = Window.partitionBy("Store").orderBy("Date")
        
        dataset = dataset.withColumn("lag1", lag("Weekly_Sales", 1).over(window))
        dataset = dataset.withColumn("lag4", lag("Weekly_Sales", 4).over(window))

        '''Adding 'Rolling Mean' features'''
        window_4_weeks = Window.partitionBy("Store").orderBy("Date").rowsBetween(-4, -1)
        dataset = dataset.withColumn("RollingMean_4Weeks", avg("Weekly_Sales").over(window_4_weeks))

        '''Dropping NULL values'''
        dataset = dataset.dropna(subset=["lag1", "lag4", "RollingMean_4Weeks"])

        return dataset

    except Exception as e:
        print(f"Exception in feature_engineering: {e}")

def process_data(config):
    try:
        '''EDA of the data is already done,
        so data preparation, cleaning, and feature engineering is done based on EDA.'''

        spark = setup_spark(config["spark"])

        features, stores, train = load_data(spark, config["datapaths"])
      
        dataset = build_dataset(features, stores, train)

        cleaned_dataset = clean_data(dataset)

        training_data = feature_engineering(cleaned_dataset)
        
        return training_data

    except Exception as e:
        print(f"Exception in process_data: {e}")


# import yaml
# with open(r"/Users/viltujujhajiya/Desktop/Walmart-Sales-Data-Analysis-and-Forecasting-main/config.yaml") as file:
#     config = yaml.safe_load(file)

# process_data(config)

    