import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
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
        dataset = dataset.orderBy(f.col("Date").asc())

        return dataset

    except Exception as e:
        print(f"Exception in build_dataset: {e}")


def clean_data(dataset):
    try:
        '''Removing irrelevant features'''
        dataset = dataset.drop(f.col("MarkDown1"), f.col("MarkDown2"), f.col("MarkDown3"), f.col("MarkDown4"), f.col("MarkDown5"))

        '''Replacing "NA" with NULL value'''
        for c in dataset.columns:
            if dict(dataset.dtypes)[c] == "string":
                dataset = dataset.withColumn(c, f.when(f.col(c) == "NA", None).otherwise(f.col(c)))
        
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
                f.sum("Weekly_Sales").alias("Weekly_Sales"),
                f.first("IsHoliday").alias("IsHoliday"),
                f.first("Temperature").alias("Temperature"),
                f.first("Fuel_Price").alias("Fuel_Price"),
                f.first("CPI").alias("CPI"),
                f.first("Unemployment").alias("Unemployment"),
                f.first("Type").alias("Type"),
                f.first("Size").alias("Size"),
            )
        )

        '''Change 'CPI' and 'Unemployment' data types from string to double'''
        dataset = dataset.withColumn("CPI", f.col("CPI").cast("double"))
        dataset = dataset.withColumn("Unemployment", f.col("Unemployment").cast("double"))

        '''Creating date-wise features'''
        dataset = dataset.withColumn("DayOfWeek", f.dayofweek("Date"))
        dataset = dataset.withColumn("Month", f.month("Date"))
        dataset = dataset.withColumn("Year", f.year("Date"))
        dataset = dataset.withColumn("WeekOfYear", f.weekofyear("Date"))
        dataset = dataset.withColumn("WeekOfMonth",((f.dayofmonth("Date") - 1) / 7).cast("int") + 1)

        '''Adding 'lag' features'''
        window = Window.partitionBy("Store").orderBy("Date")
        
        dataset = dataset.withColumn("lag1", f.lag("Weekly_Sales", 1).over(window))
        dataset = dataset.withColumn("lag4", f.lag("Weekly_Sales", 4).over(window))

        '''Adding 'Rolling Mean' features'''
        window_4_weeks = Window.partitionBy("Store").orderBy("Date").rowsBetween(-4, -1)
        dataset = dataset.withColumn("RollingMean_4Weeks", f.avg("Weekly_Sales").over(window_4_weeks))

        '''Dropping NULL values'''
        dataset = dataset.dropna(subset=["lag1", "lag4", "RollingMean_4Weeks"])

        return dataset

    except Exception as e:
        print(f"Exception in feature_engineering: {e}")


def feature_encoding(data_config, data):
    try:
        '''Type encoding'''
        type_encoding = data_config["Type_encoding"]
        type_mapping = f.create_map(*[f.lit(x) for pair in type_encoding.items() for x in pair])
        data = data.withColumn("Type", type_mapping[f.col("Type")])

        '''IsHoliday encoding'''
        data = data.withColumn("IsHoliday", f.when(f.col("IsHoliday"), 1).otherwise(0))

        return data

    except Exception as e:
            print(f"Exception in feature_encoding: {e}")


def process_data(config):
    try:
        '''EDA of the data is already done,
        so data preparation, cleaning, and feature engineering is done based on EDA.'''

        spark = setup_spark(config["spark"])

        features, stores, train = load_data(spark, config["datapaths"])
      
        dataset = build_dataset(features, stores, train)

        cleaned_dataset = clean_data(dataset)

        processed_data = feature_engineering(cleaned_dataset)

        training_data = feature_encoding(config["data"], processed_data)

        try:
            training_data = training_data.select(config["data"]["features"])
        except:
            pass

        return training_data

    except Exception as e:
        print(f"Exception in process_data: {e}")


import yaml
with open(r"/Users/viltujujhajiya/Desktop/Walmart-Sales-Data-Analysis-and-Forecasting-main/config.yaml") as file:
    config = yaml.safe_load(file)

process_data(config)

    