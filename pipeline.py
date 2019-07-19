import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyspark as ps
import pyspark.sql.types as types
from pyspark.sql.functions import col, countDistinct
from pyspark.sql.functions import to_timestamp
from sklearn.model_selection import train_test_split


class CleanYelpData:
    '''
    data pipeline for fraud detector case study
    holds full dataframe and numerified data
    provides these, and also train/test split on the numerified data
    '''

    def __init__(self, use_spark=False):
        '''
        initialize with file path to the fraud dataset
        '''
        self.business_df = None
        self.review_df = None
        self.user_df = None
        self.tip_df = None
        self.checkin_df = None
        self.use_spark = use_spark

        if use_spark:

            self.spark = (ps.sql.SparkSession
                          .builder
                          .master('local[4]')
                          .appName('CleanYelpData')
                          .getOrCreate()
                          )

    def read_data(self, data_dir_path='data/', desired_data=['business', 'review']):
        if self.use_spark:
            self.read_data_spark(data_dir_path, desired_data)
        else:
            self.read_data_pd(data_dir_path, desired_data)

    def read_data_spark(self, data_dir_path='data/', desired_data=['business', 'review']):
        '''
        Summary: 
        ~~~~~~~~~~~~~~~
        Read in the desired data from raw json files

        Params:
        ~~~~~~~~~~~~~~~
        data_dir_path: 
        path to the folder that holds the data, in
        this case just 'data/'

        desired_data: list of strings
        name of desired data types out of the following - [business, 
        review, user, tip, checkin]
        '''
        for item in desired_data:
            file_name = data_dir_path + item + '.json'

            if item == 'business':
                self.business_df = self.spark.read.json(file_name)
                # Make "star" columns unique on business_df and review_df to avoid confusion
                self.business_df = self.business_df.withColumnRenamed("stars", "avg_stars")
            elif item == 'review':
                self.review_df = self.spark.read.json(file_name)
                # Make "star" columns unique on business_df and review_df to avoid confusion
                self.review_df = self.review_df.withColumnRenamed("stars", "review_stars")
                self.review_df = self.review_df.withColumnRenamed("business_id", "business_id_r")
            elif item == 'user':
                self.user_df = self.spark.read.json(file_name)
            elif item == 'checkin':
                self.checkin_df = self.spark.read.json(file_name)
            else:
                self.tip_df = self.spark.read.json(file_name)

    def read_data_pd(self, data_dir_path='data/', desired_data=['business', 'review']):
        '''
        Summary: 
        ~~~~~~~~~~~~~~~
        Read in the desired data from raw json files

        Params:
        ~~~~~~~~~~~~~~~
        data_dir_path: 
        path to the folder that holds the data, in
        this case just 'data/'

        desired_data: list of strings
        name of desired data types out of the following - [business, 
        review, user, tip, checkin]
        '''
        for item in desired_data:
            file_name = data_dir_path + item + '.json'

            if item == 'business':
                print('Reading business data...\n')
                self.business_df = pd.read_json(file_name)
                # Make "star" columns unique on business_df and review_df to avoid confusion
                self.business_df = self.business_df.rename(columns={"stars": "avg_stars"})
            elif item == 'review':
                print('Reading review data...\n')
                self.review_df = pd.read_json(file_name)
                # Make "star" columns unique on business_df and review_df to avoid confusion
                self.review_df = self.review_df.rename(columns={"stars": "review_stars"})
                self.review_df = self.review_df.rename(columns={"business_id": "business_id"})
            elif item == 'user':
                print('Reading user data...\n')
                self.user_df = pd.read_json(file_name)
            elif item == 'checkin':
                print('Reading checkin data...\n')
                self.checkin_df = pd.read_json(file_name)
            else:
                print('Reading tip data...\n')
                self.tip_df = pd.read_json(file_name)

    def close_spark(self):
        # End the spark session to free up memory
        self.spark.close()

    def query_business_review_geo(self, desired_geo='AZ'):
        if self.use_spark:
            self.query_business_review_geo_spark(desired_geo)
        else:
            self.query_business_review_geo_pd(desired_geo)

    def query_business_review_geo_spark(self, desired_geo='AZ'):
        # Filter business_df down to only businesses in Arizona
        self.business_df = self.business_df.filter((self.business_df.state == desired_geo) &
                                                   (self.business_df.categories.like('%Restaurants%'))
                                                   )
        # Join business and review df's
        self.bus_review_df = self.review_df.join(self.business_df,
                                                 self.review_df.business_id_r == self.business_df.business_id,
                                                 how='left')
        # Drop duplicated business id column
        self.bus_review_df = self.bus_review_df.drop(self.bus_review_df.business_id_r)

    def query_business_review_geo_pd(self, desired_geo='AZ'):
        # Filter business_df down to only businesses in Arizona
        self.business_df = self.business_df[self.business_df.state == desired_geo]
        self.business_df = self.business_df[self.business_df.categories.str.contains('Restaurants')]

        # Join business and review df's
        self.bus_review_df = self.review_df.join(self.business_df, on='business_id',
                                                 how='left')
        # Drop duplicated business id column
        self.bus_review_df = self.bus_review_df.drop(self.bus_review_df.business_id_r)

    def pickle_test_set(self, path='data/bus_review_df.pkl'):
        self.bus_review_df.to_pickle(path)

    def convert_spark_to_pandas(self):
        self.bus_review_df = self.bus_review_df.select("*").toPandas()
        close_spark()

    def get_train_test_data(self):
        '''
        train/test split for the numerified dataset
        No parameters
        returns: same as sklearn train_test_split
        '''
        X = self.X
        y = self.DF['label']
        return train_test_split(X, y, stratify=y, random_state=42)

    def preprocessing(df):
        '''
        creates model prediction input from a pandas df
        (for use on incoming cases)
        This redirects to 'create_numerified_data 
        (so the webserver code did not need to be changed)
        Parameters: Pandas DF 
        returns: Pandas DF
        '''
        return create_numerified_df(df)


if __name__ == '__main__':
    # pipe = CleanYelpData()
    # pipe.read_data()
    # pipe.query_business_review_geo()
    # pipe.convert_spark_to_pandas()
    # print(type(pipe.bus_review_df))

    pipe = CleanYelpData(use_spark=False)
    print('Reading in data from json files...\n')
    pipe.read_data()
    print('Querying data...\n')
    pipe.query_business_review_geo()
    print(type(pipe.bus_review_df))
    pipe.pickle_test_set()
