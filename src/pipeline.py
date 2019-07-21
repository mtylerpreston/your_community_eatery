import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyspark as ps
import pyspark.sql.types as types
from pyspark.sql.functions import col, countDistinct
from pyspark.sql.functions import to_timestamp
from sklearn.model_selection import train_test_split
import os


class CleanYelpData:
    '''
    data pipeline for fraud detector case study
    holds full dataframe and numerified data
    provides these, and also train/test split on the numerified data
    '''

    def __init__(self, use_spark=False, low_memory=True):
        '''
        initialize with file path to the fraud dataset
        '''
        self.business_df = None
        self.review_df = None
        self.user_df = None
        self.tip_df = None
        self.checkin_df = None
        self.use_spark = use_spark
        self.low_memory = low_memory

        if use_spark:

            self.spark = (ps.sql.SparkSession
                          .builder
                          .master('local[4]')
                          .appName('CleanYelpData')
                          .getOrCreate()
                          )

    def read_data(self, data_dir_path='data/', desired_data=['business', 'review']):
        if self.low_memory:
            self.read_data_pd_low_memory(data_dir_path, desired_data)
        elif self.use_spark:
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
                self.business_df = pd.read_json(file_name, lines=True)
                # Make "star" columns unique on business_df and review_df to avoid confusion
                self.business_df = self.business_df.rename(columns={"stars": "avg_stars"})
            elif item == 'review':
                print('Reading review data...\n')
                self.review_df = pd.read_json(file_name, lines=True)
                # Make "star" columns unique on business_df and review_df to avoid confusion
                self.review_df = self.review_df.rename(columns={"stars": "review_stars"})
                self.review_df = self.review_df.rename(columns={"business_id": "business_id"})
            elif item == 'user':
                print('Reading user data...\n')
                self.user_df = pd.read_json(file_name, lines=True)
            elif item == 'checkin':
                print('Reading checkin data...\n')
                self.checkin_df = pd.read_json(file_name, lines=True)
            else:
                print('Reading tip data...\n')
                self.tip_df = pd.read_json(file_name, lines=True)

    def read_data_pd_low_memory(self, data_dir_path='data/', desired_data=['business', 'review'], chunksize=100000):

        for item in desired_data:
            file_name = data_dir_path + item + '.json'

            if item == 'business':
                '''
                Given the low memory issue, this function will need to perform the full 
                pipeline functionality at least for the specific use of this project.
                Read business.json > clean business_df > drop unneeded columns >
                    > drop nan rows > query to restaurants in AZ
                This is needed in order for review to be able to save the relevant chunks.
                '''
                print('Reading business data...\n')
                self.business_df = pd.read_json(file_name, lines=True)
                # Make "star" columns unique on business_df and review_df to avoid confusion
                self.business_df = self.business_df.rename(columns={"stars": "avg_stars"})

                # Query down to pertinent rows to save memory, this will modify
                # self.business_df
                self.query_business_review_geo_pd()

            elif item == 'review':
                print('Reading review data...\n')
                # Read the data from json file in chunks, save to file if pertains to restaurant in AZ
                reader = pd.read_json(file_name, lines=True, chunksize=chunksize)

                for idx, chunk in enumerate(reader):
                    # Make "star" columns unique on chunk to avoid confusion
                    chunk = chunk.rename(columns={"stars": "review_stars"})

                    # Merge chunk onto business_df to get the relevent rows, this creates
                    # self.bus_review_df
                    self.merge_chunks(chunk)

                    # Save relevant rows to a json file in a specific directory
                    self.bus_review_df.to_json('data/chunks/chunk' + str(idx) + '.json')

            elif item == 'user':
                print('Reading user data...\n')
                self.user_df = pd.read_json(file_name, lines=True)
            elif item == 'checkin':
                print('Reading checkin data...\n')
                self.checkin_df = pd.read_json(file_name, lines=True)
            else:
                print('Reading tip data...\n')
                self.tip_df = pd.read_json(file_name, lines=True)

    def concatenate_chunks(self):
        self.bus_review_df = pd.DataFrame()
        for chunk in os.listdir('data/chunks/'):
            if chunk.endswith(".json") and chunk.startswith('chunk'):
                self.bus_review_df = self.bus_review_df.append(pd.read_json('data/chunks/' + chunk))

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
        # Clean business_df
        # Drop messy/unneeded columns
        col_drop = ['attributes', 'hours', 'review_count', 'postal_code']
        self.business_df.drop(columns=col_drop, inplace=True)

        # Drop nan rows
        self.business_df = self.business_df[~self.business_df.categories.isnull()]

        # Filter business_df down to only businesses in Arizona
        self.business_df = self.business_df[self.business_df.state == desired_geo]
        self.business_df = self.business_df[self.business_df.categories.str.contains('Restaurants')]

    def merge_chunks(self, chunk):
        # Merge business and review df's
        self.bus_review_df = pd.merge(chunk, self.business_df, on='business_id', how='inner')

    def merge_data_frame(self):
        # merge business and review df's
        self.bus_review_df = pd.merge(self.review_df, self.business_df, on='business_id', how='left')

        # Drop duplicated business id column
        self.bus_review_df = self.bus_review_df.drop(self.bus_review_df.business_id_r)

    def persist_test_set(self, path='data/bus_review_df'):
        # try:
        self.bus_review_df.to_json(path + '.json', orient='records')
        # except:
        #     self.bus_review_df.to_pickle(path + '.pkl')

    def convert_spark_to_pandas(self):
        self.bus_review_df = self.bus_review_df.select("*").toPandas()
        close_spark()


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
    pipe.merge_data_frame()
    print(type(pipe.bus_review_df))
    pipe.persist_test_set()
