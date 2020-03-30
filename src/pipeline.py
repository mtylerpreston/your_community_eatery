import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

import pyspark as ps
import pyspark.sql.types as types
from pyspark.sql.functions import col, countDistinct
from pyspark.sql.functions import to_timestamp

from sklearn.model_selection import train_test_split

import pickle


class YourCommunityEatery:
    '''
    data pipeline for fraud detector case study
    holds full dataframe and numerified data
    provides these, and also train/test split on the numerified data
    '''

    def __init__(self, save_memory=True):
        '''
        initialize with file path to the fraud dataset
        '''
        self.business_df = None
        self.review_df = None
        self.user_df = None
        self.tip_df = None
        self.checkin_df = None
        self.save_memory = save_memory

        # Use of spark is deprecated for now but may use in future
        self.use_spark = False
        if self.use_spark:
            self.spark = (ps.sql.SparkSession
                          .builder
                          .master('local[4]')
                          .appName('CleanYelpData')
                          .getOrCreate()
                          )

    def read_data(self, data_dir_path='data/', desired_data=['business', 'review'],
                  business_type='Restaurants', state='AZ'):
        '''
        Read the desired data into memory.

        Params
        ~~~~~~~~~~~~~~~
        data_dir_path: type - String
        Directory relative to working directory where data is stored. Default is 'data/'
        which works for programs run from the main repo directory.

        desired_data: type - List of strings
        The type of data desired from the Yelp dataset. The options are as follows:
        'business', 'review', 'user', 'checkin', 'tip', 'photo'
        However, you must ensure that you have all of Yelp's json files in the
        directory for this to work. As of July 2019, this project only uses business
        and review.

        business_type: type - string
        This is a Yelp-category of business for which to retrieve data. For 
        this project we are using 'Restaurants'.

        region: type = string
        This is the state for which to retrieve data. For this project we are using 'AZ'.
        Note that not all states are available in the dataset. Also, in any given state,
        the data set only includes a specific metropolitan area. For instance, although we
        are selecting 'AZ' for this study, it only includes the Phoenix metropolitan area.
        Some examples of states that are in the data set: NC, NV, PA, OH, IL, ON (Ontario), 
        AB (Alberta)
        '''

        self.business_type = business_type
        self.state = state

        if self.save_memory:
            self._read_data_pd_save_memory(data_dir_path, desired_data)
        elif self.use_spark:
            self._read_data_spark(data_dir_path, desired_data)
        else:
            self._read_data_pd(data_dir_path, desired_data)

    def filter_business(self):
        if self.use_spark:
            self.filter_business_spark(self.state)
        else:
            self.filter_business_pd(self.state)

    def merge_data_frame(self):
        '''
        This method is only used for cpu's with sufficient memory to avoid reading in chunks
        '''
        # merge business and review df's
        self.bus_review_df = pd.merge(self.review_df, self.business_df, on='business_id', how='left')

        # Drop duplicated business id column
        self.bus_review_df = self.bus_review_df.drop(self.bus_review_df.business_id_r)

    def calculate_density(self):
        num_ratings = len(self.bus_review_df.review_id)
        num_items = self.bus_review_df.business_id.nunique()
        num_users = self.bus_review_df.user_id.nunique()
        return num_ratings / (num_items * num_users)

    def filter_low_density_users(self, threshold=4):
        ''' 
        Filter the data to eliminate users with reviews less than a 
        specified threshold. Performs operation inplace on self.bus_review_df.
        '''
        groups = self.bus_review_df.groupby('user_id')
        selection = groups.review_id.nunique() >= threshold
        selection = selection[selection]
        self.bus_review_df = self.bus_review_df[self.bus_review_df.user_id.isin(selection.index)]

    def filter_low_density_items(self, threshold=50):
        ''' 
        Filter the data to eliminate businesses with reviews less than a 
        specified threshold. Performs operation inplace on self.bus_review_df.
        '''
        groups = self.bus_review_df.groupby('business_id')
        selection = groups.user_id.nunique() >= threshold
        selection = selection[selection]
        self.bus_review_df = self.bus_review_df[self.bus_review_df.business_id.isin(selection.index)]

    def filter_by_review_date(self, threshold=2013):
        ''' 
        Filter the data to eliminate reviews older than a specified
        threshold. Performs operation inplace on self.bus_review_df.

        Threshold must be a year. Specific month/date functionality
        is not included at this time.
        '''
        threshold = str(threshold)
        threshold = datetime.strptime(threshold, '%Y')
        selection = self.bus_review_df.date >= threshold
        self.bus_review_df = self.bus_review_df[selection]

    def factorize(self):
        # Map user and business ids to integers so that Surprise can handle them
        self.bus_review_df['iid'] = pd.factorize(self.bus_review_df.business_id)[0]
        self.bus_review_df['uid'] = pd.factorize(self.bus_review_df.user_id)[0]

    def persist_subject_data(self, path='data/bus_review_df.json'):
        # with open(path + '.pkl', 'wb') as file:
        #     pickle.dump(self.bus_review_df, file)
        self.bus_review_df.to_json(path, orient='records')

    def recommend(self, selections=['In-N-Out Burger', 'Chick-fil-A', 'The Stand', 'Whataburger'], k=3):
        '''
        Use the similarity matrix to find most similar restaurants to the four
        that the user specified in their selections. 

        Params:
        ~~~~~~~~~~~~
        similarity_matrix: type - Pandas Dataframe
        Similarity matrix from our fitted production model

        data_df: type - Pandas Dataframe
        Dataframe that has all of the pertinent data so that we can ensure we don't 
        the same chain of restaurants to someone just because they had a different
        location.

        item_map: type - Dict
        Dictionary that maps the business_id, name, and i_business_id (for surprise), with
        business_id as keys, and tupes of name and i_business_id as values.

        n: type - in
        Number of recommendations to provide

        selections: type - list of strings
        Names of restaurants that user selected for recommendations.
        '''
        # Create new dataframe for holding the vectors of similarities for
        # the user's selections
        similarity_matrix = self._compute_similarities()

        item_vectors = pd.DataFrame()
        for key, val in item_map.items():
            if val[0] in selections:
                item_vectors[val[1]] = similarity_matrix.iloc[:, val[1]]

        # Take mean similarity across columns and sort by it to
        # raise the best picks to the top
        item_vectors['mean_similarity'] = item_vectors.mean(axis=1)
        item_vectors.sort_values(by='mean_similarity', ascending=False, inplace=True)

        # Make sure that the picks don't have the same exact name
        name_mask = data_df.name.isin(selections)
        picks = item_vectors[~name_mask].index.unique()[:k]
        print(picks)

        # picks = item_vectors[~name_mask].iloc[:n, :].index
        recs = data_df[['business_id', 'name', 'i_business_id']].drop_duplicates(['i_business_id'])
        recs = recs[recs.i_business_id.isin(picks)]

        return recs

    def _compute_similarities(self):
        '''
        Compute similarity matrix from a fitted model and convert it to a 
        Pandas Dataframe. Using Pandas won't use any additional memory, and it lends more
        functionality.
        '''
        similarity_matrix = fitted_model.compute_similarities()
        return pd.DataFrame(similarity_matrix)

    def _read_data_spark(self, data_dir_path='data/', desired_data=['business', 'review']):
        '''
        Summary: 
        ~~~~~~~~~~~~~~~
        This is deprecated but being retained for future reference and possible use.
        This will read data into Spark dataframes which can allow for faster computation and
        help avoid memory issues.

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

    def _read_data_pd(self, data_dir_path='data/', desired_data=['business', 'review']):
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
                print('Reading business data...')
                self.business_df = pd.read_json(file_name, lines=True)
                # Make "star" columns unique on business_df and review_df to avoid confusion
                self.business_df = self.business_df.rename(columns={"stars": "avg_stars"})
            elif item == 'review':
                print('Reading review data...')
                self.review_df = pd.read_json(file_name, lines=True)
                # Make "star" columns unique on business_df and review_df to avoid confusion
                self.review_df = self.review_df.rename(columns={"stars": "review_stars"})
                self.review_df = self.review_df.rename(columns={"business_id": "business_id"})
            elif item == 'user':
                print('Reading user data...')
                self.user_df = pd.read_json(file_name, lines=True)
            elif item == 'checkin':
                print('Reading checkin data...')
                self.checkin_df = pd.read_json(file_name, lines=True)
            else:
                print('Reading tip data...')
                self.tip_df = pd.read_json(file_name, lines=True)

    def _read_data_pd_save_memory(self, data_dir_path='data/', desired_data=['business', 'review'], chunksize=100000):
        '''
        Given the high demands of loading the full json files into memory, 
        this function will need to perform a variety of tasks in the overall
        pipeline functionality at least for the specific use of this project. 
        It is important that business be loaded before review in this case.

        Process flow:
        Read business.json > clean business_df > drop unneeded columns >
            > drop nan rows > query to restaurants in AZ
        '''
        for item in desired_data:
            file_name = data_dir_path + 'yelp_academic_dataset_' + item + '.json'

            if item == 'business':

                print('Reading business data...')
                self.business_df = pd.read_json(file_name, lines=True)
                # Make "star" columns unique on business_df and review_df to avoid confusion
                self.business_df = self.business_df.rename(columns={"stars": "avg_stars"})

                # Query down to pertinent rows to save memory and filter down to businesses
                # of the desired type and region, this will modify self.business_df
                self._filter_business_pd()
            elif item == 'user':
                # Not deployed in this project
                print('Reading user data...')
                self.user_df = pd.read_json(file_name, lines=True)
            elif item == 'checkin':
                # Not deployed in this project
                print('Reading checkin data...')
                self.checkin_df = pd.read_json(file_name, lines=True)
            elif item == 'tip':
                # Not deployed in this project
                print('Reading tip data...')
                self.tip_df = pd.read_json(file_name, lines=True)
            elif item == 'review':
                # As of July 2019, this is specifically built for the use on this project
                print('Reading review data...')
                # Read the data from json file in chunks
                reader = pd.read_json(file_name, lines=True, chunksize=chunksize)

                self.bus_review_df = pd.DataFrame()
                for idx, chunk in enumerate(reader):
                    if idx % 10 == 0:
                        print(f'{idx/85}% complete')
                    # Make "star" columns unique on chunk to avoid confusion
                    chunk = chunk.rename(columns={"stars": "review_stars"})

                    # Merge chunk onto filtered business_df returning a chunk of bus_review_df
                    # and append it onto the growing self.bus_review_df
                    chunk = self._merge_chunk(chunk)
                    self.bus_review_df = self.bus_review_df.append(chunk)


    def _merge_chunk(self, chunk):
        # Merge business and review df's
        return pd.merge(chunk, self.business_df, on='business_id', how='inner')

    def _close_spark(self):
        # End the spark session to free up memory
        self.spark.close()

    def _filter_business_spark(self):
        # Filter business_df down to only businesses in Arizona
        self.business_df = self.business_df.filter((self.business_df.state == self.state) &
                                                   (self.business_df.categories.like('%Restaurants%'))
                                                   )
        # Join business and review df's
        self.bus_review_df = self.review_df.join(self.business_df,
                                                 self.review_df.business_id_r == self.business_df.business_id,
                                                 how='left')
        # Drop duplicated business id column
        self.bus_review_df = self.bus_review_df.drop(self.bus_review_df.business_id_r)

    def _filter_business_pd(self):
        # Clean business_df
        # Drop messy/unneeded columns
        col_drop = ['attributes', 'hours', 'review_count', 'postal_code']
        self.business_df.drop(columns=col_drop, inplace=True)

        # Drop nan rows
        self.business_df = self.business_df[~self.business_df.categories.isnull()]

        # Filter business_df down to only businesses in Arizona
        if self.state is not None:
            self.business_df = self.business_df[self.business_df.state == self.state]

        # Filter business_df down to only businesses of desired type
        self.business_df = self.business_df[self.business_df.categories.str.contains(self.business_type)]


if __name__ == '__main__':

    pipe = YourCommunityEatery(save_memory=True)
    print('Reading in data from json files...\n')
    pipe.read_data(data_dir_path='../data/')
    print('Persisting data...\n')
    pipe.persist_subject_data('../data/test_df')
    print('Complete')
