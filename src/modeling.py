import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # Spark library and functions
# import pyspark as ps
# import pyspark.sql.types as types
# from pyspark.sql.functions import col, countDistinct
# from pyspark.sql.functions import to_timestamp

# Sklearn Modeling
from sklearn.model_selection import train_test_split

# Surprise modeling
from surprise import SVD
import surprise
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate

# Persistance
import pickle

# Housekeeping
from io import StringIO

test = False

if test:
    # Only taking a small sample for testing
    print('Reading file')
    reader = pd.read_json('../data/bus_review_df.json',
                          lines=True,
                          chunksize=1000)
    print(type(reader))
    print('Taking chunk from file')
    for df in reader:
        df = df
        break
else:
    print('Reading file')
    df = pd.read_json('../data/bus_review_df.json', orient='records')

# Map user and business ids to numbers
print('Mapping to unique ids')
df['i_business_id'] = pd.factorize(df.business_id)[0]
df['i_user_id'] = pd.factorize(df.user_id)[0]
print(df.i_user_id)


train_df.drop(columns=['cool', 'date', 'review_id', 'categories', 'city',
                       'is_open', 'latitude', 'longitude', 'name', 'state',
                       'funny', 'text', 'useful', 'avg_stars', 'address',
                       'user_id', 'business_id'],
              inplace=True)

# #Train test split for production models
# y = df['stars']
# X = df.drop('stars', axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# identifier_df_train = X_train[['user_id', 'business_id']]
# identifier_df_test = X_test[['user_id', 'business_id']]


# A reader is needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df, reader)

all_predictions = []
all_ratings = []

# Iterate over all algorithms
for algorithm in [SVD(), surprise.SlopeOne(), surprise.NMF(),
                  surprise.NormalPredictor(), surprise.KNNBaseline(),
                  # surprise.KNNBasic(), surprise.KNNWithMeans(),
                  # surprise.KNNWithZScore(), surprise.BaselineOnly(),
                  surprise.CoClustering()]:

    # Take a look at cross validation results to compare model types
    print('Modeling: {}'.format(str(algorithm)))
    cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


#     # fit the model
#     alg_name = str(algorithm)[str(algorithm).find('ization')+8 : str(algorithm).find('obj')-1]
#     print('Fitting algorithm {}'.format(alg_name))
#     algorithm.fit(train_set)

#     # run predictions over
#     print('Predicting algorithm {}'.format(alg_name))
#     model_predictions = []
#     model_ratings = []
#     for idx, row in X_test.iterrows():
#         prediction = algorithm.predict(row[0], row[1])
#         model_predictions.append(prediction)
#         model_ratings.append(prediction[3])

#     # Pickle the models
#     pred_pkl_file = alg_name + '_predictions.pkl'
#     ratings_pkl_file = alg_name + '_ratings.pkl'
#     pickle.dump(model_predictions, open(pred_pkl_file, 'wb'))
#     pickle.dump(model_ratings, open(ratings_pkl_file, 'wb'))
