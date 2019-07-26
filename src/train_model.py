import pandas as pd
import numpy as np

# Surprise modeling
from surprise import SVD
import surprise
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV, train_test_split

# Persistance
import pickle

# Housekeeping
from io import StringIO

test = False

print('Reading file')
df = pd.read_json('../data/trimmed_df.json', orient='records')

# take a small sample for testing
if test:
    df = df.iloc[:10000, :]

# Take the necessary data from the df in the order that we need to pass to surprise
train_df = pd.DataFrame()
train_df['uid'] = df['uid']
train_df['iid'] = df['iid']
train_df['review_stars'] = df['review_stars']

# Wipe original df to save memory, for now
df = None

# Train test split for production models
y = df['review_stars']
X = df.drop('review_stars', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

identifier_df_train = X_train[['user_id', 'business_id']]
identifier_df_test = X_test[['user_id', 'business_id']]


# A reader is needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df, reader)
# trainset, testset = train_test_split(data, test_size=.1)

# Instantiate model with desired hyperparameters
alg = surprise.KNNBaseline(k=100,
                           min_k=4,
                           bsl_options={'method': 'als', 'reg': 1},
                           sim_options={'name': 'cosine',
                                        'min_support': 1,
                                        'user_based': False,
                                        'shrinkage': 50},
                           random_state=2)


similarities = alg.compute_similarities()

# Glean model name from object for handling purposes
alg_name = str(alg)
alg_name = alg_name[alg_name.find('.') + 1:]
alg_name = alg_name[alg_name.find('.') + 1:]
alg_name = alg_name[alg_name.find('.') + 1:]
alg_name = alg_name[:alg_name.find('object') - 1]

# Pickle similarity matrix
similarities_file = '../models/' + alg_name + '_similarities.pkl'


pickle.dump(cv, open(similarities_file, 'wb'))
