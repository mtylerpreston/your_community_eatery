import pandas as pd
import numpy as np

# Surprise modeling
from surprise import SVD
import surprise
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV, train_test_split

# Project specific
from functions import build_item_map

# Persistance
import pickle


test = False

print('Reading file...')
df = pd.read_json('../data/processed_df.json', orient='records')

# take a small sample for testing
if test:
    df = df.iloc[:10000, :]

# Build the item map for our recommender and pickle it
print('Building item map...')
item_map = build_item_map(df)
with open('../website/models/item_map.pkl', 'wb') as file:
    pickle.dump(item_map, file)

# Take the necessary data from the df in the order that we need to pass to surprise
train_df = pd.DataFrame()
train_df['uid'] = df['uid']
train_df['iid'] = df['iid']
train_df['review_stars'] = df['review_stars']

# Wipe original df to save memory, for now
df = None

# A reader is needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df, reader)

# Taking only a minimal split as this is for the purposes
# of getting a completed similarity matrix
trainset, testset = train_test_split(data, test_size=0.1)

# Instantiate model with desired hyperparameters
alg = surprise.KNNBaseline(k=100,
                           min_k=4,
                           bsl_options={'method': 'als', 'reg_i': 5, 'reg_u': 10},
                           sim_options={'name': 'cosine',
                                        'min_support': 2,
                                        'user_based': False,
                                        'shrinkage': 50},
                           random_state=2)
# alg = surprise.KNNBaseline()

alg.fit(trainset)

similarities = alg.compute_similarities()
similarities = pd.DataFrame(similarities)

# Glean model name from object for handling purposes
alg_name = str(alg)
alg_name = alg_name[alg_name.find('.') + 1:]
alg_name = alg_name[alg_name.find('.') + 1:]
alg_name = alg_name[alg_name.find('.') + 1:]
alg_name = alg_name[:alg_name.find('object') - 1]

# Pickle similarity matrix
similarities_file = '../website/models/' + alg_name + '_similarities.pkl'
pickle.dump(similarities, open(similarities_file, 'wb'))
