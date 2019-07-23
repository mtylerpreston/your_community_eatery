import pandas as pd
import numpy as np

# Sklearn Modeling
from sklearn.model_selection import train_test_split

# Surprise modeling
from surprise import SVD
import surprise
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV

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

# Map user and business ids to numbers
print('Mapping to unique ids')
df['i_business_id'] = pd.factorize(df.business_id)[0]
df['i_user_id'] = pd.factorize(df.user_id)[0]

# Take the necessary data from the df in the order that we need to pass to surprise
train_df = pd.DataFrame()
train_df['i_user_id'] = df['i_user_id']
train_df['i_business_id'] = df['i_business_id']
train_df['review_stars'] = df['review_stars']

# Wipe original df to save memory, for now
df = None

# #Train test split for production models
# y = df['stars']
# X = df.drop('stars', axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# identifier_df_train = X_train[['user_id', 'business_id']]
# identifier_df_test = X_test[['user_id', 'business_id']]


# A reader is needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))
train_set = Dataset.load_from_df(train_df, reader)

# Perform gridsearch
algorithm = SVD

if test:
    parameters_grid = {'n_factors': [50, 100],
                       'n_epochs': [10, 20],
                       'lr_all': [.005],
                       'reg_all': [.01],
                       'random_state': [2],
                       'verbose': [True]}
else:
    parameters_grid = {'n_factors': [3, 5, 10, 15, 20, 25],
                       'n_epochs': [10, 20, 50, 100],
                       'lr_all': [.001, .005, .01],
                       'reg_all': [.005, .002, .006, .01],
                       'random_state': [2],
                       'verbose': [True]}

grid = GridSearchCV(algo_class=algorithm,
                    param_grid=parameters_grid,
                    cv=3,
                    n_jobs=-1,
                    joblib_verbose=2)
grid.fit(train_set)


# Print best params and result


# Pickle the best model, params, and result
model_file = '../models/top_grid_SVD_model1.pkl'
param_file = '../model_results/top_grid_SVD_params.pkl'
score_file = '../model_results/top_grid_SVD_score.pkl'

try:
    print(grid.best_estimator['rmse'])
    pickle.dump(grid.best_estimator['rmse'], open(model_file, 'wb'))
except:
    print('Failed to pickle model.')

print(grid.best_params['rmse'])
pickle.dump(grid.best_params['rmse'], open(param_file, 'wb'))
print(grid.best_score['rmse'])
pickle.dump(grid.best_score['rmse'], open(score_file, 'wb'))
