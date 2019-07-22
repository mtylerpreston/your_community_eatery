import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

def _get_body_length_columns(DF_in):
    body_lenght_ranges = pd.DataFrame()
    body_lenght_ranges['body_length_0'] = (DF_in['body_length'] == 0)*1
    body_lenght_ranges['body_length_100'] = DF_in['body_length'].between(1, 100)*1
    body_lenght_ranges['body_length_200'] = DF_in['body_length'].between(101, 200)*1
    body_lenght_ranges['body_length_long'] = (DF_in['body_length']>200)*1
    DF_in.drop('body_length',axis=1,inplace=True)
    return pd.concat((DF_in,body_lenght_ranges),axis=1)

def _get_user_type_dummies(DF_in):
    user_types = pd.DataFrame()
    user_types['user_type_1'] = (DF_in['user_type']==1)*1
    user_types['user_type_2'] = (DF_in['user_type']==2)*1
    user_types['user_type_3'] = (DF_in['user_type']==3)*1
    user_types['user_type_4'] = (DF_in['user_type']==4)*1
    DF_in.drop('user_type',axis=1,inplace=True)
    return pd.concat((DF_in,user_types),axis=1)

def _get_currency_dummies(DF_in):
    #currency_dummies = pd.get_dummies(DF_in.currency)
    currency_dummies = pd.DataFrame()
    curr_columns=['AUD', 'CAD', 'EUR', 'GBP', 'MXN', 'NZD', 'USD']
    for curr in curr_columns:
        currency_dummies[curr] = (DF_in['currency'] == curr)*1
    DF_in.drop('currency',axis=1,inplace=True)
    return pd.concat((DF_in,currency_dummies),axis=1)

def preprocessing(df):
    newdf = df.copy()
    newdf = newdf[['body_length','currency','fb_published','has_analytics','has_logo','user_type']]
    newdf = _get_body_length_columns(newdf)
    newdf = _get_user_type_dummies(newdf)
    newdf = _get_currency_dummies(newdf)
    return newdf

def training(df):
    newdf = df.copy()
    newdf['fraud'] = newdf.acct_type.apply(lambda x: ('fraud' in x)*1)
    newdf = newdf[['fraud','body_length','currency','fb_published','has_analytics','has_logo','user_type']]
    newdf = _get_body_length_columns(newdf)
    newdf = _get_user_type_dummies(newdf)
    newdf = _get_currency_dummies(newdf)
    X = newdf.drop('fraud',axis=1)
    y = newdf.fraud
    return train_test_split(X,y, stratify=y, random_state=42)