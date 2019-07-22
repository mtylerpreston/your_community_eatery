'''
Data pipeline for fraud detector 
'''
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

class FraudData:
    '''
    data pipeline for fraud detector case study
    '''
    def __init__(self, file_path):
        self.raw = pd.read_json(file_path)
        self.DF = self.raw.copy()
        _add_label(self.DF)
        enrich_input_data(self.DF)
        # the numerified Dataset with labels:
        self.X = create_numerified_df(self.DF)
        self.XY = self.X.copy()
        self.XY['label'] = self.DF['label']

    def get_enriched_df(self):
        '''
        the raw data plus derived additional,data, like datetime info
        Returns: pandas DF
        '''
        return self.DF

    def get_numerified_data(self):
        '''
        provides the input data for model training. 
        All data numerical
        '''
        return self.X

    def get_train_test_data(self):
        X = self.X
        y = self.DF['label']
        return train_test_split(X,y, stratify=y, random_state=42)
                               
def get_prediction_intput(DF):
    enrich_input_data(DF)
    return create_numerified_df(DF)

def enrich_input_data(DF):
    # translate to datetime 
    DF['user_creation_dt'] = DF.user_created.apply(lambda x: 
                                    pd.to_datetime(x,unit='s'))
    DF['event_creation_dt'] = DF.event_created.apply(lambda x: 
                                    pd.to_datetime(x,unit='s'))
    # datime components
    DF['uc_year'] = DF['user_creation_dt'].apply(lambda x: x.year)
    DF['uc_month'] = DF['user_creation_dt'].apply(lambda x: x.month)
    DF['uc_week'] = DF['user_creation_dt'].apply(lambda x: x.week)
    DF['uc_weekday'] = DF['user_creation_dt'].apply(lambda x: x.weekday())
    DF['uc_day'] = DF['user_creation_dt'].apply(lambda x: x.day)
    DF['uc_hour'] = DF['user_creation_dt'].apply(lambda x: x.hour)

    DF['ec_year'] = DF['event_creation_dt'].apply(lambda x: x.year)
    DF['ec_month'] = DF['event_creation_dt'].apply(lambda x: x.month)
    DF['ec_week'] = DF['event_creation_dt'].apply(lambda x: x.week)
    DF['ec_weekday'] = DF['event_creation_dt'].apply(lambda x: x.weekday())
    DF['ec_day'] = DF['event_creation_dt'].apply(lambda x: x.day)
    DF['ec_hour'] = DF['event_creation_dt'].apply(lambda x: x.hour)

    DF['user_event_delta'] = (DF['event_creation_dt'] - 
                    DF['user_creation_dt'])/np.timedelta64(1, 'M')

def create_numerified_df(DF_in):
    num_df = _get_currency_dummies(DF_in)
    num_df = num_df.join(_get_body_length_columns(DF_in))
    num_df = num_df.join(_get_channel_dummies(DF_in))
    num_df = num_df.join(_get_user_type_dummies(DF_in))
    num_df = num_df.join(_get_ticket_dummies(DF_in))
    return num_df

def preprocessing(df):
    return create_numerified_df(df)

def _add_label(DF):
    DF['label'] = DF.acct_type.apply(lambda x: ('fraud' in x)*1)
    return DF
            
def _get_currency_dummies(DF_in):
    #currency_dummies = pd.get_dummies(DF_in.currency)
    currency_dummies = pd.DataFrame()
    curr_columns=['AUD', 'CAD', 'EUR', 'GBP', 'MXN', 'NZD', 'USD']
    for curr in curr_columns:
        currency_dummies[curr] = (DF_in['currency'] == curr)*1
    return currency_dummies

def _get_body_length_columns(DF_in):
    body_lenght_ranges = pd.DataFrame()
    body_lenght_ranges['body_length_0'] = (DF_in['body_length'] == 0)*1
    body_lenght_ranges['body_length_100'] = DF_in['body_length'].between(1, 100)*1
    body_lenght_ranges['body_length_200'] = DF_in['body_length'].between(101, 200)*1
    body_lenght_ranges['body_length_long'] = (DF_in['body_length']>200)*1
    return body_lenght_ranges

def _get_channel_dummies(DF_in):
    channels = pd.DataFrame()
    channels['channel_0'] = (DF_in['channels'] == 0)*1
    channels['channel_8'] = (DF_in['channels'] == 8)*1
    channels['channel_other'] = (~DF_in['channels'].isin([0, 8]))*1
    return channels

def _get_user_type_dummies(DF_in):
    user_types = pd.DataFrame()
    user_types['user_type_1'] = (DF_in['user_type']==1)*1
    user_types['user_type_2'] = (DF_in['user_type']==2)*1
    user_types['user_type_3'] = (DF_in['user_type']==3)*1
    user_types['user_type_4'] = (DF_in['user_type']==4)*1
    return user_types

def _get_ticket_dummies(DF_in):
    idx= DF_in.set_index(['object_id']).ticket_types.apply(pd.Series).stack().index

    tmp = pd.DataFrame(DF_in.set_index(['object_id']).ticket_types.apply(pd.Series).stack().values.tolist(),index=idx).reset_index().drop('level_1',1)
    tmp_sum = tmp.groupby('event_id').agg({'cost': ['min', 'max'],'quantity_total' : 'sum' }).reset_index()
    tmp_sum.columns = [' '.join(col).strip() for col in tmp_sum.columns.values]
    
    df = DF_in.merge(tmp_sum, how='left', left_on = 'object_id', right_on = 'event_id')
    df['ticket_tiers_num'] = df['ticket_types'].apply(lambda x: len(x))
    
    for col in ['cost min', 'cost max','quantity_total sum']:
        df[col]=df[col].apply(lambda x: 0 if pd.isnull(x) else x)
    return df[['cost min', 'cost max',
       'quantity_total sum','ticket_tiers_num']]
