import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=Warning)


def load_dataset(asset, back_period=10, train=False,X_scaler=None,y_scaler=None, prev_period=2, weights=[]):
    data = load_data(asset, train=train)
    data_with_features, feature_labels, target_labels = construct_features(data, back_period,prev_period,weights) # {asset:df}
    X,y,stamp,X_scaler,y_scaler = data_processor(data_with_features,feature_labels,target_labels,train,X_scaler,y_scaler)
    return X,y,stamp,X_scaler,y_scaler,feature_labels, target_labels

def load_data(asset, train):
    data_path = 'train' if train else 'test'
    df = pd.read_parquet(f'../tick_data/{asset}/{data_path}/')
    df['exchange_time'] =  pd.to_datetime(df['exchange_time'])
    # 涨停价处理
    df['AskPrice1'] = df['AskPrice1'].where(df['AskPrice1'] != 0,df['BidPrice1'])
    df['BidPrice1'] = df['BidPrice1'].where(df['BidPrice1'] != 0,df['AskPrice1'])
    df['mid_price'] = (df['AskPrice1']+ df['BidPrice1'] )/2
    return df

def construct_features(df, back_period, prev_period, weights):
    calc_voi(df, prev_period, weights)
    df['Return_H'] = df['mid_price'].diff(back_period)
    df['Next_Return'] = df['Return_H'].shift(-back_period)
    df.dropna(inplace=True)
    feature_labels = ['VOI','rolling_voi']
    # feature_labels = ['VOI']
    target_labels = ['Next_Return'] 
    return df, feature_labels, target_labels

def weighted_sum(window_values, weights):
    return np.sum(window_values * weights)

def calc_voi(df, prev_period, weights):
    df['delta_vol'] = df['BidVolume1'] - df['AskVolume1']
    df['sum_vol'] = df['BidVolume1'] + df['AskVolume1']
    df['VOI'] = df['delta_vol'] / df['sum_vol']
    df['rolling_voi'] = 0
    for i in range(prev_period):
        df['rolling_voi'] += df['VOI'].shift(i) * weights[-i-1]
    df.dropna(inplace=True)

def get_Xy_stamp(df,feature_labels,target_labels,ad_labels=['exchange_time']):
    X = df[feature_labels].values
    y = df[target_labels].values
    ad_stamp = df[ad_labels]
    return X, y, ad_stamp

def data_processor(df, feature_labels,target_labels,train,X_scaler,y_scaler):
    X,y,stamp = get_Xy_stamp(df, feature_labels, target_labels)
    if train:
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_scaler.fit(X)
        y_scaler.fit(y)
    X = X_scaler.transform(X)
    y = y_scaler.transform(y)
    return X,y,stamp,X_scaler,y_scaler