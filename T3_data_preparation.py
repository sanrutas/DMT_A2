from T2_data_exploration import load
import numpy as np
import pandas as pd

def clean(df):
    # several records had booking price = 0, which should be an error / mistake
    df = df[~((df['gross_bookings_usd'] == 0) & (df['booking_bool'] == 1))]
    
    df['1day_price'] = df['price_usd'] / df['srch_length_of_stay']
    df['log_price'] = np.log1p(df['1day_price'])
    
    # transform is like pyspark window as it keeps all rows for every group
    group_mean = df.groupby(['prop_id', 'srch_saturday_night_bool', 'promotion_flag'])['log_price'].transform('mean')
    group_std = df.groupby(['prop_id', 'srch_saturday_night_bool', 'promotion_flag'])['log_price'].transform('std')

    z_score = (df['log_price'] - group_mean) / group_std
 
    df = df.loc[(z_score.abs() <= 3) | group_std.isna() | (group_std == 0)]

    return df.reset_index(drop=True)

def add_features(df):
    
    df = df.copy().sort_values(['srch_id', 'date_time'])
    df['weekday'] = df['date_time'].dt.weekday
    df['month'] = df['date_time'].dt.month
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df.drop(columns=['weekday'], inplace=True)
    df.drop(columns=['month'], inplace=True)

    df['price_relative_to_search'] = df['1day_price'] / df.groupby('srch_id')['1day_price'].transform('mean')
    df['price_relative_to_hotel'] = df['1day_price'] / df.groupby('prop_id')['1day_price'].transform('mean')

if __name__=="__main__":
    df, = load(["train"])
    df['date_time'] = pd.to_datetime(df['date_time'])

    df = clean(df)
    df = add_features(df)

    print(df)