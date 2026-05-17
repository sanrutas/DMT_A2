from config import *
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from T2_data_exploration import load
from T3_data_preparation import clean


def plot_price_over_time(df):

    # example how some searches for the same hotels may have abnormal price values. Those should be deleted.
    df = df.loc[df['prop_id'] == 21315]
    df = df.sort_values(by='date_time', ascending=True)

    plt.figure(figsize=(16, 8))

    plt.plot(df['date_time'],  df['1day_price'])

    plt.xlabel('dates')
    plt.ylabel('USD')
    plt.title('Time series of room price by date of search')

    plt.show()

def plot_price_density_by_categories(df, category_cols):
    
    # filter out extreme price values to get visible shapes of general distributions
    df = df[df['1day_price'] < 500]

    for category_col in category_cols:
        plt.figure(figsize=(10, 6))

        for value in sorted(df[category_col].dropna().unique()):
            subset = df.loc[df[category_col] == value, "1day_price"].dropna()
            plt.hist(subset, bins=50, density=True, alpha=0.5, label=f'{category_col}={value}')            

        plt.title(f'1 day price density by {category_col}')
        plt.xlabel('1day_price')
        plt.ylabel('Density')
        plt.legend(title=category_col)
        plt.show()

if __name__=="__main__":
    df, = load(["train"])
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['1day_price'] = df['price_usd'] / df["srch_length_of_stay"]

    plot_price_over_time(df)
    plot_price_density_by_categories(df, ['srch_saturday_night_bool', 'promotion_flag'])
    
    # test
    df = clean(df) 
    plot_price_over_time(df)

