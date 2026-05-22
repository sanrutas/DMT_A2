from config import *
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from T2_data_exploration import load
from T3_data_preparation import clean_train_only
import os


def plot_price_over_time(df):

    # example how some searches for the same hotels may have abnormal price values. Those should be deleted.
    df = df.loc[df['prop_id'] == 21315]
    df = df.sort_values(by='date_time', ascending=True)

    plt.figure(figsize=(16, 8))

    plt.plot(df['date_time'],  df['price_per_day'])

    plt.xlabel('dates')
    plt.ylabel('USD')
    plt.title('Time series of room price by date of search')

    plt.show()

def plot_prices_over_time(df, prop_ids, type):

    # example how some searches for the same hotels may have abnormal price values. Those should be deleted.

    fig, axes = plt.subplots(3, 4, figsize=(14, 6), constrained_layout=True)
    axes = axes.flatten()

    fig.suptitle("Price history of some hotels (prop_id), type", fontsize=16)

    # prop_ids = df["prop_id"].drop_duplicates()[:12]

    for i, pid in enumerate(prop_ids):
        prop_id_subset = df.loc[df['prop_id'] == pid].sort_values(by='date_time', ascending=True)
        ax = axes[i]
        ax.plot(prop_id_subset["date_time"], prop_id_subset["price_per_day"])
        ax.set_title(f"prop_id = {pid}")
        ax.set_xlabel("date")
        ax.set_ylabel("Price USD")
        ax.tick_params(axis="x", rotation=45)

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/prices_over_time_{type}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_price_density_by_categories(df, category_cols):
    
    os.makedirs("plots", exist_ok=True)
    for category_col in category_cols:
        plt.figure(figsize=(10, 6))

        for value in sorted(df[category_col].dropna().unique()):
            subset = df.loc[df[category_col] == value, "price_per_day"].dropna()
            plt.hist(subset, bins=50, density=True, alpha=0.5, label=f'{category_col}={value}')            

        plt.title(f'1 day price density by {category_col}')
        plt.xlabel('price_per_day')
        plt.ylabel('Density')
        plt.legend(title=category_col)
        
        plt.savefig(f"plots/daily_price_density_by_{category_col}.png", dpi=300, bbox_inches='tight')
        plt.show()

if __name__=="__main__":
    print("loading df", flush=True)
    df, = load(["train"])
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['price_per_day'] = df['price_usd'] / df["srch_length_of_stay"]
    prop_ids = df["prop_id"].drop_duplicates().head(12).tolist()
    
    plot_prices_over_time(df, prop_ids, type="pre-clean")
    df = clean_train_only(df)

    plot_price_density_by_categories(df, ['srch_saturday_night_bool', 'promotion_flag'])
    
    # test
    plot_prices_over_time(df, prop_ids, type="post-clean")

