
from config import *
import pandas as pd

continuous_cols = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'srch_query_affinity_score', 'orig_destination_distance', 'gross_bookings_usd', 'comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff']
categorical_cols = ['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'srch_destination_id', 'prop_starrating', 'position', 'prop_brand_bool', 'promotion_flag', 'srch_saturday_night_bool', 'random_bool', 'click_bool', 'booking_bool', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'comp1_rate', 'comp1_inv', 'comp2_rate', 'comp2_inv', 'comp3_rate', 'comp3_inv', 'comp4_rate', 'comp4_inv', 'comp5_rate', 'comp5_inv', 'comp6_rate', 'comp6_inv', 'comp7_rate', 'comp7_inv', 'comp8_rate', 'comp8_inv']

def load(names=["train", "test", "example"]):
    global DATASET_PATHS
    dfs = []
    for name in names:
        df = pd.read_csv(DATASET_PATHS[name])
        dfs.append(df)
    return tuple(dfs)


def summarize(df):
    num_summary = df[continuous_cols].describe().T.round(3)

    num_summary['missing'] = df[continuous_cols].isna().sum()
    num_summary['missing_pct'] = round(df[continuous_cols].isna().mean() * 100, 2)
    num_summary['count'] = num_summary['count'].astype(int)

    num_summary = num_summary[['count', 'missing', 'missing_pct', 'min', 'max', 'mean', 'std',]]

    df['date_time'] = pd.to_datetime(df['date_time'])

    time_row = pd.DataFrame([{
        'count': df['date_time'].count(),
        'missing': df['date_time'].isna().sum(),
        'missing_pct': round(df['date_time'].isna().mean() * 100, 2),
        'min': df['date_time'].min().date(),
        'max': df['date_time'].max().date(),
    }], index=['date'])

    num_summary = pd.concat([num_summary, time_row])

    num_latex = num_summary.to_latex(
        longtable=True,
        escape=True,
        caption='Numeric variable summary statistics (+ date)',
        label='tab:numeric_summary_stats',
        float_format='%.3f'
    )

    with open('nummeric_summary_stats.tex', 'w') as f:
        f.write(num_latex)


    cat_summary = df[categorical_cols].astype('category').describe().T
    cat_summary['missing_count'] = df[categorical_cols].isna().sum()
    cat_summary['missing_pct'] = round(df[categorical_cols].isna().mean() * 100, 2)
    cat_summary = cat_summary[['count', 'missing_count', 'missing_pct', 'unique', 'top', 'freq']]


    cat_latex = cat_summary.to_latex(
        longtable=True,
        escape=True,
        caption='Categoric variable summary statistics',
        label='tab:categoric_summary_stats',
        float_format='%.3f'
    )

    with open('categoric_summary_stats.tex', 'w') as f:
        f.write(cat_latex)

    num_summary = num_summary.sort_values(by='missing_pct', ascending=False)
    cat_summary = cat_summary.sort_values(by='missing_pct', ascending=False)
    print(num_summary)
    print(cat_summary)


if __name__=="__main__":
    df, = load(["train"])
    summarize(df)
    """
    A handful of columns have >90% missing values:

    compX_rate, compX_inv, compX_rate_percent_diff - Not sure if these columns compare 
    the same exact hotel room prices across different platforms or just something that's "similar".
    Also, it's not stated if the client sees this information, I believe he shouldn't.
    Either way I suspect most of these will have insignificant importance and so we will have to make a decision 
    to keep only those that are significant or don't use those at all. 

    visitor_hist_starrating, visitor_hist_adr_usd - information on user's past purchases. I expect significant importance

    srch_query_affinity_score = log prob of click on a hotel on In internet search


    """