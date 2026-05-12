import numpy as np
import pandas as pd


TARGET_COLUMNS = [
    "position",
    "click_bool",
    "booking_bool",
    "gross_booking_usd",
    "gross_bookings_usd",
    "relevance",
]

MISSING_INDICATOR_COLUMNS = [
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "srch_query_affinity_score",
    "orig_destination_distance",
]

COMP_MISSING_INDICATOR_COLUMNS = [
    f"comp{i}_{suffix}"
    for i in range(1, 9)
    for suffix in ["rate", "inv", "rate_percent_diff"]
]

COMPETITORS = range(1, 9)

QUERY_RELATIVE_COLUMNS = [
    "price_usd",
    "price_per_night",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "orig_destination_distance",
    "srch_query_affinity_score",
    "score2ma",
    "log_price",
    "comp_z_score",
    "ump",
    "starrating_diff",
    # "price_diff",
    # "total_fee",
    # "per_fee",
    # "price_per_person",
    # "stay_value"
]


def clean_common(df):
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["srch_length_of_stay_safe"] = df["srch_length_of_stay"].replace(0, np.nan)
    df["price_per_night"] = df["price_usd"] / df["srch_length_of_stay_safe"]
    df["log_price"] = np.log1p(df["price_per_night"])
    return df


def clean_train_only(df):
    df = df.copy()
    if "gross_bookings_usd" in df.columns and "booking_bool" in df.columns:
        df = df[~((df["gross_bookings_usd"] == 0) & (df["booking_bool"] == 1))]
    if "gross_booking_usd" in df.columns and "booking_bool" in df.columns:
        df = df[~((df["gross_booking_usd"] == 0) & (df["booking_bool"] == 1))]
    return df


def add_query_relative_features(df):
    grouped = df.groupby("srch_id")
    features = {}

    for col in QUERY_RELATIVE_COLUMNS:
        values = grouped[col]
        search_mean = values.transform("mean")
        search_std = values.transform("std")

        features[f"{col}_rank_asc_by_search"] = values.rank(method="average", ascending=True)
        features[f"{col}_rank_desc_by_search"] = values.rank(method="average", ascending=False)
        # features[f"{col}_minus_search_mean"] = df[col] - search_mean
        # features[f"{col}_div_search_mean"] = df[col] / search_mean
        features[f"{col}_zscore_by_search"] = (df[col] - search_mean) / search_std
        # features[f"is_min_{col}_in_search"] = (df[col] == values.transform("min")).astype(int)
        # features[f"is_max_{col}_in_search"] = (df[col] == values.transform("max")).astype(int)

    return pd.concat([df, pd.DataFrame(features)], axis=1)


def add_competitor_features(df):
    comp_rates = df[[f"comp{i}_rate" for i in COMPETITORS]].astype(float)
    comp_rate_diffs = df[[f"comp{i}_rate_percent_diff" for i in COMPETITORS]].astype(float)
    comp_rate_diffs.columns = comp_rates.columns

    comp_data_count = comp_rates.notna().sum(axis=1)
    df["comp_data_count"] = comp_data_count
    # df["comp_price_win_share"] = comp_rates.eq(1).sum(axis=1) / comp_data_count
    # df["comp_price_loss_share"] = comp_rates.eq(-1).sum(axis=1) / comp_data_count
    # comp_price_advantages = comp_rates * comp_rate_diffs
    # df["comp_price_advantage_mean"] = comp_price_advantages.mean(axis=1)

    comp_prices = (
        comp_rates
        .mul(comp_rate_diffs.fillna(0))
        .div(100)
        .add(1)
        .mul(df["price_usd"], axis=0)
    )
    comp_prices.columns = [f"comp{i}_price" for i in COMPETITORS]
    comp_prices = comp_prices.where(comp_rates.notna().to_numpy())
    comp_mean = comp_prices.mean(axis=1)
    comp_std = comp_prices.std(axis=1).replace(0, np.nan)
    df["comp_z_score"] = (df["price_usd"] - comp_mean) / comp_std
    return df


def add_winner_writeup_features(df):
    grouped = df.groupby("srch_id")
    mean_price = grouped["price_usd"].transform("mean")
    mean_starrating = grouped["prop_starrating"].transform("mean")
    mean_score2 = grouped["prop_location_score2"].transform("mean")

    df["count_window"] = grouped["prop_id"].transform("size")
    df["ump"] = np.exp(df["prop_log_historical_price"]) - df["price_usd"]
    df["price_diff"] = df["price_usd"] - mean_price
    df["starrating_diff"] = df["prop_starrating"] - mean_starrating
    df["score2ma"] = df["prop_location_score2"] - mean_score2
    df["per_fee"] = df["price_usd"] / df["guests"].clip(lower=1)
    df["total_fee"] = df["price_usd"] * df["srch_length_of_stay"] * df["srch_room_count"]
    return df


def add_base_features(df):
    df = clean_common(df)

    df["month"] = df["date_time"].dt.month
    df["weekday"] = df["date_time"].dt.weekday
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df["stay_value"] = df["price_usd"] * df["srch_room_count"]
    df["guests"] = df["srch_adults_count"] + df["srch_children_count"]
    df["price_per_person"] = df["price_usd"] / df["guests"].clip(lower=1)
    df = add_winner_writeup_features(df)

    for col in MISSING_INDICATOR_COLUMNS + COMP_MISSING_INDICATOR_COLUMNS:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    df["price_rank"] = df.groupby("srch_id")["price_usd"].rank(method="average", ascending=True)
    df["price_rank_pct"] = df.groupby("srch_id")["price_usd"].rank(method="average", ascending=True, pct=True)
    df["price_relative_to_search"] = df["price_per_night"] / df.groupby("srch_id")["price_per_night"].transform("mean")
    df = add_competitor_features(df)
    df = add_query_relative_features(df)

    return df.reset_index(drop=True)


def prepare_train(df):
    df = clean_train_only(df)
    return add_base_features(df)


def prepare_test(df):
    return add_base_features(df)


def model_feature_columns(df):
    blocked = set(TARGET_COLUMNS + ["date_time", "srch_id"])
    return [
        col for col in df.columns
        if col not in blocked and pd.api.types.is_numeric_dtype(df[col])
    ]


def clean(df):
    return clean_common(df).reset_index(drop=True)


def add_features(df):
    return add_base_features(df)


if __name__ == "__main__":
    from T2_data_exploration import load

    df, = load(["train"])
    df = prepare_train(df)
    print(df)
