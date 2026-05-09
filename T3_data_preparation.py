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

    for col in MISSING_INDICATOR_COLUMNS + COMP_MISSING_INDICATOR_COLUMNS:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    df["price_rank"] = df.groupby("srch_id")["price_usd"].rank(method="average", ascending=True)
    df["price_rank_pct"] = df.groupby("srch_id")["price_usd"].rank(method="average", ascending=True, pct=True)
    df["price_relative_to_search"] = df["price_per_night"] / df.groupby("srch_id")["price_per_night"].transform("mean")

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
