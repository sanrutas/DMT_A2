import numpy as np
from T3_data_preparation import TARGET_COLUMNS, add_base_features, model_feature_columns


LEAKAGE_COLUMNS = TARGET_COLUMNS


def add_relevance(df):
    df = df.copy()
    df["relevance"] = np.where(
        df["booking_bool"] == 1,
        5,
        np.where(df["click_bool"] == 1, 1, 0),
    )
    return df


def add_features(df):
    return add_base_features(df)
