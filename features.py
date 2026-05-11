import numpy as np
from T3_data_preparation import TARGET_COLUMNS, add_base_features, model_feature_columns


LEAKAGE_COLUMNS = TARGET_COLUMNS

HISTORICAL_PRIOR_KEYS = [
    ["prop_id"],
    # ["srch_destination_id"],
    # ["prop_country_id"],
    # ["site_id"],
    # ["visitor_location_country_id"],
]

PRIOR_TARGETS = [
    ("position", "pos_mean")
    # ("click_bool", "click_rate"),
    # ("booking_bool", "booking_rate"),
    # ("relevance", "relevance_mean"),
]

PRIOR_HISTORY_COLUMNS = list(dict.fromkeys(
    [col for keys in HISTORICAL_PRIOR_KEYS for col in keys]
    + [target for target, _ in PRIOR_TARGETS]
))


def add_relevance(df):
    df = df.copy()
    df["relevance"] = np.where(
        df["booking_bool"] == 1,
        5,
        np.where(df["click_bool"] == 1, 1, 0),
    )
    return df


def historical_prior_alpha(keys):
    return 100 if len(keys) > 1 else 50


def historical_prior_prefix(keys):
    return "hist_" + "_".join(keys)


def add_historical_priors(history, df):
    result = df
    global_means = {target: history[target].mean() for target, _ in PRIOR_TARGETS}

    for keys in HISTORICAL_PRIOR_KEYS:
        prefix = historical_prior_prefix(keys)
        count_col = f"{prefix}_count"
        alpha = historical_prior_alpha(keys)

        grouped = history.groupby(keys, dropna=False)
        priors = grouped.size().rename(count_col).reset_index()
        target_sums = grouped[[target for target, _ in PRIOR_TARGETS]].sum().reset_index()
        priors = priors.merge(target_sums, on=keys, how="left")

        for target, suffix in PRIOR_TARGETS:
            prior_col = f"{prefix}_{suffix}"
            priors[prior_col] = (
                priors[target] + alpha * global_means[target]
            ) / (priors[count_col] + alpha)

        prior_cols = [count_col] + [f"{prefix}_{suffix}" for _, suffix in PRIOR_TARGETS]
        result = result.merge(priors[keys + prior_cols], on=keys, how="left")
        result[count_col] = result[count_col].fillna(0)

        for target, suffix in PRIOR_TARGETS:
            result[f"{prefix}_{suffix}"] = result[f"{prefix}_{suffix}"].fillna(global_means[target])

    return result


def add_features(df):
    return add_base_features(df)
