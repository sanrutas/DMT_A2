import numpy as np
from T3_data_preparation import TARGET_COLUMNS, add_base_features


LEAKAGE_COLUMNS = TARGET_COLUMNS
COUNTRY_IMPUTATION_COLUMNS = [
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "srch_query_affinity_score",
    "orig_destination_distance",
    "prop_location_score2",
]

HISTORICAL_PRIORS = [
    (["prop_id"], "position"),
]

PRIOR_HISTORY_COLUMNS = list(dict.fromkeys(
    [col for keys, target in HISTORICAL_PRIORS for col in keys + [target]]
    + ["prop_country_id"]
    + COUNTRY_IMPUTATION_COLUMNS
))


def add_relevance(df):
    df = df.copy()
    df["relevance"] = np.where(
        df["booking_bool"] == 1,
        5,
        np.where(df["click_bool"] == 1, 1, 0),
    )
    return df

def add_historical_priors(history, df):
    result = df
    prior_groups = {}
    global_means = {target: history[target].mean() for _, target in HISTORICAL_PRIORS}
    alpha = 50

    for keys, target in HISTORICAL_PRIORS:
        prior_groups.setdefault(tuple(keys), []).append(target)

    for key_tuple, targets in prior_groups.items():
        keys = list(key_tuple)
        prefix = "hist_" + "_".join(keys)
        count_col = f"{prefix}_count"

        grouped = history.groupby(keys, dropna=False)
        priors = grouped.size().rename(count_col).reset_index()
        target_sums = grouped[targets].sum().reset_index()
        priors = priors.merge(target_sums, on=keys, how="left")
        prior_cols = [count_col]

        for target in targets:
            prior_col = f"{prefix}_{target}_mean"
            priors[prior_col] = (
                priors[target] + alpha * global_means[target]
            ) / (priors[count_col] + alpha)
            prior_cols.append(prior_col)

        result = result.merge(priors[keys + prior_cols], on=keys, how="left")
        result[count_col] = result[count_col].fillna(0)

        for target in targets:
            prior_col = f"{prefix}_{target}_mean"
            result[prior_col] = result[prior_col].fillna(global_means[target])

    return result


def add_country_imputations(history, df):
    result = df
    country_medians = history.groupby("prop_country_id")[COUNTRY_IMPUTATION_COLUMNS].median()

    for col in COUNTRY_IMPUTATION_COLUMNS:
        imputed_col = f"{col}_country_imputed"
        country_values = result["prop_country_id"].map(country_medians[col])
        result[imputed_col] = result[col].fillna(country_values).fillna(history[col].median())

    return result


def add_features(df):
    return add_base_features(df)
