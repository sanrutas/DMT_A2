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
    (["prop_id"], "position", "non-random"),
    (["prop_id"], "click_bool", "non-random"),
    (["prop_id"], "booking_bool", "non-random"),
    (["prop_id"], "click_bool", "random"),
    # (["prop_id"], "booking_bool", "random"),
    # (["prop_id"], "srch_children_count", "non-random"),
    # (["prop_id"], "srch_adults_count", "non-random")
]

PRIOR_HISTORY_COLUMNS = list(dict.fromkeys(
    [col for keys, target, search_type in HISTORICAL_PRIORS for col in keys + [target]]
    + ["prop_country_id", "random_bool"]
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
    alpha = 50

    for keys, target, search_type in HISTORICAL_PRIORS:
        prior_groups.setdefault((tuple(keys), search_type), []).append(target)

    for (key_tuple, search_type), targets in prior_groups.items():
        keys = list(key_tuple)
        search_value = {"non-random": 0, "random": 1}[search_type]
        search_history = history[history["random_bool"] == search_value]
        global_means = {target: search_history[target].mean() for target in targets}
        search_suffix = search_type.replace("-", "_")
        prefix = "hist_" + "_".join(keys) + f"_{search_suffix}"
        count_col = f"{prefix}_count"

        grouped = search_history.groupby(keys, dropna=False)
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


def historical_prior_columns():
    prior_groups = {}

    for keys, target, search_type in HISTORICAL_PRIORS:
        prior_groups.setdefault((tuple(keys), search_type), []).append(target)

    cols = []
    for (key_tuple, search_type), targets in prior_groups.items():
        search_suffix = search_type.replace("-", "_")
        prefix = "hist_" + "_".join(key_tuple) + f"_{search_suffix}"
        cols.append(f"{prefix}_count")

        for target in targets:
            cols.append(f"{prefix}_{target}_mean")

    return cols


def add_oof_historical_priors(df, group_col="srch_id", n_folds=5, random_state=42):
    result = df.copy()
    base = df.copy()
    groups = result[group_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(random_state)
    rng.shuffle(groups)
    group_folds = np.array_split(groups, n_folds)
    prior_cols = historical_prior_columns()

    for fold_groups in group_folds:
        fold_mask = result[group_col].isin(fold_groups)
        history = base.loc[~fold_mask, PRIOR_HISTORY_COLUMNS]
        fold = add_historical_priors(history, base.loc[fold_mask])
        result.loc[fold_mask, prior_cols] = fold[prior_cols].to_numpy()

    return result


def add_country_imputations(history, df):
    result = df
    country_medians = history.groupby("prop_country_id")[COUNTRY_IMPUTATION_COLUMNS].median()

    for col in COUNTRY_IMPUTATION_COLUMNS:
        imputed_col = f"{col}_country_imputed"
        country_values = result["prop_country_id"].map(country_medians[col])
        result[imputed_col] = result[col].fillna(country_values).fillna(history[col].median())

    return result


def new_features(df):
    df = df.copy()
    grouped = df.groupby("srch_id")
    value_denominator = 1 + df["log_price"]

    df["price_usd_div_search_median"] = df["price_usd"] / grouped["price_usd"].transform("median")
    df["price_per_night_div_search_median"] = df["price_per_night"] / grouped["price_per_night"].transform("median")
    df["review_value"] = df["prop_review_score"] / value_denominator
    df["location2_value"] = df["prop_location_score2_country_imputed"] / value_denominator
    df["quality_value"] = (df["prop_review_score"] + df["prop_location_score2_country_imputed"]) / value_denominator
    df["premium_review"] = (df["price_rank_pct"] > 0.75).astype(int) * (df["prop_review_score"] >= 4).astype(int)
    df["premium_location2"] = (df["price_rank_pct"] > 0.75).astype(int) * (df["prop_location_score2_country_imputed"] >= grouped["prop_location_score2_country_imputed"].transform("median")).astype(int)
    df["promotion_x_price_rank_pct"] = df["promotion_flag"] * df["price_rank_pct"]
    df["promotion_x_review"] = df["promotion_flag"] * df["prop_review_score"]
    df["promotion_x_location2"] = df["promotion_flag"] * df["prop_location_score2_country_imputed"]


    df["hist_prop_id_non_random_count_log1p"] = np.log1p(df["hist_prop_id_non_random_count"])
    
    df["hist_booking_mean_x_log_count"] = df["hist_prop_id_non_random_booking_bool_mean"] * df["hist_prop_id_non_random_count_log1p"]
    df["hist_click_mean_x_log_count"] = df["hist_prop_id_non_random_click_bool_mean"] * df["hist_prop_id_non_random_count_log1p"]
    df["hist_booking_mean_rank_desc_by_search"] = grouped["hist_prop_id_non_random_booking_bool_mean"].rank(method="average", ascending=False)
    df["hist_count_rank_pct_by_search"] = grouped["hist_prop_id_non_random_count"].rank(method="average", ascending=True, pct=True)
    # df["hist_booking_mean_minus_search_mean"] = df["hist_prop_id_non_random_booking_bool_mean"] - grouped["hist_prop_id_non_random_booking_bool_mean"].transform("mean")


    # df["hist_prop_id_non_random_count_sqrt"] = np.sqrt(df["hist_prop_id_non_random_count"])
    # df["hist_prop_id_non_random_count_cap_25"] = df["hist_prop_id_non_random_count"].clip(upper=25)
    # df["hist_prop_id_non_random_count_cap_100"] = df["hist_prop_id_non_random_count"].clip(upper=100)
    # df["missing_location2_x_price_rank_pct"] = df["prop_location_score2_missing"] * df["price_rank_pct"]
    # df["missing_location2_x_review"] = df["prop_location_score2_missing"] * df["prop_review_score"]
    # df["low_history_high_review"] = (df["hist_prop_id_non_random_count"] < 25).astype(int) * (df["prop_review_score"] >= 4).astype(int)
    # df["low_history_high_location2"] = (df["hist_prop_id_non_random_count"] < 25).astype(int) * (df["prop_location_score2_country_imputed"] >= grouped["prop_location_score2_country_imputed"].transform("median")).astype(int)

    return df


def add_features(df):
    return add_base_features(df)
