import numpy as np
import pandas as pd


COMPARISON_COLUMNS = [
    "price_usd",
    "price_per_night",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_brand_bool",
    "promotion_flag",
    "hist_prop_id_non_random_count",
    "hist_prop_id_non_random_booking_bool_mean",
    "orig_destination_distance",
]


def add_model_rank(df):
    ranked = df.sort_values(
        ["srch_id", "score", "prop_id"],
        ascending=[True, False, True],
    ).copy()
    ranked["model_rank"] = ranked.groupby("srch_id").cumcount() + 1
    return ranked


def booking_rank_summary(booked):
    return pd.DataFrame([
        {"metric": "searches_with_booking", "value": len(booked)},
        {"metric": "booked_in_top_1", "value": (booked["model_rank"] <= 1).mean()},
        {"metric": "booked_in_top_3", "value": (booked["model_rank"] <= 3).mean()},
        {"metric": "booked_in_top_5", "value": (booked["model_rank"] <= 5).mean()},
        {"metric": "booked_missed_top_5", "value": (booked["model_rank"] > 5).mean()},
        {"metric": "mean_booked_rank", "value": booked["model_rank"].mean()},
        {"metric": "median_booked_rank", "value": booked["model_rank"].median()},
    ])


def booking_rank_distribution(booked):
    distribution = booked["model_rank"].value_counts().sort_index().reset_index()
    distribution.columns = ["model_rank", "searches"]
    distribution["share"] = distribution["searches"] / len(booked)
    return distribution


def booked_vs_top_ranked(ranked):
    columns = ["srch_id", "prop_id", "model_rank", "score"] + COMPARISON_COLUMNS
    booked = ranked.loc[ranked["booking_bool"] == 1, columns].copy()
    top = ranked.groupby("srch_id", sort=False).head(1)[columns].copy()

    booked = booked.rename(columns={
        col: f"booked_{col}" for col in columns if col != "srch_id"
    })
    top = top.rename(columns={
        col: f"top_{col}" for col in columns if col != "srch_id"
    })

    comparison = booked.merge(top, on="srch_id", how="left")
    comparison["missed_top_1"] = comparison["booked_model_rank"] > 1
    comparison["missed_top_3"] = comparison["booked_model_rank"] > 3
    comparison["missed_top_5"] = comparison["booked_model_rank"] > 5

    for col in COMPARISON_COLUMNS:
        comparison[f"{col}_booked_minus_top"] = (
            comparison[f"booked_{col}"] - comparison[f"top_{col}"]
        )

    comparison["booked_more_expensive_than_top"] = (
        comparison["booked_price_usd"] > comparison["top_price_usd"]
    )
    comparison["booked_lower_review_score_than_top"] = (
        comparison["booked_prop_review_score"] < comparison["top_prop_review_score"]
    )
    comparison["booked_missing_location_score2"] = comparison["booked_prop_location_score2"].isna()
    comparison["top_missing_location_score2"] = comparison["top_prop_location_score2"].isna()
    comparison["booked_independent"] = comparison["booked_prop_brand_bool"] == 0
    comparison["top_independent"] = comparison["top_prop_brand_bool"] == 0
    comparison["booked_promoted"] = comparison["booked_promotion_flag"] == 1
    comparison["top_promoted"] = comparison["top_promotion_flag"] == 1
    comparison["booked_less_historical_count_than_top"] = (
        comparison["booked_hist_prop_id_non_random_count"]
        < comparison["top_hist_prop_id_non_random_count"]
    )
    comparison["booked_historical_count_zero"] = (
        comparison["booked_hist_prop_id_non_random_count"] == 0
    )
    comparison["booked_farther_than_top"] = (
        comparison["booked_orig_destination_distance"]
        > comparison["top_orig_destination_distance"]
    )

    return comparison.sort_values(["missed_top_5", "booked_model_rank"], ascending=[False, False])


def miss_summary(comparison):
    misses = comparison[comparison["missed_top_5"]].copy()
    rows = [
        {"metric": "top_5_miss_count", "value": len(misses)},
        {"metric": "top_5_miss_share", "value": len(misses) / len(comparison)},
        {"metric": "booked_more_expensive_than_top", "value": misses["booked_more_expensive_than_top"].mean()},
        {"metric": "booked_lower_review_score_than_top", "value": misses["booked_lower_review_score_than_top"].mean()},
        {"metric": "booked_missing_location_score2", "value": misses["booked_missing_location_score2"].mean()},
        {"metric": "top_missing_location_score2", "value": misses["top_missing_location_score2"].mean()},
        {"metric": "booked_independent", "value": misses["booked_independent"].mean()},
        {"metric": "top_independent", "value": misses["top_independent"].mean()},
        {"metric": "booked_promoted", "value": misses["booked_promoted"].mean()},
        {"metric": "top_promoted", "value": misses["top_promoted"].mean()},
        {"metric": "booked_less_historical_count_than_top", "value": misses["booked_less_historical_count_than_top"].mean()},
        {"metric": "booked_historical_count_zero", "value": misses["booked_historical_count_zero"].mean()},
        {"metric": "booked_farther_than_top", "value": misses["booked_farther_than_top"].mean()},
        {"metric": "median_price_usd_booked_minus_top", "value": misses["price_usd_booked_minus_top"].median()},
        {"metric": "median_review_score_booked_minus_top", "value": misses["prop_review_score_booked_minus_top"].median()},
        {"metric": "median_location_score2_booked_minus_top", "value": misses["prop_location_score2_booked_minus_top"].median()},
        {"metric": "median_history_count_booked_minus_top", "value": misses["hist_prop_id_non_random_count_booked_minus_top"].median()},
        {"metric": "median_distance_booked_minus_top", "value": misses["orig_destination_distance_booked_minus_top"].median()},
    ]
    return pd.DataFrame(rows)


def feature_gap_summary(comparison):
    rows = []
    for col in COMPARISON_COLUMNS:
        delta = comparison[f"{col}_booked_minus_top"]
        miss_delta = comparison.loc[comparison["missed_top_5"], f"{col}_booked_minus_top"]
        rows.append({
            "feature": col,
            "all_mean_booked": comparison[f"booked_{col}"].mean(),
            "all_mean_top": comparison[f"top_{col}"].mean(),
            "all_median_booked_minus_top": delta.median(),
            "miss_mean_booked": comparison.loc[comparison["missed_top_5"], f"booked_{col}"].mean(),
            "miss_mean_top": comparison.loc[comparison["missed_top_5"], f"top_{col}"].mean(),
            "miss_median_booked_minus_top": miss_delta.median(),
        })
    return pd.DataFrame(rows)


def report_text(rank_summary, miss_stats):
    rank_values = rank_summary.set_index("metric")["value"]
    miss_values = miss_stats.set_index("metric")["value"].replace({np.nan: 0})
    return "\n".join([
        "# Validation Booking Error Analysis",
        "",
        "## Booked Hotel Rank",
        f"- Searches with booking: {rank_values['searches_with_booking']:.0f}",
        f"- Booked hotel in top 1: {rank_values['booked_in_top_1']:.4f}",
        f"- Booked hotel in top 3: {rank_values['booked_in_top_3']:.4f}",
        f"- Booked hotel in top 5: {rank_values['booked_in_top_5']:.4f}",
        f"- Mean booked rank: {rank_values['mean_booked_rank']:.2f}",
        "",
        "## Top-5 Misses",
        f"- Miss count: {miss_values['top_5_miss_count']:.0f}",
        f"- Booked more expensive than model top hotel: {miss_values['booked_more_expensive_than_top']:.4f}",
        f"- Booked lower review score than model top hotel: {miss_values['booked_lower_review_score_than_top']:.4f}",
        f"- Booked missing location score2: {miss_values['booked_missing_location_score2']:.4f}",
        f"- Booked independent hotel: {miss_values['booked_independent']:.4f}",
        f"- Booked promoted hotel: {miss_values['booked_promoted']:.4f}",
        f"- Booked historically unseen: {miss_values['booked_historical_count_zero']:.4f}",
        f"- Booked farther than model top hotel: {miss_values['booked_farther_than_top']:.4f}",
    ])


def save_validation_error_analysis(df, output_dir):
    ranked = add_model_rank(df)
    booked = ranked[ranked["booking_bool"] == 1].copy()
    comparison = booked_vs_top_ranked(ranked)
    rank_summary = booking_rank_summary(booked)
    rank_distribution = booking_rank_distribution(booked)
    miss_stats = miss_summary(comparison)
    gap_stats = feature_gap_summary(comparison)

    rank_summary.to_csv(output_dir / "validation_booking_rank_summary.csv", index=False)
    rank_distribution.to_csv(output_dir / "validation_booking_rank_distribution.csv", index=False)
    comparison.to_csv(output_dir / "validation_booked_vs_top_ranked.csv", index=False)
    miss_stats.to_csv(output_dir / "validation_top5_miss_summary.csv", index=False)
    gap_stats.to_csv(output_dir / "validation_booked_vs_top_feature_gaps.csv", index=False)
    (output_dir / "validation_error_analysis.md").write_text(
        report_text(rank_summary, miss_stats)
    )
