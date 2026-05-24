import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker as CatBoostRankerModel

from config import DATASET_PATHS
from features import PRIOR_HISTORY_COLUMNS, add_relevance
from model_catboost import add_model_features, catboost_labels, ndcg_at_5
from model_position import add_estimated_position_predictions, load_position_estimator
from split import group_train_val_split
from T3_data_preparation import clean_train_only


MODEL_PATH = Path("artifacts/catboost/model.cbm")
POSITION_MODEL_PATH = Path("artifacts/lgbm/position_model.txt")
OUTPUT_DIR = Path("artifacts/catboost/bias_analysis")
SCORED_CACHE_PATH = OUTPUT_DIR / "validation_scored.csv"
MITIGATION_OFFSETS = [round(x * 0.025, 3) for x in range(0, 21)]


def load_catboost_model(model_path):
    model_path = Path(model_path)
    model = CatBoostRankerModel()
    model.load_model(model_path)
    feature_cols = json.loads(model_path.with_suffix(".features.json").read_text())
    return model, feature_cols


def prepare_validation_data(position_model_path):
    train = pd.read_csv(DATASET_PATHS["train"])
    train = add_relevance(clean_train_only(train))
    train, val = group_train_val_split(train, group_col="srch_id", val_size=0.2, random_state=42)
    split_history = train[PRIOR_HISTORY_COLUMNS].copy()
    val = add_model_features(split_history, val)
    position_model = load_position_estimator(position_model_path)
    val = add_estimated_position_predictions(val, position_model.predict(val))
    val["catboost_label"] = catboost_labels(val)
    return val


def add_bias_columns(df):
    df = df.copy()

    df["children_slice"] = np.where(
        df["srch_children_count"] > 0,
        "searches_with_children",
        "searches_without_children",
    )
    df["stay_length_slice"] = pd.cut(
        df["srch_length_of_stay"],
        bins=[0, 1, 3, 7, float("inf")],
        labels=["stay_1_night", "stay_2_3_nights", "stay_4_7_nights", "stay_8_plus_nights"],
    )
    df["price_tier_slice"] = pd.cut(
        df["price_rank_pct"],
        bins=[0, 0.33, 0.67, 1],
        labels=["low_price_within_search", "mid_price_within_search", "high_price_within_search"],
        include_lowest=True,
    )
    return df


def auxiliary_slice_metrics(df):
    rows = []
    for col in ["children_slice", "stay_length_slice"]:
        for name, part in df.groupby(col, sort=False, observed=True):
            rows.append({
                "group": col,
                "slice": name,
                "ndcg_at_5": ndcg_at_5(part),
            })
    return pd.DataFrame(rows)


def ranked_for_score(df, score_col):
    ranked = df.sort_values(
        ["srch_id", score_col, "prop_id"],
        ascending=[True, False, True],
    ).copy()
    ranked["mitigation_rank"] = ranked.groupby("srch_id").cumcount() + 1
    return ranked


def ideal_dcg_by_search(df):
    ideal = df.sort_values(
        ["srch_id", "catboost_label", "prop_id"],
        ascending=[True, False, True],
    ).copy()
    ideal["ideal_rank"] = ideal.groupby("srch_id").cumcount() + 1
    ideal_top5 = ideal[ideal["ideal_rank"] <= 5].copy()
    ideal_discounts = np.log2(ideal_top5["ideal_rank"] + 1)
    ideal_top5["ideal_dcg_part"] = ideal_top5["catboost_label"].map({0: 0, 1: 1, 5: 5}) / ideal_discounts
    return ideal_top5.groupby("srch_id")["ideal_dcg_part"].sum()


def overall_ndcg_from_ranked(ranked, ideal_dcg):
    top5 = ranked[ranked["mitigation_rank"] <= 5].copy()
    discounts = np.log2(top5["mitigation_rank"] + 1)
    top5["dcg_part"] = top5["catboost_label"].map({0: 0, 1: 1, 5: 5}) / discounts
    dcg = top5.groupby("srch_id")["dcg_part"].sum()
    ndcg = (dcg / ideal_dcg).fillna(0)
    return float(ndcg.mean())


def price_top5_hit_rates_from_ranked(ranked):
    booked = ranked[ranked["booking_bool"] == 1]
    high = booked[booked["price_tier_slice"] == "high_price_within_search"]
    low = booked[booked["price_tier_slice"] == "low_price_within_search"]
    high_top5 = (high["mitigation_rank"] <= 5).mean()
    low_top5 = (low["mitigation_rank"] <= 5).mean()
    return high_top5, low_top5


def mitigation_metrics_row(label, offset, df, score_col, ideal_dcg):
    ranked = ranked_for_score(df, score_col)
    high_top5, low_top5 = price_top5_hit_rates_from_ranked(ranked)
    return {
        "variant": label,
        "high_price_offset": offset,
        "overall_ndcg_at_5": overall_ndcg_from_ranked(ranked, ideal_dcg),
        "high_price_top5_hit_rate": high_top5,
        "low_price_top5_hit_rate": low_top5,
        "disparity_gap_low_minus_high": low_top5 - high_top5,
    }


def tune_high_price_offset(df):
    rows = []
    ideal_dcg = ideal_dcg_by_search(df)
    for offset in MITIGATION_OFFSETS:
        adjusted = df.copy()
        adjusted["adjusted_score"] = adjusted["score"]
        high_price = adjusted["price_tier_slice"] == "high_price_within_search"
        adjusted.loc[high_price, "adjusted_score"] = (
            adjusted.loc[high_price, "adjusted_score"] + offset
        )
        row = mitigation_metrics_row("grid", offset, adjusted, "adjusted_score", ideal_dcg)
        rows.append(row)

    grid = pd.DataFrame(rows)
    baseline = grid.iloc[0]
    candidates = grid[grid["high_price_top5_hit_rate"] > baseline["high_price_top5_hit_rate"]].copy()
    candidates["ndcg_loss"] = baseline["overall_ndcg_at_5"] - candidates["overall_ndcg_at_5"]
    candidates["gap_reduction"] = (
        baseline["disparity_gap_low_minus_high"]
        - candidates["disparity_gap_low_minus_high"]
    )
    candidates = candidates.sort_values(
        ["ndcg_loss", "disparity_gap_low_minus_high"],
        ascending=[True, True],
    )
    return grid, candidates.iloc[0]


def price_mitigation_table(df):
    grid, best = tune_high_price_offset(df)
    ideal_dcg = ideal_dcg_by_search(df)
    adjusted = df.copy()
    adjusted["adjusted_score"] = adjusted["score"]
    high_price = adjusted["price_tier_slice"] == "high_price_within_search"
    adjusted.loc[high_price, "adjusted_score"] = (
        adjusted.loc[high_price, "adjusted_score"] + best["high_price_offset"]
    )

    before = mitigation_metrics_row("before", 0.0, df, "score", ideal_dcg)
    after = mitigation_metrics_row(
        "after",
        best["high_price_offset"],
        adjusted,
        "adjusted_score",
        ideal_dcg,
    )
    table = pd.DataFrame([before, after])
    return table, grid


def write_report(mitigation_table, auxiliary_metrics, output_dir):
    lines = [
        "# CatBoost Validation Bias Analysis",
        "",
        "## Price Mitigation Summary",
        mitigation_table.to_string(index=False),
        "",
        "## Auxiliary Validation Slices",
        auxiliary_metrics.to_string(index=False),
    ]
    (output_dir / "bias_analysis.md").write_text("\n".join(lines))


def main(
    model_path=MODEL_PATH,
    position_model_path=POSITION_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    save_scored=False,
    use_scored_cache=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_scored_cache:
        print("Loading cached validation predictions...", flush=True)
        val = pd.read_csv(output_dir / "validation_scored.csv")
    else:
        print("Loading saved CatBoost model...", flush=True)
        model, feature_cols = load_catboost_model(model_path)

        print("Preparing validation data...", flush=True)
        val = prepare_validation_data(position_model_path)

        print("Scoring validation data...", flush=True)
        val["score"] = model.predict(val[feature_cols])

    val = add_bias_columns(val)
    auxiliary_metrics = auxiliary_slice_metrics(val)
    auxiliary_metrics.to_csv(output_dir / "auxiliary_slice_metrics.csv", index=False)

    print("Tuning high-price post-processing offset...", flush=True)
    mitigation_table, mitigation_grid = price_mitigation_table(val)
    mitigation_table.to_csv(output_dir / "price_mitigation_summary.csv", index=False)
    mitigation_grid.to_csv(output_dir / "price_mitigation_grid.csv", index=False)

    if save_scored:
        val.to_csv(output_dir / "validation_scored.csv", index=False)
    write_report(mitigation_table, auxiliary_metrics, output_dir)

    print(mitigation_table.to_string(index=False), flush=True)
    print(f"Wrote mitigation summary to {output_dir / 'price_mitigation_summary.csv'}", flush=True)
    print(f"Wrote auxiliary slices to {output_dir / 'auxiliary_slice_metrics.csv'}", flush=True)
    print(f"Wrote bias analysis to {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--position-model-path", default=POSITION_MODEL_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--save-scored", action="store_true")
    parser.add_argument("--use-scored-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.model_path,
        args.position_model_path,
        args.output_dir,
        args.save_scored,
        args.use_scored_cache,
    )
