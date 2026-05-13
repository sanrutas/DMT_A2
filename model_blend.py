import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_submission import make_submission


OUTPUT_DIR = Path("artifacts/blend")
LGBM_DIR = Path("artifacts/lgbm")
CATBOOST_DIR = Path("artifacts/catboost")
BLEND_WEIGHTS = np.round(np.arange(0, 1.01, 0.05), 2)
CATBOOST_WEIGHT = 0.5
LABEL_GAIN = {
    0: 0,
    1: 1,
    5: 5,
}


def ndcg_at_5(df, score_col="score"):
    searches = df["srch_id"].drop_duplicates()
    predicted = df.sort_values(["srch_id", score_col], ascending=[True, False]).copy()
    predicted["rank"] = predicted.groupby("srch_id").cumcount() + 1
    predicted = predicted[predicted["rank"] <= 5]
    predicted_gain = predicted["relevance"].map(LABEL_GAIN)
    predicted_dcg = (
        (predicted_gain / np.log2(predicted["rank"] + 1))
        .groupby(predicted["srch_id"])
        .sum()
    )

    ideal = df.sort_values(["srch_id", "relevance"], ascending=[True, False]).copy()
    ideal["rank"] = ideal.groupby("srch_id").cumcount() + 1
    ideal = ideal[ideal["rank"] <= 5]
    ideal_gain = ideal["relevance"].map(LABEL_GAIN)
    ideal_dcg = (
        (ideal_gain / np.log2(ideal["rank"] + 1))
        .groupby(ideal["srch_id"])
        .sum()
    )

    scores = predicted_dcg.div(ideal_dcg).reindex(searches).fillna(0)
    return float(scores.mean())


def rank_scores(df, score_col):
    return df.groupby("srch_id")[score_col].rank(method="average", pct=True)


def load_predictions(path, score_col):
    df = pd.read_csv(path)
    return df.rename(columns={"score": score_col})


def load_validation_predictions():
    lgbm = load_predictions(LGBM_DIR / "validation_predictions.csv", "lgbm_score")
    catboost = load_predictions(CATBOOST_DIR / "validation_predictions.csv", "catboost_score")
    lgbm = lgbm.drop(columns=["lgbm_label"])
    catboost = catboost[["srch_id", "prop_id", "catboost_score"]]
    return lgbm.merge(catboost, on=["srch_id", "prop_id"], how="inner")


def load_test_predictions():
    lgbm = load_predictions(LGBM_DIR / "test_predictions.csv", "lgbm_score")
    catboost = load_predictions(CATBOOST_DIR / "test_predictions.csv", "catboost_score")
    return lgbm.merge(catboost, on=["srch_id", "prop_id"], how="inner")


def add_rank_scores(df):
    df = df.copy()
    df["lgbm_rank_score"] = rank_scores(df, "lgbm_score")
    df["catboost_rank_score"] = rank_scores(df, "catboost_score")
    return df


def add_blend_score(df, catboost_weight=CATBOOST_WEIGHT):
    lgbm_weight = 1 - catboost_weight
    df = df.copy()
    df["score"] = (
        catboost_weight * df["catboost_rank_score"]
        + lgbm_weight * df["lgbm_rank_score"]
    )
    return df


def evaluate_blends(val):
    val = add_rank_scores(val)
    rows = []

    for catboost_weight in BLEND_WEIGHTS:
        lgbm_weight = round(1 - catboost_weight, 2)
        predictions = add_blend_score(val, catboost_weight)
        rows.append({
            "catboost_weight": catboost_weight,
            "lgbm_weight": lgbm_weight,
            "validation_ndcg_at_5": ndcg_at_5(predictions),
        })

    results = pd.DataFrame(rows).sort_values("validation_ndcg_at_5", ascending=False)
    best = results.iloc[0]
    predictions = add_blend_score(val, best["catboost_weight"])
    return results, predictions


def save_predictions(df, path, label_cols=False):
    cols = [
        "srch_id",
        "prop_id",
        "lgbm_score",
        "catboost_score",
        "lgbm_rank_score",
        "catboost_rank_score",
        "score",
    ]
    if label_cols:
        cols = [
            "srch_id",
            "prop_id",
            "click_bool",
            "booking_bool",
            "relevance",
            "lgbm_score",
            "catboost_score",
            "lgbm_rank_score",
            "catboost_rank_score",
            "score",
        ]
    df[cols].to_csv(path, index=False)


def save_model_params(best, path):
    params = {
        "blend": {
            "score_inputs": ["catboost_rank_score", "lgbm_rank_score"],
            "catboost_weight": best["catboost_weight"],
            "lgbm_weight": best["lgbm_weight"],
            "validation_ndcg_at_5": best["validation_ndcg_at_5"],
        },
        "inputs": {
            "lgbm_validation_predictions": str(LGBM_DIR / "validation_predictions.csv"),
            "catboost_validation_predictions": str(CATBOOST_DIR / "validation_predictions.csv"),
            "lgbm_test_predictions": str(LGBM_DIR / "test_predictions.csv"),
            "catboost_test_predictions": str(CATBOOST_DIR / "test_predictions.csv"),
        },
    }
    path.write_text(json.dumps(params, indent=2))


def load_catboost_weight():
    params = json.loads((OUTPUT_DIR / "model_params.json").read_text())
    return params["blend"]["catboost_weight"]


def print_results(results):
    lgbm_score = results.loc[results["catboost_weight"] == 0, "validation_ndcg_at_5"].iloc[0]
    catboost_score = results.loc[results["catboost_weight"] == 1, "validation_ndcg_at_5"].iloc[0]
    best = results.iloc[0]

    print(f"LightGBM validation NDCG@5: {lgbm_score:.6f}", flush=True)
    print(f"CatBoost validation NDCG@5: {catboost_score:.6f}", flush=True)
    print(
        "Best blend validation NDCG@5: "
        f"{best['validation_ndcg_at_5']:.6f} "
        f"(catboost_weight={best['catboost_weight']:.2f}, "
        f"lgbm_weight={best['lgbm_weight']:.2f})",
        flush=True,
    )


def blend_validation_predictions():
    print("Loading validation predictions...", flush=True)
    val = load_validation_predictions()
    results, best_predictions = evaluate_blends(val)
    best = results.iloc[0]

    results.to_csv(OUTPUT_DIR / "blend_results.csv", index=False)
    save_predictions(best_predictions, OUTPUT_DIR / "best_validation_predictions.csv", label_cols=True)
    save_model_params(best, OUTPUT_DIR / "model_params.json")
    print_results(results)
    return best["catboost_weight"]


def blend_test_predictions(catboost_weight=CATBOOST_WEIGHT):
    print("Loading test predictions...", flush=True)
    test = load_test_predictions()
    test = add_rank_scores(test)
    test = add_blend_score(test, catboost_weight)
    save_predictions(test, OUTPUT_DIR / "test_predictions.csv")
    make_submission(test, output_path="submission.csv")


def main(run="both"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    catboost_weight = CATBOOST_WEIGHT

    if run in ["valid", "both"]:
        catboost_weight = blend_validation_predictions()

    if run == "test":
        catboost_weight = load_catboost_weight()

    if run in ["test", "both"]:
        blend_test_predictions(catboost_weight)

    print(f"Wrote blend outputs to {OUTPUT_DIR}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["valid", "test", "both"], default="both")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.run)
