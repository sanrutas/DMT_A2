import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import DATASET_PATHS
from features import add_features, add_relevance
from split import group_train_val_split
from T3_data_preparation import clean_train_only, model_feature_columns


PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "label_gain": [0, 1, 5],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
}

OUTPUT_DIR = Path("artifacts/lgbm")
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100


class LightGBMRanker:
    def __init__(self, booster, feature_cols):
        self.booster = booster
        self.feature_cols = feature_cols

    def predict(self, df):
        return self.booster.predict(
            df[self.feature_cols],
            num_iteration=self.booster.best_iteration,
        )


def lgbm_labels(df):
    return np.where(df["booking_bool"] == 1, 2, np.where(df["click_bool"] == 1, 1, 0))


def sorted_by_search(df):
    return df.sort_values("srch_id").reset_index(drop=True)


def group_sizes(df):
    return df.groupby("srch_id", sort=False).size().to_numpy()


def lgbm_feature_columns(df):
    blocked = {"lgbm_label", "score"}
    return [col for col in model_feature_columns(df) if col not in blocked]


def make_dataset(df, feature_cols):
    df = sorted_by_search(df)
    labels = lgbm_labels(df)
    dataset = lgb.Dataset(
        df[feature_cols],
        label=labels,
        group=group_sizes(df),
        free_raw_data=False,
    )
    return df, dataset


def train_model(train_df, val_df=None):
    feature_cols = lgbm_feature_columns(train_df)
    _, train_data = make_dataset(train_df, feature_cols)
    valid_sets = [train_data]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=50)]

    if val_df is not None:
        _, val_data = make_dataset(val_df, feature_cols)
        valid_sets.append(val_data)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS))

    booster = lgb.train(
        PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return LightGBMRanker(booster, feature_cols)


def dcg(labels, k=5):
    labels = np.asarray(labels)[:k]
    gains = np.asarray(PARAMS["label_gain"])[labels]
    discounts = np.log2(np.arange(2, len(labels) + 2))
    return np.sum(gains / discounts)


def ndcg_at_5(df):
    scores = []

    for _, group in df.groupby("srch_id", sort=False):
        predicted = group.sort_values("score", ascending=False)["lgbm_label"].to_numpy()
        ideal = group.sort_values("lgbm_label", ascending=False)["lgbm_label"].to_numpy()
        ideal_dcg = dcg(ideal, k=5)
        scores.append(dcg(predicted, k=5) / ideal_dcg if ideal_dcg > 0 else 0.0)

    return float(np.mean(scores))


def add_scores(model, df):
    scored = df.copy()
    scored["score"] = model.predict(scored)
    return scored


def save_predictions(df, path, label_cols=False):
    cols = ["srch_id", "prop_id", "score"]
    if label_cols:
        cols = [
            "srch_id",
            "prop_id",
            "click_bool",
            "booking_bool",
            "relevance",
            "lgbm_label",
            "score",
        ]
    df[cols].to_csv(path, index=False)


def save_feature_importance(model, path):
    importance = pd.DataFrame({
        "feature": model.feature_cols,
        "importance_gain": model.booster.feature_importance(importance_type="gain"),
        "importance_split": model.booster.feature_importance(importance_type="split"),
    })
    importance = importance.sort_values("importance_gain", ascending=False)
    importance.to_csv(path, index=False)


def save_model_params(model, path, validation_ndcg):
    model_params = {
        "params": PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": model.booster.best_iteration,
        "best_score": model.booster.best_score,
        "validation_ndcg_at_5": validation_ndcg,
        "features": model.feature_cols,
    }
    path.write_text(json.dumps(model_params, indent=2))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATASET_PATHS["train"])
    test = pd.read_csv(DATASET_PATHS["test"])

    train = clean_train_only(train)
    train = add_relevance(train)
    train, val = group_train_val_split(train, group_col="srch_id", val_size=0.2, random_state=42)

    train = add_features(train)
    val = add_features(val)
    test = add_features(test)

    train["lgbm_label"] = lgbm_labels(train)
    val["lgbm_label"] = lgbm_labels(val)

    model = train_model(train, val)

    val = add_scores(model, val)
    test = add_scores(model, test)
    validation_ndcg = ndcg_at_5(val)

    save_predictions(val, OUTPUT_DIR / "validation_predictions.csv", label_cols=True)
    save_predictions(test, OUTPUT_DIR / "test_predictions.csv")
    save_feature_importance(model, OUTPUT_DIR / "feature_importances.csv")
    save_model_params(model, OUTPUT_DIR / "model_params.json", validation_ndcg)

    print(f"Validation NDCG@5: {validation_ndcg:.6f}")
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
