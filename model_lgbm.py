import json
import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    DATASET_PATHS,
    POSITION_BOOST_ROUND,
    POSITION_EARLY_STOPPING_ROUNDS,
    POSITION_FOLDS,
    POSITION_PARAMS,
)
from features import PRIOR_HISTORY_COLUMNS, add_features, add_historical_priors, add_relevance
from make_submission import make_submission
from model_position import (
    add_estimated_position_features,
    add_estimated_position_predictions,
    load_position_estimator,
    save_position_estimator,
)
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
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "min_gain_to_split": 0.05,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
}

OUTPUT_DIR = Path("artifacts/lgbm")
POSITION_MODEL_PATH = OUTPUT_DIR / "position_model.txt"
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100


class LightGBMRanker:
    def __init__(self, booster, feature_cols):
        self.booster = booster
        self.feature_cols = feature_cols

    def predict(self, df):
        return self.booster.predict(
            df[self.feature_cols],
            num_iteration=self.booster.best_iteration or self.booster.current_iteration(),
        )


def lgbm_labels(df):
    return np.where(df["booking_bool"] == 1, 2, np.where(df["click_bool"] == 1, 1, 0))


def make_dataset(df, feature_cols):
    df = df.sort_values("srch_id").reset_index(drop=True)
    labels = df["lgbm_label"].to_numpy() if "lgbm_label" in df.columns else lgbm_labels(df)
    dataset = lgb.Dataset(
        df[feature_cols],
        label=labels,
        group=df.groupby("srch_id", sort=False).size().to_numpy(),
        free_raw_data=True,
    )
    return df, dataset


def train_model(train_df, val_df=None, num_boost_round=NUM_BOOST_ROUND):
    blocked = {"lgbm_label", "score"}
    feature_cols = [col for col in model_feature_columns(train_df) if col not in blocked]
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
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return LightGBMRanker(booster, feature_cols)


def add_model_features(history, df):
    df = add_features(df)
    return add_historical_priors(history, df)


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


def save_model_params(model, path, validation_ndcg, use_position_estimator):
    model_params = {
        "params": PARAMS,
        "use_position_estimator": use_position_estimator,
        "position_params": POSITION_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "position_boost_round": POSITION_BOOST_ROUND,
        "position_folds": POSITION_FOLDS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "position_early_stopping_rounds": POSITION_EARLY_STOPPING_ROUNDS,
        "best_iteration": model.booster.best_iteration,
        "current_iteration": model.booster.current_iteration(),
        "best_score": model.booster.best_score,
        "validation_ndcg_at_5": validation_ndcg,
        "features": model.feature_cols,
    }
    path.write_text(json.dumps(model_params, indent=2))


def add_cached_estimated_position_features(train_df, predict_dfs, position_model):
    result_train = add_estimated_position_predictions(train_df, position_model.predict(train_df))
    result_predict_dfs = [
        add_estimated_position_predictions(df, position_model.predict(df))
        for df in predict_dfs
    ]
    return result_train, result_predict_dfs


def main(train_full=True, retrain_pos_model=False, use_position_estimator=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data for validation split...", flush=True)
    train = pd.read_csv(DATASET_PATHS["train"])

    train = add_relevance(clean_train_only(train))
    train, val = group_train_val_split(train, group_col="srch_id", val_size=0.2, random_state=42)
    gc.collect()

    print("Building validation-split features...", flush=True)
    split_history = train[PRIOR_HISTORY_COLUMNS].copy()
    train = add_model_features(split_history, train)
    val = add_model_features(split_history, val)
    del split_history
    gc.collect()

    if use_position_estimator:
        if retrain_pos_model:
            print("Adding estimated-position features for validation split...", flush=True)
            train, (val,), position_model = add_estimated_position_features(train, [val])
        else:
            print("Loading cached estimated-position model for validation split...", flush=True)
            position_model = load_position_estimator(POSITION_MODEL_PATH)
            train, (val,) = add_cached_estimated_position_features(train, [val], position_model)
        del position_model
        gc.collect()

    train["lgbm_label"] = lgbm_labels(train)
    val["lgbm_label"] = lgbm_labels(val)

    print("Training validation model...", flush=True)
    model = train_model(train, val)

    print("Scoring validation split...", flush=True)
    val["score"] = model.predict(val)
    validation_ndcg = ndcg_at_5(val)
    final_rounds = model.booster.best_iteration or NUM_BOOST_ROUND

    save_predictions(val, OUTPUT_DIR / "validation_predictions.csv", label_cols=True)

    if not train_full:
        save_feature_importance(model, OUTPUT_DIR / "feature_importances.csv")
        save_model_params(
            model,
            OUTPUT_DIR / "model_params.json",
            validation_ndcg,
            use_position_estimator,
        )
        print(f"Validation NDCG@5: {validation_ndcg:.6f}")
        print(f"Wrote validation outputs to {OUTPUT_DIR}")
        return

    del train, val, model
    gc.collect()

    print("Loading training data for final model...", flush=True)
    full_train = pd.read_csv(DATASET_PATHS["train"])
    full_train = add_relevance(clean_train_only(full_train))
    full_history = full_train[PRIOR_HISTORY_COLUMNS].copy()
    print("Building final-training features...", flush=True)
    full_train = add_model_features(full_history, full_train)
    if use_position_estimator:
        if retrain_pos_model:
            print("Adding estimated-position features for final training data...", flush=True)
            full_train, _, final_position_model = add_estimated_position_features(full_train, [])
            save_position_estimator(final_position_model, POSITION_MODEL_PATH)
        else:
            print("Loading cached estimated-position model for final training data...", flush=True)
            final_position_model = load_position_estimator(POSITION_MODEL_PATH)
            full_train, _ = add_cached_estimated_position_features(full_train, [], final_position_model)
    full_train["lgbm_label"] = lgbm_labels(full_train)
    full_train = full_train.drop(
        columns=[
            "position",
            "click_bool",
            "booking_bool",
            "gross_booking_usd",
            "gross_bookings_usd",
            "relevance",
        ],
        errors="ignore",
    )

    print(f"Training final model for {final_rounds} rounds...", flush=True)
    final_model = train_model(full_train, num_boost_round=final_rounds)
    del full_train
    gc.collect()

    print("Loading and featurizing test data...", flush=True)
    test = pd.read_csv(DATASET_PATHS["test"])
    test = add_model_features(full_history, test)
    del full_history
    gc.collect()
    if use_position_estimator:
        test = add_estimated_position_predictions(test, final_position_model.predict(test))
        del final_position_model
        gc.collect()

    print("Scoring test data...", flush=True)
    test["score"] = final_model.predict(test)

    print("Writing outputs...", flush=True)
    save_predictions(test, OUTPUT_DIR / "test_predictions.csv")
    make_submission(test, output_path="submission.csv")
    save_feature_importance(final_model, OUTPUT_DIR / "feature_importances.csv")
    save_model_params(
        final_model,
        OUTPUT_DIR / "model_params.json",
        validation_ndcg,
        use_position_estimator,
    )

    print(f"Validation NDCG@5: {validation_ndcg:.6f}")
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
