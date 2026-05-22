import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    DATASET_PATHS,
    ESTIMATED_POSITION_COLUMNS,
    POSITION_BLOCKED_COLUMNS,
    POSITION_BOOST_ROUND,
    POSITION_EARLY_STOPPING_ROUNDS,
    POSITION_FOLDS,
    POSITION_PARAMS,
)
from features import (
    PRIOR_HISTORY_COLUMNS,
    add_country_imputations,
    add_features,
    add_historical_priors,
    add_oof_historical_priors,
    add_relevance,
    new_features,
)
from split import group_train_val_split
from T3_data_preparation import clean_train_only, model_feature_columns


OUTPUT_DIR = Path("artifacts/position")
MODEL_PATH = Path("artifacts/lgbm/position_model.txt")
VALIDATION_MODEL_PATH = OUTPUT_DIR / "validation_position_model.txt"


class LightGBMPositionEstimator:
    def __init__(self, booster, feature_cols):
        self.booster = booster
        self.feature_cols = feature_cols

    def predict(self, df):
        return self.booster.predict(
            df[self.feature_cols],
            num_iteration=self.booster.best_iteration or self.booster.current_iteration(),
        )


def save_position_estimator(estimator, model_path):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    estimator.booster.save_model(model_path)
    model_path.with_suffix(".features.json").write_text(json.dumps(estimator.feature_cols, indent=2))


def load_position_estimator(model_path):
    model_path = Path(model_path)
    booster = lgb.Booster(model_file=str(model_path))
    feature_cols = json.loads(model_path.with_suffix(".features.json").read_text())
    return LightGBMPositionEstimator(booster, feature_cols)


def position_labels(df):
    group_size = df.groupby("srch_id")["position"].transform("size")
    denom = (group_size - 1).clip(lower=1)
    return ((df["position"] - 1) / denom).clip(0, 1)


def position_feature_columns(df):
    blocked = {"lgbm_label", "score"} | set(ESTIMATED_POSITION_COLUMNS) | POSITION_BLOCKED_COLUMNS
    return [col for col in model_feature_columns(df) if col not in blocked]


def add_model_features(history, df):
    df = add_features(df)
    df = add_country_imputations(history, df)
    df = add_historical_priors(history, df)
    return new_features(df)


def add_oof_model_features(history, df):
    df = add_features(df)
    df = add_country_imputations(history, df)
    df = add_oof_historical_priors(df)
    return new_features(df)


def train_position_estimator(
    train_df,
    num_boost_round=POSITION_BOOST_ROUND,
    params=None,
    init_model_path=None,
):
    if params is None:
        params = POSITION_PARAMS
    fit_df = train_df[train_df["random_bool"] == 0]
    train_fit, val_fit = group_train_val_split(
        fit_df,
        group_col="srch_id",
        val_size=0.2,
        random_state=42,
    )
    if init_model_path is None:
        feature_cols = position_feature_columns(fit_df)
    else:
        feature_cols = json.loads(Path(init_model_path).with_suffix(".features.json").read_text())
    train_data = lgb.Dataset(
        train_fit[feature_cols],
        label=position_labels(train_fit),
        free_raw_data=True,
    )
    val_data = lgb.Dataset(
        val_fit[feature_cols],
        label=position_labels(val_fit),
        reference=train_data,
        free_raw_data=True,
    )
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        init_model=init_model_path,
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(POSITION_EARLY_STOPPING_ROUNDS),
        ],
    )
    return LightGBMPositionEstimator(booster, feature_cols)


def add_estimated_position_predictions(df, predictions):
    df = df.copy()
    group_size = df.groupby("srch_id")["prop_id"].transform("size")
    df["estimated_position_pct"] = np.clip(predictions, 0, 1)
    df["estimated_position"] = 1 + df["estimated_position_pct"] * (group_size - 1)
    df["estimated_position_inverse"] = 1 / df["estimated_position"]
    df["estimated_position_rank"] = df.groupby("srch_id")["estimated_position"].rank(
        method="average",
        ascending=True,
    )
    df["estimated_position_rank_pct"] = df.groupby("srch_id")["estimated_position"].rank(
        method="average",
        ascending=True,
        pct=True,
    )
    return df


def add_estimated_position_features(train_df, predict_dfs):
    groups = train_df["srch_id"].drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(42)
    rng.shuffle(groups)
    group_folds = np.array_split(groups, POSITION_FOLDS)
    train_predictions = pd.Series(index=train_df.index, dtype=float)

    for i, fold_groups in enumerate(group_folds, start=1):
        print(f"Training estimated-position fold {i}/{POSITION_FOLDS}...", flush=True)
        val_mask = train_df["srch_id"].isin(fold_groups)
        estimator = train_position_estimator(train_df.loc[~val_mask])
        train_predictions.loc[val_mask] = estimator.predict(train_df.loc[val_mask])
        del estimator
        gc.collect()

    print("Training estimated-position model for held-out/test rows...", flush=True)
    estimator = train_position_estimator(train_df)
    result_train = add_estimated_position_predictions(train_df, train_predictions.to_numpy())
    result_predict_dfs = [
        add_estimated_position_predictions(df, estimator.predict(df))
        for df in predict_dfs
    ]
    return result_train, result_predict_dfs, estimator


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def save_feature_importance(estimator, path):
    importance = pd.DataFrame({
        "feature": estimator.feature_cols,
        "importance_gain": estimator.booster.feature_importance(importance_type="gain"),
        "importance_split": estimator.booster.feature_importance(importance_type="split"),
    })
    importance = importance.sort_values("importance_gain", ascending=False)
    importance.to_csv(path, index=False)


def save_model_params(estimator, path, validation_rmse=None, resumed_from=None, resume_rounds=None):
    params = {
        "params": POSITION_PARAMS,
        "num_boost_round": POSITION_BOOST_ROUND,
        "resumed_from": str(resumed_from) if resumed_from is not None else None,
        "resume_rounds": resume_rounds,
        "early_stopping_rounds": POSITION_EARLY_STOPPING_ROUNDS,
        "folds": POSITION_FOLDS,
        "best_iteration": estimator.booster.best_iteration,
        "current_iteration": estimator.booster.current_iteration(),
        "best_score": estimator.booster.best_score,
        "validation_rmse": validation_rmse,
        "features": estimator.feature_cols,
    }
    path.write_text(json.dumps(params, indent=2))


def save_predictions(df, path):
    cols = [
        "srch_id",
        "prop_id",
        "position",
        "estimated_position_pct",
        "estimated_position",
        "estimated_position_inverse",
        "estimated_position_rank",
        "estimated_position_rank_pct",
    ]
    df[cols].to_csv(path, index=False)


def prepare_validation_data():
    print("Loading training data for validation split...", flush=True)
    train = pd.read_csv(DATASET_PATHS["train"])
    train = add_relevance(clean_train_only(train))
    train, val = group_train_val_split(train, group_col="srch_id", val_size=0.2, random_state=42)
    gc.collect()

    print("Building validation-split features...", flush=True)
    split_history = train[PRIOR_HISTORY_COLUMNS].copy()
    train = add_oof_model_features(split_history, train)
    val = add_model_features(split_history, val)
    del split_history
    gc.collect()

    return train, val


def prepare_training_data():
    print("Loading training data for final position model...", flush=True)
    train = pd.read_csv(DATASET_PATHS["train"])
    train = add_relevance(clean_train_only(train))
    history = train[PRIOR_HISTORY_COLUMNS].copy()

    print("Building final-training features...", flush=True)
    train = add_oof_model_features(history, train)
    del history
    gc.collect()

    return train


def train_validation_model(resume=False, resume_rounds=POSITION_BOOST_ROUND):
    train, val = prepare_validation_data()

    init_model_path = VALIDATION_MODEL_PATH if resume else None
    if resume:
        print(f"Resuming validation position model from {init_model_path}...", flush=True)
    else:
        print("Training validation position model...", flush=True)
    estimator = train_position_estimator(
        train,
        num_boost_round=resume_rounds,
        init_model_path=init_model_path,
    )

    print("Scoring validation split...", flush=True)
    predictions = estimator.predict(val)
    val = add_estimated_position_predictions(val, predictions)
    validation_rmse = rmse(position_labels(val), val["estimated_position_pct"])

    save_predictions(val, OUTPUT_DIR / "validation_predictions.csv")
    save_feature_importance(estimator, OUTPUT_DIR / "validation_feature_importances.csv")
    save_position_estimator(estimator, VALIDATION_MODEL_PATH)
    save_model_params(
        estimator,
        OUTPUT_DIR / "validation_model_params.json",
        validation_rmse,
        resumed_from=init_model_path,
        resume_rounds=resume_rounds if resume else None,
    )

    print(f"Validation position RMSE: {validation_rmse:.6f}", flush=True)
    print(f"Wrote validation outputs to {OUTPUT_DIR}", flush=True)
    return validation_rmse


def train_test_model(resume=False, resume_rounds=POSITION_BOOST_ROUND):
    train = prepare_training_data()

    init_model_path = MODEL_PATH if resume else None
    if resume:
        print(f"Resuming final position model from {init_model_path}...", flush=True)
    else:
        print("Training final position model...", flush=True)
    estimator = train_position_estimator(
        train,
        num_boost_round=resume_rounds,
        init_model_path=init_model_path,
    )

    save_position_estimator(estimator, MODEL_PATH)
    save_feature_importance(estimator, OUTPUT_DIR / "feature_importances.csv")
    save_model_params(
        estimator,
        OUTPUT_DIR / "model_params.json",
        resumed_from=init_model_path,
        resume_rounds=resume_rounds if resume else None,
    )

    print(f"Wrote final position model to {MODEL_PATH}", flush=True)


def main(run="both", resume=False, resume_rounds=POSITION_BOOST_ROUND):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if run in ["valid", "both"]:
        train_validation_model(resume, resume_rounds)

    if run in ["test", "both"]:
        train_test_model(resume, resume_rounds)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["valid", "test", "both"], default="both")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-rounds", type=int, default=POSITION_BOOST_ROUND)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.run, args.resume, args.resume_rounds)
