import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ablation import ablated_feature_columns
from config import (
    ABLATION,
    ABLATION_FEATURE_IMPORTANCE_PATH,
    ABLATION_TOP_K,
    ESTIMATED_POSITION_COLUMNS,
    POSITION_BLOCKED_COLUMNS,
    POSITION_BOOST_ROUND,
    POSITION_EARLY_STOPPING_ROUNDS,
    POSITION_FOLDS,
    POSITION_PARAMS,
)
from split import group_train_val_split
from T3_data_preparation import model_feature_columns


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
    feature_cols = [col for col in model_feature_columns(df) if col not in blocked]
    return ablated_feature_columns(
        feature_cols,
        ABLATION,
        ABLATION_TOP_K,
        ABLATION_FEATURE_IMPORTANCE_PATH,
    )


def train_position_estimator(train_df, num_boost_round=POSITION_BOOST_ROUND, params=None):
    if params is None:
        params = POSITION_PARAMS
    fit_df = train_df[train_df["random_bool"] == 0]
    train_fit, val_fit = group_train_val_split(
        fit_df,
        group_col="srch_id",
        val_size=0.2,
        random_state=42,
    )
    feature_cols = position_feature_columns(fit_df)
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
    groups = train_df["srch_id"].drop_duplicates().to_numpy()
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
