import gc
import json
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd

from config import (
    DATASET_PATHS,
    POSITION_BOOST_ROUND,
    POSITION_EARLY_STOPPING_ROUNDS,
    POSITION_PARAMS,
)
from features import add_features, add_oof_historical_priors
from model_position import position_feature_columns, position_labels
from split import group_train_val_audit_split
from T3_data_preparation import clean_train_only


OUTPUT_DIR = Path("artifacts/lgbm")
TUNING_RESULTS_PATH = OUTPUT_DIR / "position_tuning_results.csv"
TUNING_BEST_PARAMS_PATH = OUTPUT_DIR / "position_best_params.json"
N_TRIALS = 50


def load_position_data():
    print("Loading training data...", flush=True)
    df = pd.read_csv(DATASET_PATHS["train"])
    df = clean_train_only(df)

    print("Building position-estimator features...", flush=True)
    df = add_features(df)
    df = add_oof_historical_priors(df)
    return df


def suggest_position_params(trial):
    params = POSITION_PARAMS.copy()
    params.update({
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127, 255]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 1000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12]),
    })
    return params


def current_position_trial_params():
    return {
        "learning_rate": POSITION_PARAMS["learning_rate"],
        "num_leaves": POSITION_PARAMS["num_leaves"],
        "min_data_in_leaf": POSITION_PARAMS["min_data_in_leaf"],
        "feature_fraction": POSITION_PARAMS["feature_fraction"],
        "bagging_fraction": POSITION_PARAMS["bagging_fraction"],
        "bagging_freq": POSITION_PARAMS["bagging_freq"],
        "min_gain_to_split": POSITION_PARAMS["min_gain_to_split"],
        "lambda_l1": 1e-8,
        "lambda_l2": 1e-8,
        "max_depth": -1,
    }


def make_position_datasets(df):
    fit_df = df[df["random_bool"] == 0]
    train_fit, val_fit, _ = group_train_val_audit_split(
        fit_df,
        group_col="srch_id",
        val_size=0.2,
        audit_size=0.1,
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
    return train_data, val_data


def tune_position_estimator(df, n_trials=N_TRIALS):
    train_data, val_data = make_position_datasets(df)

    def objective(trial):
        booster = lgb.train(
            suggest_position_params(trial),
            train_data,
            num_boost_round=POSITION_BOOST_ROUND,
            valid_sets=[val_data],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(POSITION_EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=0),
            ],
        )
        gc.collect()
        return booster.best_score["valid"]["rmse"]

    study = optuna.create_study(direction="minimize")
    study.enqueue_trial(current_position_trial_params())
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    return study


def save_study(study):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = study.trials_dataframe()
    results.to_csv(TUNING_RESULTS_PATH, index=False)

    best = POSITION_PARAMS.copy()
    best.update(study.best_params)
    best_params = {
        "best_value": study.best_value,
        "best_params": best,
        "best_trial_params": study.best_params,
        "position_boost_round": POSITION_BOOST_ROUND,
        "position_early_stopping_rounds": POSITION_EARLY_STOPPING_ROUNDS,
    }
    TUNING_BEST_PARAMS_PATH.write_text(json.dumps(best_params, indent=2))


def main(n_trials=N_TRIALS):
    df = load_position_data()
    study = tune_position_estimator(df, n_trials=n_trials)
    save_study(study)
    print(f"Best RMSE: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Wrote tuning results to {TUNING_RESULTS_PATH}")


if __name__ == "__main__":
    main()
