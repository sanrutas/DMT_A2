import gc
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from datetime import datetime

from config import DATASET_PATHS
from features import PRIOR_HISTORY_COLUMNS, add_relevance
from split import group_train_val_split
from T3_data_preparation import clean_train_only
from saturnas_changes_old_version_lgbm.model_lgbm_SVD_saturnas import (
    FIXED_PARAMS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    add_model_features,
    lgbm_labels,
    ndcg_at_5,
    train_model
)
from SVD_Saturnas import fit_svd_features, apply_svd_features

N_TRIALS = 30
OUTPUT_DIR = Path("artifacts/lgbm")

def param_grid(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 100, 200),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.01, 0.1),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 15.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 1.0, log=True),
    }

def run_trial(trial, train_split, val):
    tunable = param_grid(trial)
    params = {**FIXED_PARAMS, **tunable}

    model = train_model(train_split, val, num_boost_round=NUM_BOOST_ROUND, params=params)

    print("Scoring validation split...", flush=True)
    val_scored = val.copy()
    val_scored["score"] = model.predict(val_scored)
    validation_ndcg = ndcg_at_5(val_scored)

    trial.set_user_attr("best_iteration", model.booster.best_iteration)

    return validation_ndcg


def tune(svd=False):
    print(f"Running LGBM TUNING, BOOST_ROUND={NUM_BOOST_ROUND}, EARLY_STOP={EARLY_STOPPING_ROUNDS}, "
        f"svd={svd}\nloading training data...", flush=True) 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = add_relevance(clean_train_only(pd.read_csv(DATASET_PATHS["train"])))
    train_split, val = group_train_val_split(train, group_col="srch_id", val_size=0.2, random_state=42)
    del train
    gc.collect()

    svd_artifacts = None
    if svd:
        print("Fitting SVD on train split...", flush=True)
        svd_artifacts = fit_svd_features(train_split)
        train_split = apply_svd_features(train_split, *svd_artifacts)
        val = apply_svd_features(val, *svd_artifacts)

    print("Building features...", flush=True)
    split_history = train_split[PRIOR_HISTORY_COLUMNS].copy()
    train_split = add_model_features(split_history, train_split)
    val = add_model_features(split_history, val)
    del split_history
    gc.collect()

    train_split["lgbm_label"] = lgbm_labels(train_split)
    val["lgbm_label"] = lgbm_labels(val)

    print(f"Starting Optuna study ({N_TRIALS} trials)...", flush=True)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: run_trial(trial, train_split, val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    best_params = best_trial.params
    best_iteration = best_trial.user_attrs["best_iteration"]
    best_ndcg = best_trial.value

    print(f"\nBest NDCG@5: {best_ndcg:.6f}")
    print(f"Best iteration: {best_iteration}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    output = {
        "tunable_params": best_params,
        "best_iteration": best_iteration,
        "best_ndcg_at_5": best_ndcg,
    }

    BEST_PARAMS_PATH = OUTPUT_DIR / f"best_params_{datetime.now():%Y%m%d_%H%M%S}.json"
    BEST_PARAMS_PATH.write_text(json.dumps(output, indent=2))
    print(f"Saved best params to {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    tune()