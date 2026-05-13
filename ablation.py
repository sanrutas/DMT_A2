import gc
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import (
    ABLATION_FEATURE_IMPORTANCE_PATH,
    ABLATION_FEATURES_PATH,
    ABLATION_MAX_FEATURES,
    ABLATION_RESULTS_PATH,
    DATASET_PATHS,
)
from features import PRIOR_HISTORY_COLUMNS, add_relevance
from model_lgbm import (
    add_model_features,
    add_oof_model_features,
    load_ablated_features,
    lgbm_labels,
    ndcg_at_5,
    save_feature_importance,
    train_model,
)
from split import group_train_val_audit_split
from T3_data_preparation import clean_train_only


OUTPUT_DIR = Path("artifacts/lgbm")


def format_duration(seconds):
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def prepare_validation_data():
    print("Loading training data for validation split...", flush=True)
    train = pd.read_csv(DATASET_PATHS["train"])
    train = add_relevance(clean_train_only(train))
    train, val, _ = group_train_val_audit_split(
        train,
        group_col="srch_id",
        val_size=0.2,
        audit_size=0.1,
        random_state=42,
    )
    gc.collect()

    print("Building validation-split features...", flush=True)
    split_history = train[PRIOR_HISTORY_COLUMNS].copy()
    train = add_oof_model_features(split_history, train)
    val = add_model_features(split_history, val)
    del split_history
    gc.collect()

    train["lgbm_label"] = lgbm_labels(train)
    val["lgbm_label"] = lgbm_labels(val)
    return train, val


def validation_score(model, val):
    val["score"] = model.predict(val)
    return ndcg_at_5(val)


def candidate_features(path, ablated_features):
    importance = pd.read_csv(path).sort_values("importance_gain", ascending=False)
    ablated_features = set(ablated_features)
    features = [
        feature
        for feature in importance["feature"].tolist()
        if feature not in ablated_features
    ]
    return features[:ABLATION_MAX_FEATURES]


def append_ablated_feature(feature, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(f"{feature}\n")


def append_result(row, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def expected_end_time(trial, total_trials, trial_durations):
    average_trial_time = sum(trial_durations) / len(trial_durations)
    remaining = average_trial_time * (total_trials - trial)
    return datetime.now() + timedelta(seconds=remaining)


def log_progress(start_time, expected_end):
    elapsed = time.monotonic() - start_time
    print(
        f"Runtime {format_duration(elapsed)} | expected end {expected_end:%Y-%m-%d %H:%M:%S}",
        flush=True,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()

    train, val = prepare_validation_data()
    ablated_features = load_ablated_features(ABLATION_FEATURES_PATH)

    print(f"Training baseline without {len(ablated_features)} ablated features...", flush=True)
    baseline_model = train_model(train, val, excluded_features=ablated_features)
    best_score = validation_score(baseline_model, val)
    save_feature_importance(baseline_model, ABLATION_FEATURE_IMPORTANCE_PATH)
    print(f"Baseline validation NDCG@5: {best_score:.6f}", flush=True)
    del baseline_model
    gc.collect()

    features = candidate_features(ABLATION_FEATURE_IMPORTANCE_PATH, ablated_features)
    trial_durations = []

    for trial, feature in enumerate(features, start=1):
        trial_start = time.monotonic()
        excluded_features = ablated_features + [feature]
        print(f"Trial {trial}/{len(features)}: training without {feature}...", flush=True)

        model = train_model(train, val, excluded_features=excluded_features)
        score = validation_score(model, val)
        improved = score > best_score
        duration = time.monotonic() - trial_start
        trial_durations.append(duration)
        expected_end = expected_end_time(trial, len(features), trial_durations)

        row = {
            "trial": trial,
            "feature": feature,
            "removed": improved,
            "validation_ndcg_at_5": score,
            "previous_best_ndcg_at_5": best_score,
            "best_iteration": model.booster.best_iteration,
            "current_iteration": model.booster.current_iteration(),
            "trial_runtime_seconds": round(duration, 2),
            "elapsed_seconds": round(time.monotonic() - start_time, 2),
            "time_finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "expected_end": expected_end.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if improved:
            best_score = score
            ablated_features.append(feature)
            append_ablated_feature(feature, ABLATION_FEATURES_PATH)
            save_feature_importance(model, ABLATION_FEATURE_IMPORTANCE_PATH)
            print(f"Accepted {feature}; validation NDCG@5 improved to {score:.6f}", flush=True)
        else:
            print(f"Kept {feature}; validation NDCG@5 stayed at {best_score:.6f}", flush=True)

        append_result(row, ABLATION_RESULTS_PATH)
        del model
        gc.collect()
        log_progress(start_time, expected_end)

    print(f"Done. Best validation NDCG@5: {best_score:.6f}", flush=True)
    print(f"Ablated features written to {ABLATION_FEATURES_PATH}", flush=True)
    print(f"Trial results written to {ABLATION_RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
