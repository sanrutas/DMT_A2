import json
import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import DATASET_PATHS
from features import PRIOR_HISTORY_COLUMNS, add_features, add_historical_priors, add_relevance
from make_submission import make_submission
from split import group_train_val_split
from T3_data_preparation import clean_train_only, model_feature_columns
from saturnas_changes_old_version_lgbm.SVD_Saturnas import fit_svd_features, apply_svd_features

TUNED_PARAMS = {
    'learning_rate': 0.012868288646766658,
    'num_leaves': 148,
    'min_data_in_leaf': 273,
    'feature_fraction': 0.8027712051774001,
    'bagging_fraction': 0.5697934280175004,
    'min_gain_to_split': 0.04330139640082351,
    'lambda_l1': 9.862622190490258,
    'lambda_l2': 0.02329476795812801
}
# TUNED_PARAMS = {
#     "learning_rate": 0.05,
#     "num_leaves": 63,
#     "min_data_in_leaf": 100,
#     "feature_fraction": 0.6,
#     "bagging_fraction": 0.7,
#     "min_gain_to_split": 0.05,
#     "lambda_l1": 0.1,
#     "lambda_l2": 0.1,
# }
FIXED_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "label_gain": [0, 1, 5],
    "verbosity": -1,
    "seed": 42,
    "bagging_freq": 1,
}

PARAMS = {
    **TUNED_PARAMS,
    **FIXED_PARAMS,
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


def train_model(train_df, val_df=None, num_boost_round=NUM_BOOST_ROUND, params=PARAMS):
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
        params,
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


# def dcg(labels, k=5):
#     labels = np.asarray(labels)[:k]
#     gains = np.asarray(PARAMS["label_gain"])[labels]
#     discounts = np.log2(np.arange(2, len(labels) + 2))
#     return np.sum(gains / discounts)


# def ndcg_at_5(df):
#     scores = []

#     for _, group in df.groupby("srch_id", sort=False):
#         predicted = group.sort_values("score", ascending=False)["lgbm_label"].to_numpy()
#         ideal = group.sort_values("lgbm_label", ascending=False)["lgbm_label"].to_numpy()
#         ideal_dcg = dcg(ideal, k=5)
#         scores.append(dcg(predicted, k=5) / ideal_dcg if ideal_dcg > 0 else 0.0)

#     return float(np.mean(scores))

def ndcg_at_5(df, k=5):
    label_gain = np.array(PARAMS["label_gain"])

    df = df.sort_values(["srch_id", "score"], ascending=[True, False])
    df["rank"] = df.groupby("srch_id").cumcount()

    top_k = df[df["rank"] < k].copy()
    top_k["gain"] = label_gain[top_k["lgbm_label"].to_numpy()]
    top_k["discount"] = np.log2(top_k["rank"] + 2)
    top_k["dcg"] = top_k["gain"] / top_k["discount"]

    predicted_dcg = top_k.groupby("srch_id")["dcg"].sum()

    df2 = df.sort_values(["srch_id", "lgbm_label"], ascending=[True, False])
    df2["rank"] = df2.groupby("srch_id").cumcount()
    top_k2 = df2[df2["rank"] < k].copy()
    top_k2["gain"] = label_gain[top_k2["lgbm_label"].to_numpy()]
    top_k2["discount"] = np.log2(top_k2["rank"] + 2)
    top_k2["dcg"] = top_k2["gain"] / top_k2["discount"]

    ideal_dcg = top_k2.groupby("srch_id")["dcg"].sum()

    ndcg = (predicted_dcg / ideal_dcg.replace(0, np.nan)).dropna()
    return float(ndcg.mean())


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
        "current_iteration": model.booster.current_iteration(),
        "best_score": model.booster.best_score,
        "validation_ndcg_at_5": validation_ndcg,
        "features": model.feature_cols,
    }
    path.write_text(json.dumps(model_params, indent=2))

def train_validation_model(train, svd, params):
    train = pd.read_csv(DATASET_PATHS["train"])
    train = add_relevance(clean_train_only(train))

    train_split, val = group_train_val_split(train, group_col="srch_id", val_size=0.2, random_state=42)
    gc.collect()

    svd_artifacts = None
    if svd:
        print("Fitting SVD on train split...", flush=True)
        svd_artifacts = fit_svd_features(train_split)
        train_split = apply_svd_features(train_split, *svd_artifacts)
        val = apply_svd_features(val, *svd_artifacts)

    print("Building validation-split features...", flush=True)
    split_history = train_split[PRIOR_HISTORY_COLUMNS].copy()
    train_split = add_model_features(split_history, train_split)
    val = add_model_features(split_history, val)
    del split_history
    gc.collect()

    train_split["lgbm_label"] = lgbm_labels(train_split)
    val["lgbm_label"] = lgbm_labels(val)

    print("Training validation model...", flush=True)
    model = train_model(train_split, val, num_boost_round=NUM_BOOST_ROUND, params=params)
    del train_split
    gc.collect()

    print("Scoring validation split...", flush=True)
    val["score"] = model.predict(val)
    validation_ndcg = ndcg_at_5(val)
    final_rounds = model.booster.best_iteration or NUM_BOOST_ROUND
    
    return model, val, validation_ndcg, final_rounds, svd_artifacts

def train_final_model(train, final_rounds, svd, svd_artifacts, params):
    if svd:
        print("Refitting SVD on full training data...", flush=True)
        svd_artifacts = fit_svd_features(train)
        train = apply_svd_features(train, *svd_artifacts)

    full_history = train[PRIOR_HISTORY_COLUMNS].copy()

    print("Building final-training features...", flush=True)
    train = add_model_features(full_history, train)
    train["lgbm_label"] = lgbm_labels(train)
    train = train.drop(
        columns=["position", "click_bool", "booking_bool", "gross_booking_usd", "gross_bookings_usd", "relevance"],
        errors="ignore",
    )

    print(f"Training final model for {final_rounds} rounds...", flush=True)
    final_model = train_model(train, None, num_boost_round=final_rounds, params=params)
    del train
    gc.collect()

    print("Loading and featurizing test data...", flush=True)
    test = pd.read_csv(DATASET_PATHS["test"])
    test = add_model_features(full_history, test)
    if svd:
        test = apply_svd_features(test, *svd_artifacts)
    del full_history
    gc.collect()

    print("Scoring test data...", flush=True)
    test["score"] = final_model.predict(test)

    return final_model, test

def main(train_full=False, svd=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running LGBM, BOOST_ROUND={NUM_BOOST_ROUND}, EARLY_STOP={EARLY_STOPPING_ROUNDS}, "
        f"train_full={train_full}, svd={svd}\nloading training data...", flush=True) 
    
    train = add_relevance(clean_train_only(pd.read_csv(DATASET_PATHS["train"])))

    model, val, validation_ndcg, final_rounds, svd_artifacts = train_validation_model(train, svd, params=PARAMS)
    save_predictions(val, OUTPUT_DIR / "validation_predictions.csv", label_cols=True)

    if not train_full:
        save_feature_importance(model, OUTPUT_DIR / "feature_importances.csv")
        save_model_params(model, OUTPUT_DIR / "model_params.json", validation_ndcg)
        print(f"Validation NDCG@5: {validation_ndcg:.6f}")
        print(f"Wrote validation outputs to {OUTPUT_DIR}")
        return

    del val, model
    gc.collect()

    final_model, test = train_final_model(train, final_rounds, svd, svd_artifacts, params=PARAMS)

    print("Writing outputs...", flush=True)
    save_predictions(test, OUTPUT_DIR / "test_predictions.csv")
    make_submission(test, output_path="submission.csv")
    save_feature_importance(final_model, OUTPUT_DIR / "feature_importances.csv")
    save_model_params(final_model, OUTPUT_DIR / "model_params.json", validation_ndcg)

    print(f"Validation NDCG@5: {validation_ndcg:.6f}")
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main(True)
