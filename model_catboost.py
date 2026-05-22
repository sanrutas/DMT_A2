import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker as CatBoostRankerModel
from catboost import Pool

from config import DATASET_PATHS
from error_analysis import save_validation_error_analysis
from features import (
    PRIOR_HISTORY_COLUMNS,
    add_country_imputations,
    add_features,
    add_historical_priors,
    add_oof_historical_priors,
    add_relevance,
    new_features,
)
from make_submission import make_submission
from model_position import (
    MODEL_PATH as POSITION_MODEL_PATH,
    VALIDATION_MODEL_PATH as VALIDATION_POSITION_MODEL_PATH,
    add_estimated_position_predictions,
    load_position_estimator,
)
from split import group_train_val_split
from T3_data_preparation import clean_train_only, model_feature_columns


PARAMS = {
    "loss_function": "PairLogitPairwise",
    "eval_metric": "NDCG:top=5",
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 50,
    "allow_writing_files": False,
    "task_type": "GPU",
    "border_count": 254,
}

OUTPUT_DIR = Path("artifacts/catboost")
MODEL_PATH = OUTPUT_DIR / "model.cbm"
NUM_ITERATIONS = 30000
EARLY_STOPPING_ROUNDS = 2000
GOLDEN_BORDER_COUNT = 1024
GOLDEN_FEATURES = [
    "orig_destination_distance",
    "srch_booking_window",
]
LABEL_GAIN = {
    0: 0,
    1: 1,
    5: 5,
}
CATEGORICAL_FEATURES = [
    "site_id",
    "visitor_location_country_id",
    "prop_country_id",
    "prop_id",
    "srch_destination_id",
    "prop_brand_bool",
    "promotion_flag",
    "srch_saturday_night_bool",
    "random_bool",
    "month",
    "weekday",
]


class CatBoostRanker:
    def __init__(self, model, feature_cols, cat_feature_cols):
        self.model = model
        self.feature_cols = feature_cols
        self.cat_feature_cols = cat_feature_cols

    def predict(self, df):
        return self.model.predict(df[self.feature_cols])


def catboost_labels(df):
    return np.where(df["booking_bool"] == 1, 5, np.where(df["click_bool"] == 1, 1, 0))


def cat_feature_columns(feature_cols):
    return [col for col in CATEGORICAL_FEATURES if col in feature_cols]


def golden_feature_quantization(feature_cols):
    return [
        f"{feature_cols.index(feature)}:border_count={GOLDEN_BORDER_COUNT}"
        for feature in GOLDEN_FEATURES
    ]


def make_pool(df, feature_cols, cat_feature_cols):
    df = df.sort_values("srch_id").reset_index(drop=True)
    labels = (
        df["catboost_label"].to_numpy()
        if "catboost_label" in df.columns
        else catboost_labels(df)
    )
    pool = Pool(
        df[feature_cols],
        label=labels,
        cat_features=cat_feature_cols,
        group_id=df["srch_id"].to_numpy(),
    )
    return df, pool


def train_model(train_df, val_df=None, iterations=NUM_ITERATIONS):
    blocked = {"catboost_label", "score"}
    feature_cols = [col for col in model_feature_columns(train_df) if col not in blocked]
    cat_feature_cols = cat_feature_columns(feature_cols)
    _, train_pool = make_pool(train_df, feature_cols, cat_feature_cols)

    params = PARAMS.copy()
    params["iterations"] = iterations
    params["per_float_feature_quantization"] = golden_feature_quantization(feature_cols)
    model = CatBoostRankerModel(**params)

    if val_df is None:
        model.fit(train_pool)
    else:
        _, val_pool = make_pool(val_df, feature_cols, cat_feature_cols)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )

    return CatBoostRanker(model, feature_cols, cat_feature_cols)


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


def dcg(labels, k=5):
    labels = np.asarray(labels)[:k]
    gains = np.asarray([LABEL_GAIN[label] for label in labels])
    discounts = np.log2(np.arange(2, len(labels) + 2))
    return np.sum(gains / discounts)


def ndcg_at_5(df):
    scores = []

    for _, group in df.groupby("srch_id", sort=False):
        predicted = group.sort_values("score", ascending=False)["catboost_label"].to_numpy()
        ideal = group.sort_values("catboost_label", ascending=False)["catboost_label"].to_numpy()
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
            "catboost_label",
            "score",
        ]
    df[cols].to_csv(path, index=False)


def save_feature_importance(model, path):
    importance = pd.DataFrame({
        "feature": model.feature_cols,
        "importance": model.model.get_feature_importance(type="PredictionValuesChange"),
    })
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(path, index=False)


def save_model_params(model, path, validation_ndcg, use_position_estimator):
    model_params = {
        "params": PARAMS,
        "use_position_estimator": use_position_estimator,
        "position_model_path": str(POSITION_MODEL_PATH),
        "validation_position_model_path": str(VALIDATION_POSITION_MODEL_PATH),
        "golden_features": GOLDEN_FEATURES,
        "golden_border_count": GOLDEN_BORDER_COUNT,
        "per_float_feature_quantization": golden_feature_quantization(model.feature_cols),
        "num_iterations": NUM_ITERATIONS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "tree_count": model.model.tree_count_,
        "best_iteration": model.model.get_best_iteration(),
        "best_score": model.model.get_best_score(),
        "validation_ndcg_at_5": validation_ndcg,
        "features": model.feature_cols,
        "categorical_features": model.cat_feature_cols,
    }
    path.write_text(json.dumps(model_params, indent=2))


def save_model(model):
    model.model.save_model(MODEL_PATH)
    MODEL_PATH.with_suffix(".features.json").write_text(
        json.dumps(model.feature_cols, indent=2)
    )
    MODEL_PATH.with_suffix(".categorical_features.json").write_text(
        json.dumps(model.cat_feature_cols, indent=2)
    )


def add_cached_estimated_position_features(train_df, predict_dfs, position_model):
    result_train = add_estimated_position_predictions(train_df, position_model.predict(train_df))
    result_predict_dfs = [
        add_estimated_position_predictions(df, position_model.predict(df))
        for df in predict_dfs
    ]
    return result_train, result_predict_dfs


def train_validation_model(use_position_estimator=True):
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

    if use_position_estimator:
        print("Loading validation estimated-position model...", flush=True)
        position_model = load_position_estimator(VALIDATION_POSITION_MODEL_PATH)
        train, (val,) = add_cached_estimated_position_features(train, [val], position_model)
        del position_model
        gc.collect()

    train["catboost_label"] = catboost_labels(train)
    val["catboost_label"] = catboost_labels(val)

    print("Training validation model...", flush=True)
    model = train_model(train, val)

    print("Scoring validation split...", flush=True)
    val["score"] = model.predict(val)
    validation_ndcg = ndcg_at_5(val)
    final_iterations = model.model.tree_count_

    save_predictions(val, OUTPUT_DIR / "validation_predictions.csv", label_cols=True)
    save_validation_error_analysis(val, OUTPUT_DIR)
    save_feature_importance(model, OUTPUT_DIR / "validation_feature_importances.csv")
    return model, validation_ndcg, final_iterations


def train_test_model(
    final_iterations=NUM_ITERATIONS,
    validation_ndcg=None,
    use_position_estimator=True,
):
    print("Loading training data for final model...", flush=True)
    full_train = pd.read_csv(DATASET_PATHS["train"])
    full_train = add_relevance(clean_train_only(full_train))
    full_history = full_train[PRIOR_HISTORY_COLUMNS].copy()
    print("Building final-training features...", flush=True)
    full_train = add_oof_model_features(full_history, full_train)
    if use_position_estimator:
        print("Loading cached estimated-position model for final training data...", flush=True)
        final_position_model = load_position_estimator(POSITION_MODEL_PATH)
        full_train = add_estimated_position_predictions(
            full_train,
            final_position_model.predict(full_train),
        )
    full_train["catboost_label"] = catboost_labels(full_train)
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

    print(f"Training final model for {final_iterations} iterations...", flush=True)
    final_model = train_model(full_train, iterations=final_iterations)
    del full_train
    gc.collect()

    print("Loading and featurizing test data...", flush=True)
    test = pd.read_csv(DATASET_PATHS["test"])
    test = add_model_features(full_history, test)
    del full_history
    gc.collect()
    if use_position_estimator:
        test = add_estimated_position_predictions(
            test,
            final_position_model.predict(test),
        )
        del final_position_model
        gc.collect()

    print("Scoring test data...", flush=True)
    test["score"] = final_model.predict(test)

    print("Writing outputs...", flush=True)
    save_predictions(test, OUTPUT_DIR / "test_predictions.csv")
    make_submission(test, output_path="submission.csv")
    save_feature_importance(final_model, OUTPUT_DIR / "feature_importances.csv")
    save_model(final_model)
    save_model_params(
        final_model,
        OUTPUT_DIR / "model_params.json",
        validation_ndcg,
        use_position_estimator,
    )

    if validation_ndcg is not None:
        print(f"Validation NDCG@5: {validation_ndcg:.6f}")
    print(f"Wrote outputs to {OUTPUT_DIR}")


def main(run="both", use_position_estimator=True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation_ndcg = None
    final_iterations = NUM_ITERATIONS

    if run in ["valid", "both"]:
        model, validation_ndcg, final_iterations = train_validation_model(
            use_position_estimator,
        )
        if run == "valid":
            save_feature_importance(model, OUTPUT_DIR / "feature_importances.csv")
            save_model(model)
            save_model_params(
                model,
                OUTPUT_DIR / "model_params.json",
                validation_ndcg,
                use_position_estimator,
            )
        print(f"Validation NDCG@5: {validation_ndcg:.6f}")
        print(f"Wrote validation outputs to {OUTPUT_DIR}")
        del model
        gc.collect()

    if run in ["test", "both"]:
        train_test_model(final_iterations, validation_ndcg, use_position_estimator)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["valid", "test", "both"], default="both")
    parser.add_argument("--no-position-estimator", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.run, not args.no_position_estimator)
