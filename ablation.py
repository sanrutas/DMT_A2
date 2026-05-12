import pandas as pd


def ablated_feature_columns(feature_cols, enabled, top_k, importance_path):
    if not enabled:
        return feature_cols

    importance = pd.read_csv(importance_path).sort_values("importance_gain", ascending=False)
    top_features = set(importance["feature"].head(top_k))
    return [col for col in feature_cols if col in top_features]
