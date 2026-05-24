DATASET_PATHS = {
    'train': 'data/training_set_VU_DM.csv',
    'test': 'data/test_set_VU_DM.csv',
    'example': 'data/submission_sample.csv',
}

ABLATION = False
ABLATION_MAX_FEATURES = 50
ABLATION_FEATURES_PATH = "artifacts/lgbm/ablated_features.txt"
ABLATION_FEATURE_IMPORTANCE_PATH = "artifacts/lgbm/validation_feature_importances.csv"
ABLATION_RESULTS_PATH = "artifacts/lgbm/ablation_results.csv"

POSITION_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "min_gain_to_split": 0.05,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
}
POSITION_BOOST_ROUND = 3000
POSITION_EARLY_STOPPING_ROUNDS = 100
POSITION_FOLDS = 3
ESTIMATED_POSITION_COLUMNS = [
    "estimated_position_pct",
    "estimated_position",
    "estimated_position_inverse",
    "estimated_position_rank",
    "estimated_position_rank_pct",
]
POSITION_BLOCKED_COLUMNS = {
    "random_bool",
    "hist_prop_id_non_random_position_mean",
}
