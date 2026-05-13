import argparse
from pathlib import Path

import pandas as pd


PREDICTION_PATHS = {
    "lgbm": Path("artifacts/lgbm/test_predictions.csv"),
    "catboost": Path("artifacts/catboost/test_predictions.csv"),
    "blend": Path("artifacts/blend/test_predictions.csv"),
}


def rank_by_score(df, score_col="score"):
    return df.sort_values(["srch_id", score_col], ascending=[True, False])


def make_submission(test_df, output_path="submission.csv"):
    ranked = rank_by_score(test_df)
    submission = ranked[["srch_id", "prop_id"]]
    submission.to_csv(output_path, index=False)
    return submission


def prediction_path(model, predictions=None):
    if predictions is not None:
        return Path(predictions)
    return PREDICTION_PATHS[model]


def submission_output_path(model, output=None):
    if output is not None:
        return output
    return f"submission_{model}.csv"


def main(model, predictions=None, output=None):
    path = prediction_path(model, predictions)
    output_path = submission_output_path(model, output)
    df = pd.read_csv(path)
    make_submission(df, output_path)
    print(f"Wrote {output_path} from {path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=PREDICTION_PATHS.keys(), required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.model, args.predictions, args.output)
