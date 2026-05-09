from config import DATASET_PATHS
from features import add_features, add_relevance
from metrics import ndcg_at_k
from model_lgbm import train_model
from split import group_train_val_split
from T3_data_preparation import clean_train_only
import pandas as pd


def rank_by_score(df, score_col="score"):
    return df.sort_values(["srch_id", score_col], ascending=[True, False])


def make_submission(test_df, output_path="submission.csv"):
    ranked = rank_by_score(test_df)
    submission = ranked[["srch_id", "prop_id"]].rename(
        columns={"srch_id": "SearchId", "prop_id": "PropertyId"}
    )
    assert list(submission.columns) == ["SearchId", "PropertyId"]
    submission.to_csv(output_path, index=False)
    return submission


def main():
    train = pd.read_csv(DATASET_PATHS["train"])
    test = pd.read_csv(DATASET_PATHS["test"])

    train = clean_train_only(train)
    train = add_relevance(train)
    train, val = group_train_val_split(train, group_col="srch_id", val_size=0.2, random_state=42)

    train = add_features(train)
    val = add_features(val)
    test = add_features(test)

    model = train_model(train)

    val["score"] = model.predict(val)
    score = ndcg_at_k(val, label_col="relevance", score_col="score", group_col="srch_id", k=5)
    print(f"Validation NDCG@5: {score:.6f}")

    test["score"] = model.predict(test)
    make_submission(test, output_path="submission.csv")
    print("Wrote submission.csv")


if __name__ == "__main__":
    main()
