def rank_by_score(df, score_col="score"):
    return df.sort_values(["srch_id", score_col], ascending=[True, False])


def make_submission(test_df, output_path="submission.csv"):
    ranked = rank_by_score(test_df)
    submission = ranked[["srch_id", "prop_id"]]
    submission.to_csv(output_path, index=False)
    return submission
