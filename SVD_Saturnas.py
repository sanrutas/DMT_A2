import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from config import DATASET_PATHS
from features import add_relevance
from make_submission import make_submission
from split import group_train_val_split
from T3_data_preparation import clean_train_only

 # globals
OUTPUT_DIR = Path("SVD_outputs")
K = 10
SVD_col_1 = "srch_destination_id"
SVD_col_2 = "prop_id"

def dcg(gains, k=5):
    gains = np.asarray(gains)[:k]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return np.sum(gains / discounts)


def ndcg_at_5(df):
    scores = []
    for _, group in df.groupby("srch_id", sort=False):
        predicted = group.sort_values("score", ascending=False)["relevance"].to_numpy()
        ideal = group.sort_values("relevance", ascending=False)["relevance"].to_numpy()
        ideal_dcg = dcg(ideal, k=5)
        scores.append(dcg(predicted, k=5) / ideal_dcg if ideal_dcg > 0 else 0.0)
    return float(np.mean(scores))


def get_full_dest_prop_matrix(train_df):

    agg = train_df.groupby([SVD_col_1, SVD_col_2])["relevance"].sum().reset_index()

    dest_ids = sorted(agg[SVD_col_1].unique())
    prop_ids = sorted(agg[SVD_col_2].unique())
    dest_to_idx = {d: i for i, d in enumerate(dest_ids)}
    prop_to_idx = {p: i for i, p in enumerate(prop_ids)}

    rows = agg[SVD_col_1].map(dest_to_idx).values
    cols = agg[SVD_col_2].map(prop_to_idx).values
    vals = agg["relevance"].values.astype(np.float32)

    full_dest_prop_matrix = sp.csr_matrix((vals, (rows, cols)), shape=(len(dest_ids), len(prop_ids)))

    return full_dest_prop_matrix, dest_to_idx, prop_to_idx


def fit_svd(full_dest_prop_matrix, k):

    U, sigma, Vt = svds(full_dest_prop_matrix.astype(np.float32), k=k)
    order = np.argsort(-sigma)
    U = U[:, order]
    sigma = sigma[order]
    Vt = Vt[order, :]

    dest_emb = U * sigma # shape (n_dests, k)
    prop_emb = Vt.T # shape (n_props, k)
    # dest_emb * prop_emb approximates full matrix 
    print(f'dest{dest_emb.shape}, prop{prop_emb.shape}')

    return dest_emb, prop_emb


def score_rows(df, dest_emb, prop_emb, dest_to_idx, prop_to_idx):
    """Score each row by dot product of its dest and prop embeddings.
    Unseen IDs get score 0."""
    k = dest_emb.shape[1]
    n = len(df)

    dest_idx = df[SVD_col_1].map(dest_to_idx).values
    prop_idx = df[SVD_col_2].map(prop_to_idx).values

    # Build matrices, using zeros for unseen IDs
    dest_vecs = np.zeros((n, k), dtype=np.float32)
    prop_vecs = np.zeros((n, k), dtype=np.float32)
    dest_seen = ~pd.isna(dest_idx)
    prop_seen = ~pd.isna(prop_idx)
    dest_vecs[dest_seen] = dest_emb[dest_idx[dest_seen].astype(int)]
    prop_vecs[prop_seen] = prop_emb[prop_idx[prop_seen].astype(int)]

    scores = (dest_vecs * prop_vecs).sum(axis=1)
    # Rows where either ID is unseen → score is 0 (no info)
    return scores, dest_vecs, prop_vecs

def fit_svd_features(train_df):
    svd_matrix, dest_to_idx, prop_to_idx = get_full_dest_prop_matrix(train_df)
    dest_emb, prop_emb = fit_svd(svd_matrix, K)
    return dest_emb, prop_emb, dest_to_idx, prop_to_idx


def apply_svd_features(df, dest_emb, prop_emb, dest_to_idx, prop_to_idx):
    scores, dest_vecs, prop_vecs = score_rows(df, dest_emb, prop_emb, dest_to_idx, prop_to_idx)

    new_cols = {}
    for i in range(K):
        new_cols[f"SVD_{SVD_col_2}{i}"] = prop_vecs[:, i]   # prop_vecs → prop name
        new_cols[f"SVD_{SVD_col_1}{i}"] = dest_vecs[:, i]   # dest_vecs → dest name
    new_cols["SVD_score"] = scores

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(new_cols)], axis=1)


def main(train_full=False):
    print(f"Running SVD with k={K}, train_full={train_full}, loading training data...", flush=True)
    train = pd.read_csv(DATASET_PATHS["train"])
    train = add_relevance(clean_train_only(train)).dropna(subset=[SVD_col_1, SVD_col_2])

    if train_full:
        print("Building full interaction matrix...", flush=True)
        full_dest_prop_matrix, dest_to_idx, prop_to_idx = get_full_dest_prop_matrix(train)
        print(
            f"shape: {full_dest_prop_matrix.shape}, "
            f"nnz: {full_dest_prop_matrix.nnz}, "
            f"pct: {full_dest_prop_matrix.nnz / (full_dest_prop_matrix.shape[0] * full_dest_prop_matrix.shape[1]):.5%}",
            flush=True,
        )
        del train
        gc.collect()

        dest_emb, prop_emb = fit_svd(full_dest_prop_matrix, K)

        print("Loading and scoring test data...", flush=True)
        test = pd.read_csv(DATASET_PATHS["test"])
        test["score"], dest_vecs, prop_vecs = score_rows(test, dest_emb, prop_emb, dest_to_idx, prop_to_idx)

        test_dest_seen = test[SVD_col_1].isin(dest_to_idx).mean()
        test_prop_seen = test[SVD_col_2].isin(prop_to_idx).mean()
        print(f"train dataset contains {test_dest_seen:.1%} of srch_destination_id in test df", flush=True)
        print(f"train dataset contains {test_prop_seen:.1%} of prop_id in test df", flush=True)

        test[["srch_id", SVD_col_2, "score"]].to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
        make_submission(test, output_path=f"{OUTPUT_DIR}/submission_svd.csv")
        print(f"Wrote outputs to {OUTPUT_DIR}", flush=True)
    else:
        train, val = group_train_val_split(train, group_col="srch_id", val_size=0.2, random_state=42)
        gc.collect()        

        print("Building full interaction matrix...", flush=True)
        full_dest_prop_matrix, dest_to_idx, prop_to_idx = get_full_dest_prop_matrix(train)
        print(
            f"shape: {full_dest_prop_matrix.shape}, "
            f"nnz: {full_dest_prop_matrix.nnz}, "
            f"pct: {full_dest_prop_matrix.nnz / (full_dest_prop_matrix.shape[0] * full_dest_prop_matrix.shape[1]):.5%}",
            flush=True,
        )
        del train
        gc.collect()

        dest_emb, prop_emb = fit_svd(full_dest_prop_matrix, K)

        print("Scoring validation set...", flush=True)
        val["score"], dest_vecs, prop_vecs = score_rows(val, dest_emb, prop_emb, dest_to_idx, prop_to_idx)

        val_dest_seen = val[SVD_col_1].isin(dest_to_idx).mean()
        val_prop_seen = val[SVD_col_2].isin(prop_to_idx).mean()
        print(f"train dataset contains {val_dest_seen:.1%} of srch_destination_id in val df", flush=True)
        print(f"train dataset contains {val_prop_seen:.1%} of prop_id in val df", flush=True)

        val_ndcg = ndcg_at_5(val)
        print(f"Val NDCG@5: {val_ndcg:.5f}", flush=True)


if __name__ == "__main__":
    main(train_full=False)