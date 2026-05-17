import numpy as np


def _dcg(relevance, k=5):
    relevance = np.asarray(relevance)[:k]
    gains = (2 ** relevance) - 1
    discounts = np.log2(np.arange(2, len(relevance) + 2))
    return np.sum(gains / discounts)


def ndcg_at_k(df, label_col="relevance", score_col="score", group_col="srch_id", k=5):
    scores = []

    for _, group in df.groupby(group_col):
        predicted = group.sort_values(score_col, ascending=False)[label_col].to_numpy()
        ideal = group.sort_values(label_col, ascending=False)[label_col].to_numpy()

        ideal_dcg = _dcg(ideal, k)
        if ideal_dcg > 0:
            scores.append(_dcg(predicted, k) / ideal_dcg)
        else:
            scores.append(0.0)

    return float(np.mean(scores))
