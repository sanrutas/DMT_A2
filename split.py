import numpy as np


def group_train_val_split(df, group_col="srch_id", val_size=0.2, random_state=42):
    groups = df[group_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(random_state)
    rng.shuffle(groups)

    n_val = int(round(len(groups) * val_size))
    val_groups = set(groups[:n_val])

    val_mask = df[group_col].isin(val_groups)
    train_df = df.loc[~val_mask].copy()
    val_df = df.loc[val_mask].copy()

    assert set(train_df[group_col]).isdisjoint(set(val_df[group_col]))

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def group_train_val_audit_split(
    df,
    group_col="srch_id",
    val_size=0.2,
    audit_size=0.1,
    random_state=42,
):
    groups = df[group_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(random_state)
    rng.shuffle(groups)

    n_audit = int(round(len(groups) * audit_size))
    n_val = int(round(len(groups) * val_size))
    audit_groups = set(groups[:n_audit])
    val_groups = set(groups[n_audit:n_audit + n_val])

    audit_mask = df[group_col].isin(audit_groups)
    val_mask = df[group_col].isin(val_groups)
    train_df = df.loc[~audit_mask & ~val_mask].copy()
    val_df = df.loc[val_mask].copy()
    audit_df = df.loc[audit_mask].copy()

    assert set(train_df[group_col]).isdisjoint(set(val_df[group_col]))
    assert set(train_df[group_col]).isdisjoint(set(audit_df[group_col]))
    assert set(val_df[group_col]).isdisjoint(set(audit_df[group_col]))

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        audit_df.reset_index(drop=True),
    )
