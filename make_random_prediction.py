import pandas as pd
from config import DATASET_PATHS
import random

from make_submission import make_submission

def main():
    test = pd.read_csv(DATASET_PATHS["test"])
    test = test[["srch_id", "prop_id"]]

    test["score"] = test.groupby("srch_id", group_keys=False
        ).apply(lambda x: pd.Series(random.sample(range(len(x)), len(x)), index=x.index))

    make_submission(test, "random_baseline_submission.csv")


if __name__ == "__main__":
    main()