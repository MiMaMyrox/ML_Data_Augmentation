import pandas as pd
import smogn
import random
import numpy as np


def smogn_model(df: pd.DataFrame = None):
    if df is None:
        df = pd.read_csv("./data/evaluation/train.csv")

    random.seed(76344)
    np.random.seed(76344)
    augmentation = smogn.smoter(
        data=df,
        y="strength",
        k=9,
        samp_method="extreme",
        pert=0.26,
        drop_na_row=True,
        # phi
        rel_thres=0.18,
        rel_xtrm_type="high"
    )
    
    augmented_data = pd.concat([augmentation, df])
    augmented_data.to_csv('data/evaluation/smogn.csv', index=False)
    return augmented_data


if __name__ == "__main__":
    smogn_model()
