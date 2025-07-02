import pandas as pd

def linear_mix_up(df: pd.DataFrame=None, alpha=0.3, new_datapoints=0.4, label="strength"):

    if df is None:
        df = pd.read_csv("data/train.csv")
    
    number_of_points =  int(len(df) * new_datapoints) if new_datapoints <= 1. else new_datapoints

    allow_dups = number_of_points > len(df) 

    set1 = df.sample(n=number_of_points, random_state=42, replace=allow_dups).to_numpy()
    set2 = df.sample(n=number_of_points, random_state=7, replace=allow_dups).to_numpy()

    new_set = set1 * alpha + set2 * (1-alpha)

    new_df = pd.DataFrame(new_set, columns=df.columns)

    new_df = pd.concat([df, new_df]).round(2)

    new_df.to_csv("data/evaluation/lin_mix_up.csv", index=False)
    return df


if __name__ == "__main__":
    linear_mix_up(new_datapoints=1000)
