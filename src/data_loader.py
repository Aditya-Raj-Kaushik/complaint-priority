import pandas as pd


def load_complaints(path: str) -> pd.DataFrame:
    """
    Load complaints dataset tracked by DVC.
    """
    df = pd.read_csv(path)
    return df
