"""Helper utilities for data handling.
"""
from pathlib import Path
import pandas as pd


def read_clinical_csvs(folder: str) -> pd.DataFrame:
    p = Path(folder)
    dfs = []
    for f in p.glob('*.csv'):
        dfs.append(pd.read_csv(f))
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def save_parquet(df, out_path: str):
    df.to_parquet(out_path, index=False)
