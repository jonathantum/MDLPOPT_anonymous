"""build_metadata

Utilities to merge clinical CSVs and imaging metadata into a master parquet table.
"""

from pathlib import Path
import pandas as pd


def build_metadata(clinical_dir: str, image_dir: str, out_path: str) -> pd.DataFrame:
    """Scan clinical CSVs and imaging folders and produce a master metadata DataFrame.

    Args:
        clinical_dir: path to directory with clinical CSVs
        image_dir: path to directory with images (xray/mri)
        out_path: where to save the master parquet file

    Returns:
        DataFrame of merged metadata
    """
    clinical_dir = Path(clinical_dir)
    image_dir = Path(image_dir)

    # Basic placeholder implementation: user should fill with real logic
    all_clinical = []
    for p in clinical_dir.glob('*.csv'):
        df = pd.read_csv(p)
        all_clinical.append(df)
    if all_clinical:
        clinical_df = pd.concat(all_clinical, ignore_index=True)
    else:
        clinical_df = pd.DataFrame()

    # Example: add imaging availability flag
    # (real implementation should inspect filenames and link by subject ID)
    clinical_df['has_xray'] = False
    clinical_df['has_mri'] = False

    clinical_df.to_parquet(out_path, index=False)
    return clinical_df
