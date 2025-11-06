import pandas as pd
import os

ZILLOW_DATA = "zillow_data.csv"


def fetch_zip_data(zip_code: str, local_df: pd.DataFrame) -> pd.Series | None:
    """Finds ZIP-level data in the Zillow dataset."""
    if "RegionName" not in local_df.columns:
        raise KeyError("Zillow dataset missing RegionName column.")

    # Try matching by ZIP or partial string
    match = local_df[local_df["RegionName"].astype(str).str.contains(str(zip_code), na=False)]
    if match.empty:
        print(f"No match for ZIP {zip_code} in dataset.")
        return None

    row = match.iloc[0]
    latest_month = row.drop(
        labels=[col for col in row.index if not col[0].isdigit()],
        errors="ignore"
    )
    latest_value = pd.to_numeric(latest_month.dropna().iloc[-1], errors="coerce")

    row["LatestValue"] = latest_value
    return row
