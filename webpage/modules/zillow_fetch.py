import pandas as pd
import os

ZILLOW_DATA_1 = "zillow_data.csv"
ZILLOW_DATA_2 = "zillow_data2.csv"

def load_zillow_data() -> pd.DataFrame:
    """
    Loads Zillow dataset(s) from zillow_data.csv and optionally zillow_data2.csv.
    Automatically merges them and removes duplicates by RegionName.
    """
    df_list = []

    if os.path.exists(ZILLOW_DATA_1):
        print("→ Loading zillow_data.csv")
        df_list.append(pd.read_csv(ZILLOW_DATA_1, low_memory=False))

    if os.path.exists(ZILLOW_DATA_2):
        print("→ Loading zillow_data2.csv")
        df_list.append(pd.read_csv(ZILLOW_DATA_2, low_memory=False))

    if not df_list:
        raise FileNotFoundError("No Zillow dataset found (expected zillow_data.csv or zillow_data2.csv).")

    # Merge and deduplicate
    combined = pd.concat(df_list).drop_duplicates(subset=["RegionName"], keep="last")
    return combined


def fetch_zip_data(zip_code: str, local_df: pd.DataFrame | None = None) -> pd.Series | None:
    """
    Finds ZIP-level data from the Zillow dataset(s).
    If local_df is not passed, automatically loads and merges both Zillow CSVs.
    """
    if local_df is None:
        local_df = load_zillow_data()

    if "RegionName" not in local_df.columns:
        raise KeyError("Zillow dataset missing RegionName column.")

    # Match rows that contain the ZIP code as a substring (typical Zillow format)
    match = local_df[local_df["RegionName"].astype(str).str.contains(str(zip_code), na=False)]

    if match.empty:
        print(f"No match for ZIP {zip_code} in dataset.")
        return None

    row = match.iloc[0]

    # Filter only columns that look like YYYY-MM or YYYY-MM-DD for price data
    date_cols = [col for col in row.index if col[0].isdigit()]
    monthly_values = row[date_cols].dropna()

    # Extract most recent value
    latest_value = pd.to_numeric(monthly_values.iloc[-1], errors="coerce")
    row["LatestValue"] = latest_value

    return row


# Example manual test (won't run when imported)
if __name__ == "__main__":
    df = load_zillow_data()
    zip_code = input("Enter ZIP code: ").strip()
    result = fetch_zip_data(zip_code, df)
    if result is not None:
        print(result.head(15))
        print(f"Latest median value: ${result['LatestValue']:,.0f}")
