import pandas as pd
import os

LOCAL_CACHE = "local_cache.csv"
ZILLOW_DATA_1 = "zillow_data.csv"
ZILLOW_DATA_2 = "zillow_data2.csv"


def load_local_data() -> pd.DataFrame:
    """
    Loads Zillow and local cache data.
    Merges zillow_data.csv + zillow_data2.csv if both exist.
    """
    df_list = []

    if os.path.exists(ZILLOW_DATA_1):
        print("→ Loading zillow_data.csv")
        df_list.append(pd.read_csv(ZILLOW_DATA_1))

    if os.path.exists(ZILLOW_DATA_2):
        print("→ Loading zillow_data2.csv")
        df_list.append(pd.read_csv(ZILLOW_DATA_2))

    if not df_list:
        raise FileNotFoundError("No Zillow dataset found!")

    combined_zillow = pd.concat(df_list).drop_duplicates(subset=["RegionName"], keep="last")

    if os.path.exists(LOCAL_CACHE):
        print("→ Merging with local cache...")
        local_df = pd.read_csv(LOCAL_CACHE)
        combined_zillow = pd.concat([combined_zillow, local_df]).drop_duplicates(subset=["RegionName"], keep="last")

    return combined_zillow


def update_local_cache(zip_code: str, data_row: pd.Series):
    """
    Updates or creates the local cache CSV.
    """
    df_new = pd.DataFrame([data_row])
    if os.path.exists(LOCAL_CACHE):
        cache = pd.read_csv(LOCAL_CACHE)
        cache = pd.concat([cache, df_new]).drop_duplicates(subset=["RegionName"], keep="last")
    else:
        cache = df_new
    cache.to_csv(LOCAL_CACHE, index=False)
    print(f"✅ Cached new ZIP data for {zip_code}")
