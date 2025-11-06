import pandas as pd
import os

LOCAL_CACHE = "local_cache.csv"
ZILLOW_DATA = "zillow_data.csv"


def load_local_data() -> pd.DataFrame:
    if os.path.exists(LOCAL_CACHE):
        local_df = pd.read_csv(LOCAL_CACHE)
    else:
        local_df = pd.DataFrame()
    if os.path.exists(ZILLOW_DATA):
        zillow_df = pd.read_csv(ZILLOW_DATA)
        if not local_df.empty:
            zillow_df = pd.concat([zillow_df, local_df]).drop_duplicates(subset=["RegionName"], keep="last")
        return zillow_df
    else:
        raise FileNotFoundError("Zillow data CSV not found!")


def update_local_cache(zip_code: str, data_row: pd.Series):
    df_new = pd.DataFrame([data_row])
    if os.path.exists(LOCAL_CACHE):
        cache = pd.read_csv(LOCAL_CACHE)
        cache = pd.concat([cache, df_new]).drop_duplicates(subset=["RegionName"], keep="last")
    else:
        cache = df_new
    cache.to_csv(LOCAL_CACHE, index=False)
    print(f"✅ Cached new ZIP data for {zip_code}")
