import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def filter_data(state, county, bedroom = None):
    """filters the data for the given state or county"""

    file_paths = {
        1:"data/County_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        2:"data/County_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        3:"data/County_zhvi_bdrmcnt_3_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        4:"data/County_zhvi_bdrmcnt_4_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        5:"data/County_zhvi_bdrmcnt_5_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    }

    dataframes = []

    # Fileter bedroom dataset
    if bedroom is None:
        target_data = dict(file_paths)
    elif bedroom >5:
        target_data = {bedroom: file_paths[bedroom]}
    else:
        target_data = {bedroom: file_paths[bedroom]}

    for label, path in target_data.items():
        df = pd.read_csv(path)

        # Filter by county and state
        filtered = df[(df['StateName'] == state) & (df['RegionName'] == county)]

        if filtered.empty:
            print(f"No data found for {label} in {county}, {state}")
            continue

        # Select date columns
        date_cols = [col for col in filtered.columns if re.match(r'\d{4}-\d{2}', col)]

        # Transpose so dates become rows
        ts = filtered[date_cols].T
        ts.columns = [label]
        ts['Date'] = ts.index
        ts.reset_index(drop=True, inplace=True)
        ts = ts[['Date', label]]

        dataframes.append(ts)

    # Merge all bedroom categories on the Date column
    if len(dataframes) == 0:
        raise ValueError("No Data found.")

    final_df = dataframes[0]
    for df in dataframes[1:]:
        final_df = pd.merge(final_df, df, on='Date', how='outer')

    final_df['Date'] = pd.to_datetime(final_df['Date'])


    return final_df


def predict_linear_regression(dataframes):
    pass


def main() -> None:
    state_initials = "CO"
    county_name = "Mesa County"

    df = filter_data(state_initials, county_name)

    if not df.empty:
        print(df)

        # add prediction
        # add graph


if __name__ == "__main__":
    main()
