from datetime import datetime

import os
import pandas as pd
from holidays import check_public_holiday, check_weekend
import sys

DATASET_ORG_DIR = "dataset_org"
SELECTED_COLUMNS = ["STATION ID", "TIME", "BIKE STANDS", "AVAILABLE BIKE STANDS"]
WEATHER_FILE = "weather.csv"

if __name__ == "__main__":
    if not os.path.exists(DATASET_ORG_DIR):
        print(f"Directory '{DATASET_ORG_DIR}' does not exist!")
        sys.exit(1)

    # loop over all csv files
    file_list = [file for file in os.listdir(DATASET_ORG_DIR) if "dublinbikes" in file]
    combined_csv_list = []
    for file in file_list:
        print(f"processing {os.path.join(DATASET_ORG_DIR, file)}")
        df = pd.read_csv(os.path.join(DATASET_ORG_DIR, file))
        df["TIME"] = pd.to_datetime(df["TIME"])
        
        # workaround for different labels in monthly CSVs
        if "BIKE STANDS" in df:
            quarterly = True
        else:
            df.rename(columns={"BIKE_STANDS": "BIKE STANDS", "AVAILABLE_BIKE_STANDS": "AVAILABLE BIKE STANDS"}, inplace=True)
            quarterly = False
        
        df = df[SELECTED_COLUMNS]

        group_dfs = {station_id: group_df for station_id, group_df in df.groupby("STATION ID")}
        res = []
        for station_id, group_df in group_dfs.items():
            # only for old quaterly samples
            # workaround to collect 30 mins samples to match monthly samples which are every 30 mins
            if quarterly:
                group_df = group_df.iloc[::6].copy()
            
            group_df = group_df.drop(["STATION ID"], axis=1)

            # generate usage columns
            group_df["USAGE"] = group_df["AVAILABLE BIKE STANDS"].diff().abs()
            # group_df['USAGE_IN'] = group_df['USAGE'].apply(lambda x: x if x > 0 else 0)
            # group_df['USAGE_OUT'] = group_df['USAGE'].apply(lambda x: -x if x < 0 else 0)
            # group_df = group_df.drop(["USAGE"], axis=1)

            group_df.set_index("TIME", inplace=True)
            group_df = group_df.resample("H").sum() #.reset_index()

            group_df.reset_index(drop=False, inplace=True)
            res.append(group_df)
    
        group_dfs_hourly = pd.concat(res, ignore_index=True)
        group_dfs_hourly.dropna(inplace=True)
        combined_csv_list.append(group_dfs_hourly)
    
    combined_csv_df = pd.concat(combined_csv_list, ignore_index=False)

    # take total entries for that hour across all stations
    station_avg_df = combined_csv_df.groupby("TIME").sum()
    station_avg_df.reset_index(drop=False, inplace=True)

    # add weekend column
    station_avg_df["WEEKEND"] = station_avg_df["TIME"].apply(check_weekend)

    # add public holiday column
    station_avg_df["PUBLIC HOLIDAY"] = station_avg_df["TIME"].apply(check_public_holiday)
    
    # add utilization column
    station_avg_df["UTILIZATION"] = (station_avg_df["AVAILABLE BIKE STANDS"] / station_avg_df["BIKE STANDS"]).round(2)

    # load weather csv
    weather_df = pd.read_csv(os.path.join(DATASET_ORG_DIR, WEATHER_FILE))
    
    # merge both datasets
    weather_df["date"] = pd.to_datetime(weather_df["date"], format='%d-%b-%Y %H:%M')
    weather_df.set_index("date", inplace=True)
    station_avg_df.set_index("TIME", inplace=True)
    station_avg_df = pd.merge(station_avg_df, weather_df[["rain","temp","wetb","rhum","wdsp","sun","vis","clht","clamt"]], left_index=True, right_index=True, how='left')
    station_avg_df.reset_index(drop=False, inplace=True)
    station_avg_df.dropna(inplace=True)

    # add time columns
    station_avg_df.insert(0, 'WEEKDAY', station_avg_df['TIME'].dt.weekday)
    station_avg_df.insert(0, 'YEAR', station_avg_df['TIME'].dt.year)
    station_avg_df.insert(0, 'MONTH', station_avg_df['TIME'].dt.month)
    station_avg_df.insert(0, 'DAY', station_avg_df['TIME'].dt.day)
    station_avg_df.insert(0, 'HOUR', station_avg_df['TIME'].dt.hour)

    # add lagged features
    # last hour
    station_avg_df["USAGE_LAST_1H"] = station_avg_df["USAGE"].shift(periods=1).fillna(0)

    # last week
    station_avg_df["USAGE_LAST_1W"] = station_avg_df["USAGE"].shift(24*7).fillna(0)

    
    # add mean to missing data points
    station_avg_df.set_index("TIME", inplace=True)
    new_index = pd.date_range(start=station_avg_df.index.min(), end=station_avg_df.index.max(), freq="1H")
    station_avg_df = station_avg_df.reindex(new_index)

    # Fill missing datapoints with mean
    mean_values = station_avg_df.mean()
    station_avg_df = station_avg_df.fillna(mean_values)

    station_avg_df.to_csv(f"combined.csv", index_label="TIME")