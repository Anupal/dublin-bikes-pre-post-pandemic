from datetime import datetime, timedelta

import os
import pandas as pd
import sys

ALL_PUBLIC_HOLIDAYS = {
    "2018-08-06",
    "2018-10-29",
    "2018-12-25",
    "2018-12-26",
    "2019-01-01",
    "2019-03-17",
    "2019-04-22",
    "2019-05-06",
    "2019-06-03",
    "2019-08-05",
    "2019-10-28",
    "2019-12-25",
    "2019-12-26",
    "2020-01-01",
    "2020-03-17",
    "2020-04-13",
    "2020-05-04",
    "2020-06-01",
    "2020-08-03",
    "2020-10-26",
    "2020-12-25",
    "2020-12-26",
    "2021-01-01",
    "2020-03-17",
    "2020-04-05",
    "2020-05-03",
    "2020-06-07",
    "2020-08-02",
    "2020-10-25",
    "2020-12-25",
    "2020-12-26",
    "2021-01-01",
    "2021-03-17",
    "2021-03-18",
    "2021-04-18",
    "2021-05-02",
    "2021-06-06",
    "2021-08-01",
    "2021-10-31",
    "2021-12-25",
    "2021-12-26",
    "2022-01-01",
    "2022-03-17",
    "2022-03-18",
    "2022-04-18",
    "2022-05-02",
    "2022-06-06",
    "2022-08-01",
    "2022-10-31",
    "2022-12-25",
    "2022-12-26",
    "2023-01-01",
    "2023-02-06",
    "2023-03-17",
    "2023-04-10",
    "2023-05-01",
    "2023-06-05",
    "2023-08-07",
    "2023-10-30",
    "2023-12-25",
    "2023-12-26",
}

def weekends_by_year(year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    weekends = set()

    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() in [5, 6]:  # 5 is Saturday, 6 is Sunday
            weekends.add(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    return weekends

ALL_WEEKENDS = set()
for year in (2018, 2019, 2020, 2021, 2022, 2023):
    ALL_WEEKENDS.update(weekends_by_year(year))

def check_weekend(date_obj):
    return int(date_obj.strftime('%Y-%m-%d') in ALL_WEEKENDS)

def check_public_holiday(date_obj):
    return int(date_obj.strftime('%Y-%m-%d') in ALL_PUBLIC_HOLIDAYS)

DATASET_ORG_DIR = "dataset_org"
SELECTED_COLUMNS = ["STATION ID", "TIME", "BIKE STANDS", "AVAILABLE BIKE STANDS"]
WEATHER_FILE = "weather.csv"

if __name__ == "__main__":
    if not os.path.exists(DATASET_ORG_DIR):
        print(f"Directory '{DATASET_ORG_DIR}' does not exist!")
        sys.exit(1)

    # loop over all csv files
    file_list = [file for file in os.listdir(DATASET_ORG_DIR) if "dublinbikes" in file] + [file for file in os.listdir(DATASET_ORG_DIR) if "dublinbike-" in file]
    combined_csv_list = []
    print("PROCESSING FILES:")
    for file in file_list:
        print(f"{os.path.join(DATASET_ORG_DIR, file)}")
        df = pd.read_csv(os.path.join(DATASET_ORG_DIR, file))
        df["TIME"] = pd.to_datetime(df["TIME"])
        
        # workaround for different labels in monthly CSVs
        if "BIKE STANDS" in df:
            quarterly = True
        else:
            print("    + label corrections")
            df.rename(columns={"BIKE_STANDS": "BIKE STANDS", "AVAILABLE_BIKE_STANDS": "AVAILABLE BIKE STANDS"}, inplace=True)
            quarterly = False
        
        df = df[SELECTED_COLUMNS]

        group_dfs = {station_id: group_df for station_id, group_df in df.groupby("STATION ID")}
        res = []
        print("    + combine csv for all station ids")
        for station_id, group_df in group_dfs.items():
            # only for old quaterly samples
            # workaround to collect 30 mins samples to match monthly samples which are every 30 mins
            if quarterly:
                group_df = group_df.iloc[::6].copy()
            
            group_df = group_df.drop(["STATION ID"], axis=1)

            # generate usage columns
            group_df["USAGE"] = group_df["AVAILABLE BIKE STANDS"].diff().abs()

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
    
    # add mean to missing data points
    station_avg_df.set_index("TIME", inplace=True)
    new_index = pd.date_range(start=station_avg_df.index.min(), end=station_avg_df.index.max(), freq="1H")
    station_avg_df = station_avg_df.reindex(new_index)

    # Fill missing datapoints with mean
    mean_values = station_avg_df.mean()
    station_avg_df = station_avg_df.fillna(mean_values)

    # drop BIKE STANDS and UTILIZATION columns
    station_avg_df = station_avg_df.drop(["BIKE STANDS", "AVAILABLE BIKE STANDS", "UTILIZATION"], axis=1)

    station_avg_df.to_csv(f"combined.csv", index_label="TIME")