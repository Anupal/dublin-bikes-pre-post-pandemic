import os
import pandas as pd


file_list = os.listdir("dataset_org")

files = [file for file in file_list if "dublinbikes_" in file]

# print(files)
# print(files_post)

stations = {}
data_df = {}
all_stations = set()

# get station ids
for file in files:
    print(f"reading {file} ...")
    data_df[file] = pd.read_csv(os.path.join("dataset_org", file))
    stations[file] = set(data_df[file]["STATION ID"].unique())
    all_stations.update(stations[file])

print("ALL STATIONS", all_stations)

print("MISSING STATIONS")

missing_stations = set()
for file in files:
    missing_stations.update(all_stations - stations[file])
    print(file, ":", all_stations - stations[file])

print("OVERALL MISSING STATIONS:", missing_stations)