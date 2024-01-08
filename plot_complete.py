import pandas as pd
import matplotlib.pyplot as plt


PANDEMIC_DATE = pd.to_datetime("2020-03-01")
POST_PANDEMIC_DATE = pd.to_datetime("2022-04-03")


df = pd.read_csv("combined.csv")
df["TIME"] = pd.to_datetime(df["TIME"])
df.set_index("TIME", inplace=True)

station_sum_df = df.resample("W").sum()
station_sum_df.reset_index(drop=False, inplace=True)

station_avg_df = df.resample("W").mean()
station_avg_df.reset_index(drop=False, inplace=True)

plt.figure(figsize=(10, 6))

pre_pandemic = station_sum_df[station_sum_df["TIME"] <= PANDEMIC_DATE]
pandemic = station_sum_df[(PANDEMIC_DATE < station_sum_df["TIME"]) & (station_sum_df["TIME"] <= POST_PANDEMIC_DATE)]
post_pandemic = station_sum_df[POST_PANDEMIC_DATE < station_sum_df["TIME"]]

plt.plot(pre_pandemic["TIME"], pre_pandemic["USAGE"], linestyle='-', marker='', color='blue', label='Pre Pandemic', linewidth=2)
plt.plot(pandemic["TIME"], pandemic["USAGE"], linestyle='-', marker='', color='red', label='Pandemic', linewidth=2)
plt.plot(post_pandemic["TIME"], post_pandemic["USAGE"], linestyle='-', marker='', color='green', label='Post Pandemic', linewidth=2)

plt.fill_between(pre_pandemic["TIME"], pre_pandemic["USAGE"], color='lightblue', alpha=0.3)
plt.fill_between(pandemic["TIME"], pandemic["USAGE"], color='pink', alpha=0.3)
plt.fill_between(post_pandemic["TIME"], post_pandemic["USAGE"], color='lightgreen', alpha=0.3)

plt.xlabel('Week')
plt.ylabel('Bike Usage')
plt.title('Weekly Bike Usage (2018-2024)')
plt.legend()

plt.show()