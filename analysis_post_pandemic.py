import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


DATASET_FILE = "combined.csv"

PANDEMIC_DATE = pd.to_datetime("2020-03-01")
POST_PANDEMIC_DATE = pd.to_datetime("2022-04-03")

MAX_DEPTH = 50
MIN_SAMPLES_SPLIT = 10
N_ESTIMATORS = 20

if __name__ == "__main__":
    df = pd.read_csv("combined.csv")


    df["TIME"] = pd.to_datetime(df["TIME"])
    time_column = df["TIME"]

    # filter pre pandemic for training
    training_df_pre = df[df["TIME"] < PANDEMIC_DATE]

    # drop time and station id
    training_df_pre = training_df_pre.drop(["TIME"], axis=1)

    # input and output
    x_pre = training_df_pre.drop(["USAGE"], axis=1)
    y_pre = training_df_pre["USAGE"]

    # normalize x values
    x_pre = (x_pre - x_pre.min()) / (x_pre.max() - x_pre.min())

    rf_pre = RandomForestRegressor(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, n_estimators=N_ESTIMATORS).fit(x_pre, y_pre)
    
    x_pred_pre = df.drop(["USAGE", "TIME"], axis=1)
    # normalize x values
    x_pred_pre = (x_pred_pre - x_pred_pre.min()) / (x_pred_pre.max() - x_pred_pre.min())

    df['predicted_USAGE_pre'] = rf_pre.predict(x_pred_pre)


    # filter pandemic for training
    training_df_pandemic = df[df["TIME"] < POST_PANDEMIC_DATE]

    # drop time and station id
    training_df_pandemic = training_df_pandemic.drop(["TIME"], axis=1)

    # input and output
    x_pandemic = training_df_pandemic.drop(["USAGE"], axis=1)
    y_pandemic = training_df_pandemic["USAGE"]

    # normalize x values
    x_pandemic = (x_pandemic - x_pandemic.min()) / (x_pandemic.max() - x_pandemic.min())

    rf_pandemic = RandomForestRegressor(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, n_estimators=N_ESTIMATORS).fit(x_pandemic, y_pandemic)
    
    x_pred_pandemic = df.drop(["USAGE", "TIME"], axis=1)
    # normalize x values
    x_pred_pandemic = (x_pred_pandemic - x_pred_pandemic.min()) / (x_pred_pandemic.max() - x_pred_pandemic.min())

    df['predicted_USAGE_pandemic'] = rf_pandemic.predict(x_pred_pandemic)

    # change to weekly samples
    df["TIME"] = pd.to_datetime(time_column)
    df.set_index("TIME", inplace=True)
    station_sum_df = df.resample("W").sum()
    station_sum_df.reset_index(drop=False, inplace=True)

    rf_station_sum_df_post_pandemic = station_sum_df[station_sum_df["TIME"] > POST_PANDEMIC_DATE]
    rf_station_sum_df_pandemic = station_sum_df[station_sum_df["TIME"] > PANDEMIC_DATE]

    mse_rf = mean_squared_error(rf_station_sum_df_post_pandemic["USAGE"], rf_station_sum_df_post_pandemic["predicted_USAGE_pandemic"])
    mae_rf = mean_absolute_error(rf_station_sum_df_post_pandemic["USAGE"], rf_station_sum_df_post_pandemic["predicted_USAGE_pandemic"])
    print("MSE MAE RF:", mse_rf, mae_rf)

    plt.figure(figsize=(10, 6))

    plt.plot(station_sum_df["TIME"], station_sum_df["USAGE"], linestyle='-', marker='', color='blue', label='Actual', linewidth=2)
    plt.plot(rf_station_sum_df_pandemic["TIME"], rf_station_sum_df_pandemic["predicted_USAGE_pre"], linestyle='-', marker='', color='red', label='Prediction without Pandemic', linewidth=2)
    plt.plot(rf_station_sum_df_post_pandemic["TIME"], rf_station_sum_df_post_pandemic["predicted_USAGE_pandemic"], linestyle='-', marker='', color='orange', label='Prediction with Pandemic', linewidth=2)

    
    # separators
    plt.axvline(x=PANDEMIC_DATE, color='k', linestyle='--')
    plt.axvline(x=POST_PANDEMIC_DATE, color='k', linestyle='--')
    plt.text(PANDEMIC_DATE, plt.ylim()[1] * 0.95, "Pandemic", color='k')
    plt.text(POST_PANDEMIC_DATE, plt.ylim()[1] * 0.95, "Post Pandemic", color='k')
    
    plt.xlabel('Week')
    plt.ylabel('Bike Usage')
    plt.title('Bike Usage Analysis - Post-Pandemic')
    plt.legend()

    plt.show()