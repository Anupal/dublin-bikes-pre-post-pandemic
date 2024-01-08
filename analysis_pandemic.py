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

    # limit samples till pandemic end
    df = df[df["TIME"] < POST_PANDEMIC_DATE]
    
    # filter pre pandemic for training
    training_df = df[df["TIME"] < PANDEMIC_DATE]

    # drop time and station id
    training_df = training_df.drop(["TIME"], axis=1)

    # input and output
    x = training_df.drop(["USAGE"], axis=1)
    y = training_df["USAGE"]

    # normalize x values
    x = (x - x.min()) / (x.max() - x.min())

    rf = RandomForestRegressor(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, n_estimators=N_ESTIMATORS).fit(x, y)
    
    print("FEATURE IMPORTANCE")
    feature_importances = rf.feature_importances_
    res = []
    # Display the feature_importances for each input feature
    for i, column in enumerate(x.columns):
        res.append({
            "Parameter": column,
            "Importance": feature_importances[i],
        })
    
    print(pd.DataFrame(res))
    
    x_pred = df.drop(["USAGE", "TIME"], axis=1)
    # normalize x values
    x_pred = (x_pred - x_pred.min()) / (x_pred.max() - x_pred.min())

    df['predicted_USAGE'] = rf.predict(x_pred)

    # change to weekly samples
    df["TIME"] = pd.to_datetime(time_column)
    df.set_index("TIME", inplace=True)
    station_sum_df = df.resample("W").sum()
    station_sum_df.reset_index(drop=False, inplace=True)

    rf_station_sum_df = station_sum_df[station_sum_df["TIME"] > PANDEMIC_DATE]


    mse_rf = mean_squared_error(rf_station_sum_df["USAGE"], rf_station_sum_df["predicted_USAGE"])
    mae_rf = mean_absolute_error(rf_station_sum_df["USAGE"], rf_station_sum_df["predicted_USAGE"])
    print("MSE MAE RF:", mse_rf, mae_rf)

    plt.figure(figsize=(10, 6))

    plt.plot(station_sum_df["TIME"], station_sum_df["USAGE"], linestyle='-', marker='', color='blue', label='Actual', linewidth=2)
    plt.plot(rf_station_sum_df["TIME"], rf_station_sum_df["predicted_USAGE"], linestyle='-', marker='', color='red', label='Prediction', linewidth=2)

    # separators
    plt.axvline(x=PANDEMIC_DATE, color='k', linestyle='--')
    plt.text(PANDEMIC_DATE, plt.ylim()[1] * 0.95, "Pandemic", color='k')

    plt.xlabel('Week')
    plt.ylabel('Bike Usage')
    plt.title('Bike Usage Analysis - Pandemic')
    plt.legend()

    plt.show()