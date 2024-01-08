import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error


DATASET_FILE = "combined.csv"
CUTOFF_DATE = pd.to_datetime("2020-03-01")
MAX_DEPTH = 50
MIN_SAMPLES_SPLIT = 10
N_ESTIMATORS = 20
ALPHA = 0.5


if __name__ == "__main__":
    df = pd.read_csv("combined.csv")

    df["TIME"] = pd.to_datetime(df["TIME"])
    time_column = df["TIME"]

    # limit samples till pandemic beginning
    df = df[df["TIME"] < CUTOFF_DATE]

    # drop time and station id
    training_df = df.drop(["TIME"], axis=1)

    # input and output
    x = training_df.drop(["USAGE"], axis=1)
    y = training_df["USAGE"]

    # normalize x values
    x = (x - x.min()) / (x.max() - x.min())

    rf = RandomForestRegressor(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, n_estimators=N_ESTIMATORS).fit(x, y)
    lasso = Lasso(alpha=ALPHA).fit(x, y)

    # baseline model
    df['predicted_USAGE_baseline'] = df['USAGE'].mean()

    df['predicted_USAGE_rf'] = rf.predict(x)
    df['predicted_USAGE_lasso'] = lasso.predict(x)

    
    mse_lasso = mean_squared_error(df["USAGE"], df["predicted_USAGE_lasso"])
    mse_rf = mean_squared_error(df["USAGE"], df["predicted_USAGE_rf"])
    mse_baseline = mean_squared_error(df["USAGE"], df["predicted_USAGE_baseline"])

    mae_lasso = mean_absolute_error(df["USAGE"], df["predicted_USAGE_lasso"])
    mae_rf = mean_absolute_error(df["USAGE"], df["predicted_USAGE_rf"])
    mae_baseline = mean_absolute_error(df["USAGE"], df["predicted_USAGE_baseline"])

    print("MSE MAE Lasso:", mse_lasso, mae_lasso)
    print("MSE MAE RF:", mse_rf, mae_rf)
    print("MSE MAE Baseline:", mse_baseline, mae_baseline)
    
    
    # change to weekly samples
    df["TIME"] = pd.to_datetime(time_column)
    df.set_index("TIME", inplace=True)
    df = df.resample("W").sum()
    df.reset_index(drop=False, inplace=True)

    plt.figure(figsize=(10, 6))

    plt.plot(df["TIME"], df["predicted_USAGE_baseline"], linestyle='--', marker='', color='orange', label='Baseline', linewidth=2)
    plt.plot(df["TIME"], df["USAGE"], linestyle='-', marker='', color='blue', label='Actual', linewidth=2)
    plt.plot(df["TIME"], df["predicted_USAGE_lasso"], linestyle='-', marker='', color='red', label='Lasso', linewidth=2)
    plt.plot(df["TIME"], df["predicted_USAGE_rf"], linestyle='-', marker='', color='green', label='Random Forest', linewidth=2)

    # Setting labels and title
    plt.xlabel('Week')
    plt.ylabel('Bike Usage')
    plt.title('Comparison Bike Usage Prdictions (2018-2020)')
    plt.legend()

    plt.show()