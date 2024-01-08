import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np


DATASET_FILE = "combined.csv"
CUTOFF_DATE = pd.to_datetime("2020-03-01")


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


    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize RandomForestRegressor or any other model
    model = RandomForestRegressor()

    param_grid = {
        'max_depth': [10, 50, 100],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [5, 10, 20],
    }

    # Initialize RandomForestRegressor
    model = RandomForestRegressor()

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_result = grid_search.fit(x, y)

    # Print best hyperparameters and corresponding score
    print("Best parameters found:", grid_result.best_params_)
    print("Best RMSE score:", np.sqrt(-grid_result.best_score_))

    # Extract grid search results
    results = pd.DataFrame(grid_result.cv_results_)
    print(results)

    results.to_csv("rf_cv_results.csv")
