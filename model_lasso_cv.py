"""
Plots weekly data for whole dataset.
"""

import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import sys

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit
import numpy as np


DATASET_FILE = "combined.csv"
CUTOFF_DATE = pd.to_datetime("2020-03-01")
ALPHAS = [0.01, 0.1, 0.15, 0.5, 0.75, 1.0, 2, 5, 10.0]


if __name__ == "__main__":
    df = pd.read_csv("combined.csv")

    # workaround: drop all lagged features
    # df = df.drop(["USAGE_LAST_1H", "USAGE_LAST_1W", "UTILIZATION"], axis=1)


    df["TIME"] = pd.to_datetime(df["TIME"])
    time_column = df["TIME"]

    # limit samples till pandemic end
    df = df[df["TIME"] < CUTOFF_DATE]

    # drop time and station id
    training_df = df.drop(["TIME", "AVAILABLE BIKE STANDS"], axis=1)

    # input and output
    x = training_df.drop(["USAGE"], axis=1)
    y = training_df["USAGE"]

    # normalize x values
    x = (x - x.min()) / (x.max() - x.min())

    tscv = TimeSeriesSplit(n_splits=5)


    lasso_cv = LassoCV(cv=tscv, alphas=ALPHAS)


    lasso_cv.fit(x, y)

    best_alpha = lasso_cv.alpha_

    print("BEST ALPHA:", best_alpha)

    print("MSE PATH:", lasso_cv.mse_path_)

    # Get MSE scores for each alpha
    mse_scores = lasso_cv.mse_path_
    mean_mse_scores = np.mean(mse_scores, axis=1)

    # Display the MSE scores for each alpha
    for idx, alpha in enumerate(lasso_cv.alphas_):
        print(f'Alpha: {alpha}, MSE score: {mean_mse_scores[idx]}')

