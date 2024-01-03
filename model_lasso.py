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

from sklearn.linear_model import Lasso, LinearRegression


DATASET_FILE = "combined.csv"

PANDEMIC_DATE = pd.to_datetime("2020-03-01")
POST_PANDEMIC_DATE = pd.to_datetime("2021-06-30")


if __name__ == "__main__":
    df = pd.read_csv("combined.csv")

    # workaround: drop all lagged features
    df = df.drop(["USAGE_LAST_1H", "USAGE_LAST_1W", "UTILIZATION"], axis=1)


    df["TIME"] = pd.to_datetime(df["TIME"])
    time_column = df["TIME"]

    # limit samples till pandemic end
    df = df[df["TIME"] < POST_PANDEMIC_DATE]
    
    # filter pre pandemic for training
    training_df = df[df["TIME"] < PANDEMIC_DATE]

    # drop time and station id
    training_df = training_df.drop(["TIME", "AVAILABLE BIKE STANDS"], axis=1)

    # input and output
    x = training_df.drop(["USAGE"], axis=1)
    y = training_df["USAGE"]

    # normalize x values
    x = (x - x.min()) / (x.max() - x.min())

    lasso = Lasso(alpha=0.1).fit(x, y)
    x_pred = df.drop(["USAGE", "TIME", "AVAILABLE BIKE STANDS"], axis=1)
    # normalize x values
    x_pred = (x_pred - x_pred.min()) / (x_pred.max() - x_pred.min())

    df['predicted_USAGE'] = lasso.predict(x_pred)

    # change to weekly samples
    df["TIME"] = pd.to_datetime(time_column)
    df.set_index("TIME", inplace=True)
    station_sum_df = df.resample("W").sum()
    station_sum_df.reset_index(drop=False, inplace=True)

    lasso_station_sum_df = station_sum_df[station_sum_df["TIME"] > PANDEMIC_DATE]

    fig = go.Figure()
    trace_usage = go.Scatter(x=station_sum_df["TIME"], y=station_sum_df["USAGE"], mode='lines+markers', name='Usage')

    # Add lasso regression line
    trace_usage_lasso = go.Scatter(x=lasso_station_sum_df["TIME"], y=lasso_station_sum_df["predicted_USAGE"], mode='lines+markers', name='Lasso Usage')

    fig.add_traces([trace_usage, trace_usage_lasso])

    # Update layout and display the figure
    fig.update_layout(title='Original and predicted usage overall')
    fig.show()