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


df = pd.read_csv("combined.csv")
df["TIME"] = pd.to_datetime(df["TIME"])
df.set_index("TIME", inplace=True)

# filter weekdays
# df = df[df['WEEKEND'] == 1]

# take total weekly total for USAGE
station_sum_df = df.resample("W").sum()
station_sum_df.reset_index(drop=False, inplace=True)

# take average weekly for utilization
station_avg_df = df.resample("W").mean()
station_avg_df.reset_index(drop=False, inplace=True)


trace_usage = go.Scatter(x=station_sum_df["TIME"], y=station_sum_df["USAGE"], mode='lines+markers', name='Usage')
trace_utilization = go.Scatter(x=station_avg_df["TIME"], y=station_avg_df["UTILIZATION"], mode='lines+markers', name='Utilization')

fig = make_subplots(rows=2, cols=1, subplot_titles=("USAGE", "UTILIZATION"))

fig.add_trace(trace_usage, row=1, col=1)
fig.add_trace(trace_utilization, row=2, col=1)


# Update layout (optional)
fig.update_layout(title='Bike usage and station utilization')

# Display the figure
fig.show()