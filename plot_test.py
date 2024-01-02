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

# load both pre and post covid
pre_df = pd.read_csv("combined_pre.csv")
post_df = pd.read_csv("combined_post.csv")

# combine both
df = pd.concat([pre_df, post_df], ignore_index=False)
df["TIME"] = pd.to_datetime(df["TIME"])
df.set_index("TIME", inplace=True)

# filter weekdays
# df = df[df['WEEKEND'] == 1]

# take total entries for that hour across all stations
station_avg_df = df.resample("W").sum()
station_avg_df.reset_index(drop=False, inplace=True)


trace_usage_out = go.Scatter(x=station_avg_df["TIME"], y=station_avg_df["USAGE_OUT"], mode='lines+markers', name='Example')
trace_usage_in = go.Scatter(x=station_avg_df["TIME"], y=station_avg_df["USAGE_IN"], mode='lines+markers', name='Example')
trace_utilization = go.Scatter(x=station_avg_df["TIME"], y=station_avg_df["UTILIZATION"], mode='lines+markers', name='Example')

fig = make_subplots(rows=3, cols=1, subplot_titles=('USAGE OUT', 'USAGE IN', 'UTILIZATION'))

fig.add_trace(trace_usage_out, row=1, col=1)
fig.add_trace(trace_usage_in, row=2, col=1)
fig.add_trace(trace_utilization, row=3, col=1)


# Update layout (optional)
fig.update_layout(title='Multiple Plots')

# Display the figure
fig.show()