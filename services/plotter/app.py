"""

File: services/plotter/app.py
Author: David Riser
Derived From: https://plotly.com/python/plotly-express/
Date: 4/19/2021
Purpose: Read metadata from the generator service and
plot in a dash app.

"""
import argparse
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import yaml

from dash.dependencies import Input, Output


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", required=True, type=str,
                    help="Yaml configuration file")
args = parser.parse_args()

with open(args.config, "r") as cfile:
    config = yaml.safe_load(cfile)

print(config)
df = pd.read_csv(config["program"]["monitor_file"])
print(df)

all_dims = list(df.columns)
print(all_dims)

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id="dropdown",
        options=[{"label": x, "value": x}
                 for x in all_dims],
        value=all_dims[:2],
        multi=True
    ),
    dcc.Graph(id="splom"),
])

@app.callback(
    Output("splom", "figure"),
    [Input("dropdown", "value")])
def update_bar_chart(dims):
    fig = px.scatter_matrix(
        df, dimensions=dims)
    return fig

app.run_server(
    host="0.0.0.0",
    debug=True,
    port=8888
)
