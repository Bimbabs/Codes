from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.graph_objects as go

import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from pandas import DataFrame
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn import metrics


def monthly_accident(df: DataFrame):

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div(
        [
            html.H1(children='Accident Records Dashboard', style={'text-align': 'center'}),
            html.Div(
                [dbc.FormGroup([
                    dbc.Label(id='slider-value'),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Slider(id="slider", min=1, max=12, step=1, value=6,
                                           marks={
                                               1: {'label': 'Month 1', 'style': {'color': '#77b0b1'}},
                                               6: {'label': 'Month 6'},
                                               12: {'label': 'Month 12', 'style': {'color': '#f50'}}}))]
                    ),
                    html.Br(),
                    'Select GPS point size determinant: ',
                    dcc.Dropdown(
                        id="slct_gps_size", options=[
                            {"label": "Accident Severity", "value": "accident_severity"},
                            {"label": "No of Vehicles", "value": "number_of_vehicles"},
                            {"label": "No of Casualties", "value": "number_of_casualties"},
                            {"label": "First Road class", "value": "first_road_class"},
                            {"label": "Second Road class", "value": "second_road_class"},
                            {"label": "Urban or Rural Area", "value": "urban_or_rural_area"}
                        ],
                        multi=False,
                        value="accident_severity"
                    ),
                    html.Div(
                        [
                            "Number of Clusters: ",
                            dbc.Input(id='id_cluster', type="number", value=3, min=2, max=5,
                                      step=1)],
                    ),
                    html.Br(),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button('Run Clustering', id='cluster-button', color='primary',
                                           style={'width': '40%'},
                                           block=True))]
                    ),
                ],
                ),
                ], style={
                    'position': 'relative',
                    'float': 'left',
                    'width': '40%',
                    'margin-left': '15px'
                }),
            html.Div(
                [
                    dbc.Row(html.Div("Cluster algorithm: KMeans", id='cluster-output',
                                     style={'width': '80%', 'margin-top': '5%', 'margin-left': '10%'})),
                    dbc.Row([dbc.Col(children=[
                        dcc.Graph(id='accident-map', style={'width': '100vh', 'height': '100vh'})]
                    )])
                ],
                style={
                    'position': 'relative',
                    'float': 'left',
                    'width': '40%',
                    'margin-left': '25px',
                    'display': 'inline-block'
                }
            )
        ])

    @app.callback(
        Output('accident-map', 'figure'),
        Input('cluster-button', 'n_clicks'),
        [State('slider', 'value'),
         State('slct_gps_size', 'value'),
         State('id_cluster', 'value')
         ]
    )
    def update_figure(n_clicks, slider_value, gps_size, cluster_no):
        # set default values for map
        if slider_value == ' ':
            slider_value = 6
        if gps_size == ' ':
            gps_size = "accident_severity"
        if cluster_no == ' ':
            cluster_no = 3
        try:

            if gps_size == 'number_of_casualties':
                size_col = 'No of Casualties'
            elif gps_size == 'first_road_class':
                size_col = 'First Road class'
            elif gps_size == 'second_road_class':
                size_col = 'Second Road class'
            elif gps_size == 'urban_or_rural_area':
                size_col = 'Urban or Rural Area'
            elif gps_size == 'number_of_vehicles':
                size_col = 'No of Vehicles'
            else:
                size_col = 'Accident Severity'

            df['text'] = 'Month : ' + df['month_name'] + '<br>Number of Accidents: ' + df['Count'].astype(str) + \
                         '<br>GPS point determinant: ' + size_col + '<br>GPS size: ' + df[gps_size].astype(str)

            df_month_accident = df.loc[df['accident_month'] == slider_value]

            # cluster points
            df_cluster = df_month_accident.iloc[:, 0:2]
            kmeans = simple_k_means(df_cluster, cluster_no)
            cluster = kmeans['clusters'].labels_
            df_month_accident['Clusters'] = cluster.tolist()

            # show figure
            fig = go.Figure(data=go.Scattergeo(
                lon=df_month_accident['longitude'],
                lat=df_month_accident['latitude'],
                text=df_month_accident['text'],
                mode='markers',
                marker=dict(
                    size=df_month_accident[gps_size]/.3, # increased the size to triple the original number to make them bigger
                    color=df_month_accident['Clusters'],
                    line_color='rgb(122,0,0)',
                    line_width=0.5,
                    sizemode='area')
            ))

            fig.update_layout(
                geo=dict(
                    resolution=50,
                    scope='europe',
                    center={'lat': (df_month_accident.latitude.min() + df_month_accident.latitude.max()) / 2,
                            'lon': (df_month_accident.longitude.min() + df_month_accident.longitude.max()) / 2},
                    projection=go.layout.geo.Projection(
                        type='azimuthal equal area',
                        scale=8
                    ),
                    landcolor='rgb(217, 217, 217)',
                    showocean=True,
                )
            )
        except Exception as error:
            print('Caught this error: ' + repr(error))

        return fig

    return app


def processdata() -> Dict:
    """
        I processed the raw data to  select the required months for the dashboard.I added a column 'accident_month' to
        the data. this was created from the column 'accident_year'.
        Then i did a group count to get the number of accidents per month for each gps point per month
        add the 'accident_month' column to the data.
        """
    # copy data from file to dataframe
    df = pd.read_csv(Path('accidents.csv'))

    # create dataframe for map with selected columns
    df_list = [df['longitude'], df['latitude'], df['accident_year'], df['accident_month'], df['month_name'],
               df['accident_severity'], df['number_of_vehicles'], df['number_of_casualties'],
               df['first_road_class'], df['second_road_class'], df['urban_or_rural_area']]

    headers = ['longitude', 'latitude', 'accident_year', 'accident_month', 'month_name', 'accident_severity',
               'number_of_vehicles', 'number_of_casualties', 'first_road_class', 'second_road_class',
               'urban_or_rural_area']
    df_month_accident = pd.concat(df_list, axis=1, keys=headers)
    df_month_accident = df_month_accident.groupby(
        ['longitude', 'latitude', 'accident_year', 'accident_month', 'month_name', 'accident_severity',
         'number_of_vehicles',
         'number_of_casualties', 'first_road_class', 'second_road_class', 'urban_or_rural_area']). \
        size().reset_index(name="Count")

    return df_month_accident


def simple_k_means(x: pd.DataFrame, n_clusters) -> Dict:
    model = KMeans(n_clusters=n_clusters)
    score_metric = 'euclidean'
    clusters = model.fit(x)
    score = metrics.silhouette_score(x, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)


def task():
    """
    For the last assignment, there is only one task, which will use your knowledge from all previous assignments.
    If your methods of a1, a2 and a3 are well implemented, a4 will be fairly simple, so reuse the methods if possible for your own
    benefit! If you prefer, you can reimplement any logic you with in the assignment4 folder.

    For this task, feel free to create new files, modules or methods in this file as you see fit. Our test will be done by calling this
    task() method, and we expect to receive the dash app back (similar to a3) and we will run it. No other method will be called by us, so
    make sure your code is running as expected. We will basically run this code: `task().run_server(debug=True)`

    For this task, you will build a dash app where we can perform a simple form of interactivity on it. We will use the accidents.csv
    dataset. This accidents.csv dataset has information about traffic accidents in the UK, and is available to you now.
    You will show the most accident prone areas in the UK on a map, so the regular value shown in the map should let us know the number of accidents
    that occurred in a specific area of the map (which should include the accident's severity as well as a weight factor). That said, the purpose of
    this map is to be used by the police to identify how they can do things better.

    **This is raw data, so preprocess the data as per requirement. Drop columns that you feel are unnecessary for the analysis or clustering.
    Don't forget to add comments about why you did what you did**
    """
    """

    ##############################################################
    # Your task will be to Implement all the below functionalities
    ##############################################################

    1. (40pts) Implement a map with the GPS points for the accidents per month. Have a slider(#slider1) that can be used to filter accident data for the month I need.
        You are free to choose a map style, but I suggest using a scatter plot map.


    2. (10pts) Implement a dropdown to select few of the numerical columns in the dataset that can be used meaningfully to represent the size of the GPS points. 
        By default the size of the GPS point on map should be based on the value of "accident_severity" column."""
    """
    3. (30pts) You have to Cluster the points now. Be sure to have a text somewhere on the webpage that says what clustering algorithm you are using (e.g. KMeans, dbscan, etc).
        For clustering, you should run a clustering method over the dataset (it should run fairly fast for quick iteration, so make sure to use a simple clustering procedure)
        **COLOR** the GPS points based on the clustering label returned from the algorithm.

    4. (10pts) Have a button(#run_clustering) to run or rerun the clustering algorithm over the filtered dataset (filtered using #slider1 to select months).

    5. (10pts) At least one parameter of the clustering algorithm should be made available for us to tinker with as a button/slider/dropdown. 
        When I change it and click #run_clustering button, your code should rerun the clustering algorithm. 
        example: change the number of clusters in kmeans or eps value in dbscan.

        Please note: distance values should be in meters, for example dbscan uses eps as a parameter. This value should be read in mts from users and converted appropriately to be used in clustering, 
        so input_eps=100 should mean algorithm uses 100mts circle for finding core and non-core points. 

    The total points is 100pts
    """
    df_new = processdata()
    app_monthly = monthly_accident(df_new)
    return app_monthly


if __name__ == "__main__":
    task().run_server(debug=True)
