import dash
import plotly.express as px
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from assignment1.e_experimentation import *
from assignment1.a_load_file import *


##############################################
# Now let's use dash, a library built on top of flask (a backend framework for python) and plotly
# Check the documentation at https://dash.plotly.com/
# For a complete example, check https://dash-bootstrap-components.opensource.faculty.ai/examples/iris/
# Example(s). Read the comments in the following method(s)
##############################################
def dash_simple_example():
    """
    Here is a simple example from https://dash.plotly.com/layout
    The focus is to create a fig from plotly and add it to dash, but differently from using just plotly, now we can use html elements,
    such as H1 for headers, Div for dividers, and all interations (buttons, sliders, etc).
    Check dash documentation for html and core components."""

    app = dash.Dash(__name__)

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    # You create a fig just as you did in a_
    fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        dcc.Graph(
            id='example-graph',
            figure=fig  # and include the fig here as a dcc.Graph
        )
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_with_bootstrap_example():
    """Here is a simple example from https://dash.plotly.com/layout See examples of components from the bootstrap library
    at https://dash-bootstrap-components.opensource.faculty.ai/docs/components/alert/"""

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig1 = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    fig2 = px.line(df, x="Fruit", y="Amount", color="City")

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.Div(children='Dash: A web application framework for Python.'), md=4),
            dbc.Col(dbc.Button('Example Button', color='primary', style={'margin-bottom': '1em'}, block=True), md=8)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph1', figure=fig1)),
            dbc.Col(dcc.Graph(id='example-graph2', figure=fig2))
        ])
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_callback_example():
    """Here is a more complex example that uses callbacks. With this example, I believe you will suddenly perceive why dash (and webapps)
    are so much better for visual analysis."""

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        html.Hr(),
        dbc.FormGroup([
            dbc.Label("Choose Dataset"),
            dcc.Dropdown(id="dropdown", value=1,
                         options=[{"label": "First Data", "value": 1}, {"label": "Second Data", "value": 2}]),
        ]),
        dbc.FormGroup([
            dbc.Label(id='slider-value'),
            dcc.Slider(id="slider", min=1, max=10, step=0.5, value=1),
        ]),
        dbc.Button('Run Callback', id='example-button', color='primary', style={'margin-bottom': '1em'}, block=True),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph')),
            # Not including fig here because it will be generated with the callback
        ])
    ])

    @app.callback(  # See documentation or tutorial to see how to use this
        Output('example-graph', 'figure'),
        # Outputs is what you wish to update with the callback, which in this case is the figure
        [Input('example-button', 'n_clicks')],
        # Use inputs to define when this callback is called, and read from the values in the inputs as parameters in
        # the method
        [State('dropdown', 'value'),
         # Use states to read values from the interface, but values only in states will not trigger the callback when
         # changed
         State('slider',
               'value')])  # For example, here if you change the slider, this method will not be called, it will only
    # be called when you click the button
    def update_figure(n_clicks, dropdown_value, slider_value):
        df2 = df[:]
        df2.Amount = df2.Amount * slider_value
        if dropdown_value == 1:
            return px.bar(df2, x="Fruit", y="Amount", color="City", barmode="group")
        else:
            return px.line(df2, x="City", y="Amount", color="Fruit")

    @app.callback(Output('slider-value', 'children'), [Input('slider', 'value')])
    def update_slider_value(slider):
        return f'Multiplier: {slider}'

    """You can also use app.callback to get selection from any of the plotly graphs, including tables and maps, and update anything you wish.
    See some examples at https://dash-gallery.plotly.host/Portal/"""

    return app


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################


def dash_task():
    """
    There is only only one task to do, a web app with:
    1. Some nice title
    2. One visualization placeholder for dataset visualization
        a. A dropdown to allow me to select which dataset I want to see (iris, video_game and life_expectancy)
        b. Two other dropdowns for me to choose what column to put in x and what column to put in y of the visualization
        c. Another dropdown for me to choose what type of graph I want (see examples in file a_) (at least 3 choices of graphs)
        d. Feel free to change the structure of the dataset if you prefer (e.g. change life_expectancy so that
            there is one column of "year", one for "country" and one for "value")
    4. A https://dash-bootstrap-components.opensource.faculty.ai/docs/components/card/ with the number of rows being showed on the above graph
    5. Another visualization with:
        a. It will containing the figure created in the tasks in a_, b_ or c_ related to plotly's figures
        b. Add a dropdown for me to choose among 3 (or more if you wish) different graphs from a_, b_ or c_ (choose the ones you like)
        c. In this visualization, if I select data in the visualization, update some text in the page (can be a new bootstrap card with text inside)
            with the number of values selected. (see https://dash.plotly.com/interactive-graphing for examples)
    """
    df_iris = read_dataset(Path('..', '..', 'iris.csv'))
    df_life = process_life_expectancy_dataset()
    df_life = df_life[['country', 'year', 'value', 'Latitude', 'Longitude']]
    df_game = read_dataset(Path('..', '..', 'ratings_Video_Games.csv'))

    df_dict = {'Iris data': df_iris, 'Life expectancy data': df_life, 'Video game data': df_game}

    df_key = list(df_dict.keys())
    cols = df_dict[df_key[0]]

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = html.Div(
        [
            html.H2("A simple dash application", style={'text-align': 'center'}),
            html.Div([
                html.Div([
                    html.P('Select data file:', style={'color': 'black'}),
                    html.Div(children=[
                        dcc.Dropdown(
                            id='dataset-dropdown',
                            options=[{'label': file, 'value': file} for file in df_key],
                            value=list(df_dict.keys())[0],
                            searchable=False
                        ),
                    ], style=dict(
                        width='90%',
                        verticalAlign="middle"
                    )),
                    html.Br(),
                    html.P('Select column for x axis:', style={'color': 'black'}),
                    html.Div([
                        dcc.Dropdown(
                            id='x-dropdown',
                            value='sepal_length'
                        ),
                    ], style=dict(
                        width='90%',
                        verticalAlign="middle"
                    )
                    ),
                    html.Br(),
                    html.P('Select column for y axis:', style={'color': 'black'}),
                    html.Div([
                        dcc.Dropdown(
                            id='y-dropdown',
                            value='petal_length'
                        ),
                    ], style=dict(
                        width='90%',
                        verticalAlign="middle"
                    )
                    ),
                    html.Br(),
                    html.P('Select visualization', style={'color': 'black'}),
                    html.Div([
                        dcc.Dropdown(
                            id="slct_visual",
                            options=[
                                {"label": "Bar Chart", "value": "Bar Chart"},
                                {"label": "Scatter plot", "value": "Scatter plot"},
                                {"label": "Histogram", "value": "Histogram"}],
                            multi=False,
                            value='Bar Chart'
                        ),

                    ], style=dict(
                        width='90%',
                        verticalAlign="middle"
                    )
                    ),
                    html.Br(),
                    dbc.Button('View Graph', id='graph-button', outline=True, color='primary',
                               style=dict(width='70%', verticalAlign="middle"), block=True),

                ]),

            ],
                style={
                    'height': '65%',
                    'margin-left': '5vw',
                    'padding': '10px',
                    'width': '20%',
                    'vertical-align': 'center',
                    'text-align': 'left',
                    'display': 'inline-block',
                    'position': 'relative'
                }

            ),
            html.Div(children=[
                html.Div(id='output_container', children=[]),
                html.Br(),
                html.Div(
                    dbc.Card(
                        [
                            dbc.CardHeader("Number of data rows"),
                            dbc.CardBody(
                                html.P("Number of data rows", id='card_id', className="card-text")
                            ),
                        ],
                        style={'width': '30%', 'text-align': 'center', 'margin-left': '15vw'}

                    )
                ),
                html.Div(
                    dcc.Graph(id='output_graph', style={'height': '4%', 'width': '95%','margin-left': '1vw'})
                )],
                style={
                    'height': '74%',
                    'width': '70%',
                    'position': 'absolute',
                    'display': 'inline-block'

                }
            )
        ],
    )

    # set a dataframe and populate column dropdown based on selected datafile
    @app.callback(
        [Output('output_container', 'children'),
         Output('x-dropdown', 'options')],
        [Input('dataset-dropdown', 'value')]
    )
    def get_x_dropdown(file):
        if file == 'Iris data':
            container = "You chose the Iris dataset "
        elif file == 'Life expectancy data':
            container = "You chose the Life expectancy dataset "
        else:
            container = "You chose the Video game dataset "

        return container, [{'label': i, 'value': i} for i in df_dict[file]]

    @app.callback(
        Output('y-dropdown', 'options'),
        [Input('dataset-dropdown', 'value')]
    )
    def get_y_dropdown(file):
        return [{'label': i, 'value': i} for i in df_dict[file]]

    # populate the card with number of rows
    @app.callback(
        Output('card_id', 'children'),
        [Input('dataset-dropdown', 'value')])
    def update_card(input_df):
        return len(df_dict[input_df])

    # plot the graph
    @app.callback(
        Output('output_graph', 'figure'),
        [Input('graph-button', 'n_clicks')],
        [State('dataset-dropdown', 'value'),
         State('x-dropdown', 'value'),
         State('y-dropdown', 'value'),
         State('slct_visual', 'value')])
    def plot_graph(n_clicks, df_data, x_axis, y_axis, visual):
        # get selected dataset
        df = df_dict[df_data]
        # get selected x axis
        x = df[x_axis]
        # get selected y axis
        y = df[y_axis]
        # get the selected visual

        # {"label": "Heat Map",
        if visual == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis)
        elif visual == "Scatter plot":
            # {'Iris data': df_iris, 'Life expectancy data': df_life, 'Video game data': df_game}
            if df_data == 'Iris data':
                fig = px.scatter(df, x=x_axis, y=y_axis, color="species")
            elif df_data == 'Life expectancy data':
                fig = px.scatter(df, x=x_axis, y=y_axis, color="country")
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, color="user")
        else:
            fig = px.histogram(df, x=x_axis)

        return fig
    return app


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    app_ce = dash_callback_example()
    app_b = dash_with_bootstrap_example()
    app_c = dash_callback_example()
    app_t = dash_task()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # app_ce.run_server(debug=True)
    # app_b.run_server(debug=True)
    app_c.run_server(debug=True)
    # app_t.run_server(debug=True)
