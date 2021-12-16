from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

##############################################
# Example(s). Read the comments in the following method(s)
##############################################

###############
# There is a plethora of examples of how to use matplotlib on the internet.
# There are many ways of showing data through matplotlib, including sub-plots and
# many different graph types. For that see the documentation at https://matplotlib.org/.
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots


def matplotlib_line_sin_example():
    """
    As an example, the official website shows a simple line plot at
    https://matplotlib.org/3.3.2/gallery/lines_bars_and_markers/simple_plot.html
    Run this method on your own to try it out.
    """
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='About as simple as it gets, folks')
    ax.grid()
    plt.show()


def matplotlib_line_example(array_x: np.array, array_y: np.array) -> Tuple:
    """
    Of course, I can send the array to be displayed as a parameter, and return the figure generated from it,
    so that other methods can decide what to do with the figure (show, save, etc).
    """
    fig, ax = plt.subplots()
    ax.plot(array_x, array_y)
    return fig, ax


def matplotlib_line_example2():
    """
    Now we can call the other method and add more parameters to teh axis if we want before showing it
    """
    fig, ax = matplotlib_line_example(np.array([1, 2, 3, 4, 5]), np.array([5, 3, 5, 3, 5]))
    ax.set(xlabel='some linear list', ylabel='another list changing values', title='Another example')
    ax.grid()
    plt.show()


###############
# Another library we can use is plotly, which is very different from matplotlib because it uses JavaScript
# behind the scenes. This means that it works pretty well to configure front-end graphs for webapps and also
# it works pretty nicely with Jupyter Notebooks. Unfortunately, Jupyter Notebooks has a significant limitation
# of using dash (which we will use in d_dash_backend), so let's keep the assignment in .py files, but you
# are free to try both visualization libraries in Jupyter if you wish to see how it is like.
#
# In case you are using PyCharm, I suggest you to go to your run configuration (as seen in a tutorial which may
# or may not have been given when you read this) and uncheck the 'Run in a Python Console'. You may also wish
# to uncheck the 'Show plots in tool window' if you prefer for python to open a plotly app or a matplotlib default
# interface.
#

def plotly_sin_line_example_np():
    """
    In this example, we will recreate the first example, but with plotly.
    Notice that plotly has a generic constructor (go.*) for data manually made or plotly express to handle
    pandas data. Let's first see the generic one first. After it runs, hover your mouse on top of the graph
    and also check the buttons on the top right. All this can be customized and used for analysing the visualization.
    """
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig = go.Figure(data=go.Scatter(x=t, y=s))
    fig.show()


def plotly_sin_line_example_pandas():
    """
    Now, let's see the express version. Notice that although it seems that both are very similar,
    the express version allows for much more functionality when using pandas data-frames, but the generic
    is more flexible for more customised visualizations.
    """
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)
    df = pd.DataFrame(dict(t=t, s=s))

    fig = px.line(df, x="t", y="s", title='Sine wave')
    fig.show()


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################
def matplotlib_bar_chart(x: np.array) -> Tuple:
    """
    Create a matplotlib bar chart with the inputs. DO NOT PLOT IT!!
    Return the fig and ax as was shown in matplotlib_line_example.
    """
    axis = np.arange(len(x))
    fig, ax = plt.subplots()
    ax.bar(axis, x, width=0.4)
    return fig, ax


def matplotlib_pie_chart(x: np.array) -> Tuple:
    """
    Create a matplotlib pie chart with the inputs. DO NOT PLOT IT!!
    Return the fig and ax as was shown in matplotlib_line_example.
    """
    fig = plt.figure()
    ax = plt.pie(x)
    return fig, ax


def matplotlib_histogram(x: np.array, n_bins: int) -> Tuple:
    """
    Create a matplotlib histogram chart with the inputs. DO NOT PLOT IT!!
    Return the fig and ax as was shown in matplotlib_line_example.
    Note that a histogram is the distribution of the data as a bar chart split in bins
    """
    fig = plt.figure()
    ax = plt.hist(x, bins=n_bins)
    return fig, ax


def matplotlib_polar_chart(x: np.array, y: np.array) -> Tuple:
    """
    Create a matplotlib polar chart with the inputs. DO NOT PLOT IT!!
    The y input is the same as the line chart, so you need to convert it to an angle.
    Return the fig and ax as was shown in matplotlib_line_example.
    """
    fig = plt.figure()
    # convert input y to angle
    angle_y = np.cos(y)
    ax = plt.polar(angle_y, x)
    return fig, ax


def matplotlib_heatmap_chart(matrix: np.array) -> Tuple:
    """
    Create a matplotlib heatmap chart with the inputs. DO NOT PLOT IT!!
    The input is a 2D matrix (x, y). See example at the end of file.
    Return the fig and ax as was shown in matplotlib_line_example.
    """
    fig, ax = plt.subplots()
    ax.imshow(matrix)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(j, i, '%.3f' % matrix[i, j],
                           ha="center", va="center", color="w")
    fig.tight_layout()
    return fig, ax


def matplotlib_table(matrix: np.array) -> Tuple:
    """
    Create a matplotlib table with the input. DO NOT PLOT IT!!
    Return the fig and ax as was shown in matplotlib_line_example.
    """
    fig, ax = plt.subplots()
    ax.axis('off')  # remove axes
    ax = ax.table(cellText=matrix, loc='center')
    return fig, ax


def matplotlib_composite_line_bar(x: np.array) -> Tuple:
    """
    Create a composite matplotlib chart by composing a bar and line chart on top of each other
    with the input. Consider the y coordinate as the x value index.
    DO NOT PLOT IT!!
    Return the fig and ax as was shown in matplotlib_line_example.
    """
    y = []
    count = len(x)
    for i in range(count):
        y.append(i)

    fig, ax = plt.subplots()
    ax.bar(y, x, width=0.6)
    plt.xticks(rotation=0)
    ax = ax.twinx()
    ax.plot(y, x, marker='.')
    return fig, ax


# There are many other possibilities. Please, do check the documentation and examples so you
# may have a good breadth of tools for future work (in assignments, projects, and your own career)
###################################
def plotly_bar_chart(df: pd.DataFrame) -> Figure:
    """
    The input is a dataframe.
    Create a plotly bar chart with the inputs. DO NOT PLOT IT!!
    Return the fig only. Feel free to choose between px and go.
    """
    df['rowId'] = 'row' + df.index.astype(str)
    df.set_index(["rowId"], inplace=True)
    ax = df.plot(kind="bar")
    fig = ax.get_figure()
    ax.set_xlabel("row_index")
    return fig


def plotly_pie_chart(df: pd.DataFrame) -> Figure:
    """
    The input is a dataframe with x column to be used.
    Create a plotly pie chart with the inputs. DO NOT PLOT IT!!
    Return the fig only. Feel free to choose between px and go.
    """
    df['rowId'] = 'row' + df.index.astype(str)
    fig = px.pie(df, values=df.columns[0], names='rowId')
    return fig


def plotly_histogram(df: pd.DataFrame, n_bins: int) -> Figure:
    """
    The input is a dataframe with x column to be used.
    Create a plotly histogram chart with the inputs. DO NOT PLOT IT!!
    Return the fig only. Feel free to choose between px and go.
    """
    col = df.columns[0]
    fig = px.histogram(df, x=col, nbins=n_bins)
    return fig


def plotly_polar_chart(df: pd.DataFrame) -> Figure:
    """
    The input is a dataframe with x and y columns.
    Create a plotly polar chart with the inputs. DO NOT PLOT IT!!
    The y input is the same as the line chart, so you need to convert it to an angle.
    Return the fig only. Feel free to choose between px and go.
    """
    # convert y to an angle
    df['angle_y'] = np.where(df['y'] > 0, (y + 90) % 360, (y - 90) % 360)
    fig = px.line_polar(df, r='angle_y', theta='x', line_close=True)
    return fig


def plotly_heatmap_chart(df: pd.DataFrame) -> Figure:
    """
    The input is a dataframe with columns and rows. Use well the column and row names!
    Create a plotly heatmap chart with the inputs. DO NOT PLOT IT!!
    Return the fig only. Feel free to choose between px and go.
    """
    fig = px.imshow(df)
    fig.update_xaxes(side="top")
    return fig


def plotly_table(df: pd.DataFrame) -> Figure:
    """
    The input is a dataframe with columns and rows. Use well the column and row names!
    See example input at the end of the file.
    Create a plotly table with the input. DO NOT PLOT IT!!
    Return the fig only. Feel free to choose between px and go.
    """
    cols = df.columns
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.index),
                    align='left'),
        cells=dict(values=df.values,
                   align='left'))
    ])
    return fig


def plotly_contour_chart(df: pd.DataFrame) -> Figure:
    """
    The input is a dataframe with columns and rows. Use well the column and row names!
    See example input at the end of the file.
    Create a plotly contour chart with the inputs.
    DO NOT PLOT IT!!
    Return the fig only.
    """
    fig = go.Figure(data=go.Contour(z=df))
    return fig


def plotly_composite_line_bar(df: pd.DataFrame) -> Tuple:
    """
    The input is a dataframe with x and y columns.
    Create a composite matplotlib chart by composing a bar and line chart on top of each other
    with the inputs. DO NOT PLOT IT!!
    Return the fig and ax as was shown in matplotlib_line_example.
    """
    col_x = df['x'].tolist()
    col_y = df['y'].tolist()
    col_z = df['z'].tolist()
    fig, ax = plt.subplots()
    ax = px.line(df, x='x', y='y')
    ax.add_bar(x=col_x, y=col_z, width=0.05)
    return fig, ax


def plotly_subgraphs(df: pd.DataFrame) -> Figure:
    """
    The inputs is a dataframe with x1, y1, x2, y2, x3, y3, x4 and y4 as columns.
    Create 4 plots (line, scatterplot, bar chart (using only x3) and
    stacked area plots respectively) and place each of the at each quadrant (e.g. top-left,
    top-right, etc), and then output a single fig. DO NOT PLOT IT!!
    """
    fig = make_subplots(rows=2, cols=2, start_cell="top-left")
    fig.add_trace(go.Scatter(x=df["x1"], y=df["y1"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["x2"], y=df["y2"], mode='markers', ), row=1, col=2)
    fig.add_trace(go.Bar(x=df["x3"], ), row=2, col=1, )
    fig.add_trace(go.Scatter(x=df["x1"], y=df["y1"], hoverinfo='x+y', mode='lines',
                             line=dict(width=0.5, color='rgb(131, 90, 241)'),
                             stackgroup='one'), row=2, col=2, )
    fig.add_trace(go.Scatter(x=df["x1"], y=df["y2"], hoverinfo='x+y', mode='lines',
                             line=dict(width=0.5, color='rgb(127, 166, 238)'),
                             stackgroup='one'), row=2, col=2, )
    fig.add_trace(go.Scatter(x=df["x1"], y=df["y4"], hoverinfo='x+y', mode='lines',
                             line=dict(width=0.5, color='rgb(184, 247, 212)'),
                             stackgroup='one'), row=2, col=2, )
    fig.add_trace(go.Scatter(x=df["x1"], y=df["y3"], hoverinfo='x+y', mode='lines',
                             line=dict(width=0.5, color='rgb(111, 231, 219)'),
                             stackgroup='one'), row=2, col=2, )

    return fig


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.

    x = np.random.rand(50) * np.random.randint(-10, 10)
    y = np.random.rand(50) * np.random.randint(-10, 10)
    matrix_2D = np.random.rand(10, 10) * np.random.randint(-10, 10)

    fig1, _ = matplotlib_bar_chart(x)
    fig2, _ = matplotlib_pie_chart(np.abs(x))
    fig3, _ = matplotlib_histogram(x, 5)
    fig4, _ = matplotlib_polar_chart(x, y)
    matplotlib_heatmap_chart(matrix_2D)
    matplotlib_table(matrix_2D)
    matplotlib_composite_line_bar(x)

    df = pd.DataFrame(dict(x=x, y=y, z=x + y))
    df_matrix = pd.DataFrame(matrix_2D, columns=[f'grade of class {x}' for x in range(10)],
                             index=[f'professor {x}' for x in range(10)])
    plotly_bar_chart(df)
    fig_pie = plotly_pie_chart(df)
    plotly_histogram(df, 5)
    plotly_polar_chart(df)
    plotly_heatmap_chart(df_matrix)
    plotly_table(df_matrix)
    plotly_contour_chart(df_matrix)
    plotly_composite_line_bar(df)
    plotly_subgraphs(pd.DataFrame(
        dict(x1=np.random.rand(50) * np.random.randint(-10, 10), y1=np.random.rand(50) * np.random.randint(-10, 10),
             x2=np.random.rand(50) * np.random.randint(-10, 10), y2=np.random.rand(50) * np.random.randint(-10, 10),
             x3=np.random.rand(50) * np.random.randint(-10, 10), y3=np.random.rand(50) * np.random.randint(-10, 10),
             x4=np.random.rand(50) * np.random.randint(-10, 10), y4=np.random.rand(50) * np.random.randint(-10, 10),
             )))
