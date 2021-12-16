from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.widgets import Button, Slider


###############
# Interactivity in visualizations is challenging due to limitations and chunkiness of libraries.
# For example, some libraries works well in Jupyter Notebooks, but then the code makes barely any sense and becomes hard to change/update,
# defeating the purpose of using Jupyter notebooks in the first place, and other libraries provide a window of their own, but
# they are very tied to the running code, and far from the experience you'd expect from a proper visual analytics webapp
#
# We will try out some approaches to exercise in this file, but the next file will give you the proper tooling to make a
# well-rounded and efficient code for visual interactions.
##############################################
# Example(s). Read the comments in the following method(s)
##############################################
from assignment1.a_load_file import read_dataset
from assignment1.b_data_profile import *
from assignment2.c_clustering import *


def matplotlib_simple_example():
    """
    Using the same logic from before, we can add sliders or buttons to select options for interactivity.
    Matplotlib is limited to only a few options, but do take a look, since they are useful for fast prototyping and analysis

    In case you are using PyCharm, I suggest you to uncheck the 'Show plots in tool window'
    to be able to see and interact with the buttons defined below.
    This example comes from https://matplotlib.org/3.1.1/gallery/widgets/buttons.html
    """
    freqs = np.arange(2, 20, 3)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    t = np.arange(0.0, 1.0, 0.001)
    s = np.sin(2 * np.pi * freqs[0] * t)
    l, = plt.plot(t, s, lw=2)

    class Index(object):
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    return fig, ax


def matplotlib_simple_example2():
    """
    Here is another example, which also has a slider and simplifies a bit the callbacks
    """
    data = np.random.rand(10, 5)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.bar(np.arange(10).astype(str).tolist(), data[:, 0])

    class Index(object):
        ind = 0
        multiplier = 1

        def change_data(self, event, i):
            self.ind = np.clip(self.ind + i, 0, data.shape[1] - 1)
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(), data[:, self.ind] * self.multiplier)
            plt.draw()

        def change_multiplier(self, value):
            self.multiplier = value
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(), data[:, self.ind] * self.multiplier)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.1, 0.05, 0.12, 0.075])
    axnext = plt.axes([0.23, 0.05, 0.12, 0.075])
    axslider = plt.axes([0.55, 0.1, 0.35, 0.03])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(lambda event: callback.change_data(event, 1))
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(lambda event: callback.change_data(event, -1))
    slider = Slider(axslider, 'multiplier', 1, 10, 1)
    slider.on_changed(callback.change_multiplier)
    plt.show(block=True)
    return fig, ax


def plotly_slider_example():
    """
    Here is a simple example from https://plotly.com/python/sliders/ of how to include a slider in plotly
    Notice how similar it already is to GapMinder!
    """
    df = px.data.gapminder()
    fig = px.scatter(df, x="gdpPercap", y="lifeExp",
                     animation_frame="year",  # set which column makes the animation though a slider
                     size="pop",
                     color="continent",
                     hover_name="country",
                     log_x=True,
                     size_max=55,
                     range_x=[100, 100000],
                     range_y=[25, 90])

    fig["layout"].pop("updatemenus")  # optional, drop animation buttons

    return fig


def plotly_button_example():
    """
    To have buttons, plotly requires us to use go (and not px) to generate the graph.
    The button options are very restrictive, since all they can do is change a parameter from the go graph.
    In the example below, it changes the "mode" value of the graph (between lines and scatter)
    The code is a modified code example taken from https://plotly.com/python/custom-buttons/
    """
    x = np.random.rand(50) * np.random.randint(-10, 10)
    y = np.random.rand(50) * np.random.randint(-10, 10)

    fig = go.Figure()

    # Add surface trace
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(type="buttons",
                 direction="left",
                 buttons=[
                     dict(
                         label="line",  # just the name of the button
                         method="update",  # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         args=[{"mode": "markers"}],  # This is the value being updated in the visualization
                     ), dict(
                         label="scatter",  # just the name of the button
                         method="update",  # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         args=[{"mode": "line"}],  # This is the value being updated in the visualization
                     )
                 ],
                 pad={"r": 10, "t": 10}, showactive=True, x=0.11, xanchor="left", y=1.1, yanchor="top"  # Layout-related values
                 ),
        ]
    )

    fig.show()
    return fig


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################
def matplotlib_interactivity():
    """
    Do an interactive matplotlib plot where I can select which visualization I want.
    Make either a slider, a dropdown or several buttons and make so each option gives me a different visualization from
    the matplotlib figures of b_simple_usages. Return just the resulting fig as is done in plotly_slider_example.
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    iris_species = list(set(df['species']))
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.2)
    ax.title.set_text('Graph for Iris Dataset')

    class Index(object):

        def heat_map(self, event):
            col_corr = numeric_columns.corr().to_numpy()
            ax.clear()
            ax.imshow(col_corr, cmap="RdYlBu")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            for i in range(len(col_corr)):
                for j in range(len(col_corr)):
                    ax.text(j, i, '%.4f' % col_corr[i, j], ha="center", va="center", color="w")
            # fig.tight_layout()
            plt.draw()

        def bar_chart(self, event):
            max_val = []
            for nc in numeric_columns:
                max_val.append(get_column_max(df, nc))
            arr = np.array(max_val)
            ax.clear()
            ax.bar(numeric_columns.columns, arr)
            plt.draw()

        def pie_chart(self, event):
            col = np.array([len(df.query("species == species[0]")), len(df.query("species == species[1]")),
                            len(df.query("species == species[2]"))])
            ax.clear()
            ax.pie(col, labels=iris_species, autopct='%1.1f%%')
            plt.draw()

    callback = Index()
    axpie = plt.axes([0.7, 0.01, 0.1, 0.075])
    axbar = plt.axes([0.81, 0.01, 0.1, 0.075])
    axheat = plt.axes([0.59, 0.01, 0.1, 0.075])
    bheat = Button(axheat, 'HeatMap')
    bheat.on_clicked(callback.heat_map)
    bnext = Button(axbar, 'BarChart')
    bnext.on_clicked(callback.bar_chart)
    bprev = Button(axpie, 'PieChart')
    bprev.on_clicked(callback.pie_chart)
    plt.show(block=True)
    return fig


def matplotlib_cluster_interactivity():
    """
    Do an interactive matplotlib plot where I can select how many clusters I want to train from.
    Use iris dataset (just numeric columns) and k-means (feel free to reuse as/c_clustering if you wish).
    The slider (or dropdown) should range from 2 to 10. Return just the resulting fig.
    """
    # load original data, get the numeric columns and train using k-means
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    k_dict = simple_k_means(numeric_columns, 3, 'euclidean')

    # get the labels by the clustering algorithm
    data = k_dict['clusters'].labels_

    # define the figure
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.2)

    # Draw a scatter plot using the dataframe with number columns
    # x -> first column, y -> second column, clusters are defined by the label result of clustering algorithm
    ax.scatter(numeric_columns. iloc[:, 0], numeric_columns. iloc[:, 2], c=data)
    plt.draw()

    class Index(object):
        val = 0

        def train_cluster(self, val):
            self.val = slider.val
            k_dicts = simple_k_means(numeric_columns, self.val, 'euclidean')
            k_data = k_dicts['clusters'].labels_
            ax.clear()
            ax.scatter(numeric_columns.iloc[:, 0], numeric_columns.iloc[:, 2], c=k_data)
            plt.draw()

    callback = Index()
    axslider = plt.axes([0.55, 0.1, 0.35, 0.03])
    slider = Slider(ax=axslider, label='clusters', valmin=2, valmax=10, valinit=3, valstep=1, closedmax=True)
    slider.on_changed(callback.train_cluster)
    plt.show(block=True)
    return fig


def plotly_interactivity():
    """
    Do a plotly graph with all plotly 6 figs from b_simple_usages, and make 6 buttons (one for each fig).
    Change the displayed graph depending on which button I click. Return just the resulting fig.
    """
    # load original data
    df = process_life_expectancy_dataset()

    # select sample 5 years
    df = df.query("year == '2014' or year == '2015' or year == '2016' or year == '2017' or year == '2018'")

    # select year for scatter plot
    data = df.query("year == '2018'")

    # create treemap as 'year' -> 'country' -> 'values'
    fig = go.Figure(data=px.treemap(df, path=[px.Constant("all"), 'year', 'country'], values='value'))


    # create scatterplot for year 2018 with x -> 'country' and y -> value
    fig.add_trace(go.Scatter(x=data.iloc[:, 0], y=data.iloc[:, 2], name="scatter", mode='markers'))

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(type="buttons",
                 direction="left",
                 buttons=[
                     dict(
                         label="Tree Map",
                         method="update",
                         args=[{"visible": [True, False]}],
                     ), dict(
                         label="Scatter Plot",
                         method="update",
                         args=[{"visible": [False, True]},
                               {"mode": "markers"}],
                     )
                 ],
                 pad={"r": 10, "t": 10}, showactive=True, x=0.11, xanchor="left", y=1.1, yanchor="top"
                 ),
        ]
    )
    return fig


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    fig_m_i = matplotlib_interactivity()
    fig_m_ci = matplotlib_cluster_interactivity()
    fig_p = plotly_interactivity()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # matplotlib_simple_example()[0].show()
    # matplotlib_simple_example2()[0].show()
    # plotly_slider_example().show()
    # plotly_button_example().show()
    # fig_m_i.show()
    # fig_m_ci.show()
    # fig_p.show()
