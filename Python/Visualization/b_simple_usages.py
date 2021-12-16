from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from assignment2.a_classification import *
from assignment2.c_clustering import *
from a_libraries import *


##############################################
# In this file, we will use data and methods of previous assignments with visualization.
# But before you continue on, take some time to look on the internet about the many existing visualization types and their usages, for example:
# https://extremepresentation.typepad.com/blog/2006/09/choosing_a_good.html
# https://datavizcatalogue.com/
# https://plotly.com/python/
# https://www.tableau.com/learn/whitepapers/which-chart-or-graph-is-right-for-you
# Or just google "which visualization to use", and you'll find a near-infinite number of resources
#
# You may want to create a new visualization in the future, and for that I suggest using JavaScript and D3.js,
# but for the course, we will only
# use python and already available visualizations
##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
# For ALL methods return the fig and ax of matplotlib or fig from plotly!
##############################################


def matplotlib_bar_chart() -> Tuple:
    """
    Create a bar chart with a1/b_data_profile's get column max.
    Show the max of each numeric column from iris dataset as the bars
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    max_val = []
    for nc in numeric_columns:
        max_val.append(get_column_max(df, nc))
    arr = np.array(max_val)
    fig, ax = plt.subplots()
    ax.bar(numeric_columns.columns, arr)
    return fig, ax


def matplotlib_pie_chart() -> Tuple:
    """
    Create a pie chart where each piece of the chart has the number of columns which are numeric/categorical/binary
    from the output of a1/e_/process_life_expectancy_dataset
    """
    # 1. get the instance of each column type
    df_life = process_life_expectancy_dataset()
    numeric_columns = get_numeric_columns(df_life).columns.tolist()
    categorical_columns = get_text_categorical_columns(df_life)
    binary_col = get_binary_columns(df_life)

    # 2. Create an list of the column_type and an array of count
    col = np.array([len(numeric_columns), len(categorical_columns), len(binary_col)])
    mylabels = ['Numeric', 'Categorical', 'Binary']

    # 3. plot pie chart
    fig, ax = plt.subplots()
    ax = plt.pie(col, labels=mylabels, autopct='%1.1f%%')
    plt.legend(title="Column type:")
    return fig, ax


def matplotlib_histogram() -> Tuple:
    """
    Build 4 histograms as subplots in one figure with the numeric values of the iris dataset
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    xaxes = []
    arr_list = []
    for nc in numeric_columns:
        col = numeric_columns[nc].to_numpy()
        arr_list.append(col)
        xaxes.append(nc)

    for idx, a in enumerate(ax):
        a.hist(arr_list[idx])
        a.set_xlabel(xaxes[idx])

    fig.tight_layout()
    return fig, ax


def matplotlib_heatmap_chart() -> Tuple:
    """
    Remember a1/b_/pandas_profile? There is a heat map over there to analyse the correlation among columns.
    Use the pearson correlation (e.g. https://docs.scipy.org/doc/scipy-1.5.3/reference/generated/scipy.stats.pearsonr.html)
    to calculate the correlation between two numeric columns and show that as a heat map. Use the iris dataset.
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    col_corr = numeric_columns.corr().to_numpy()
    fig, ax = plt.subplots()
    ax.imshow(col_corr, cmap="RdYlBu")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(col_corr)):
        for j in range(len(col_corr)):
            ax.text(j, i, '%.4f' % col_corr[i, j],
                           ha="center", va="center", color="w")
    fig.tight_layout()
    return fig, ax


# There are many other possibilities. Please, do check the documentation and examples so you
# may have a good breadth of tools for future work (in assignments, projects, and your own career)
###################################
# Once again, for ALL methods return the fig and ax of matplotlib or fig from plotly!


def plotly_scatter_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() as the color of a scatterplot made from the original (unprocessed)
    iris dataset. Choose among the numeric values to be the x and y coordinates.
    """
    # get the labels from the result of the cluster method
    iris = cluster_iris_dataset_again()
    cluster = iris['clusters'].labels_

    # load original dataset
    df = read_dataset(Path('..', '..', 'iris.csv'))

    # create the specie column using the cluster result
    df['cluster_result'] = cluster

    # plot scatter plot using the 'petal length' and 'petal width'
    # the color is defined by the 'cluster_result' column'
    fig = px.scatter(df, x=df.columns[0], y=df.columns[2], color="cluster_result")
    return fig


def plotly_bar_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() and use x as 3 groups of bars (one for each iris species)
    and each group has multiple bars, one for each cluster, with y as the count of instances in the specific cluster/species combination.
    The grouped bar chart is like https://plotly.com/python/bar-charts/#grouped-bar-chart (search for the grouped bar chart visualization)
    """
    iris = cluster_iris_dataset_again()
    iris_list = list(set(iris['clusters'].labels_))
    y_axis = np.arange(len(iris_list))
    fig = go.Figure(data=[
        go.Bar(name='Specie'+str(iris_list[0]), x=iris_list, y=[50, 50, 50]),
        go.Bar(name='Specie'+str(iris_list[1]), x=iris_list, y=[50, 50, 50]),
        go.Bar(name='Specie'+str(iris_list[2]), x=iris_list, y=[50, 50, 50])
    ])

    # Change the bar mode
    fig.update_layout(barmode='group')
    return fig


def plotly_polar_scatterplot_chart():
    """
    Do something similar to a1/e_/process_life_expectancy_dataset, but don't drop the latitude and longitude.
    Use these two values to figure out the theta to plot values as a compass (example: https://plotly.com/python/polar-chart/).
    Each point should be one country and the radius should be thd value from the dataset (add up all years and feel free to ignore everything else)
    """
    df1 = read_dataset(Path('..', '..', 'life_expectancy_years.csv'))
    numeric_columns = get_numeric_columns(df1)
    for nc in numeric_columns:
        df1 = fix_outliers(df1, nc)
        df1 = fix_nans(df1, nc)

    df1 = pd.melt(df1, 'country', var_name='year').sort_values(['country', 'year']).reset_index(drop=True)
    df2 = pd.read_csv(Path('..', '..', 'geography.csv'), encoding='utf8')
    df3 = pd.merge(df1, df2, left_on='country', right_on='name')

    # pick necessary columns alone
    df4 = pd.concat([df3["country"], df3["year"], df3["Latitude"], df3["Longitude"]], axis=1,
                    keys=["country", "year", "Latitude", "Longitude"])

    # convert column 'year' to 'number' column
    df4[["year"]] = df4[["year"]].apply(pd.to_numeric)

    # sum up the year column by country
    df4 = df4.groupby(['country', 'Latitude', 'Longitude']).agg({'year': ['sum']})
    df4.columns = ['year']
    df4 = df4.reset_index()

    # plot scatter plot with country, Latitude and Longitude
    fig = px.scatter_polar(df4, r="Latitude", theta="Longitude", color="country")
    return fig


def plotly_table():
    """
    Show the data from a2/a_classification/decision_tree_classifier() as a table
    See https://plotly.com/python/table/ for documentation
    """
    # load data and classify with decision_tree_classifier
    iris = read_dataset(Path('../../iris.csv'))
    iris_copy = iris.copy()
    label_col = "species"
    feature_cols = iris_copy.columns.tolist()
    feature_cols.remove(label_col)
    x_iris = iris_copy[feature_cols]
    y_iris = iris_copy[label_col]
    dtc = decision_tree_classifier(x_iris, y_iris)

    # get the confusion matrix from result
    confusion_matrix = dtc['confusion_matrix']

    # convert to dataframe
    dtc_df = pd.DataFrame(confusion_matrix)

    # plot confusion matrix in table
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(dtc_df.index),
                    align='left'),
        cells=dict(values=dtc_df.values,
                   align='left'))
    ])
    return fig


def plotly_composite_line_bar():
    """
    Use the data from a1/e_/process_life_expectancy_dataset and show in a single graph on year on x and value on y where
    there are 5 line charts of 5 countries (you choose which) and one bar chart on the background with the total value of all 5
    countries added up.
    """
    df = process_life_expectancy_dataset()

    # select 5 countries
    df = df.query("country == 'Afghanistan' or country == 'Belarus' or country == 'Cameroon' or country == 'France' "
                  "or country == 'Indonesia' ")

    # plot the composite chart
    col_x = df['year'].tolist()
    col_z = df['value'].tolist()
    fig = px.line(df, x='year', y='value', color='country')
    fig.add_bar(x=col_x, y=col_z)

    return fig


def plotly_map():
    """
    Use the data from a1/e_/process_life_expectancy_dataset on a plotly map (anyone will do)
    Examples: https://plotly.com/python/maps/, https://plotly.com/python/choropleth-maps/#using-builtin-country-and-state-geometries
    Use the value from the dataset of a specific year (e.g. 1900) to show as the color in the map
    """
    df = process_life_expectancy_dataset()

    # get data for year 2018
    df = df.query("year == '2018'")

    # create column 'code' with first three characters of country name
    df['code'] = df['country'].astype(str).str[:3].str.upper()

    # plot the map
    fig = px.choropleth(df, locations="code", color="value", hover_name="country",
                        color_continuous_scale=px.colors.sequential.Plasma)

    return fig


def plotly_tree_map():
    """
    Use plotly's treemap to plot any data returned from any of a1/e_experimentation or a2 tasks
    Documentation: https://plotly.com/python/treemaps/
    """
    df = process_life_expectancy_dataset()

    # load sample years
    df = df.query("year == '2014' or year == '2015' or year == '2016' or year == '2017' or year == '2018'")

    # create map
    fig = px.treemap(df, path=[px.Constant("all"), 'year', 'country'], values='value',
                     color_continuous_scale='RdBu')
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    fig_m_bc, _ = matplotlib_bar_chart()
    fig_m_pc, _ = matplotlib_pie_chart()
    fig_m_h, _ = matplotlib_histogram()
    fig_m_hc, _ = matplotlib_heatmap_chart()

    fig_p_s = plotly_scatter_plot_chart()
    fig_p_bpc = plotly_bar_plot_chart()
    fig_p_psc = plotly_polar_scatterplot_chart()
    fig_p_t = plotly_table()
    fig_p_clb = plotly_composite_line_bar()
    fig_p_map = plotly_map()
    fig_p_treemap = plotly_tree_map()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # fig_m_bc.show()
    # fig_m_pc.show()
    # fig_m_h.show()
    # fig_m_hc.show()

    # fig_p_s.show()
    # fig_p_bpc.show()
    # fig_p_psc.show()
    # fig_p_t.show()
    # fig_p_clb.show()
    # fig_p_map.show()
    # fig_p_treemap.show()
