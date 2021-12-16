import collections
import itertools
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from assignment1.b_data_profile import *
from assignment1.c_data_cleaning import *
from assignment1.d_data_encoding import *
from assignment1.a_load_file import *

##############################################
# Example(s). Read the comments in the following method(s)
##############################################


def process_iris_dataset() -> pd.DataFrame:
    """
    In this example, I call the methods you should have implemented in the other files
    to read and preprocess the iris dataset. This dataset is simple, and only has 4 columns:
    three numeric and one categorical. Depending on what I want to do in the future, I may want
    to transform these columns in other things (for example, I could transform a numeric column
    into a categorical one by splitting the number into bins, similar to how a histogram creates bins
    to be shown as a bar chart).

    In my case, what I want to do is to *remove missing numbers*, replacing them with valid ones,
    and *delete outliers* rows altogether (I could have decided to do something else, and this decision
    will be on you depending on what you'll do with the data afterwords, e.g. what machine learning
    algorithm you'll use). I will also standardize the numeric columns, create a new column with the average
    distance between the three numeric column and convert the categorical column to a onehot-encoding format.

    :return: A dataframe with no missing values, no outliers and onehotencoded categorical columns
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    for nc in numeric_columns:
        df = fix_outliers(df, nc)
        df = fix_nans(df, nc)
        df.loc[:, nc] = standardize_column(df.loc[:, nc])

    distances = pd.DataFrame()
    for nc_combination in list(itertools.combinations(numeric_columns, 2)):
        distances[str(nc_combination)] = calculate_numeric_distance(df.loc[:, nc_combination[0]],
                                                                    df.loc[:, nc_combination[1]],
                                                                    DistanceMetric.EUCLIDEAN).values
    df['numeric_mean'] = distances.mean(axis=1)

    for cc in categorical_columns:
        ohe = generate_one_hot_encoder(df.loc[:, cc])
        df = replace_with_one_hot_encoder(df, cc, ohe, list(ohe.get_feature_names()))
    return df


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def process_iris_dataset_again() -> pd.DataFrame:
    """
    Consider the example above and once again perform a preprocessing and cleaning of the iris dataset.
    This time, use normalization for the numeric columns and use label_encoder for the categorical column.
    Also, for this example, consider that all petal_widths should be between 0.0 and 1.0, replace the wong_values
    of that column with the mean of that column. Also include a new (binary) column called "large_sepal_lenght"
    saying whether the row's sepal_length is larger (true) or not (false) than 5.0
    :return: A dataframe with the above conditions.
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    for nc in numeric_columns:
        df = fix_outliers(df, nc)
        df = fix_nans(df, nc)

    # checking for Sepal_length and creating new boolean column before normalization
    df["large_sepal_length"] = df["sepal_length"] > 5

    # Normalizing the Numeric Columns
    for nc_column in numeric_columns:
        df.loc[:, nc_column] = normalize_column(df.loc[:, nc_column])

    # replacing values of petal_width which are outside (0,1) range with its mean value
    df.loc[(df['petal_width'] < 0) | (df['petal_width'] > 1), 'petal_width'] = df['petal_width'].mean()

    # Label Encoding the categorical columns
    for cc in categorical_columns:
        le = generate_label_encoder(df.loc[:, cc])
        df = replace_with_label_encoder(df, cc, le)

    df.drop(labels='large_sepal_length', inplace=True, axis=1)

    return df


def process_amazon_video_game_dataset():
    """
    Now use the rating_Video_Games dataset following these rules:
    1. The rating has to be between 1.0 and 5.0
    2. Time should be converted from milliseconds to datetime.datetime format
    3. For the future use of this data, I don't care about who voted what, I only want the average rating per product,
        therefore replace the user column by counting how many ratings each product had (which should be a column called count),
        and the average rating (as the "review" column).
    :return: A dataframe with the above conditions. The columns at the end should be: asin,review,time,count
    """
    df = read_dataset(Path('..', '..', 'ratings_Video_Games.csv'))
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    # getting rows with incorrect review values and replacing that with nan
    df_incorrect_review = df.loc[~df["review"].isin([1, 2, 3, 4, 5]), df.columns]
    df.loc[df.index.isin(df_incorrect_review.index), 'review'] = np.nan

    # converting time from ms to datatime format
    # df['time'] = pd.to_datetime(df['time'], unit='ms')

    # grouping by usng product type i.e.'asin'
    df = df.groupby('asin').agg({'review': 'mean', 'time': 'max', 'user': 'size'}).rename(
        columns={'review': 'review', 'time': 'time', 'user': 'count'}).reset_index()
    df['review'] = df['review'].round(decimals=0).astype(int)

    return df


def process_amazon_video_game_dataset_again():
    """
    Now use the rating_Video_Games dataset following these rules (the third rule changed, and is more open-ended):
    1. The rating has to be between 1.0 and 5.0, drop any rows not following this rule
    2. Time should be converted from milliseconds to datetime.datetime format
    3. For the future use of this data, I just want to know more about the users, therefore show me how many reviews each user has,
        and a statistical analysis of each user (average, median, std, etc..., each as its own row)
    :return: A dataframe with the above conditions.
    """
    df = read_dataset(Path('..', '..', 'ratings_Video_Games.csv'))
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    # getting rows with incorrect review values and dropping those rows from the DataFrame
    df_incorrect_review = df.loc[~df["review"].isin([1, 2, 3, 4, 5]), "review"]
    df = df.drop(df_incorrect_review.index, axis=0)

    # converting time from ms to datatime format
    df['time'] = pd.to_datetime(df['time'], unit='ms')

    # grouping by usng user and getting metric values of the reviews provided by the user
    df = df.groupby('user').agg({'user': 'size', 'review': {'mean', 'median', 'std'}})
    df.columns = df.columns.map('_'.join)
    df = df.reset_index()
    # Replacing the np.nan values in review_std value with 0, since np.std(ddfo=0) "Delta Degrees of Freedom=0" is used for calculation, this leads to division by 0 for single values
    df['review_std'].fillna(value=0, inplace=True)
    return df


def process_life_expectancy_dataset():
    """
    Now use the life_expectancy_years and geography datasets following these rules:
    1. The life expectancy dataset has missing values and outliers. Fix them.
    2. The geography dataset has problems with unicode letters. Make sure your code is handling it properly.
    3. Change the format of life expectancy, so that instead of one row with all 28 years, the data has 28 rows, one for each year,
        and with a column "year" with the year and a column "value" with the original value
    4. Merge (or more specifically, join) the two datasets with the common column being the country name (be careful with wrong values here)
    5. Drop all columns except country, continent, year, value and latitude (in this hypothetical example, we wish to analyse differences
        between southern and northern hemisphere)
    6. Change the latitude column from numerical to categorical (north vs south) and pass it though a label_encoder
    7. Change the continent column to a one_hot_encoder version of it
    :return: A dataframe with the above conditions.
    """
    df1 = read_dataset(Path('..', '..', 'life_expectancy_years.csv'))
    numeric_columns = get_numeric_columns(df1)
    categorical_columns = get_text_categorical_columns(df1)
    for nc in numeric_columns:
        df1 = fix_outliers(df1, nc)
        df1 = fix_nans(df1, nc)

    df1 = pd.melt(df1, 'country', var_name='year').sort_values(['country', 'year']).reset_index(
        drop=True)  # Using pd.melt function to perform the data inversion

    df2 = pd.read_csv(Path('..', '..', 'geography.csv'), encoding='utf8')

    df3 = pd.merge(df1, df2, left_on='country', right_on='name')
    # print(df3.loc[1])

    # Dropping the unecessary columns as per the requirement
    df3 = df3.drop(
        columns=['name', 'geo', 'eight_regions', 'six_regions', 'members_oecd_g77', 'UN member since',
                 'World bank region', 'World bank, 4 income groups 2017',
                 'adjective', 'adjective_plural'])

    # Replacing the Latitude with text values
    df3.loc[df3['Latitude'] >= 0, 'Latitude'] = 'North'
    df3.loc[df3['Latitude'] != 'North', 'Latitude'] = 'South'

    # Endoding the latitude using label encoder
    le = generate_label_encoder(df3.Latitude)
    df3 = replace_with_label_encoder(df3, 'Latitude', le)

    # One hot encoding the continent value
    ohe = generate_one_hot_encoder(df3.four_regions)
    df3 = replace_with_one_hot_encoder(df3, 'four_regions', ohe, list(ohe.get_feature_names()))

    return df3


if __name__ == "__main__":
    # assert process_iris_dataset() is not None
    # assert process_iris_dataset_again() is not None
    # assert process_amazon_video_game_dataset() is not None
    # assert process_amazon_video_game_dataset_again() is not None
    # assert process_life_expectancy_dataset() is not None
    print("ok")
    df = process_life_expectancy_dataset()
    print(df)
    print(df.iloc[0, :])
    # print(df.loc[0])
