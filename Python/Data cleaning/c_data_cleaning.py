from typing import Optional
from enum import Enum
from sklearn import preprocessing
from scipy.spatial.distance import hamming
from scipy import stats

# from b_data_profile import *
from assignment1.b_data_profile import *
##############################################
# Example(s). Read the comments in the following method(s)
##############################################


class WrongValueNumericRule(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    MUST_BE_POSITIVE = 0
    MUST_BE_NEGATIVE = 1
    MUST_BE_GREATER_THAN = 2
    MUST_BE_LESS_THAN = 3


class DistanceMetric(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    EUCLIDEAN = 0
    MANHATTAN = 1


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def fix_numeric_wrong_values(df: pd.DataFrame,
                             column: str,
                             must_be_rule: WrongValueNumericRule,
                             must_be_rule_optional_parameter: Optional[float] = None) -> pd.DataFrame:
    """
    This method should fix the wrong_values depending on the logic you think best to do and using the rule passed by parameter.
    Remember that wrong values are values that are in the dataset, but are wrongly inputted (for example, a negative age).
    Here you should only fix them (and not find them) by replacing them with np.nan ("not a number", or also called "missing value")
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :param must_be_rule: one of WrongValueNumericRule identifying what rule should be followed to flag a value as a wrong value
    :param must_be_rule_optional_parameter: optional parameter for the "greater than" or "less than" cases
    :return: The dataset with fixed column
    """
    if must_be_rule.value == 0:
        df.loc[df[column] < 0] = np.nan
    elif must_be_rule.value == 1:
        df.loc[df[column] > 0] = np.nan
    elif must_be_rule.value == 2:
        df.loc[df[column] < must_be_rule_optional_parameter, column] = np.nan
    elif must_be_rule.value == 3:
        df.loc[df[column] > must_be_rule_optional_parameter, column] = np.nan
    else:
        df[column] = df[column].replace('None', np.nan)
    return df


def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix the column in respective to outliers depending on the logic you think best to do.
    Feel free to choose which logic you prefer, but if you are in doubt, use the simplest one to remove the row
    of the dataframe which is an outlier (note that the issue with this approach is when the dataset is small,
    dropping rows will make it even smaller).
    Remember that some datasets are large, and some are small, so think wisely on how to calculate outliers
    and when to remove/replace them. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The dataset with fixed column
    """

    """This method calculates the z_score for numerical columns.
       The threshold is set at 2.5. if z_score > threshold, replace with np.nan"""
    if (df[column].dtype == 'int64') or (df[column].dtype == 'float64'):
        threshold = 2.5
        z_score = np.abs(stats.zscore(df[column]))
        df.loc[np.abs(z_score > threshold), column] = np.nan
    return df


def fix_nans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix all nans (missing data) depending on the logic you think best to do
    Remember that some datasets are large, and some are small, so think wisely on when to use each possible
    removal/fix/replace of nans. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The fixed dataset
    """
    """This replaces nan with the mean of the column"""
    df[column] = df[column].fillna(get_column_mean(df, column))
    return df


def normalize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and normalise it between 0 and 1.
    :param df_column: Dataset's column
    :return: The column normalized
    """
    try:
        df_column = preprocessing.MinMaxScaler().fit_transform(df_column.values.reshape(-1, 1))
    except ValueError:
        print("Error: Text column cannot be normalized")
    else:
        return df_column


def standardize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and standardize it between -1 and 1 with its average at 0.
    :param df_column: Dataset's column
    :return: The column standardized
    """
    if (df_column.dtype == 'int64') or (df_column.dtype == 'float64'):
        df_column = np.abs(stats.zscore(df_column))
    else:
        print("Column is not a number column")

    return df_column


def calculate_numeric_distance(df_column_1: pd.Series, df_column_2: pd.Series,
                               distance_metric: DistanceMetric) -> pd.Series:
    """
    This method should calculate the distance between two numeric columns
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :param distance_metric: One of DistanceMetric, and for each one you should implement its logic
    :return: A new 'column' with the distance between the two inputted columns
    """
    dist_array = []
    rows = len(df_column_1)
    if distance_metric.value == 0:
        for row in range(rows):
            dist_array.append(np.linalg.norm(df_column_1[row] - df_column_2[row]))
    elif distance_metric.value == 1:
        for i in range(rows):
            dist_array.append(abs(df_column_1[i]) + abs(df_column_2[i]))
    return pd.Series(dist_array)


def calculate_binary_distance(df_column_1: pd.Series, df_column_2: pd.Series) -> pd.Series:
    """
    This method should calculate the distance between two binary columns.
    Choose one of the possibilities shown in class for future experimentation.
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :return: A new 'column' with the distance between the two inputted columns
    """
    rows = len(df_column_1)
    dist_array = []
    for row in range(rows):
        dist_array.append(hamming(df_column_1[row], df_column_2[row])*len(df_column_1))
    return pd.Series(dist_array)


if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 2, 3, None], 'b': [True, True, False, None], 'c': ['one', 'two', np.nan, None]})
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_LESS_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_GREATER_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_POSITIVE, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_NEGATIVE, 2) is not None
    assert fix_outliers(df, 'c') is not None
    assert fix_nans(df, 'a') is not None
    try:
        assert normalize_column(df.loc[:, 'c']) is not None
    except AssertionError:
        pass
    try:
        assert standardize_column(df.loc[:, 'a']) is not None
    except AssertionError:
        pass
    assert calculate_numeric_distance(df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.EUCLIDEAN) is not None
    assert calculate_numeric_distance(df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.MANHATTAN) is not None
    assert calculate_binary_distance(df.loc[:, 'b'], df.loc[:, 'b']) is not None
    print("ok")
