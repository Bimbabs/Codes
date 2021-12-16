import glob

import numpy as np
import pandas as pd

from imports import *
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

"""
Regression is a supervised form of machine learning, which means that it uses labeled data to train a model that 
can predict or forecast an output value for the given input/unseen features.

In this subtask, you'll be training and testing 2 different types of regression models, each with 2 different types of train-test splits and
compare the performance of same for a single different dataset which was downloaded from below given link. 

The dataset contains 36733 instances of 11 sensor measures aggregated over one hour (by means of average or sum) from a gas turbine 
located in Turkey's north western region for the purpose of studying flue gas emissions, namely CO and NOx (NO + NO2).
We will be predicting the "Carbon monoxide (CO)" emissions by this gas turbine using features that represent sensor measurements.
This dataset is split into 5 different files based on year. Read all 5 files and combine them in your code before
running regression algorithms.

Dataset1: "Gas Turbine CO and NOx Emission Data Set Data Set" 
https://archive.ics.uci.edu/ml/datasets/Gas+Turbine+CO+and+NOx+Emission+Data+Set
"""


#######################################################
# Read the comments carefully in the following method(s)
# Assignment questions are marked as "??subtask"
#######################################################


# Use case 1: 80/20 Data split - randomly
# MODEL_1:
def svm_regressor_1(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a Linear Support Vector Machine for given dataset.
    Refer the official documentation below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
    :param x: data features
    :param y: data outputs
    :return: trained_model and below given performance_scores for test dataset
    """
    # ??subtask1: Split the data into 80%-20% train-test sets.
    #             Randomize the data selection.i.e. train and test data shoud be randomly selected in 80/20 ratio.
    #             Use a random_state=42, so that we can recreate same splitting when run multiple times.
    # ??subtask2: Train your model using train set.
    # ??subtask3: Predict test output value for test set.
    # ??subtask4: Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to perform train-test split and measure below asked model performance scores.

    # performance measure https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b
    # mean_squared_error, mean_absolute_error

    # -----My code--------
    # subtask1 - randomly split dataset into train and test data
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # subtask2 - train with train set
    # random state is the pseudo random number used for generating the data
    # tol is tolerance for stopping criteria
    svm1_model = LinearSVR(random_state=0, tol=1e-5)
    svm1_model = svm1_model.fit(train_x, train_y)

    # subtask3 - Predict for test case
    predicted = svm1_model.predict(test_x)

    # subtask4 - Measure performance
    svm1_mse = mean_squared_error(test_y, predicted)
    svm1_mae = mean_absolute_error(test_y, predicted)

    return dict(model=svm1_model, mse=svm1_mse, mae=svm1_mae)


# Use case 1: 80/20 Data split - randomly
# MODEL_2:
def random_forest_regressor_1(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a Random Forest regression model for given dataset.
    Refer the official documentation below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    :param x: data features
    :param y: data outputs
    :return: trained_model and below given performance_scores for test dataset
    """
    # ??subtask1: Split the data into 80%-20% train-test sets.
    #             Randomize the data selection.i.e. train and test data shoud be randomly selected in 80/20 ratio.
    #             Use a random_state=42, so that we can recreate same splitting when run multiple times.
    # ??subtask2: Train your model using train set.
    # ??subtask3: Predict test output value for test set.
    # ??subtask4: Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to perform train-test split and measure below asked model performance scores.

    # performance measure https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b
    # mean_squared_error, mean_absolute_error

    # -----My code--------
    # subtask1 - randomly split dataset into train and test data
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # subtask2
    rfc = RandomForestRegressor(max_depth=2, random_state=0)
    rf1_model = rfc.fit(train_x, train_y)

    # subtask3 - Predict for test case
    predicted = rf1_model.predict(test_x)

    # subtask4 - Measure performance
    rf1_mse = mean_squared_error(test_y, predicted)
    rf1_mae = mean_absolute_error(test_y, predicted)

    return dict(model=rf1_model, mse=rf1_mse, mae=rf1_mae)


# Use Case 1: 80/20 Data split - randomly
def run_regression_models_1(x, y):
    """
    This method takes input features and labels and calls all the functions which trains and tests the above 2 regression models.
    :param x: data features
    :param y: data outputs
    :return: results after training and testing above 2 regression models
    """
    # ??subtask1: Drop the "Year" column from x that was added in "__main__" method during the process of file reading.
    #  It is important to remove this newly added "Year" column before training since this will add bias to data
    #  and is intended to be used only for train-test splitting in use case 2.

    x.drop(labels='Year', inplace=True, axis=1)
    r1 = svm_regressor_1(x, y)
    assert len(r1.keys()) == 3
    r2 = random_forest_regressor_1(x, y)
    assert len(r2.keys()) == 3

    return r1, r2


# Use Case 2: Data split based on year
# MODEL_3:
def svm_regressor_2(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a Linear Support Vector Machine for given dataset.
    Refer the official documentation below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
    :param x: data features
    :param y: data outputs
    :return: trained_model and below given performance_scores for test dataset
    """
    # ??subtask1: Dataset has 5 different files in a folder "pp_gas_emission" ==> ["gt_2011.csv", "gt_2012.csv",
    #             "gt_2013.csv", "gt_2014.csv", "gt_2015.csv"]
    #             Read all the files and split the data such that data form ["gt_2011.csv", "gt_2012.csv",
    #             "gt_2013.csv", "gt_2014.csv"] is used as train set
    #             and data from ["gt_2015.csv"] is used as test set.
    #             Hint: Use "Year" column which was added in "__main__" method to make this train-test split.Sklearn
    #             can't be used here. You need to do it using a logic.
    #            **Since y is pandas.Series and dosent have "Year" column, use Index of x to split y
    #            into y_train/y_test sets**
    # ??subtask2: Drop the "Year" column from x.
    #             It is important to remove this newly added "Year" column before training since this will add
    #             bias to data
    #             and is intended to be used only for train-test splitting in use case 2.
    # ??subtask3: Train your model using train set.
    # ??subtask3: Predict test output value for test set.
    # ??subtask5: Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to measure below asked model performance scores.
    # performance measure https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b
    # mean_squared_error, mean_absolute_error

    """subtask1 - train with train set"""
    # convert series y to list
    target_series = y.tolist()

    # get train data from year 2011-2014
    train_x = x[x['Year'] != "2015"]
    train_x_index = list(np.where(x['Year'] != "2015"))
    index_x_train = np.asarray(train_x_index).flatten()
    y_train = []
    for i in index_x_train:
        y_train.append(target_series[i])

    # get train data for year 2015
    test_x = x[x['Year'] == "2015"]
    test_x_index = list(np.where(x['Year'] == "2015"))
    index_x_test = np.asarray(test_x_index).flatten()
    y_test = []

    for i in index_x_test:
        y_test.append(target_series[i])

    train_y = pd.Series(y_train)
    test_y = pd.Series(y_test)

    """subtask2 - Drop the "Year" column from train and test set."""
    train_x.drop(labels='Year', inplace=True, axis=1)
    test_x.drop(labels='Year', inplace=True, axis=1)

    """
    subtask3 - train with train set
    random state is the pseudo random number used for generating the data
    tol is tolerance for stopping criteria"""
    lsvr2 = LinearSVR(random_state=0, tol=1e-5)
    svm2_model = lsvr2.fit(train_x, train_y)

    # subtask4 - Predict for test case
    predicted = svm2_model.predict(test_x)

    # subtask - Measure performance
    svm2_mse = mean_squared_error(test_y, predicted)
    svm2_mae = mean_absolute_error(test_y, predicted)

    return dict(model=svm2_model, mse=svm2_mse, mae=svm2_mae)


# Use Case 2: Data split based on year
# MODEL_3:
def random_forest_regressor_2(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a Linear Support Vector Machine for given dataset.
    Refer the official documentation below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
    :param x: data features
    :param y: data outputs
    :return: trained_model and below given performance_scores for test dataset
    """
    # ??subtask1: Dataset has 5 different files in a folder "pp_gas_emission" ==> ["gt_2011.csv", "gt_2012.csv",
    # "gt_2013.csv", "gt_2014.csv", "gt_2015.csv"]
    #             Read all the files and split the data such that data form ["gt_2011.csv", "gt_2012.csv",
    #             "gt_2013.csv", "gt_2014.csv"] is used as train set
    #             and data from ["gt_2015.csv"] is used as test set.
    #             Hint: Use "Year" column which was added in "__main__" method to make this train-test split.
    #                   Sklearn can't be used here. You need to do it using a logic.
    # ??subtask2: Drop the "Year" column from x.
    #             It is important to remove this newly added "Year" column before training since this will add
    #             bias to data
    #             and is intended to be used only for train-test splitting in use case 2.
    # ??subtask3: Train your model using train set.
    # ??subtask4: Predict test output value for test set.
    # ??subtask5: Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to measure below asked model performance scores.

    # performance measure https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b
    # mean_squared_error, mean_absolute_error

    # ---My Code --------
    """subtask1 - train with train set"""
    # convert series y to list
    target_series = y.tolist()

    # get train data from year 2011-2014
    train_x = x[x['Year'] != "2015"]
    train_x_index = list(np.where(x['Year'] != "2015"))
    index_x_train = np.asarray(train_x_index).flatten()
    y_train = []
    for i in index_x_train:
        y_train.append(target_series[i])

    # get train data for year 2015
    test_x = x[x['Year'] == "2015"]
    test_x_index = list(np.where(x['Year'] == "2015"))
    index_x_test = np.asarray(test_x_index).flatten()
    y_test = []

    for i in index_x_test:
        y_test.append(target_series[i])

    train_y = pd.Series(y_train)
    test_y = pd.Series(y_test)

    """subtask2 - Drop the "Year" column from train and test set."""
    train_x.drop(labels='Year', inplace=True, axis=1)
    test_x.drop(labels='Year', inplace=True, axis=1)

    """
    subtask3 - train with train set
    random state is the pseudo random number used for generating the data
    tol is tolerance for stopping criteria"""
    rfc2 = RandomForestRegressor(max_depth=2, random_state=0)
    rf2_model = rfc2.fit(train_x, train_y)

    # subtask4 - Predict for test case
    predicted = rf2_model.predict(test_x)

    # subtask - Measure performance
    rf2_mse = mean_squared_error(test_y, predicted)
    rf2_mae = mean_absolute_error(test_y, predicted)

    return dict(model=rf2_model, mse=rf2_mse, mae=rf2_mae)


# Use Case 2: Data split based on year
def run_regression_models_2(x, y):
    """
    This method takes input features and labels and calls all the functions which trains and tests the above 2 regression models.
    No need to do anything here.
    :param x: data features
    :param y: data outputs
    :return: results after training and testing above 2 regression models
    """
    r1 = svm_regressor_2(x, y)
    assert len(r1.keys()) == 3
    r2 = random_forest_regressor_2(x, y)
    assert len(r2.keys()) == 3

    return r1, r2


def run_regression():
    start = time.time()
    print("Regression in progress...")
    # Dataset1:
    # gt_emission_dataset = read_dataset(Path('../../iris.csv'))

    # gt_emission_dataset = pd.DataFrame({"test_col": [1, 2], "CO": [3, 4]})
    # ??subtask1 Read dataset from Folder "pp_gas_emission" which has 5 data files ["gt_2011.csv", "gt_2012.csv",
    # "gt_2013.csv", "gt_2014.csv", "gt_2015.csv"]
    #            and store it in gt_emission_dataset in above line.
    # ??subtask2 Add a new column "Year" based on the file which you are reading.
    # ??subtask3 combine the above datasets from 5 different files into a single pandas dataframe.
    # ??subtask4 Drop "Nitrogen oxides (NOx)" from the dataframe since we will not be predicting the emission of
    # Nitrogen oxides.
    # ??subtask5 Normalize the DataFrame such that all the columns have data in range[0,1]. Use methods from A1
    # for this.

    # --My Code ----
    """
    subtask1 and subtask2
    read files from pp_gas_emission folder, and add 'Year' column
    """
    path = Path('../../pp_gas_emission')
    dir_list = glob.glob(os.path.join(path, "*.csv"))
    dataframes_list = []
    for file in dir_list:
        data = pd.read_csv(file)
        year = file[-8:]
        year = year[:4]
        data['Year'] = year
        dataframes_list.append(data)
    """
    subtask3 - combine all dataframes into a single dataframe
    """
    pp_raw_data = pd.concat(dataframes_list)
    """
    subtask4 - combine all dataframes into a single dataframe
    """
    pp_raw_data.drop(labels='NOX', inplace=True, axis=1)

    """
    subtask5 - Normalize the DataFrame such that all the columns have data in range[0,1].
    """
    # fix outliers and nans
    numeric_columns = get_numeric_columns(pp_raw_data)
    for nc in numeric_columns:
        pp_raw_data = fix_outliers(pp_raw_data, nc)
        pp_raw_data = fix_nans(pp_raw_data, nc)

    # Normalizing the Numeric Columns
    for nc_column in numeric_columns:
        pp_raw_data.loc[:, nc_column] = normalize_column(pp_raw_data.loc[:, nc_column])

    """
    fix wrong values outside of [0, 1] by replacing with the mean of the column. This is done after normalizing the 
    column so that the new replacement value falls between 0 and 1
    """
    for nc_column in numeric_columns:
        pp_raw_data.loc[(pp_raw_data[nc_column] < 0) | (pp_raw_data[nc_column] > 1), nc_column]\
            = pp_raw_data[nc_column].mean()

    # create copy for usecase1
    gt_emission_dataset_copy1 = pp_raw_data.copy()
    output_col = "CO"
    feature_cols = gt_emission_dataset_copy1.columns.tolist()
    feature_cols.remove(output_col)
    x_gt_emission = gt_emission_dataset_copy1[feature_cols]
    y_gt_emission = gt_emission_dataset_copy1[output_col]

    result1, result2 = run_regression_models_1(x_gt_emission, y_gt_emission)
    # Observe both the results and notice which model is preforming better in this use case.
    """
    ****My Observation*******
    The mean square error is an indication of how close a regression line is to a set of points.
    The SVM regressor1 has a lower mean square value than the Random Forest Regressor. 
    This shows that the SVM is a better model
    """
    print(f"{10*'*'}Dataset1{gt_emission_dataset_copy1.shape}, usecase1:{10*'*'}\nSVM regressor 1: {result1}\nRandom Forest regressor 1: {result2}\n")

    # create copy for usecase2
    gt_emission_dataset_copy2 = pp_raw_data.copy()
    output_col = "CO"
    feature_cols = gt_emission_dataset_copy2.columns.tolist()
    feature_cols.remove(output_col)
    x_gt_emission = gt_emission_dataset_copy2[feature_cols]
    y_gt_emission = gt_emission_dataset_copy2[output_col]

    result1, result2 = run_regression_models_2(x_gt_emission, y_gt_emission)
    # Observe both the results and notice which model is preforming better in this use case.
    """
    ****My Observation*******
    The SVM regressor2 is a better model than the Random Forest Regressor because it has a lower mse value
    """
    print(f"{10*'*'}Dataset1{gt_emission_dataset_copy1.shape}, usecase2:{10*'*'}\nSVM regressor 2: {result1}\nRandom Forest regressor 2: {result2}\n")

    end = time.time()
    run_time = round(end - start, 4)
    print("Regression ended...")
    print(f"{30*'-'}\nRegression run_time:{run_time}s\n{30*'-'}\n")


if __name__ == "__main__":
    run_regression()

#Since we are using same parameter names (x: pd.DataFrame, y: pd.Series) in most methods, remember to copy the df passed as parameter 
# and work with the df_copy to avoid warnings and unexpected results. Its a standard practice!
