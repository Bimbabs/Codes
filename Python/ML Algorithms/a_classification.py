from assignment2.imports import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

"""
Classification is a supervised form of machine learning, which means that it uses labeled data to train a model that 
can predict the category/class of the given input/unseen data.

In this subtask, you'll be training and testing 4 different types of classification models and compare the 
performance of same for 3 different datasets. Dataset1: Iris dataset from Assignment1 before normalization. Dataset2: 
Iris dataset from Assignment1 after normalization. Notice what change normalization does to the prediction resutls! 
Dataset3: Life_expectancy dataset from Assignment1 after label encoding and normalization. """


#######################################################
# Read the comments carefully in the following method(s)
# Assignment questions are marked as "??subtask"
#######################################################


# MODEL_1:
def decision_tree_classifier(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a Decision Tree classifier for given dataset.
    Refer the official documention below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    :param x: data features
    :param y: data labels/class
    :return: trained_model and below given performace_scores for test dataset
    """

    # ??subtask1: Split the data into 80%-20% train-test sets.
    #             Randomize the data selection.i.e. train and test data shoud be randomly selected in 80/20 ratio.
    #             Use a random_state=42, so that we can recreate same splitting when run multiple times.
    # ??subtask2: Train your model using train set.
    # ??subtask3: Predict test labels/classes for test set.
    # ??subtask4: Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to perform train-test split and measure below asked model performance scores.

    # performance measure https://towardsdatascience.com/performance-metrics-for-classification-machine-learning
    # -problems-97e7e774a007 confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

    # -----My code--------
    # subtask1 - randomly split dataset into train and test data
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # subtask2 - train with train set
    dt_model = DecisionTreeClassifier()
    dt_model = dt_model.fit(train_x, train_y)

    # subtask3 - Predict for test case
    predicted_species = dt_model.predict(test_x)

    # subtask4 - Measure performance
    dt_confusion_matrix = confusion_matrix(test_y, predicted_species)
    dt_accuracy = accuracy_score(test_y, predicted_species)

    # The average parameter is required for multiclass targets. The value 'None' returns the scores for each class
    # zero_division sets the value to return as 0 when denominator is zero
    dt_precision = precision_score(test_y, predicted_species, average=None, zero_division=0)
    dt_recall = recall_score(test_y, predicted_species, average=None, zero_division=0)
    dt_f1_score = f1_score(test_y, predicted_species, average=None, zero_division=0)

    # --------------------

    return dict(model=dt_model, confusion_matrix=dt_confusion_matrix, accuracy=dt_accuracy, precision=dt_precision,
                recall=dt_recall, f1_score=dt_f1_score)


# MODEL_2:
def random_forest_classifier(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a random forest classifier for given data.
    Refer the official documention below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    In this api, "n_estimator" decides number of trees in the forest, lower value means lower trees and computation
    with the default values of these parameters, the model may take a lot time for computation.
    If necessary, change the "n_estimators", "max_depth" and "max_leaf_nodes" to accelerate the model
    training, but don't forget to comment why you did and any consequences of setting them!

    :param x: data features
    :param y: data labels/class
    :return: trained_model and below given performace_scores for test dataset
    """

    # ??subtask1: Split the data into 80%-20% train-test sets.
    #             Randomize the data selection.i.e. train and test data shoud be randomly selected in 80/20 ratio.
    #             Use a random_state=42, so that we can recreate same splitting when run multiple times.
    # ??subtask2: Train your model using train set.
    # ??subtask3: Predict test labels/classes for test set.
    # ??subtask4: Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to perform train-test split and measure below asked model performance scores.

    # performance measure https://towardsdatascience.com/performance-metrics-for-classification-machine-learning
    # -problems-97e7e774a007
    # confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

    # -----My code--------
    # subtask1 - randomly split dataset into train and test data
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # subtask2
    rfc = RandomForestClassifier(n_estimators=2)  # defines the number of trees in the forest

    rf_model = rfc.fit(train_x, train_y)

    # subtask3 - Predict for test case
    predicted = rfc.predict(test_x)

    # subtask4 - Measure performance
    rf_confusion_matrix = confusion_matrix(test_y, predicted)
    rf_accuracy = accuracy_score(test_y, predicted)

    # The average parameter is required for multiclass targets. The value 'None' returns the scores for each class
    # zero_division sets the value to return as 0 when denominator is zero
    rf_precision = precision_score(test_y, predicted, average=None, zero_division=0)
    rf_recall = recall_score(test_y, predicted, average=None, zero_division=0)
    rf_f1_score = f1_score(test_y, predicted, average=None, zero_division=0)

    return dict(model=rf_model, confusion_matrix=rf_confusion_matrix, accuracy=rf_accuracy, precision=rf_precision,
                recall=rf_recall, f1_score=rf_f1_score)


# MODEL_3:
def knn_classifier(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a K-Nearest Neighbors(KNN) classifier for given data.
    Refer the official documention below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    :param x: data features
    :param y: data labels/class
    :return: trained_model and below given performace_scores for test dataset
    """

    # ??subtask1: Split the data into 80%-20% train-test sets.
    #             Randomize the data selection.i.e. train and test data shoud be randomly selected in 80/20 ratio.
    #             Use a random_state=42, so that we can recreate same splitting when run multiple times.
    # ??subtask2: Train your model using train set.
    # ??subtask3: Predict test labels/classes for test set.
    # ??subtask4: Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to perform train-test split and measure below asked model performance scores.

    # performance measure https://towardsdatascience.com/performance-metrics-for-classification-machine-learning-
    # problems-97e7e774a007
    # confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

    # -----My code--------
    # subtask1 - randomly split dataset into train and test data
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # subtask2
    knn = KNeighborsClassifier(n_neighbors=5)
    knn_model = knn.fit(train_x, train_y)

    # subtask3 - Predict for test case
    predicted = knn_model.predict(test_x)

    # subtask4 - Measure performance
    knn_confusion_matrix = confusion_matrix(test_y, predicted)
    knn_accuracy = accuracy_score(test_y, predicted)

    knn_precision = precision_score(test_y, predicted, average=None, zero_division=0)
    knn_recall = recall_score(test_y, predicted, average=None, zero_division=0)
    knn_f1_score = f1_score(test_y, predicted, average=None, zero_division=0)

    return dict(model=knn_model, confusion_matrix=knn_confusion_matrix, accuracy=knn_accuracy, precision=knn_precision,
                recall=knn_recall, f1_score=knn_f1_score)


# MODEL_4:
def naive_bayes_classifier(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a Naive Bayes classifier for given data.
    Refer the official documention below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/naive_bayes.html

    :param x: data features
    :param y: data labels/class
    :return: trained_model and below given performace_scores for test dataset
    """

    # ??subtask1: Split the data into 80%-20% train-test sets.
    #             Randomize the data selection.i.e. train and test data shoud be randomly selected in 80/20 ratio.
    #             Use a random_state=42, so that we can recreate same splitting when run multiple times.
    # ??subtask2: Train your model using train set.
    # ??subtask3: Predict test labels/classes for test set.
    # ??subtask4: Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to perform train-test split and measure below asked model performance scores.

    # performance measure https://towardsdatascience.com/performance-metrics-for-classification-machine-learning-
    # problems-97e7e774a007
    # confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    # -----My code--------
    # subtask1 - randomly split dataset into train and test data
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # subtask2
    nbc = GaussianNB()
    nbc = nbc.fit(train_x, train_y)

    # subtask3 - Predict for test case
    nv_model = nbc.predict(test_x)

    # subtask4 - Measure performance
    nv_confusion_matrix = confusion_matrix(test_y, nv_model)
    nv_accuracy = accuracy_score(test_y, nv_model)

    nv_precision = precision_score(test_y, nv_model, average=None, zero_division=0)
    nv_recall = recall_score(test_y, nv_model, average=None, zero_division=0)
    nv_f1_score = f1_score(test_y, nv_model, average=None, zero_division=0)

    return dict(model=nv_model, confusion_matrix=nv_confusion_matrix, accuracy=nv_accuracy, precision=nv_precision,
                recall=nv_recall, f1_score=nv_f1_score)


def run_classification_models(x, y):
    """
    This method takes input features and labels and calls all the functions which trains and tests the above 4
    Classification models.
    No need to do anything here.
    """
    r1 = decision_tree_classifier(x, y)
    assert len(r1.keys()) == 6
    r2 = random_forest_classifier(x, y)
    assert len(r2.keys()) == 6
    r3 = knn_classifier(x, y)
    assert len(r3.keys()) == 6
    r4 = naive_bayes_classifier(x, y)
    assert len(r4.keys()) == 6

    return r1, r2, r3, r4


def run_classification():
    start = time.time()
    print("Classification in progress...")
    # Dataset1:
    iris = read_dataset(Path('../../iris.csv'))
    iris_copy = iris.copy()
    label_col = "species"
    feature_cols = iris_copy.columns.tolist()
    feature_cols.remove(label_col)
    x_iris = iris_copy[feature_cols]
    y_iris = iris_copy[label_col]
    result1, result2, result3, result4 = run_classification_models(x_iris, y_iris)

    # Observe all 4 results and notice which model is preforming better.
    """
    For the first Dataset, all the models performed the same. 
    They all had the same accuracy, precision, recall and f1_score.
    None was better than the other
    """
    print(
        f"{10 * '*'}Dataset1:{iris.shape}{10 * '*'}\nDecision Tree: {result1}\nRandom forest: {result2}\nKNN: {result3}\nNaive Bayes: {result4}\n")

    # Dataset2:
    iris_again = process_iris_dataset_again()
    """
    ??subtask1: process iris_again dataset returned from A1 process_iris_dataset_again() method so that all the categorical columns are label encoded.
    ??subtask2: make sure that all the feature columns are normalized to range [0,1] except label encoded "species" column. These are lables to be predicted
    Use methods from Assignment1 for this.
    """
    label_col = "species"
    feature_cols = iris_again.columns.tolist()
    feature_cols.remove(label_col)
    x_iris_again = iris_again[feature_cols]

    # -----My Code -------
    """
    subtask1
    The species column is encoded and stored as pd.series in y_iris_again
    """
    iris_again = replace_with_label_encoder(iris_again, label_col, generate_label_encoder(iris_again[label_col]))
    y_iris_again = iris_again[label_col]

    """
        subtask2- Normalizing the Numeric Columns       
        The feature cols in x_iris_again are normalized to [0,1]
    """
    for nc_column in feature_cols:
        x_iris_again.loc[:, nc_column] = normalize_column(x_iris_again.loc[:, nc_column])

    result1, result2, result3, result4 = run_classification_models(x_iris_again, y_iris_again)
    # Observe all 4 results and notice which model is preforming better.
    """
    *****My observation********
    For the first Dataset, all the models performed the same. 
    They all had the same accuracy, precision, recall and f1_score.
    None was better than the other
    """
    print(
        f"{10 * '*'}Dataset2:{iris_again.shape}{10 * '*'}\nDecision Tree: {result1}\nRandom forest: {result2}\nKNN: {result3}\nNaive Bayes: {result4}\n")

    # Dataset3:
    life_expectancy = process_life_expectancy_dataset()
    """
    ??subtask1: process life_expectancy dataset returned from A1 process_life_expectancy_dataset() method so that all the categorical columns are label encoded.
    ??subtask2: make sure that all the feature columns are normalized to range [0,1] except label encoded "Latitude" column. These are lables to be predicted. (0/1-North/South)
    Use methods from Assignment1 for this.
    """
    label_col = "Latitude"
    feature_cols = life_expectancy.columns.tolist()
    feature_cols.remove(label_col)

    # ------My Code----------------------
    """    
    subtask1: process life_expectancy dataset returned from A1 process_life_expectancy_dataset() method 
    so that all the categorical columns are label encoded.
    """
    categorical_columns = get_text_categorical_columns(life_expectancy)
    for cc in categorical_columns:
        le = generate_label_encoder(life_expectancy.loc[:, cc])
        life_expectancy = replace_with_label_encoder(life_expectancy, cc, le)

    """    
    subtask2: make sure that all the feature columns are normalized to range [0,1] except label encoded "Latitude" 
    column. These are labels to be predicted. (0/1-North/South)
    """
    for nc_column in feature_cols:
        life_expectancy.loc[:, nc_column] = normalize_column(life_expectancy.loc[:, nc_column])

    x_life_expectancy = life_expectancy[feature_cols]
    y_life_expectancy = life_expectancy[label_col]

    result1, result2, result3, result4 = run_classification_models(x_life_expectancy, y_life_expectancy)

    # Observe all 4 results and notice which model is preforming better.
    """
    ******My Observation********
    Accuracy: Naive Bayes performed the poorest with an accuracy of 0.44. The other 3 had high accuracies of 0.99 but 
    considering the number of decimal places, Decision Tree has the better accuracy.
    Precision: Decision Tree has the highest precision followed by KNN. Naive Bayes has the lowest precision
    Recall: KNN has the highest recall  
    """
    print(
        f"{10 * '*'}Dataset3:{life_expectancy.shape}{10 * '*'}\nDecision Tree: {result1}\nRandom forest: {result2}\nKNN: {result3}\nNaive Bayes: {result4}\n")

    end = time.time()
    run_time = round(end - start, 4)
    print("Classification ended...")
    print(f"{30 * '-'}\nClassification run_time:{run_time}s\n{30 * '-'}\n")


if __name__ == "__main__":
    run_classification()

# Since we are using same parameter names (x: pd.DataFrame, y: pd.Series) in most methods, remember to copy
# the df passed as parameter
# and work with the df_copy to avoid warnings and unexpected results. Its a standard practice!
