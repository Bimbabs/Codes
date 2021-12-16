from imports import *

"""
The below method should:
?? subtask1  Handle any dataset (if you think worthwhile, you should do some pre-processing)
?? subtask2  Generate a (classification, regression or clustering) model based on the label_column 
             and return the one with best score/accuracy

The label_column can be categorical, numerical or None
-If categorical, run through ML classifiers in "a_classification" file and return the one with highest accuracy: 
    DecisionTree, RandomForestClassifier, KNeighborsClassifier or NaiveBayes
-If numerical, run through these ML regressors in "b_regression" file and return the one with least MSE error: 
    svm_regressor_1(), random_forest_regressor_1()
-If None, run through simple_k_means() and custom_clustering() and return the one with highest silhouette score.
(https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
"""


def generate_model(df: pd.DataFrame, label_column: Optional[pd.Series] = None) -> Dict:
    # model_name is the type of task that you are performing.
    # Use sensible names for model_name so that we can understand which ML models if executed for given df.
    # ex: Classification_random_forest or Regression_SVM.
    # model is trained model from ML process
    # final_score will be Accuracy in case of Classification, MSE in case of Regression and silhouette score in case of clustering.
    # your code here.
    return dict(model_name=None, model=None, final_score=None)


def run_custom():
    start = time.time()
    print("Custom modeling in progress...")
    df = pd.DataFrame()  # Markers will run your code with a separate dataset unknown to you.
    result = generate_model(df)
    print(f"result:\n{result}\n")

    end = time.time()
    run_time = round(end - start)
    print("Custom modeling ended...")
    print(f"{30 * '-'}\nCustom run_time:{run_time}s\n{30 * '-'}\n")


if __name__ == "__main__":
    run_custom()
