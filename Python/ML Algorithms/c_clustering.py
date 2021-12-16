from assignment2.imports import *
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import OPTICS, DBSCAN, AffinityPropagation, SpectralClustering
from sklearn import metrics
"""
Clustering is an unsupervised form of machine learning. It uses unlabeled data and returns the similarity/dissimilarity
 between rows of the data.
See https://scikit-learn.org/stable/modules/clustering.html for an overview of methods in sklearn.
"""


#######################################################
# Read the comments carefully in the following method(s)
# Assignment questions are marked as "??subtask"
#######################################################


def simple_k_means(x: pd.DataFrame, n_clusters=3, score_metric='euclidean') -> Dict:
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit(x)

    # There are many methods of deciding a score of a cluster model. Here is one example:
    score = metrics.silhouette_score(x, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)


def iris_clusters() -> Dict:
    """
    Let's use the iris dataset and clusterise it:
    """
    iris = process_iris_dataset_again()
    # iris.drop("large_sepal_length", axis=1, inplace=True)

    # Let's generate the clusters considering only the numeric columns first
    no_species_column = simple_k_means(iris.iloc[:, :4])

    ohe = generate_one_hot_encoder(iris['species'])
    df_ohe = replace_with_one_hot_encoder(iris, 'species', ohe, list(ohe.get_feature_names()))

    # Notice that here I have binary columns, but I am using euclidean distance to do the clustering AND
    # score evaluation
    # This is pretty bad
    no_binary_distance_clusters = simple_k_means(df_ohe)

    # Finally, lets use just a label encoder for the species.
    # It is still bad to change the labels to numbers directly because the distances between them does not make sense
    le = generate_label_encoder(iris['species'])
    df_le = replace_with_label_encoder(iris, 'species', le)
    labeled_encoded_clusters = simple_k_means(df_le)

    # See the result for yourself:
    r1 = round(no_species_column['score'], 2)
    r2 = round(no_binary_distance_clusters['score'], 2)
    r3 = round(labeled_encoded_clusters['score'], 2)
    print(f"Clustering Scores:\nno_species_column:{r1}, no_binary_distance_clusters:{r2}, labeled_encoded_clusters:{r3}")

    return max(r1, r2, r3)


##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def custom_clustering(x: pd.DataFrame) -> Dict:
    """
    As you saw before, it is much harder to apply the right distance metrics. Take a look at:
    https://scikit-learn.org/stable/modules/clustering.html
    and check the metric used for each implementation. You will notice that suppositions were made,
    which makes harder to apply these clustering algorithms as-is due to the metrics used.
    Also go into each and notice that some of them there is a way to choose a distance/similarity/affinity metric.
    You don't need to check how each technique is implemented (code/math), but do use the information from the clustering
    lecture and check the parameters of the method (especially if there is any distance metric available among them).
    Chose one of them which is, in your opinion, generic enough, and justify your choice with a comment in the code (1 sentence).
    The return of this method should be the model, a score (e.g. silhouette
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) and the result of clustering the
    input dataset.
    """
    # the number of clusters is defined as 3, the total number of species. This is to override the
    # default n_clusters=2 for AgglomerativeClustering. I chose this algorithm because the number of clusters
    # does not need to be defined. it has a default value when none is provided
    model = AgglomerativeClustering(n_clusters=3)
    clusters = model.fit(x)
    score = metrics.silhouette_score(x, model.labels_, metric=model.affinity)
    return dict(model=model, score=score, clusters=clusters)


def cluster_iris_dataset_again() -> Dict:
    """
    Run the df returned form process_iris_dataset_again() method of A1 e_experimentation file through the custom_clustering and discuss (3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    # The new dataset has been processed with label encoder
    df1 = process_iris_dataset_again()
    ohe = generate_one_hot_encoder(df1['species'])
    df_ohe = replace_with_one_hot_encoder(df1, 'species', ohe, list(ohe.get_feature_names()))
    c = custom_clustering(df_ohe)
    """
    Result Discussion
    * if number of clusters is not defined, there is default value set as 2. This means points that are similar will be classified as same,
      even if they are different
    * As the number of clusters increase, the score tends to 0 which implies overlapping of the points
    * After testing with different number of clusters (2, 3, 4) using different distance metrics (euclidean and manhattan),
    result showed that the score of model with n_clusters_=3 is ~0.5, which is mid range showing that the clusters are 
    evenly dispersed
    
    Limitation
    The default n_clusters_=2. This means that cluster number was be provided to the model for more accuracy 
    """
    return dict(model=c['model'], score=c['score'], clusters=c['clusters'])


def cluster_amazon_video_game() -> Dict:
    """
    Run the df returned from process_amazon_video_game_dataset() methods of A1 e_experimentation file through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df2 = process_amazon_video_game_dataset()
    numeric_columns = get_numeric_columns(df2)
    for nc in numeric_columns:
        df2 = fix_outliers(df2, nc)
        df2 = fix_nans(df2, nc)

    # Normalizing the Numeric Columns
    for nc_column in numeric_columns:
        df2.loc[:, nc_column] = normalize_column(df2.loc[:, nc_column])

    # label encoding for user column
    le = generate_label_encoder(df2['asin'])
    df_le = replace_with_label_encoder(df2, 'asin', le)
    c = custom_clustering(df_le)
    return dict(model=c['model'], score=c['score'], clusters=c['clusters'])


def cluster_amazon_video_game_again() -> Dict:
    """
    Run the df returned from process_amazon_video_game_dataset_again() methods of A1 e_experimentation file through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    # the limitation experienced is the size of the data. I had to take a sample of the data in order to get some result
    df3 = process_amazon_video_game_dataset_again()
    numeric_columns = get_numeric_columns(df3)
    for nc in numeric_columns:
        df3 = fix_outliers(df3, nc)
        df3 = fix_nans(df3, nc)

    # Normalizing the Numeric Columns
    for nc_column in numeric_columns:
        df3.loc[:, nc_column] = normalize_column(df3.loc[:, nc_column])

    # label encoding for user column
    le = generate_label_encoder(df3['user'])
    df_le = replace_with_label_encoder(df3, 'user', le)
    c = custom_clustering(df_le)
    return dict(model=c['model'], score=c['score'], clusters=c['clusters'])


def cluster_life_expectancy() -> Dict:
    """
    Run the df returned from process_amazon_video_game_dataset_again() methos of A1 e_experimentation file through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df4 = process_life_expectancy_dataset()
    # Take a sample of the dataset to reduce training time
    df4 = df4.sample(frac=0.5, replace=True, random_state=1)
    numeric_columns = get_numeric_columns(df4)
    for nc in numeric_columns:
        df4 = fix_outliers(df4, nc)
        df4 = fix_nans(df4, nc)

    # Normalizing the Numeric Columns
    for nc_column in numeric_columns:
        df4.loc[:, nc_column] = normalize_column(df4.loc[:, nc_column])

    # I initially used one hot encoding for the country column. I ended up with about 190 columns. This increased
    # the complexity and made the traing time very long. this was why I decided to use the label encoder instead
    le = generate_label_encoder(df4['country'])
    df4 = replace_with_label_encoder(df4, 'country', le)

    # label encoding for year column. This is because the years are sequential
    le = generate_label_encoder(df4['year'])
    df4 = replace_with_label_encoder(df4, 'year', le)
    c = custom_clustering(df4)
    return dict(model=c['model'], score=c['score'], clusters=c['clusters'])


def run_clustering():
    start = time.time()
    print("Clustering in progress...")
    assert iris_clusters() is not None
    assert len(cluster_iris_dataset_again().keys()) == 3
    assert len(cluster_amazon_video_game().keys()) == 3
    assert len(cluster_amazon_video_game_again().keys()) == 3
    assert len(cluster_life_expectancy().keys()) == 3

    end = time.time()
    run_time = round(end - start, 4)
    print("Clustering ended...")
    print(f"{30*'-'}\nClustering run_time:{run_time}s\n{30*'-'}\n")


if __name__ == "__main__":
    run_clustering()
