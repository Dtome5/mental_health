from mental_health_model import cluster_df, df_clean, clusters, oh_df
from mental_health_clean import df, non_standard
from prince import MCA
from collections import Counter
from scipy.stats import chi2_contingency
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from kmodes import kprototypes
from sklearn.cluster import DBSCAN

clust_df1 = cluster_df[cluster_df["cluster"] == 0].reset_index(drop=True)
clust_df2 = cluster_df[cluster_df["cluster"] == 1].reset_index(drop=True)
clust_df1.to_csv("cluster_one.csv")
clust_df2.to_csv("cluster_two.csv")
clust1 = pd.DataFrame()
clust2 = pd.DataFrame()


# calculate distance matrix
def create_dm(dataset):
    # if the input dataset is a dataframe, we take out the values as a numpy.
    # If the input dataset is a numpy array, we use it as is.
    if type(dataset).__name__ == "DataFrame":
        dataset = dataset.values
    lenDataset = len(dataset)
    distance_matrix = np.zeros(lenDataset * lenDataset).reshape(lenDataset, lenDataset)
    for i in range(lenDataset):
        for j in range(lenDataset):
            x1 = dataset[i].reshape(1, -1)
            x2 = dataset[j].reshape(1, -1)
            distance = kprototypes.matching_dissim(x1, x2)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix


# calculate silhouette scores
silhouette_scores = dict()
K = range(2, 10)
distance_matrix = create_dm(df_clean)
for k in K:
    untrained_model = KModes(n_clusters=k, n_init=4)
    trained_model = untrained_model.fit(df_clean)
    cluster_labels = trained_model.labels_
    score = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
    silhouette_scores[k] = score
print("The k and associated Silhouette scores are: ", silhouette_scores)


def value_count_df(df):
    result = {}
    for column in df.columns:
        value_counts = df[column].value_counts().to_dict()
        result[column] = [{value: count} for value, count in value_counts.items()]
    result_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in result.items()]))
    return result_df


clust1 = value_count_df(clust_df1)
clust2 = value_count_df(clust_df2)
clust1.to_csv("percentages1.csv")
clust2.to_csv("percentages2.csv")
# get Chi square values
Chi2 = []
for col in df_clean.columns:
    n = df_clean[col].sum().sum()
    contingency = pd.crosstab(df_clean[col], cluster_df["cluster"])
    chi2, p, dof, freq = chi2_contingency(contingency)
    print(df_clean[col].shape)
    r, k = df_clean[col].shape[0], cluster_df["cluster"].nunique()
    cramers_v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))
    Chi2.append(
        {
            "feature": col,
            "chisq": chi2,
            "degrees of freedom": dof,
            "significant": "yes" if p < 0.05 else "no",
            "cramer": cramers_v,
        }
    )
feat_imp = pd.DataFrame(Chi2)
feat_imp.to_csv("feature importance.csv")
print(feat_imp)
