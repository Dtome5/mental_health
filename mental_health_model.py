from mental_health_clean import df, df_clean, non_standard
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import silhouette_score
from kmodes.kmodes import KModes
import pandas as pd


# Find the maximum length of lists in the column
def to_lists(df, column):
    max_len = df[column].apply(len).max()
    columns = [column + "_" + str(n) for n in range(1, max_len)]
    return columns


def dummy(dataframe, columns):
    exploded = dataframe[columns].explode()
    dummies = pd.get_dummies(exploded).groupby(level=0).max()
    # print(dummies)
    dataframe = dataframe.drop(columns, axis=1)
    dataframe = pd.concat([dataframe, dummies], axis=1)
    return dummies


# dummy(df_clean, non_standard[1])


# Create new columns for each position
def gen_pos_cols(dataframe, list_column):
    max_length = dataframe[list_column].apply(len).max()
    for i in range(max_length):
        column_name = f"{list_column}_{i+1}"
        dataframe[column_name] = dataframe[list_column].apply(
            lambda x: x[i] if i < len(x) else "none"
        )

    return dataframe


# one hot encode list columns
for i in non_standard[1:4] + [non_standard[8]]:
    exploded = df_clean[i].explode()
    dummies = pd.get_dummies(exploded).groupby(level=0).max()
    dummies.columns = dummies.columns + i
    df_clean = pd.concat([df_clean, dummies], axis=1)
oh = OneHotEncoder(sparse_output=False)
oh_df = pd.DataFrame()

for i in df_clean.columns:
    oh_df = pd.concat([oh_df, pd.DataFrame(oh.fit_transform(df_clean[[i]]))], axis=1)

df_clean.to_csv("df_clean.csv")

# use labelencoder for the columns
encoders = []
for i in df_clean.columns:
    le = LabelEncoder()
    df_clean[i] = le.fit_transform(df_clean[i])
    encoders.append(le)

# fit kmodes and create dataframe for the clusters
Kmd = KModes(n_clusters=2, init="huang", random_state=0)
Kmd.fit(df_clean)
clusters = Kmd.predict(df_clean)
cluster_df = pd.concat([df, pd.Series(clusters, name="cluster")], axis=1)
