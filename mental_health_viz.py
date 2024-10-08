from mental_health_model import cluster_df, df_clean
from mental_health_clean import df, non_standard
from prince import MCA
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


clust_df1 = cluster_df[cluster_df["cluster"] == 0].reset_index(drop=True)
clust_df2 = cluster_df[cluster_df["cluster"] == 1].reset_index(drop=True)
clust_df1.to_csv("cluster_one.csv")
clust_df2.to_csv("cluster_two.csv")
cmap = plt.get_cmap("tab20")

age_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
ages1 = clust_df1["what is your age?"]
ages2 = clust_df2["what is your age?"]
counts1 = ages1.value_counts()
counts2 = ages2.value_counts()
ax1.pie(
    counts1,
    labels=ages1.unique(),
    explode=[0, 0, 0, 0.1, 0.2, 0.3],
    autopct="%1.1f%%",
    textprops={"fontsize": 6},
    colors=[cmap(color) for color in range(ages1.nunique())],
)
ax1.set_title("Cluster 1")
ax2.pie(
    counts2,
    labels=ages2.unique(),
    explode=[0, 0, 0, 0.1, 0.2, 0.3],
    autopct="%1.1f%%",
    textprops={"fontsize": 6},
    colors=[cmap(color) for color in range(ages2.nunique())],
)
ax2.set_title("Cluster 2")
plt.suptitle("Cluster Age Distributions")
plt.savefig("ages.png")

gender_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
genders1 = clust_df1["what is your gender?"]
genders2 = clust_df2["what is your gender?"]
counts1 = clust_df1["what is your gender?"].value_counts()
counts2 = clust_df2["what is your gender?"].value_counts()
ax1.pie(
    counts1,
    labels=genders1.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 8},
    colors=[cmap(color) for color in range(genders1.nunique())],
)
ax1.set_title("Cluster 1")
ax2.pie(
    counts2,
    labels=genders2.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 8},
    colors=[cmap(color) for color in range(genders2.nunique())],
)
ax2.set_title("Cluster 2")
plt.suptitle("Gender Distribution by Cluster")
plt.savefig("genders.png")

Countries_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
countries1 = clust_df1[non_standard[7]]
countries2 = clust_df2[non_standard[7]]
counts1 = clust_df1[non_standard[7]].value_counts()
counts2 = clust_df2[non_standard[7]].value_counts()
cmap1 = plt.get_cmap("tab20")
cmap2 = plt.get_cmap("tab20b")

combined_colors = [cmap1(i) for i in range(20)] + [cmap2(i) for i in range(20)]
bars1 = ax1.bar(
    counts1.index,
    counts1.values,
    color=combined_colors,
)
ax1.set_xticks([])
ax1.legend(
    bars1,
    counts1.index,
    title="Countries",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=4,
    title_fontsize=8,
)
ax1.set_title("Cluster 1")
bars2 = ax2.bar(
    counts2.index,
    counts2.values,
    color=combined_colors,
)
ax2.set_xticks([])
ax2.legend(
    bars2,
    counts2.index,
    title="Countries",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=4,
    title_fontsize=8,
)
ax2.set_title("Cluster 2")
plt.suptitle("Countries Distribution Across Cluster")
plt.tight_layout()
plt.savefig("countries.png")

company_size_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
company_size1 = clust_df1[non_standard[0]]
company_size2 = clust_df2[non_standard[0]]
counts1 = clust_df1[non_standard[0]].value_counts()
counts2 = clust_df2[non_standard[0]].value_counts()
ax1.pie(
    counts1,
    labels=company_size1.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 6},
    colors=[cmap(color) for color in range(company_size1.nunique())],
)
ax1.set_title("Cluster 1")
ax2.pie(
    counts2,
    labels=company_size2.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 6},
    colors=[cmap(color) for color in range(company_size2.nunique())],
)
ax2.set_title("Cluster 2")
plt.suptitle("Company size by Cluster")
plt.savefig("Company sizes.png")

roles_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
roles1 = clust_df1[non_standard[8]]
roles2 = clust_df2[non_standard[8]]
counts1 = clust_df1[non_standard[8]].explode().value_counts()
counts2 = clust_df2[non_standard[8]].explode().value_counts()
bars1 = ax1.bar(
    counts1.index,
    counts1.values,
    color=[cmap(i) for i in range(roles1.nunique())],
)
ax1.set_xticks([])
ax1.legend(
    bars1,
    counts1.index,
    title="roles",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=6,
    title_fontsize=8,
)
ax1.set_title("Cluster 1")
bars2 = ax2.bar(
    counts2.index,
    counts2.values,
    color=[cmap(i) for i in range(roles2.nunique())],
)
ax2.set_xticks([])
ax2.legend(
    bars2,
    counts2.index,
    title="roles",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=6,
    title_fontsize=8,
)
ax2.set_title("Cluster 2")
plt.suptitle("Work Position Distributition Across Clusters")
plt.tight_layout()
plt.savefig("roles.png")

conditions_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=300)
conditions1 = clust_df1[non_standard[1]]
conditions2 = clust_df2[non_standard[1]]
counts1 = clust_df1[non_standard[1]].explode().value_counts()
counts2 = clust_df2[non_standard[1]].explode().value_counts()
bars1 = ax1.bar(
    counts1.index,
    counts1.values,
    color=[cmap(i) for i in range(conditions1.nunique())],
)
ax1.set_xticks([])
ax1.legend(
    bars1,
    counts1.index,
    title="conditions",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=6,
    title_fontsize=6,
)
ax1.set_title("Cluster 1")
bars2 = ax2.bar(
    counts2.index,
    counts2.values,
    color=[cmap(i) for i in range(conditions2.nunique())],
)
ax2.set_xticks([])
ax2.legend(
    bars2,
    counts2.index,
    title="conditions",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=6,
    title_fontsize=6,
)
ax2.set_title("Cluster 2")
plt.suptitle("Mental Health Conditions by Cluster")
plt.tight_layout()
plt.savefig("conditions.png")


past_condition_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
past_condition1 = clust_df1["have you had a mental health disorder in the past?"]
past_condition2 = clust_df2["have you had a mental health disorder in the past?"]
counts1 = clust_df1["have you had a mental health disorder in the past?"].value_counts()
counts2 = clust_df2["have you had a mental health disorder in the past?"].value_counts()
ax1.pie(
    counts1,
    labels=past_condition1.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 6},
    colors=[cmap(color) for color in range(past_condition1.nunique())],
)
ax1.set_title("Cluster 1")
ax2.pie(
    counts2,
    labels=past_condition2.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 6},
    colors=[cmap(color) for color in range(past_condition2.nunique())],
)
ax2.set_title("Cluster 2")
plt.suptitle("Past Condition by Cluster")
plt.figtext(0.5, 0.41, "Figure 8", ha="center", fontsize=12)
plt.savefig("Past Conditions")

fam_hist_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
fam_hist1 = clust_df1["do you have a family history of mental illness?"]
fam_hist2 = clust_df2["do you have a family history of mental illness?"]
counts1 = clust_df1["do you have a family history of mental illness?"].value_counts()
counts2 = clust_df2["do you have a family history of mental illness?"].value_counts()
ax1.pie(
    counts1,
    labels=fam_hist1.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 6},
    colors=[cmap(color) for color in range(fam_hist1.nunique())],
)
ax1.set_title("Cluster 1")
ax2.pie(
    counts2,
    labels=fam_hist2.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 6},
    colors=[cmap(color) for color in range(fam_hist2.nunique())],
)
ax2.set_title("Cluster 2")
plt.suptitle("Familiy History of Mental Health conditions by Cluster")
plt.figtext(0.5, 0.41, "Figure 8", ha="center", fontsize=12)
plt.savefig("Family history")


men_phys_fig, (ax1) = plt.subplots(figsize=(10, 5), dpi=300)
men_phys1 = cluster_df[
    "would you be willing to bring up a physical health issue with a potential employer in an interview?"
]
# men_phys2 = clust_df2[
#     "would you be willing to bring up a physical health issue with a potential employer in an interview?"
# ]
counts1 = clust_df1[
    "would you be willing to bring up a physical health issue with a potential employer in an interview?"
].value_counts()
# counts2 = clust_df2[
#     "did you feel that your previous employers took mental health as seriously as physical health?"
# ].value_counts()
ax1.pie(
    counts1,
    labels=men_phys1.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 6},
    colors=[cmap(color) for color in range(men_phys1.nunique())],
)
ax1.set_title("Cluster 1")
# ax2.pie(
#     counts2,
#     labels=men_phys2.unique(),
#     autopct="%1.1f%%",
#     textprops={"fontsize": 6},
#     colors=[cmap(color) for color in range(men_phys2.nunique())],
# )
# ax2.set_title("Cluster 2")
plt.suptitle(
    "Participants who felt their employers were taking mental healt as seriously as physical health "
)
plt.savefig("men_phys.png")

benefits_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
benefits1 = clust_df1[
    "does your employer provide mental health benefits as part of healthcare coverage?"
]
benefits2 = clust_df2[
    "does your employer provide mental health benefits as part of healthcare coverage?"
]
counts1 = clust_df1[
    "does your employer provide mental health benefits as part of healthcare coverage?"
].value_counts()
counts2 = clust_df2[
    "does your employer provide mental health benefits as part of healthcare coverage?"
].value_counts()
ax1.pie(
    counts1,
    labels=benefits1.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 6},
    colors=[cmap(color) for color in range(benefits1.nunique())],
)
ax1.set_title("Cluster 1")
ax2.pie(
    counts2,
    labels=benefits2.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 6},
    colors=[cmap(color) for color in range(benefits2.nunique())],
)
ax2.set_title("Cluster 2")
plt.suptitle("Mental Health Benefits Provision by Cluster")
plt.savefig("Mental health benefits.png")

interview_fig, (ax1) = plt.subplots(figsize=(10, 5), dpi=300)
interview2 = clust_df2[
    "would you be willing to bring up a physical health issue with a potential employer in an interview?"
]
counts2 = clust_df2[
    "would you be willing to bring up a physical health issue with a potential employer in an interview?"
].value_counts()
ax1.pie(
    counts2,
    labels=interview2.unique(),
    autopct="%1.1f%%",
    colors=[cmap(color) for color in range(benefits1.nunique())],
)
ax1.set_title("Cluster 2")
plt.suptitle("Reluctance to Disclose Mental Health Condition During Interview")
plt.savefig("Mental health interview.png")


treatment_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
treatment1 = clust_df1[
    "have you ever sought treatment for a mental health issue from a mental health professional?"
].replace({1: "Yes", 0: "no"})
treatment2 = clust_df2[
    "have you ever sought treatment for a mental health issue from a mental health professional?"
].replace({1: "Yes", 0: "no"})
counts1 = clust_df1[
    "have you ever sought treatment for a mental health issue from a mental health professional?"
].value_counts()
counts2 = clust_df2[
    "have you ever sought treatment for a mental health issue from a mental health professional?"
].value_counts()
ax1.pie(
    counts1,
    labels=treatment1.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 9},
    colors=[cmap(color) for color in range(treatment1.nunique())],
)
ax1.set_title("Cluster 1")
ax2.pie(
    counts2,
    labels=treatment2.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 9},
    colors=[cmap(color) for color in range(treatment2.nunique())],
)
ax2.set_title("Cluster 2")
plt.suptitle("Percentage of Participants Seeking Mental Health Treatment")
plt.savefig("Mental health treatment.png")


no_jobs_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
no_jobs1 = clust_df1["which of the following best describes your work position?"].apply(
    len
)
no_jobs2 = clust_df2["which of the following best describes your work position?"].apply(
    len
)
counts1 = (
    clust_df1["which of the following best describes your work position?"]
    .apply(len)
    .value_counts()
)
counts2 = (
    clust_df2["which of the following best describes your work position?"]
    .apply(len)
    .value_counts()
)
ax1.pie(
    counts1,
    labels=no_jobs1.unique(),
    autopct="%1.1f%%",
    textprops={"fontsize": 6},
    explode=[0, 0, 0, 0.0, 0.1, 0.25, 0.4, 0.5],
    colors=[cmap(color) for color in range(no_jobs1.nunique())],
)
ax1.set_title("Cluster 1")
ax2.pie(
    counts2,
    labels=no_jobs2.unique(),
    autopct="%1.1f%%",
    explode=[0, 0, 0, 0, 0.1, 0.15, 0.3, 0.4, 0.5],
    textprops={"fontsize": 6},
    colors=[cmap(color) for color in range(no_jobs2.nunique())],
)
ax2.set_title("Cluster 2")
plt.suptitle("Job Counts by Cluster")
plt.savefig("no_jobs.png")


condition_roles_fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
explode = cluster_df.explode(non_standard[8]).explode(non_standard[1])
no_unique = explode[non_standard[1]].nunique()
crosstab = pd.crosstab(explode[non_standard[8]], explode[non_standard[1]]).sort_index(
    axis=1
)
crosstab.plot(
    kind="bar",
    stacked=True,
    ax=ax1,
    color=[cmap(color / no_unique) for color in range(no_unique)],
)
plt.title("Work Condition by Category")
plt.xlabel(non_standard[8])
plt.ylabel("Count")
plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)
plt.tight_layout()
plt.savefig("workcondition.png")


# mc = MCA(n_components=2, random_state=0)
# mcs = mc.fit_transform(df)
mc = MCA(n_components=2, random_state=0)
mca_coords = mc.fit(df_clean).transform(df_clean)

fig, ax1 = plt.subplots()
scatter = ax1.scatter(
    mca_coords.iloc[:, 0],
    mca_coords.iloc[:, 1],
    c=cluster_df["cluster"],
    cmap="coolwarm",
    s=50,
    alpha=0.7,
)
ax1.set_title("MCA Plot")
plt.savefig("cluster.png")
