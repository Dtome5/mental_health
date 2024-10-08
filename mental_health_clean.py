import pandas as pd
import re

data = pd.read_csv(r"mental-heath-in-tech-2016_20161114.csv")
df = pd.DataFrame(data)

# convert column headings to lowercase for standardisation
df.columns = df.columns.str.lower()

# fill the columns containing mental health conditions
df["if yes, what condition(s) have you been diagnosed with?"] = df[
    "if yes, what condition(s) have you been diagnosed with?"
].fillna("none")
df["if so, what condition(s) were you diagnosed with?"] = df[
    "if so, what condition(s) were you diagnosed with?"
].fillna("none")
df["if maybe, what condition(s) do you believe you have?"] = df[
    "if maybe, what condition(s) do you believe you have?"
].fillna("none")
# delete columns with more than half values missing
df = df.dropna(axis=1, thresh=df.shape[0] // 2)

# a list of columns to drop because of either complex data or unneeded data
to_drop = [
    "do you have previous employers?",
    "why or why not?",
    "why or why not?.1",
    "what us state or territory do you live in?",
    "what us state or territory do you work in?",
    "is your employer primarily a tech company/organization?",
]

# removing self employed respondents
df = df[df["are you self-employed?"] != 1]
df = df.drop("are you self-employed?", axis=1)
df = df.drop(to_drop, axis=1)

# replacing outliers in age with the mean and scaling the column
question_age = "what is your age?"
df.loc[(df[question_age] > 90), question_age] = int(df[question_age].mean())
df.loc[(df[question_age] < 18), question_age] = int(df[question_age].mean())
df[question_age] = df[question_age].astype(str)


# function to convert age to ranges
def age_class(x):
    if int(x) < 25:
        return "18-24"
    elif int(x) >= 25 and int(x) < 34:
        return "25-34"
    elif int(x) >= 34 and int(x) < 45:
        return "35-44"
    elif int(x) >= 45 and int(x) < 55:
        return "45-54"
    elif int(x) >= 55 and int(x) < 65:
        return "55-64"
    elif int(x) >= 56 and int(x) < 64:
        return "65-74"
    else:
        return "50+"


df[question_age] = df[question_age].apply(age_class)


# Function to standardise the gender column
def gender(x):
    if isinstance(x, str):
        if x[0].lower() == "m" or x[-5:] == " male":
            return "m"
        elif x[0].lower() == "f" or "female" in x.lower():
            return "f"
    else:
        return "o"


df["what is your gender?"] = df["what is your gender?"].apply(gender)


# function to get a list of the respondents roles at the organisation
def process_string(s):
    parts = s.split("|")
    return tuple(sorted([re.sub(r"\s*\(.*?\)", "", part).strip() for part in parts]))


standard_cols = df.select_dtypes(include=["object"])
standard_cols = standard_cols.columns[standard_cols.nunique() <= 6]
for i in standard_cols:
    df.fillna({i: df[i].mode()[0]}, inplace=True)

non_standard = [
    "how many employees does your company or organization have?",
    "if yes, what condition(s) have you been diagnosed with?",
    "if so, what condition(s) were you diagnosed with?",
    "if maybe, what condition(s) do you believe you have?",
    "have you ever sought treatment for a mental health issue from a mental health professional?",
    "what is your age?",
    "what country do you live in?",
    "what country do you work in?",
    "which of the following best describes your work position?",
]

df[non_standard[1:4]] = df[non_standard[1:4]].map(process_string)
df[non_standard[8]] = df[non_standard[8]].apply(lambda x: tuple(x.split("|")))
df = df.reset_index(drop=True)
# print(df[non_standard[8]])
df_clean = df.copy()
