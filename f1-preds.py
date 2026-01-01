import pandas as pd
import fastf1 as ff1
import os
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/f1_training_data_complete.csv")

df = df.sort_values(['Season', 'Round Number'])

cols_to_drop = ["Q1", "Q2", "Q3"]

for col in cols_to_drop:
    df = df.drop(columns=[col], errors="ignore")

df["Position"] = pd.to_numeric(df["ClassifiedPosition"], errors="coerce")
df["Position"] = df["Position"].fillna(20).astype(int)
df = df.drop(columns=["ClassifiedPosition"], errors="ignore")

df["Season Strength"] = df.groupby(["Season", "DriverId"])["Points"].transform(lambda x: x.cumsum().shift(1))
df["Season Strength"] = df["Season Strength"].fillna(0)

df["Driver Confidence"] = df.groupby("DriverId")["Position"].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
df["Driver Confidence"] = df["Driver Confidence"].fillna(20)

df["Rain"] = df["Rain"].astype(int)

team_mapping = {
    "ferrari": "ferrari",
    "mercedes": "mercedes",
    "red_bull": "red_bull",
    "mclaren": "mclaren",
    "alpine": "alpine",
    "renault": "alpine",
    "aston_martin": "aston_martin",
    "force_india": "aston_martin",
    "racing_point": "aston_martin",
    "rb": "rb",
    "toro_rosso": "rb",
    "alphatauri": "rb",
    "alfa_romeo": "sauber",
    "sauber": "sauber",
    "haas": "haas",
    "williams": "williams",
    "audi": "sauber",
    "cadillac": "haas"
}

df["TeamId_Cleaned"] = df["TeamId"].replace(team_mapping)

setup_order = ["Low Downforce", "Medium Downforce", "High Downforce"]
order_encoder = OrdinalEncoder(categories=[setup_order])
df["Downforce Setup Encoded"] = order_encoder.fit_transform(df[["Downforce Setup"]])

ohe = OneHotEncoder(sparse_output=False)
cols_to_encode = ["Circuit Type", "TeamId_Cleaned"]
encoded_features = ohe.fit_transform(df[cols_to_encode])
encoded_feature_names = ohe.get_feature_names_out(cols_to_encode)
print(encoded_feature_names)

encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

final_df = df.copy()

columns_to_drop = [
    'TeamColor', 'FirstName', 'LastName', 'FullName', 'HeadshotUrl', 
    'CountryCode', 'DriverNumber', 'BroadcastName', 'Abbreviation', 'TeamName',
    'Time', 'Status', 'Points', 'Laps', 'Race Name', 'Circuit Name', 
    'Circuit Type', 'Downforce Setup', 'TeamId', 'TeamId_Cleaned'
]

final_df = final_df.drop(columns=columns_to_drop, errors="ignore")

print("Final List of Columns: ", final_df.columns)