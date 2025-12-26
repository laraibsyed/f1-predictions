import pandas as pd
import fastf1 as ff1
import os
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

df = pd.read_csv("data/f1_training_data_complete.csv")

cols_to_drop = ["Q1", "Q2", "Q3"]

for col in cols_to_drop:
    df = df.drop(columns=[col], errors="ignore")

df["Season Strength"] = df.groupby(["Season", "DriverId"])["Points"].cumsum()
df["Driver Confidence"] = df.groupby("DriverId")["Position"].shift(1).rolling(window=3, min_periods=1).mean()
df["Driver Confidence"] = df["Driver Confidence"].fillna(20)

df["Position"] = pd.to_numeric(df["ClassifiedPosition"], errors="coerce")
df["Position"] = df["Position"].fillna(20).astype(int)
df = df.drop(columns=["ClassifiedPosition"], errors="ignore")

setup_order = ["Low Downforce", "Medium Downforce", "High Downforce"]
order_encoder = OrdinalEncoder(categories=[setup_order])
df["Downforce Setup Encoded"] = order_encoder.fit_transform(df[["Downforce Setup"]])

ohe = OneHotEncoder(sparse_output=False)
cols_to_encode = ["Circuit Type", "TeamId"]
encoded_features = ohe.fit_transform(df[cols_to_encode])
encoded_feature_names = ohe.get_feature_names_out(cols_to_encode)
print(encoded_feature_names)