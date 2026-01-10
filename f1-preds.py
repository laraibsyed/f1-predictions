import pandas as pd
import fastf1 as ff1
import os
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import xgboost as xgb

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

final_df = final_df.sort_values(['Season', 'Round Number'])

numeric_df = final_df.drop(columns=["DriverId", "TeamId", "TeamId_Cleaned"], errors="ignore").select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10)) 
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt='.2f', 
    linewidths=.5
)
plt.title('Feature Correlation Matrix')
plt.show()

print("Final List of Columns: ", final_df.columns)

train_df = final_df[final_df["Season"] <= 2024]
test_df = final_df[final_df["Season"] == 2025]

train_df = train_df.sort_values(["Season", "Round Number"])
test_df = test_df.sort_values(["Season", "Round Number"])

train_query = train_df.groupby(["Season", "Round Number"]).size().to_list()
test_query = test_df.groupby(["Season", "Round Number"]).size().to_list()

drop_cols = ["Season", "Round Number", "Position", "DriverId"]
x_train = train_df.drop(columns=drop_cols)
y_train = 21 - train_df["Position"]

X_test = test_df.drop(columns=drop_cols)
y_test = 21 - test_df["Position"]

model = xgb.XGBRanker(objective='rank:pairwise', learning_rate=0.1, n_estimators=100, max_depth=6)
model.fit(x_train, y_train, group=train_query)

preds = model.predict(X_test)

test_df["Predicted Score"] = preds 

test_df["Predicted Position"] = (
    test_df.groupby(["Season", "Round Number"])["Predicted Score"]
    .rank(ascending=False, method="first")
    .astype(int)
)

matches = test_df[
    (test_df["Predicted Position"] == 1) & 
    (test_df["Position"] == 1)
]

accuracy = len(matches) / len(test_query)
print(f"🏆 Winner Prediction Accuracy: {accuracy:.2%}")

print("\n--- Example Race Prediction ---")
example_race = test_df[test_df["Round Number"] == 1][["DriverId", "Predicted Position", "Position"]]
print(example_race.sort_values("Predicted Position").head(5))

importance = model.feature_importances_
feature_names = x_train.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importance)
plt.title("What Matters to the Model?")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.show()