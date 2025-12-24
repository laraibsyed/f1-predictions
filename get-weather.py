import pandas as pd
import fastf1 as ff1
import os

if not os.path.exists('cache'):
    os.makedirs('cache')
ff1.Cache.enable_cache('cache')

input_file = "data/f1_training_data_final.csv" 
print(f"Reading {input_file}...")
df = pd.read_csv(input_file)


unique_races = df[['Season', 'Round Number']].drop_duplicates().values
print(f"Found {len(unique_races)} races to scan for weather.")

weather_data = []

for season, round_num in unique_races:
    try:
        print(f"Fetching weather: {int(season)} Round {int(round_num)}")
        
        session = ff1.get_session(int(season), int(round_num), 'R')
        session.load(weather=True, laps=False, telemetry=False, messages=False)

        w = session.weather_data
        if w is not None and not w.empty:
            rainfall = w['Rainfall'].any()
            air_temp = w['AirTemp'].mean()
        else:
            rainfall = False
            air_temp = 25.0

        weather_data.append({
            'Season': season,
            'Round Number': round_num,
            'Rain': rainfall,
            'AirTemp': air_temp
        })
        
    except Exception as e:
        print(f"  Failed to get weather for {season} R{round_num}: {e}")
        weather_data.append({
            'Season': season,
            'Round Number': round_num,
            'Rain': False,
            'AirTemp': 25.0
        })

print("Merging weather data...")
weather_df = pd.DataFrame(weather_data)

final_df = pd.merge(df, weather_df, on=['Season', 'Round Number'], how='left')

final_df.to_csv("data/f1_training_data_complete.csv", index=False)
print("✅ Success! Weather data added.")
print("Columns are now:", final_df.columns.tolist())