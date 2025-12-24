import os
import fastf1 as ff1
import pandas as pd

if not os.path.exists('cache'):
    os.makedirs('cache')
if not os.path.exists('data'):
    os.makedirs('data')

ff1.Cache.enable_cache('cache')

def get_weather_data(session):
    weather = session.weather_data
    if weather is not None and not weather.empty:
        rainfall = weather['Rainfall'].any()
        air_temp = weather['AirTemp'].mean()
        return rainfall, air_temp
    else:
        return False, 25.0

def collect_past_race_data(start_year, end_year):
    all_results = []

    for year in range(start_year, end_year + 1):
        try:
            schedule = ff1.get_event_schedule(year, include_testing=False)

            if 'EventFormat' in schedule.columns:
                races = schedule[schedule['EventFormat'].isin(['conventional', 'sprint', 'sprint_shootout'])]
            else:
                races = schedule

            print(f"--- Collecting data for {year} ---")

            for _, race in races.iterrows():
                round_number = race['RoundNumber']
                race_name = race['EventName']
                circuit_name = race['Location']

                try:
                    print(f"  Fetching: {race_name}")
                    session = ff1.get_session(year, round_number, 'R')
                    session.load(weather=True, telemetry=False, messages=False)

                    results = session.results
                    rainfall, air_temp = get_weather_data(session)

                    results['Season'] = year
                    results['Round Number'] = round_number
                    results['Race Name'] = race_name
                    results['Circuit Name'] = circuit_name
                    results['Rain'] = rainfall
                    results['AirTemp'] = air_temp

                    all_results.append(results)

                except Exception as e:
                    print(f"  Error in Round {round_number}: {e}")

        except Exception as e:
            print(f"Error fetching schedule for {year}: {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()


circuit_types = {
    "Melbourne" : {"Type": "Street Circuit", "Setup": "High Downforce"},
    "Sakhir" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Shanghai" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Baku" : {"Type": "Street Circuit", "Setup": "Low Downforce"},
    "Barcelona" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Monte Carlo" : {"Type": "Street Circuit", "Setup": "High Downforce"},
    "Montreal" : {"Type": "Hybrid Circuit", "Setup": "Low Downforce"},
    "Le Castellet" : {"Type": "Track Circuit", "Setup": "Medium Downforce"},
    "Spielberg" : {"Type": "Track Circuit", "Setup": "Low Downforce"},
    "Silverstone" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Hockenheim" : {"Type": "Track Circuit", "Setup": "Medium Downforce"},
    "Budapest" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Spa-Francorchamps" : {"Type": "Track Circuit", "Setup": "Medium Downforce"},
    "Monza" : {"Type": "Track Circuit", "Setup": "Low Downforce"},
    "Singapore" : {"Type": "Street Circuit", "Setup": "High Downforce"},
    "Sochi" : {"Type": "Track Circuit", "Setup": "Medium Downforce"},
    "Suzuka" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Austin" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Mexico City" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Sao Paulo" : {"Type": "Track Circuit", "Setup": "Medium Downforce"},
    "Yas Marina" : {"Type": "Track Circuit", "Setup": "Medium Downforce"},
    "Yas Island" : {"Type": "Track Circuit", "Setup": "Medium Downforce"},
    "Mugello" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Nurburgring" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Portimao" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Imola" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Istanbul" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Zandvoort" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Lusail" : {"Type": "Track Circuit", "Setup": "High Downforce"},
    "Jeddah" : {"Type": "Street Circuit", "Setup": "Low Downforce"},
    "Miami" : {"Type": "Street Circuit", "Setup": "Medium Downforce"}, 
    "Monaco" : {"Type": "Street Circuit", "Setup": "High Downforce"},
    "Marina Bay" : {"Type": "Street Circuit", "Setup": "High Downforce"},
    "Las Vegas" : {"Type": "Street Circuit", "Setup": "Low Downforce"},
    "Miami Gardens" : {"Type": "Street Circuit", "Setup": "Medium Downforce"}
}

# df = collect_past_race_data(2018, 2025) 
def add_circuit_info(df, circuit_types):
    df = df.drop(columns=["Type_x", "Setup_x", "Type_y", "Setup_y", "Type", "Setup"], errors="ignore")
    circuit_df = (pd.DataFrame.from_dict(circuit_types, orient="index").reset_index().rename(columns={"index": "Circuit Name"}))
    df = df.merge(circuit_df, on="Circuit Name", how="left")
    return df

df = pd.read_csv("data/f1_data.csv")

df["Circuit Name"] = df["Circuit Name"].replace({
    "Montréal": "Montreal",
    "São Paulo": "Sao Paulo",
    "Nürburgring": "Nurburgring",
    "Portimão": "Portimao"
})

df = add_circuit_info(df, circuit_types)

df.rename(columns={"Type": "Circuit Type", "Setup": "Downforce Setup"}, inplace=True)

output_path = "data/f1_training_data_final.csv"
df.to_csv(output_path, index=False)

print(f"✅ Done! File saved to {output_path}")

missing_info = df[df["Circuit Type"].isnull()]["Circuit Name"].unique()
if len(missing_info) > 0:
    print(f"⚠️ Warning: The following tracks were not found in your dictionary: {missing_info}")

print(df.columns)