import os
import fastf1 as ff1
import pandas as pd

if not os.path.exists('cache'):
    os.makedirs('cache')
ff1.Cache.enable_cache('cache')

def collect_past_race_data(start_year, end_year):
    all_results = []

    for year in range(start_year, end_year + 1):
        try:
            schedule = ff1.get_event_schedule(year, include_testing=False)

            for _, race in schedule.iterrows():
                round_number = race['RoundNumber']
                race_name = race['EventName']
                circuit_name = race['Location']

                session = ff1.get_session(year, round_number, 'R')
                session.load()

                results = session.results
                results['Season'] = year
                results['Round Number'] = round_number
                results['Race Name'] = race_name
                results['Circuit Name'] = circuit_name

                all_results.append(results)

        except Exception as e:
            print(f"Skipping {year} round {round_number}: {e}")

    df = pd.concat(all_results, ignore_index=True)
    df.to_csv("data/f1_data.csv", index=False)

    return df

df = pd.read_csv("data/f1_data.csv")
circuit_types = {
    "Melbourne" : {
        "Type": "Street Circuit",
        "Setup": "High Downforce"
    },
    "Sakhir" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Shanghai" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Baku" : {
        "Type": "Street Circuit",
        "Setup": "Low Downforce"
    },
    "Barcelona" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Monte Carlo" : {
        "Type": "Street Circuit",
        "Setup": "High Downforce"
    },
    "Montreal" : {
        "Type": "Hybrid Circuit",
        "Setup": "Low Downforce"
    },
    "La Castellet" : {
        "Type": "Track Circuit",
        "Setup": "Medium Downforce"
    },
    "Spielberg" : {
        "Type": "Track Circuit",
        "Setup": "Low Downforce"
    },
    "Silverstone" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Hockenheim" : {
        "Type": "Track Circuit",
        "Setup": "Medium Downforce"
    },
    "Budapest" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Spa-Francorchamps" : {
        "Type": "Track Circuit",
        "Setup": "Medium Downforce"
    },
    "Monza" : {
        "Type": "Track Circuit",
        "Setup": "Low Downforce"
    },
    "Singapore" : {
        "Type": "Street Circuit",
        "Setup": "High Downforce"
    },
    "Sochi" : {
        "Type": "Track Circuit",
        "Setup": "Medium Downforce"
    },
    "Suzuka" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Austin" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Mexico City" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Sao Paulo" : {
        "Type": "Track Circuit",
        "Setup": "Medium Downforce"
    },
    "Yas Marina" : {
        "Type": "Track Circuit",
        "Setup": "Medium Downforce"
    },
    "Yas Island" : {
        "Type": "Track Circuit",
        "Setup": "Medium Downforce"
    },
    "Mugello" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Nurburgring" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Portimao" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Imola" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Istanbul" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Zandvoort" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Lusail" : {
        "Type": "Track Circuit",
        "Setup": "High Downforce"
    },
    "Jeddah" : {
        "Type": "Street Circuit",
        "Setup": "Low Downforce"
    },
    "Miami" : {
        "Type": "Street Circuit",
        "Setup": "Medium Downforce"
    }, 
    "Monaco" : {
        "Type": "Street Circuit",
        "Setup": "High Downforce"
    },
    "Marina Bay" : {
        "Type": "Street Circuit",
        "Setup": "High Downforce"
    },
    "Las Vegas" : {
        "Type": "Street Circuit",
        "Setup": "Low Downforce"
    },
    "Miami Garden" : {
        "Type": "Street Circuit",
        "Setup": "Medium Downforce"
    }
}

def add_circuit_info(df, circuit_types):
    df = df.drop(columns=["Type_x", "Setup_x", "Type_y", "Setup_y", "Type", "Setup"], errors="ignore")
    circuit_df = (pd.DataFrame.from_dict(circuit_types, orient="index").reset_index().rename(columns={"index": "Circuit Name"}))
    df = df.merge(circuit_df, on="Circuit Name", how="left")
    return df

df = add_circuit_info(df, circuit_types)
df.rename(columns={"Type": "Circuit Type", "Setup": "Downforce Setup"}, inplace=True)