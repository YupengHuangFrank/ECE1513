# %%
import csv
import pandas as pd
import re
from os import listdir
from os.path import isfile, join
from pathlib import Path

# %%
# Get the distribution of data in a file
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def write_data_distro():
    all_records = []
    mypath = f'../usc-x-24-us-election'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files.sort(key=natural_keys)
    
    for file in files:
        if file.endswith('.csv.gz'):
            print(f'Processing {file}')
            df = pd.read_csv(file, compression='gzip')
            dict = df["date"].value_counts().to_dict()
            for key, value in dict.items():
                all_records.append([file, key, value])

    with open('data_distro.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(all_records) 

def write_data_distro_sorted():
    data_distro = pd.read_csv('data_distro.csv')
    data_distro['date'] = pd.to_datetime(data_distro['date'])
    data_distro_sorted = data_distro.sort_values(by=['date'])
    data_distro_sorted.to_csv('data_distro_sorted.csv', index=False)

# %%
# Restructure the data to be date ordered files
def get_file_names_for_date(date):
    data_distro_sorted = pd.read_csv('data_distro_sorted.csv')
    data_distro_sorted['date'] = pd.to_datetime(data_distro_sorted['date'])
    data_distro_filtered = data_distro_sorted[data_distro_sorted['date'] == pd.to_datetime(date)]
    return data_distro_filtered['fileName'].tolist()

def get_english_data_for_date(date):
    file_names = get_file_names_for_date(date)
    if len(file_names) == 0:
        return None

    print(f'Here are the file names for date {date}: {file_names}')
    data_frames = []
    file_path = '../usc-x-24-us-election/'
    for file_name in file_names:
        df = pd.read_csv(f'{file_path}{file_name}', compression='gzip')
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] == date) & (df['lang'] == 'en')]
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if len(data_frames) > 0 else None

def restructure_file(start_date, 
                    end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    for date in date_range:
        data_frame = get_english_data_for_date(date)
        if data_frame is None or len(data_frame) == 0:
            print(f"No English data for {date}")
            continue
        yield date, data_frame

# %%
all_dates = restructure_file("2024-05-01", "2024-11-30")
for date, df in all_dates:
    date = date.strftime("%Y-%m-%d")
    path = Path(f'Date-Ordered-Data/{date}.csv.gz')
    df.to_csv(path, index=False, compression='gzip')
