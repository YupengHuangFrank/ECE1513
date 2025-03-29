# %%
import contractions
import csv
import emoji
import pandas as pd
import re
from os import listdir
from os.path import isfile, join
from pathlib import Path
from transformers import AutoTokenizer

def preprocess_internal(df,
               tokenize,  # tokenize here is a function. Example shown below
               to_lower=True,
               remove_urls=True,
               remove_emojis=True,
               remove_contractions=True,
               remove_punctuation=True):
    print("Preprocessing started")
    if df is None or df.empty:
        return None

    df['text_cleaned'] = df['text'].copy()
    if (to_lower):
        df['text_cleaned'] = df['text'].str.lower()
    if (remove_urls):
        df['text_cleaned'] = df['text_cleaned'].str.replace(r'https?://\S+|www\.\S+', '', regex=True)
    if (remove_emojis):
        df['text_cleaned'] = df['text_cleaned'].apply(lambda x: emoji.demojize(x))
    if (remove_contractions):
        df['text_cleaned'] = df['text_cleaned'].apply(lambda x: contractions.fix(x))
    if (remove_punctuation):
        df['text_cleaned'] = df['text_cleaned'].str.replace(r'[^\w\s\']', '', regex=True)

    encoded_outputs = tokenize(df['text_cleaned'].tolist())
    df["input_ids"] = encoded_outputs["input_ids"]
    df["attention_mask"] = encoded_outputs["attention_mask"]
    df["token_type_ids"] = encoded_outputs["token_type_ids"]

    result_df = df[["id", "user", "date", "input_ids", "attention_mask", "token_type_ids"]]
    return result_df

# %%
# Get the distribution of data in a file
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
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
# Sample helper
def sample_by_size(data_frame, sample_size):
    if (data_frame == None or len(data_frame) == 0):
        return None
    total_rows = len(data_frame)

    sample_size = min(sample_size, total_rows)
    print(f"Date: {date}, Sample size: {sample_size}")

    return data_frame.sample(n=sample_size, replace=False)

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
# Sample and preprocess the data
def get_data_frames_for_date(date):
    date = date.strftime("%Y-%m-%d")
    path = Path(f'Date-Ordered-Data/{date}.csv.gz')
    if not isfile(path):
        return None

    df = pd.read_csv(path, compression='gzip')
    return df

def sample_and_preprocess(start_date, 
                        end_date, 
                        sample_size,
                        tokenize,
                        to_lower=True,
                        remove_urls=True,
                        remove_emojis=True,
                        remove_contractions=True,
                        remove_punctuation=True):
    date_range = pd.date_range(start=start_date, end=end_date)
    for date in date_range:
        data_frames = get_data_frames_for_date(date)
        if data_frames is None or len(data_frames) == 0:
            print(f"No data for {date}")
            continue
        sampled_df = sample_by_size(data_frames, sample_size)
        preprocessed_df = preprocess_internal(sampled_df,
                                            tokenize,
                                            to_lower=to_lower,
                                            remove_urls=remove_urls,
                                            remove_emojis=remove_emojis,
                                            remove_contractions=remove_contractions,
                                            remove_punctuation=remove_punctuation)
        yield preprocessed_df

# %%
# Example tokenizer function
def bert_tokenize(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(
        text,
        padding="max_length",     # ensures all sequences are the same length
        truncation=True,          # shortens long sequences
    )

# %%
all_dates = restructure_file("2024-05-01", "2024-11-30")
for date, df in all_dates:
    date = date.strftime("%Y-%m-%d")
    path = Path(f'Date-Ordered-Data/{date}.csv.gz')
    df.to_csv(path, index=False, compression='gzip')
