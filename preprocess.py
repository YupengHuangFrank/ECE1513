# %%
import contractions
import csv
import emoji
import itertools
import pandas as pd
import random
import re
from os import listdir
from os.path import isfile, join
from transformers import AutoTokenizer

def preprocess_internal(df,
               tokenize,  # tokenize here is a function. Example shown below
               to_lower=True,
               remove_urls=True,
               remove_emojis=True,
               remove_contractions=True,
               remove_punctuation=True):
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
# Sampling
def sample_frames_by_date_size_english(list_frames, date, sample_size):
    if (list_frames == None or len(list_frames) == 0):
        return None

    iteration = []
    total_rows = 0
    columns = list_frames[0].columns

    # Create a generator for all rows in the list of dataframes to random sample without
    # having a single object that is too large to fit in memory
    for df in list_frames:
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] == date) & (df['lang'] == 'en')]
        total_rows += len(df)
        iteration = itertools.chain(iteration, df.iterrows())
    
    # Either get the sample size or the total number of rows in the dataframes 
    # in case sample size is larger than the total number of rows
    sample_size = min(sample_size, total_rows)
    print(f"Date: {date}, Sample size: {sample_size}")

    # Create a mask of random indices to sample from the dataframes
    sample_indices = random.sample(range(total_rows), sample_size)
    sample_indices.sort()
    next_index = 0
    result = []
    for i, row in enumerate(iteration):
        if i == sample_indices[next_index]:
            next_index += 1
            result.append(row[1])
        if next_index >= sample_size:
            break

    return pd.DataFrame(result, columns=columns)

# %%
def get_file_names_for_date(date):
    data_distro_sorted = pd.read_csv('data_distro_sorted.csv')
    data_distro_sorted['date'] = pd.to_datetime(data_distro_sorted['date'])
    data_distro_filtered = data_distro_sorted[data_distro_sorted['date'] == pd.to_datetime(date)]
    return data_distro_filtered['fileName'].tolist()

def get_data_frames_for_date(date):
    file_names = get_file_names_for_date(date)
    if len(file_names) == 0:
        return None

    print(f'Here are the file names for date {date}: {file_names}')
    data_frames = []
    file_path = '../usc-x-24-us-election/'
    for file_name in file_names:
        df = pd.read_csv(f'{file_path}{file_name}', compression='gzip')
        data_frames.append(df)
    return data_frames

# %%
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
        sampled_df = sample_frames_by_date_size_english(data_frames, date, sample_size)
        if data_frames is None or len(sampled_df) == 0:
            print(f"No English data for {date}")
            continue
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
result = sample_and_preprocess("2024-06-12", "2024-06-12", 50000, bert_tokenize)
saved_result = []
for i in result:
    saved_result.append(i)
    print(i.head())

# %%
print(saved_result[0].shape)
# %%
