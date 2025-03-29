# %%
import contractions
import emoji
import pandas as pd
from os.path import isfile
from pathlib import Path
from transformers import AutoTokenizer

# %%
# Sample helper
def sample_by_size(data_frame, sample_size):
    if (data_frame == None or len(data_frame) == 0):
        return None
    total_rows = len(data_frame)

    sample_size = min(sample_size, total_rows)
    print(f"Date: {date}, Sample size: {sample_size}")

    return data_frame.sample(n=sample_size, replace=False)


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
