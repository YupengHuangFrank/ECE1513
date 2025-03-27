# %%
import pandas as pd
import emoji
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def preprocess(file_name,
               nrows,
               compression,
               to_lower=True,
               remove_urls=True,
               remove_emojis=True,
               remove_contractions=True,
               remove_punctuation=True,
               keep_english=True,
               tokenize=True,
               remove_stopwords=True,
               lemmatize_text=True):
    df = pd.read_csv(file_name, nrows=nrows, compression=compression)

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
    if (keep_english):
        df = df[df['lang'] == 'en']
    if (tokenize):
        df['tokens'] = df['text_cleaned'].apply(lambda x: word_tokenize(x))
    if (remove_stopwords):
        stop_words = set(stopwords.words('english'))
        df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])
    if (lemmatize_text):
        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

        def lemmatize_text(text):
            pos_tags = nltk.pos_tag(text)
            
            lemmatized_words = []
            for word, tag in pos_tags:
                # Map the POS tag to WordNet POS tag
                pos = wordnet_map.get(tag[0].upper(), wordnet.NOUN)
                lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
                lemmatized_words.append(lemmatized_word)
            return lemmatized_words

    df['tokens'] = df['tokens'].apply(lemmatize_text)
    result_df = df[["id", "user", "tokens", "date"]]
    return result_df

# %%
result_df = preprocess('../usc-x-24-us-election/part_1/may_july_chunk_1.csv.gz', 50000, 'gzip')
print(result_df.head())

# %%
