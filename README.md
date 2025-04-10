# ECE1513
The models are trained with the data source here:
[https://www.kaggle.com/datasets/kazanova/sentiment140/data](https://www.kaggle.com/datasets/kazanova/sentiment140/data)
[[https://www.kaggle.com/datasets/kazanova/sentiment140/data](https://www.kaggle.com/datasets/sripaadsrinivasan/tweets-about-the-upcoming-us-electionaugtooct/data?select=us_election-edit.csv)]([https://www.kaggle.com/datasets/kazanova/sentiment140/data](https://www.kaggle.com/datasets/sripaadsrinivasan/tweets-about-the-upcoming-us-electionaugtooct/data?select=us_election-edit.csv))

In order to run the preprocessing step, run the following powershell command in the usc-x-24-us-election folder, you need the folder next to this repository as well:
Get-ChildItem -Recurse -File | Move-Item -Destination .

The tokenizer in use is a custom function. Follow the sample set in the example section

The final result of the preprocessed text is a cleaned text with steps of your choosing and the following 6 columns:
- id
- user
- date
- input_ids
- attention_mask
- token_type_ids

## Files and what they do:
### reorganize_data.py:
This is a collection of functions used for reorganize the data in a easy to process format

### data_distro_aggregated.csv:
This is an examination of the distribution of English data by date

### data_distro.csv/data_distro_sorted.csv
These are two files that contain file to date mappings

### Date-Ordered-Data
This folder contains English tweets ordered by date

### LSTM.ipynb
This is the implementation of the LSTM model. The path specified in the file is specific to the Drive directory in the google colab that the model is developed.

### word2vec.model
The word2vec mapping that was generated during the tokenization and training of the LSTM model.

### updated_distilbert_model (2).ipynb.ipynb
This is the implementation of the model distilbert-base-uncased. The path specified in the file is specific to the Drive directory in the google colab where the model is developed.

### BERTweet.ipynb
This is the implementation of the model vinai/bertweet-base. The path specified in the file is specific to the Drive directory in the google colab where the model is developed.

### Twitter-Roberta.ipynb
This is the implementation of the model cardiffnlp/twitter-roberta-base-sentiment. The path specified in the file is specific to the Drive directory in the google colab where the model is developed.
