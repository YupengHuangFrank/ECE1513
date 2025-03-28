# ECE1513
The models are trained with the data source here:
[https://www.kaggle.com/datasets/kazanova/sentiment140/data](https://www.kaggle.com/datasets/kazanova/sentiment140/data)

The storage place of the models:
LSTM:
[https://drive.google.com/drive/folders/1Afay_8kTpUCKHE5MIwXeedC9hauX4-UI?usp=sharing](https://drive.google.com/drive/folders/1Afay_8kTpUCKHE5MIwXeedC9hauX4-UI?usp=sharing)

Bert model 2:
[https://colab.research.google.com/drive/1f15zuTp9t2g2_-bfpxCffF5W9jL5JpM8?usp=sharing#scrollTo=V7Y6UdOULj-g](https://colab.research.google.com/drive/1f15zuTp9t2g2_-bfpxCffF5W9jL5JpM8?usp=sharing#scrollTo=V7Y6UdOULj-g)

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