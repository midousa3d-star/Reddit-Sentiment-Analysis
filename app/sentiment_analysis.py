from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

import pandas as pd

from dataframe_operations import merge_data_and_results

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# Post sentiment analysis

# evaluate negative, neutral and positive score given text 
def polarity_scores_roberta(text):
    # Set the maximum length for BERT-based models (typically 512 tokens)
    max_length = 512

    encoded_text = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2],
    }
    return scores_dict


# generating the list of sentiment results from the titles to be made into dataframe
def sentiment_results_df(dataframe):
    results_data = []

    for i in range(len(dataframe)):
        results = polarity_scores_roberta(dataframe['text'][i])
        results_data.append(results)

    results_df = pd.DataFrame(results_data)

    # make the sentiment column

    # Use idxmax to determine the sentiment
    results_df['sentiment'] = results_df[['roberta_neg', 'roberta_neu', 'roberta_pos']].idxmax(axis=1)

    # Map the column names to sentiment labels
    results_df['sentiment'] = results_df['sentiment'].map({
        'roberta_neg': 'negative',
        'roberta_neu': 'neutral',
        'roberta_pos': 'positive'
    })

    return results_df



# Comment sentiment analysis

# input: comment, score, post id
# output: comment, score, post id, roberta_neg, roberta_neu, roberta_pos, sentiment (one with the maximum value)


def sentiment_results_all_comments(dataframe):

    results_list = []

    # looping through all comments
    for i in range(len(dataframe)):
        
        results = polarity_scores_roberta(dataframe['text'][i])
        results_list.append(results)

    # make the df
    results_df = pd.DataFrame(results_list)

    # make the sentiment column

    # Use idxmax to determine the sentiment
    results_df['sentiment'] = results_df[['roberta_neg', 'roberta_neu', 'roberta_pos']].idxmax(axis=1)

    # Map the column names to sentiment labels
    results_df['sentiment'] = results_df['sentiment'].map({
        'roberta_neg': 'negative',
        'roberta_neu': 'neutral',
        'roberta_pos': 'positive'
    })

    # combine data and results
    df = merge_data_and_results(dataframe, results_df)  

    return df