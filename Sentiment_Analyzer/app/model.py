#%%
from transformers import pipeline

sentiment_analyzer = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                              task="sentiment-analysis")

def predict_sentiment(text: str):
    result = sentiment_analyzer(text)
    return result[0] 