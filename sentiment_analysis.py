from textblob import TextBlob

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

def analyze_sentiment(df):
    df['sentiment'] = df['clean_text'].apply(get_sentiment)  # Ensure sentiment is added
    return df
