import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize




def preprocess_text(df):
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    df['clean_text'] = df['review_text'].apply(clean_text)
    return df
