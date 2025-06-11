from scripts.data_preparation import load_and_clean_data
from scripts.eda import explore_data
from scripts.text_preprocessing import preprocess_text
from scripts.sentiment_analysis import analyze_sentiment
from scripts.model_training import train_model
from scripts.evaluation import evaluate_model

if __name__ == "__main__":
    # Step 1: Load and clean data
    df = load_and_clean_data("data/reviews.csv")

    # Step 2: EDA
    explore_data(df)

    # Step 3: Text Preprocessing
    df = preprocess_text(df)

    # Step 4: Sentiment Analysis
    df = analyze_sentiment(df)  # Ensure sentiment column is present here

    # Step 5: Train Model
    model, vectorizer, X_test_vec, y_test = train_model(df)

    # Step 6: Evaluate Model
    evaluate_model(model, X_test_vec, y_test)

    # Show sentiment distribution graph after the sentiment analysis
    explore_data(df)  # Ensure plotting after the sentiment column is available
