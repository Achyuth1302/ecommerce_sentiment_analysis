import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    print("Data Info:\n")
    print(df.info())
    print("\nMissing Values:\n")
    print(df.isnull().sum())

    # Check if sentiment column exists before plotting
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts()

        # Plot sentiment distribution using Seaborn for better style
        plt.figure(figsize=(6, 4))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.show()