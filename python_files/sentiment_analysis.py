import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(input_csv, output_csv):
    """Performs sentiment analysis on transcriptions and saves to a new CSV."""
    df = pd.read_csv(input_csv)

    sentiments = []
    for text in df["Transcription"]:
        score = sia.polarity_scores(text)["compound"]
        sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
        sentiments.append(sentiment)

    df["Sentiment"] = sentiments
    df.to_csv(output_csv, index=False)
    print(f"Sentiment analysis completed: {output_csv}")

input_csv = "files_audio_video/output_audios_csv/dataset1_final.csv"
output_csv = '/transcript_with_speakers.csv'

if __name__ == "__main__":
    analyze_sentiment(input_csv, output_csv)

