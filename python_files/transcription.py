import whisper
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

model = whisper.load_model("base")

def transcribe_audio(audio_path, output_csv):
    """Transcribes an audio file and processes it into 5-second segments."""
    result = model.transcribe(audio_path)
    segments = result["segments"]

    grouped_segments = group_into_5s_segments(segments)

    df = pd.DataFrame({
        "Timestamp": [t for t, _ in grouped_segments],
        "Transcription": [text for _, text in grouped_segments]
    })
    df.to_csv(output_csv, index=False)
    print(f"Transcription completed: {output_csv}")

# Process Transcription into 5-Second Segments
def group_into_5s_segments(segments):
    grouped_segments = []
    current_segment_text = []
    segment_start_time = 0
    for segment in segments:
        start_time = round(segment["start"], 2)
        end_time = round(segment["end"], 2)
        text = segment["text"]

        # If the current segment exceeds 5 seconds, start a new one
        if start_time - segment_start_time >= 5:
            if current_segment_text:
                grouped_segments.append((
                    f"{segment_start_time}s - {segment_start_time+5}s",
                    " ".join(current_segment_text)
                ))
            current_segment_text = []
            segment_start_time = start_time

        current_segment_text.append(text)

    # Add the last segment
    if current_segment_text:
        grouped_segments.append((
            f"{segment_start_time}s - {segment_start_time+5}s",
            " ".join(current_segment_text)
        ))

    return grouped_segments
