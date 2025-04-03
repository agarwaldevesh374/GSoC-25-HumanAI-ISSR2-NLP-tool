import os
from audio_extraction import extract_audio
from transcription import transcribe_audio
from sentiment_analysis import analyze_sentiment

# directories

INPUT_FOLDER = r"files_audio_video\input_videos"
OUTPUT_FOLDER = r"files_audio_video\output_audios_csv"

def process_all_videos():
    """Processes all videos one by one."""
    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".mp4")]

    for video in video_files:
        video_path = os.path.join(INPUT_FOLDER, video)
        base_name = os.path.splitext(video)[0]

        audio_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.wav")
        csv_transcription = os.path.join(OUTPUT_FOLDER, f"{base_name}_transcription.csv")
        csv_final = os.path.join(OUTPUT_FOLDER, f"{base_name}_final.csv")

        print(f"Processing {video}...")

        extract_audio(video_path, audio_path)
        transcribe_audio(audio_path, csv_transcription)
        analyze_sentiment(csv_transcription, csv_final)

        # Delete intermediate files after processing
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Deleted intermediate file: {audio_path}")

        if os.path.exists(csv_transcription):
            os.remove(csv_transcription)
            print(f"Deleted intermediate file: {csv_transcription}")

        print(f"Finished processing: {video}")

if __name__ == "__main__":
    process_all_videos()
