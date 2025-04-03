import os
import ffmpeg

def extract_audio(video_path, output_audio_path):
    """Extracts audio from video and saves as a WAV file."""
    input_video = ffmpeg.input(video_path)
    audio = input_video.audio.output(output_audio_path, format="wav")  # Changed to WAV format
    ffmpeg.run(audio, overwrite_output=True, quiet=True)
    return output_audio_path

if __name__ == "__main__":
    video_folder = r"files_audio_video\input_videos"  # Update this path
    output_folder = r"files_audio_video\output_audios_csv"  # Update this path

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all video files in the specified folder
    for filename in os.listdir(video_folder):
        if filename.endswith(('.mp4', '.mkv', '.avi')):  # Add other video formats as needed
            video_path = os.path.join(video_folder, filename)
            output_audio_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.wav")
            extract_audio(video_path, output_audio_path)
            print(f"Extracted audio from {filename} to {output_audio_path}")
