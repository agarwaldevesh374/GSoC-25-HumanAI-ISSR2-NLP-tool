import os
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition

# Function to perform speaker diarization
def perform_speaker_diarization(audio_file, window_size=5.0, step_size=1.0):
    """
    Perform speaker diarization on an audio file.
    
    Args:
        audio_file (str): Path to the audio file
        window_size (float): Size of the sliding window in seconds
        step_size (float): Step size for the sliding window in seconds
    
    Returns:
        list: List of (start_time, end_time, speaker_id) tuples
    """
    # Load the audio file
    signal, fs = torchaudio.load(audio_file)
    signal = signal.squeeze(0)
    
    # Convert to mono if stereo
    if len(signal.shape) > 1 and signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True).squeeze(0)
    
    # Load the speaker embedding model
    speaker_embeddings = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    
    # Calculate the number of windows
    audio_length = signal.shape[0] / fs
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    
    # Extract embeddings for each window
    embeddings = []
    timestamps = []
    
    for start_sample in range(0, signal.shape[0] - window_samples, step_samples):
        end_sample = start_sample + window_samples
        window = signal[start_sample:end_sample]
        
        # Extract embedding for the current window
        emb = speaker_embeddings.encode_batch(window.unsqueeze(0))
        embeddings.append(emb.squeeze().cpu().detach())
        
        # Calculate timestamps
        start_time = start_sample / fs
        end_time = end_sample / fs
        timestamps.append((start_time, end_time))
    
    # Convert embeddings to a tensor
    embeddings_tensor = torch.stack(embeddings)
    
    # Perform clustering to identify speakers
    # Using a simple cosine similarity-based clustering
    num_windows = embeddings_tensor.shape[0]
    similarity_matrix = torch.zeros((num_windows, num_windows))
    
    for i in range(num_windows):
        for j in range(num_windows):
            similarity_matrix[i, j] = torch.nn.functional.cosine_similarity(
                embeddings_tensor[i].unsqueeze(0),
                embeddings_tensor[j].unsqueeze(0)
            )
    
    # Apply threshold-based clustering
    threshold = 0.75  # Adjust this threshold based on your needs
    speaker_labels = [-1] * num_windows
    current_speaker = 0
    
    for i in range(num_windows):
        if speaker_labels[i] == -1:
            speaker_labels[i] = current_speaker
            for j in range(i + 1, num_windows):
                if similarity_matrix[i, j] > threshold and speaker_labels[j] == -1:
                    speaker_labels[j] = current_speaker
            current_speaker += 1
    
    # Generate result with timestamps and speaker labels
    result = []
    for i in range(num_windows):
        start_time, end_time = timestamps[i]
        result.append((start_time, end_time, f"speaker_{speaker_labels[i]}"))
    
    return result

# Function to merge adjacent segments with the same speaker
def merge_speaker_segments(diarization_result):
    """
    Merge adjacent segments with the same speaker.
    
    Args:
        diarization_result (list): List of (start_time, end_time, speaker_id) tuples
    
    Returns:
        list: Merged list of (start_time, end_time, speaker_id) tuples
    """
    if not diarization_result:
        return []
    
    merged_result = [diarization_result[0]]
    
    for segment in diarization_result[1:]:
        start_time, end_time, speaker_id = segment
        prev_start, prev_end, prev_speaker = merged_result[-1]
        
        if speaker_id == prev_speaker and abs(start_time - prev_end) < 0.5:  # Merge if gap < 0.5s
            merged_result[-1] = (prev_start, end_time, speaker_id)
        else:
            merged_result.append(segment)
    
    return merged_result

# Example usage
if __name__ == "__main__":
    # Path to your audio file
    audio_file = "cleaned_audio/dataset1_cleaned.wav" # Update path of audio file
    
    # Perform diarization
    diarization_result = perform_speaker_diarization(audio_file)
    
    # Merge adjacent segments with the same speaker
    merged_result = merge_speaker_segments(diarization_result)
    
    # Print the results
    print("Speaker Diarization Results:")
    for start_time, end_time, speaker_id in merged_result:
        print(f"{start_time:.2f}s - {end_time:.2f}s: {speaker_id}")