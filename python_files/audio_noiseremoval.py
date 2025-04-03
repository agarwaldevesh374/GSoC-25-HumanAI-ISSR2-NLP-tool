import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import time
from tqdm import tqdm  # For progress bar

def remove_noise_spectral_subtraction(input_file, output_file, noise_length=1.0, reduction_factor=1.5):
    """
    Remove noise from audio using spectral subtraction method.
    
    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to save the processed audio file
        noise_length (float): Length of audio (in seconds) to use for noise profile estimation
        reduction_factor (float): Factor to scale the noise profile (higher = more aggressive noise reduction)
    """
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    
    # Ensure we have enough audio for noise profile
    noise_samples = min(int(noise_length * sr), len(y) // 4)  # Use at most 1/4 of audio for noise
    
    # Estimate noise profile from the first noise_length seconds
    noise_profile = y[:noise_samples]
    
    # Compute noise spectrum
    noise_fft = np.abs(librosa.stft(noise_profile))
    noise_spectrum = np.mean(noise_fft, axis=1)
    
    # Compute STFT of the entire signal
    stft = librosa.stft(y)
    stft_mag = np.abs(stft)
    stft_phase = np.angle(stft)
    
    # Apply spectral subtraction
    stft_mag_reduced = np.maximum(
        stft_mag - noise_spectrum.reshape(-1, 1) * reduction_factor, 
        0.01 * stft_mag
    )
    
    # Reconstruct the signal
    stft_reduced = stft_mag_reduced * np.exp(1j * stft_phase)
    y_reduced = librosa.istft(stft_reduced)
    
    # Save the processed audio
    sf.write(output_file, y_reduced, sr)
    
    return y_reduced, sr

def remove_noise_butterworth_filter(input_file, output_file, lowcut=100, highcut=8000, order=5):
    """
    Remove noise using Butterworth bandpass filter.
    
    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to save the processed audio file
        lowcut (int): Low frequency cutoff
        highcut (int): High frequency cutoff
        order (int): Filter order
    """
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    
    # Design Butterworth bandpass filter
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter
    y_filtered = signal.filtfilt(b, a, y)
    
    # Save the processed audio
    sf.write(output_file, y_filtered, sr)
    
    return y_filtered, sr

def try_import_noisereduce():
    """Try to import noisereduce, return True if successful."""
    try:
        import noisereduce
        return True
    except ImportError:
        print("Warning: noisereduce package not found. Method 3 will not be available.")
        print("Install with: pip install noisereduce")
        return False

def remove_noise_noisereduce(input_file, output_file, noise_length=1.0, prop_decrease=0.75):
    """
    Remove noise using the noisereduce library (more advanced method).
    
    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to save the processed audio file
        noise_length (float): Length of audio (in seconds) to use for noise profile estimation
        prop_decrease (float): The proportion to decrease the noise by (1.0 = completely remove)
        verbose (bool): Whether to print progress
    """
    import noisereduce as nr
    
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)
    
    # Ensure we have enough audio for noise profile
    noise_samples = min(int(noise_length * sr), len(y) // 4)  # Use at most 1/4 of audio for noise
    
    # Estimate noise profile from the first noise_length seconds
    noise_clip = y[:noise_samples]
    
    # Apply noise reduction
    y_reduced = nr.reduce_noise(
        y=y, 
        sr=sr,
        y_noise=noise_clip,
        prop_decrease=prop_decrease,
        # verbose=verbose
    )
    
    # Save the processed audio
    sf.write(output_file, y_reduced, sr)
    
    return y_reduced, sr

def process_audio_folder(input_folder, output_folder, method='noisereduce', **kwargs):
    """
    Process all audio files in a folder.
    
    Args:
        input_folder (str): Path to folder containing audio files
        output_folder (str): Path to folder for saving processed files
        method (str): Noise removal method ('spectral', 'butterworth', 'noisereduce')
        **kwargs: Additional parameters for the selected method
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all audio files in the input folder
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = [f for f in os.listdir(input_folder) 
                  if os.path.splitext(f.lower())[1] in audio_extensions]
    
    if not audio_files:
        print(f"No audio files found in {input_folder}")
        return
    
    print(f"Found {len(audio_files)} audio files. Processing...")
    
    # Select method function
    if method == 'spectral':
        noise_func = remove_noise_spectral_subtraction
    elif method == 'butterworth':
        noise_func = remove_noise_butterworth_filter
    elif method == 'noisereduce':
        # Check if noisereduce is available
        if not try_import_noisereduce():
            print("Falling back to spectral subtraction method.")
            noise_func = remove_noise_spectral_subtraction
        else:
            noise_func = remove_noise_noisereduce
    else:
        print(f"Unknown method: {method}. Using spectral subtraction.")
        noise_func = remove_noise_spectral_subtraction
    
    # Process each file with progress bar
    start_time = time.time()
    success_count = 0
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        input_path = os.path.join(input_folder, audio_file)
        # Create output filename with same extension
        file_name, file_ext = os.path.splitext(audio_file)
        output_file = f"{file_name}_cleaned{file_ext}"
        output_path = os.path.join(output_folder, output_file)
        
        try:
            # Process the file
            noise_func(input_path, output_path, **kwargs)
            success_count += 1
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete: {success_count}/{len(audio_files)} files successfully processed")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Cleaned audio files saved to: {output_folder}")

if __name__ == "__main__":
    # Set your input and output folders
    input_folder = "files_audio_video/output_audios_csv"  # Folder containing noisy audio files
    output_folder = "cleaned_audio"  # Folder to save processed noise-removed files
    
    # Choose one of the following methods:
    
    # Method 1: Spectral Subtraction
    # process_audio_folder(input_folder, output_folder, method='spectral', 
    #                     noise_length=1.0, reduction_factor=1.5)
    
    # Method 2: Butterworth Bandpass Filter
    # process_audio_folder(input_folder, output_folder, method='butterworth',
    #                     lowcut=100, highcut=8000, order=5)
    
    # Method 3: Advanced Noise Reduction (requires noisereduce package)
    process_audio_folder(input_folder, output_folder, method='noisereduce',
                         noise_length=1.0, prop_decrease=0.75)