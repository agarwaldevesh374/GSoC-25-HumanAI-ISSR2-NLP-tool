import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from datetime import timedelta


def process_transcript_with_diarization(transcript_df, diarization_results):
    """
    Combine transcript data with speaker diarization results.
    
    Args:
        transcript_df (DataFrame): DataFrame with columns 'Timestamp', 'Transcription', 'Sentiment'
        diarization_results (list): List of (start_time, end_time, speaker_id) tuples
    
    Returns:
        DataFrame: Transcript data with speaker information added
    """
    # Convert transcript timestamps to seconds
    def parse_timestamp(ts_str):
        parts = ts_str.split('-')
        start_time = float(parts[0].strip().rstrip('s'))
        end_time = float(parts[1].strip().rstrip('s'))
        return start_time, end_time
    
    transcript_df['start_time'] = 0.0
    transcript_df['end_time'] = 0.0
    transcript_df['speaker_id'] = None
    
    for idx, row in transcript_df.iterrows():
        start_time, end_time = parse_timestamp(row['Timestamp'])
        transcript_df.at[idx, 'start_time'] = start_time
        transcript_df.at[idx, 'end_time'] = end_time
        
        # Find the most likely speaker for this time range
        speaker_counts = {}
        for d_start, d_end, speaker in diarization_results:
            # Check overlap between transcript time and diarization segment
            if max(start_time, d_start) < min(end_time, d_end):
                overlap_time = min(end_time, d_end) - max(start_time, d_start)
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + overlap_time
        
        # Assign the speaker with the maximum overlap
        if speaker_counts:
            transcript_df.at[idx, 'speaker_id'] = max(speaker_counts, key=speaker_counts.get)
    
    return transcript_df

def plot_speaker_interaction_analysis(df):
    """
    Generate speaker interaction analysis visualizations.
    
    Args:
        df (DataFrame): DataFrame with columns 'Timestamp', 'Transcription', 'Sentiment', 'speaker_id'
    """
    # 1. Speaker Sentiment Analysis
    sentiment_counts = pd.crosstab(
        df['speaker_id'], 
        df['Sentiment'], 
        normalize='index'
    ) * 100  # Convert to percentage
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sentiment_counts, cmap="Blues", annot=True, fmt=".1f")
    plt.title("Speaker Sentiment Analysis (%)")
    plt.xlabel("Sentiment")
    plt.ylabel("Speaker")
    plt.tight_layout()
    
    # 2. Speaker Word Count Analysis
    df['word_count'] = df['Transcription'].apply(lambda x: len(str(x).split()))
    speaker_words = df.groupby('speaker_id')['word_count'].sum().reset_index()
    speaker_words = speaker_words.sort_values('word_count', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='speaker_id', y='word_count', data=speaker_words)
    plt.title("Words Spoken by Each Speaker")
    plt.xlabel("Speaker")
    plt.ylabel("Word Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 3. Speaker Interaction Heatmap
    # Find who speaks after whom
    df = df.sort_values('start_time')
    df['next_speaker'] = df['speaker_id'].shift(-1)
    interaction_matrix = pd.crosstab(df['speaker_id'], df['next_speaker'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix, cmap="Blues", annot=True, fmt="d")
    plt.title("Speaker Interaction Heatmap (Who Speaks After Whom)")
    plt.xlabel("Next Speaker")
    plt.ylabel("Current Speaker")
    plt.tight_layout()
    
    # 4. Speaker Timeline Visualization
    plt.figure(figsize=(12, 6))
    speakers = df['speaker_id'].unique()
    
    for i, speaker in enumerate(speakers):
        speaker_segments = df[df['speaker_id'] == speaker]
        for _, segment in speaker_segments.iterrows():
            plt.plot([segment['start_time'], segment['end_time']], [i, i], 
                     linewidth=6, alpha=0.7)
    
    plt.yticks(range(len(speakers)), speakers)
    plt.title("Speaker Timeline")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Speaker")
    plt.grid(True, axis='x')
    plt.tight_layout()
    
    # 5. Interaction Network Graph (using networkx if available)
    try:
        import networkx as nx
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes (speakers) - Filter out None values
        speakers = df['speaker_id'].dropna().unique()
        for speaker in speakers:
            G.add_node(speaker)

        # Add edges (interactions) - Drop rows with None values
        df = df.dropna(subset=['speaker_id', 'next_speaker'])  # Ensure no None values
        for _, row in df.iterrows():
            G.add_edge(row['speaker_id'], row['next_speaker'], weight=1)

        
        # Merge parallel edges
        for u, v, data in G.edges(data=True):
            if G.has_edge(v, u):
                G[u][v]['bidirectional'] = True
                G[v][u]['bidirectional'] = True
        
        # Plotting
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, 
                              edge_color=[G[u][v].get('bidirectional', False) and 'r' or 'b' for u, v in G.edges()])
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12)
        
        plt.title("Speaker Interaction Network")
        plt.axis('off')
        plt.tight_layout()
    except ImportError:
        print("NetworkX not available for network graph visualization")
    
    plt.show()

# Main function to combine diarization and transcript analysis
def analyze_audio_with_transcript(audio_file, transcript_df):
    """
    Complete analysis pipeline:
    1. Perform speaker diarization
    2. Combine with transcript
    3. Generate visualizations
    
    Args:
        audio_file (str): Path to audio file
        transcript_df (DataFrame): DataFrame with transcript data
    """
    from speaker_diarization import perform_speaker_diarization, merge_speaker_segments
    
    # Perform diarization
    diarization_result = perform_speaker_diarization(audio_file)
    
    # Merge adjacent segments with the same speaker
    merged_result = merge_speaker_segments(diarization_result)
    
    # Combine with transcript
    df_with_speakers = process_transcript_with_diarization(transcript_df, merged_result)
    
    # Generate visualizations
    plot_speaker_interaction_analysis(df_with_speakers)
    
    return df_with_speakers

# Example usage
if __name__ == "__main__":

    # Load transcript data
    transcript_df = pd.read_csv('files_audio_video/output_audios_csv/dataset1_final.csv') # Update path of csv file
    
    # Path to your audio file
    audio_file = "cleaned_audio/dataset1_cleaned.wav" # Update path of cleaned/noise-removed audio .wav file
    
    # Run the analysis
    result_df = analyze_audio_with_transcript(audio_file, transcript_df)
    
    # Save the results with speaker information
    result_df.to_csv('transcript_with_speakers.csv', index=False)
