# GSoC-25-HumanAI-ISSR2-Screening-Test--NLP-tool
#### (Python/Machine Learning/Natural Language Processing)
## GSoC-2025 @ HumanAI > ISSR > "Communication Analysis Tool for Human-AI Interaction Driving Simulator Experiments"

### Description of the project:
The project basically focuses on building up of a fully-functional prototype of the AI-driven communication analysis tool, which explores and analyzes the high-fidelity audiovisual data of communication between the individuals involved in driving simulation. It helps to better understand the group dynamics, improvising safety measures, and incorporating (AI and machine learning / natural language processing) in the real-world driving scenarios. The project involves advanced NLP and visualization techniques. The key features of the tool includes - Data ingestion, Data processing, Key variable extraction, Speaker identification, Timestamping, Content analysis, Dynamic Reporting, User interface, and User-centric customization.


## Step-by-step Procedure to run the project ->

### Step 0:
Create a virtual environment, with **Python** **3.11** version
# Make sure to use Python "11" in the (.venv)

```powershell
python -m venv .venv
```
then
```
.\Scripts\activate 
```

*#virtual-environment activated!*


### Step 1:
Run the following command to install all necessary Python libraries:
```powershell
cd <your-folder-name>
pip install -r requirements.txt

```

### Step 2: Add Your Video Files
Put all video files in the `files_audio_video/input_videos` directory.

### Step 3: Run the Processing Pipeline
Run the following command to process all videos in the `input_videos/` folder:
```powershell
python "/python_files/process_all_videos.py"
```
This will:
- Extract audio from each video. --> (audio_extraction.py / audio_noiseremoval.py)
- Transcribe the audio into text. --> (transcription.py)
- Perform sentiment analysis (Positive, Negative, Neutral). --> (sentiment_analysis.py)
- Save the results in `files_audio_video/output_audios_csv/`.

#### Expected Output
Each processed video will generate a CSV file in `files_audio_video/output_audios_csv/` with the following format:

| Timestamp | Transcription | Sentiment |
|-----------|--------------|-----------|
| 0-5 sec   | Hello team!  | Positive  |
| 5-10 sec  | I don't agree with this... | Negative |
| 10-15 sec | Let's find a solution. | Neutral |

---
### Before visualization, run these files -->
```
cd python_files
```

### Step 3.1: Run 'audio_extraction.py' file
```
python audio_extraction.py
```
### Step 3.2: Run 'audio_noiseremoval.py' file
```
python audio_noiseremoval.py
```
### Step 3.3: Run 'speaker_diarization.py' file
```
python speaker_diarization.py
```

### Step 4: Visualize various plots
To analyze a specific processed file, run:
```powershell
python python_files/visualization_plots.py 'your_video_filename>.csv'  
# eg.-python .\python_files\visualization_plots.py dataset1_final.csv
```

This will generate following:
![Figure_1](https://github.com/user-attachments/assets/f302b7ce-29d6-487f-aad4-36e58f8d76c0)

![Figure_2](https://github.com/user-attachments/assets/b5a2e5fd-6319-40f7-a68a-89d26b29499a)

![Figure_3](https://github.com/user-attachments/assets/58bb5a58-4637-41a9-8474-40da604226bc)

![Figure_4](https://github.com/user-attachments/assets/a76be531-226f-4dcb-8b9c-c1b6314702e1)

![Figure_5](https://github.com/user-attachments/assets/5ae1c492-8814-442b-949f-e845e5ce9505)

---

