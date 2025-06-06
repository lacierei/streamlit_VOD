import whisper
import ffmpeg
import numpy as np
import pandas as pd
import os
from pyAudioAnalysis import audioBasicIO
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
# Fix potentiel pour certains chemins mal gérés
os.environ["PYTHONPATH"] = ""
def transcribe_video(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result["segments"]

def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

def analyze_audio_energy(video_path):
    audio_path = "temp_audio.wav"
    ffmpeg.input(video_path).output(audio_path, ac=1, ar="16000").overwrite_output().run()
    [Fs, x] = audioBasicIO.read_audio_file(audio_path)
    x = x[:, 0] if x.ndim > 1 else x
    segment_size = Fs
    rms_values = [np.sqrt(np.mean(x[i:i+segment_size]**2)) for i in range(0, len(x), segment_size)]
    os.remove(audio_path)
    return rms_values

def merge_and_score(segments, scenes, audio_energy):
    timeline = pd.DataFrame({
        "second": list(range(len(audio_energy))),
        "audio_score": audio_energy
    })

    timeline["scene_cut"] = 0
    for start, end in scenes:
        timeline.loc[(timeline["second"] >= start) & (timeline["second"] <= end), "scene_cut"] += 1

    timeline["text_score"] = 0
    for seg in segments:
        start = int(seg['start'])
        end = int(seg['end'])
        for t in range(start, end + 1):
            if t < len(timeline):
                timeline.loc[t, "text_score"] += 1

    timeline["score"] = (
        0.5 * timeline["audio_score"].apply(lambda x: min(x, 0.1)) +
        0.3 * timeline["scene_cut"] +
        0.2 * timeline["text_score"]
    )

    return timeline

def summarize_top_moments(timeline, top_n=5):
    best_moments = timeline.sort_values(by="score", ascending=False).head(top_n)
    return best_moments[["second", "score"]]

def analyze_vod(video_path):
    segments = transcribe_video(video_path)
    scenes = detect_scenes(video_path)
    audio_energy = analyze_audio_energy(video_path)
    timeline = merge_and_score(segments, scenes, audio_energy)
    top_moments = summarize_top_moments(timeline)
    return top_moments
