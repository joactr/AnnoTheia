import os
import subprocess

def extract_wav_from_video(video_path, temp_dir):
    """Extracts audio 16 kHz mono-channel waveform from video.
    Args:
        video_path: path where the video clip is stored.
        temp_dir: temporary directory where the audio waveforms will be stored.
    Returns:
        str: path to the resulting audio waveform file.
    """
    video_id = os.path.basename(os.path.realpath(video_path)).split('.')[0]
    wav_path = os.path.join(os.getcwd(), temp_dir, f"{video_id}.wav")

    # Gets working directory and extracts audio
    subprocess.call([
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        "-loglevel",
        "quiet",
        wav_path,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    return wav_path
