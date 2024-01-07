import os
import joblib
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

import python_speech_features
import scipy.io.wavfile as scipy_wavfile

def process_wav(video_path):
    # -- creating a temporary file containing the audio waveform
    subprocess.call([
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "./temp.wav",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,)

    # -- laoding audio waveform
    _, audio_waveform = scipy_wavfile.read("./temp.wav")

    # -- extracting MFCCs at 100fps
    mfccs = python_speech_features.mfcc(
        audio_waveform,
        samplerate=16000,
        numcep=13,
        winlen=0.025,
        winstep=0.010,
   )

   # -- audiovisual temproral alignment
   face_crops_seq = np.load(face_crop_path)["data"]

   # -- assuming video at 25fps and audio at 100fps
   max_mfccs_length = face_crops_seq.shape[0] * 4

   # -- if it smaller, we apply padding
   if mfccs.shape[0] < max_mfccs_length:
       pad_amount = max_mfccs_length - mfccs.shape[0]
       mfccs = np.pad(mfccs, ((0,pad_amount), (0,0)), 'wrap')

   # contrary, we cut the tail of the sequence
   mfccs = mfccs[:max_mfccs_length, :]

   # -- saving MFCCs in a compressed npz file
   np.savez_compressed(output_path, data=np.array(mfccs))

   # -- removing temporary file
   os.remove("./temp.wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracting the 13-component MFCCs TalkNet-ASD is expecting.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--video-dir", required=True, type=str)
    parser.add_argument("--face-crops-dir", required=True, type=str)
    parser.add_argument("--mfccs-output-dir", required=True, type=str)
    args = parser.parse_args()

    # -- creating output directory structure
    os.makedirs(args.mfccs_output_dir, exist_ok=True)

    datasets = sorted(os.listdir(args.video_dir))
    for dataset in tqdm(datasets, leave=False):
        output_dataset_dir = os.path.join(args.mfccs_output_dir, dataset)
        os.makedirs(output_dataset_dir, exist_ok=True)

        dataset_dir = os.path.join(args.video_dir, dataset)
        speakers = sorted(os.listdir(dataset_dir))
        for speaker in tqdm(speakers, leave=False):
            output_speaker_dir = os.path.join(output_dataset_dir, speaker)
            os.makedirs(output_speaker_dir, exist_ok=True)

            speaker_dir = os.path.join(dataset_dir, speaker)
            videos = sorted(os.listdir(speaker_dir))
            for video in tqmd(videos):
                face_crop_path = os.path.join(args.face_crops_dir, dataset, speaker, video.split(".")[0]+".npz")
                output_path = os.path.join(output_speaker_dir, video.split(".")[0]+".npz")
                video_path = os.path.join(speaker_dir, video)

                # -- processing video
                process_video(video_path)
