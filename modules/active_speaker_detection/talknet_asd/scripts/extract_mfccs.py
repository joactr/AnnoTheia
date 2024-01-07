import os
import joblib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import *

from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor

def process_wav(wav_path):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecting faces and facial landmarks from videos.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--video-dir", required=True, type=str)
    parser.add_argument("--left-index", type=int, default=0)
    parser.add_argument("--right-index", type=int, default=961)
    parser.add_argument("--mfccs-output-dir", required=True, type=str)
    args = parser.parse_args()

    # -- creating output directory structure
    os.makedirs(args.mfccs_output_dir, exist_ok=True)

    datasets = sorted(os.listdir(args.video_dir))
    for dataset in datasets:
        output_dataset_dir = os.path.join(args.mfccs_output_dir, dataset)
        os.makedirs(output_dataset_dir, exist_ok=True)

        dataset_dir = os.path.join(args.video_dir, dataset)
        speakers = sorted(os.listdir(dataset_dir))
        for speaker in speakers:
            output_speaker_dir = os.path.join(output_dataset_dir, speaker)
            os.makedirs(output_speaker_dir, exist_ok=True)

            speaker_dir = os.path.join(dataset_dir, speaker)
            videos = sorted(os.listdir(speaker_dir))
            #TODO
            """
            for video in videos:
                output_path = os.path.join(output_speaker_dir, video.split(".")[0]+".npz")
                video_path = os.path.join(speaker_dir, video)

                # -- processing video
                process_video(video_path)
            """
