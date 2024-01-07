import os
import joblib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import *

from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor

def process_video(video_path):
    videoID = video_path.split(os.sep)[-1].split(".")[0]
    frame_generator = load_video(video_path)

    if os.path.exists(output_path):
        print(f"Video {video_path} has already been processed.")
    else:
        frame_idx = 0
        face_sequence = []
        while True:
            try:
                frame = frame_generator.__next__()
            except StopIteration:
                break

            print(f"\tProcessing frame {frame_idx} from video {videoID}", end="\r")
            detected_faces = face_detector(frame, rgb=False)

            if len(detected_faces) == 0:
                face_sequence.append( np.zeros((112, 112, 3)) )
            else:
                landmarks, scores = landmark_detector(frame, detected_faces, rgb=False)
                transformed_frame, transformed_landmarks = affine_transform(
                    frame,
                    landmarks[0],
                    np.load(args.mean_face_path),
                    grayscale=False,
                )

                start_idx = 0; stop_idx = 68
                crop_height = 112; crop_width = 112

                face_sequence.append(
                    cut_patch(
                        transformed_frame,
                        transformed_landmarks[start_idx:stop_idx],
                        crop_height//2,
                        crop_width//2,
                    ),
                )

            frame_idx += 1

        np.savez_compressed(output_path, data=np.array(face_sequence))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecting faces and facial landmarks from videos.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--video-dir", required=True, type=str)
    parser.add_argument("--left-index", type=int, default=0)
    parser.add_argument("--right-index", type=int, default=961)
    parser.add_argument("--mean-face-path", type=str, default="./scripts/20words_mean_face.npy")
    parser.add_argument("--face-crops-output-dir", required=True, type=str)
    args = parser.parse_args()

    # -- building face detector and alignment
    face_detector = RetinaFacePredictor(
        threshold=0.8, device=args.cuda_device,
        model=RetinaFacePredictor.get_model('resnet50')
    )

    landmark_detector = FANPredictor(
        device=args.cuda_device, model=FANPredictor.get_model('2dfan2_alt')
    )

    # -- creating output directory structure
    os.makedirs(args.face_crops_output_dir, exist_ok=True)

    datasets = sorted(os.listdir(args.video_dir))
    for dataset in datasets:
        output_dataset_dir = os.path.join(args.face_crops_output_dir, dataset)
        os.makedirs(output_dataset_dir, exist_ok=True)

        dataset_dir = os.path.join(args.video_dir, dataset)
        speakers = sorted(os.listdir(dataset_dir))
        for speaker in speakers:
            output_speaker_dir = os.path.join(output_dataset_dir, speaker)
            os.makedirs(output_speaker_dir, exist_ok=True)

            speaker_dir = os.path.join(dataset_dir, speaker)
            videos = sorted(os.listdir(speaker_dir))
            for video in videos:
                output_path = os.path.join(output_speaker_dir, video.split(".")[0]+".npz")
                video_path = os.path.join(speaker_dir, video)

                # -- processing video
                process_video(video_path)
