import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from utils import load_video, affine_transform, cut_patch

from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor

def process_video(video_path, output_path, face_detector, landmark_detector, args):
    videoID = video_path.split(os.sep)[-1].split(".")[0]
    frame_generator = load_video(video_path)

    if os.path.exists(output_path):
        return f"Video {video_path} has already been processed."
    else:
        frame_idx = 0
        face_sequence = []
        total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame in tqdm(frame_generator, total=total_frames, desc=f"Processing {videoID}", leave=False):
            detected_faces = face_detector(frame, rgb=False)

            if len(detected_faces) == 0:
                # If there are no faces detected, use a black image as the face crop.
                face_sequence.append(np.zeros((112, 112)))
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

                face_crop = cut_patch(
                    transformed_frame,
                    transformed_landmarks[start_idx:stop_idx],
                    crop_height//2,
                    crop_width//2,
                )

                face_crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_sequence.append(face_crop_gray)

            frame_idx += 1
        
        np.savez_compressed(output_path, data=np.array(face_sequence))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the 112x112 face crops TalkNet-ASD is expecting.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--video-dir", required=True, type=str)
    parser.add_argument("--mean-face-path", type=str, default="./scripts/20words_mean_face.npy")
    parser.add_argument("--face-crops-output-dir", required=True, type=str)
    args = parser.parse_args()

    face_detector = RetinaFacePredictor(
        threshold=0.8, device=args.cuda_device,
        model=RetinaFacePredictor.get_model('resnet50')
    )

    landmark_detector = FANPredictor(
        device=args.cuda_device, model=FANPredictor.get_model('2dfan2_alt')
    )

    os.makedirs(args.face_crops_output_dir, exist_ok=True)

    datasets = sorted(os.listdir(args.video_dir))
    for dataset in tqdm(datasets, desc="Processing datasets"):
        output_dataset_dir = os.path.join(args.face_crops_output_dir, dataset)
        os.makedirs(output_dataset_dir, exist_ok=True)

        dataset_dir = os.path.join(args.video_dir, dataset)
        speakers = sorted(os.listdir(dataset_dir))
        for speaker in tqdm(speakers, desc=f"Processing speakers in {dataset}", leave=False):
            output_speaker_dir = os.path.join(output_dataset_dir, speaker)
            os.makedirs(output_speaker_dir, exist_ok=True)

            speaker_dir = os.path.join(dataset_dir, speaker)
            videos = sorted(os.listdir(speaker_dir))
            for video in tqdm(videos, desc=f"Processing videos for {speaker}", leave=False):
                output_path = os.path.join(output_speaker_dir, video.split(".")[0]+".npz")
                video_path = os.path.join(speaker_dir, video)

                process_video(video_path, output_path, face_detector, landmark_detector, args)




































if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the 112x112 face crops TalkNet-ASD is expecting.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--video-dir", required=True, type=str)
    parser.add_argument("--mean-face-path", type=str, default="./scripts/20words_mean_face.npy")
    parser.add_argument("--face-crops-output-dir", required=True, type=str)
    args = parser.parse_args()

    face_detector = RetinaFacePredictor(
        threshold=0.8, device=args.cuda_device,
        model=RetinaFacePredictor.get_model('resnet50')
    )

    landmark_detector = FANPredictor(
        device=args.cuda_device, model=FANPredictor.get_model('2dfan2_alt')
    )

    os.makedirs(args.face_crops_output_dir, exist_ok=True)

    datasets = sorted(os.listdir(args.video_dir))
    total_videos = sum(len(os.listdir(os.path.join(args.video_dir, dataset, speaker)))
                       for dataset in datasets
                       for speaker in os.listdir(os.path.join(args.video_dir, dataset)))

    with tqdm(total=total_videos, desc="Processing videos") as pbar:
        for dataset in datasets:
            output_dataset_dir = os.path.join(args.face_crops_output_dir, dataset)
            os.makedirs(output_dataset_dir, exist_ok=True)

            dataset_dir = os.path.join(args.video_dir, dataset)
            speakers = sorted(os.listdir(dataset_dir))
            for speaker in tqdm(speakers, desc=f"Processing speakers in {dataset}", leave=False):
                output_speaker_dir = os.path.join(output_dataset_dir, speaker)
                os.makedirs(output_speaker_dir, exist_ok=True)

                speaker_dir = os.path.join(dataset_dir, speaker)
                videos = sorted(os.listdir(speaker_dir))
                for video in tqdm(videos, desc=f"Processing videos for {speaker}", leave=False):
                    output_path = os.path.join(output_speaker_dir, video.split(".")[0]+".npz")
                    video_path = os.path.join(speaker_dir, video)

                    process_video(video_path, output_path, face_detector, landmark_detector, args)
                    pbar.update(1)