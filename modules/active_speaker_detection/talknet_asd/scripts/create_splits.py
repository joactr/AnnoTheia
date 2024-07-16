import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TYPE_OF_SAMPLES = [
    "POSITIVE",
    "NEGATIVE_TEMPORAL_MISMATCH",
    "NEGATIVE_PARTIAL_SPEAKER_MISMATCH",
    "NEGATIVE_COMPLETE_SPEAKER_MISMATCH",
]

def get_samples_per_speaker(video_dataset_dir, video_root, audio_root):
    samples_per_speaker = {}

    speakers = sorted(os.listdir(video_dataset_dir))
    for speaker in tqdm(speakers, desc="Processing speakers"):
        speaker_dir = os.path.join(video_dataset_dir, speaker)
        videos = sorted(os.listdir(speaker_dir))
        samples_per_speaker[speaker] = []
        for video in videos:
            video_path = os.path.join(speaker_dir, video)
            audio_path = video_path.replace(video_root, audio_root)
            video_length = np.load(video_path)["data"].shape[0]
            samples_per_speaker[speaker].append((video, video_length, video_path, audio_path))

    return samples_per_speaker

def get_dataset_split(samples_per_speaker, n_samples):
    dataset = []
    speaker_ids = list(samples_per_speaker.keys())

    pbar = tqdm(total=n_samples, desc="Creating dataset split")
    attempts = 0
    max_attempts = n_samples * 10  # Limit the number of attempts to avoid infinite loops

    while len(dataset) < n_samples and attempts < max_attempts:
        attempts += 1
        sample_type = np.random.choice(TYPE_OF_SAMPLES, p=[0.5, 0.166666666, 0.166666666, 0.166666666])

        video_speaker_id = random.choice(speaker_ids)
        video_id, video_length, video_path, audio_path = random.choice(samples_per_speaker[video_speaker_id])

        window_center = random.randint(0, video_length)

        try:
            if sample_type == "POSITIVE":
                audio_id = video_id
                label_id = 1
            elif sample_type == "NEGATIVE_TEMPORAL_MISMATCH":
                audio_id = video_id
                label_id = 0
            elif sample_type == "NEGATIVE_PARTIAL_SPEAKER_MISMATCH":
                audio_speaker_id = video_speaker_id
                if len(samples_per_speaker[audio_speaker_id]) > 1:
                    audio_id = random.choice([sample[0] for sample in samples_per_speaker[audio_speaker_id] if sample[0] != video_id])
                else:
                    audio_id = None
                label_id = 0
            elif sample_type == "NEGATIVE_COMPLETE_SPEAKER_MISMATCH":
                audio_speaker_id = random.choice([speaker for speaker in speaker_ids if speaker != video_speaker_id])
                audio_id, _, _, _ = random.choice(samples_per_speaker[audio_speaker_id])
                label_id = 0
            else:
                raise ValueError(f"Unknown sample type: {sample_type}")

            new_sample = (video_id, audio_id, label_id, window_center, video_path, audio_path)
            if (new_sample not in dataset) and (new_sample[1] is not None):
                dataset.append(new_sample)
                pbar.update(1)
            
        except Exception as e:
            logging.error(f"Error creating sample: {str(e)}")
            logging.error(f"Sample type: {sample_type}")
            logging.error(f"Video speaker ID: {video_speaker_id}")
            logging.error(f"Video ID: {video_id}")
            continue

    pbar.close()

    if len(dataset) < n_samples:
        logging.warning(f"Could only generate {len(dataset)} samples out of {n_samples} requested.")

    df_dataset = pd.DataFrame(dataset, columns=["video_id", "audio_id", "label", "window_center", "video_path", "audio_path"])
    return df_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset splits for fine-tuning TalkNet-ASD to your language.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-samples", default=100_000, type=int, help="Number of samples for the training dataset. Validation and test will have a 30%% of them")
    parser.add_argument("--face-crops-dir", required=True, type=str, help="Directory where extracted face crops were stored")
    parser.add_argument("--mfccs-dir", required=True, type=str, help="Directory where MFCCs were stored")
    parser.add_argument("--splits-output-dir", required=True, type=str, help="Directory where the resulting dataset split")
    args = parser.parse_args()

    if not os.path.exists(args.splits_output_dir):
        os.makedirs(args.splits_output_dir)
        logging.info(f"Created directory: {args.splits_output_dir}")

    datasets = sorted(os.listdir(args.face_crops_dir))
    for dataset in tqdm(datasets, desc="Processing datasets"):
        dataset_dir = os.path.join(args.face_crops_dir, dataset)
        samples_per_speaker = get_samples_per_speaker(dataset_dir, args.face_crops_dir, args.mfccs_dir)

        if "train" in dataset:
            df_dataset = get_dataset_split(samples_per_speaker, args.n_samples)
        else:
            notrain_n_samples = int(args.n_samples * 0.3)
            df_dataset = get_dataset_split(samples_per_speaker, notrain_n_samples)

        output_path = os.path.join(args.splits_output_dir, f"{dataset}.csv")
        df_dataset.to_csv(output_path, index=False)
        logging.info(f"Saved dataset split to {output_path}")