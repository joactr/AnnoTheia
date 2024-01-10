import os
import random
import argparse
import numpy as np
import pandas as pd

TYPE_OF_SAMPLES = [
    "POSITVE",
    "NEGATIVE_TEMPORAL_MISMATCH",
    "NEGATIVE_PARTIAL_SPEAKER_MISMATCH",
    "NEGATIVE_COMPLETE_SPEAKER_MISMATCH",
]

def get_samples_per_speaker(video_dataset_dir, video_root, audio_root):
    samples_per_speaker = {}

    speakers = sorted(os.listdir(video_dataset_dir))
    for speaker in speakers:
        speaker_dir = os.path.join(video_dataset_dir, speaker)
        videos = sorted(os.listdir(speaker_dir))
        for video in videos:
            video_path = os.path.join(speaker_idr, video)
            audio_path = video_path.replace(video_root, audio_root)
            video_length = np.load(video_path)["data"].shape[0]

            if speaker in samples_per_speaker:
                samples_per_speaker.append([(video_id, video_length, video_path, audio_path)])
            else:
                samples_per_speaker = [(video_id, video_length, video_path, audio_path)]

    return samples_per_speaker

def get_dataset_split(samples_per_speaker, n_samples):
    dataset = []
    speaker_ids = list(samples_per_speaker.keys())

    count = 0
    repeat_type = False
    while count < n_samples:
        # -- in case we are not repeating the type of sample, we got a new one
        if not repeat_type:
            sample_type = np.random.choice(TYPE_OF_SAMPLES, p=[0.5,0.166666666,0.166666666,0.166666666])

        # -- sampling a video sample
        video_speaker_id = random.choice( speaker_ids )
        video_id, video_length, video_path, audio_path = random.choice( samples_per_speaker[speaker_id] )

        # -- obtaining a random window center
        window_center = random.randint(0, video_length)

        # -- differences between type of samples
        if sample_type == "POSITIVE":
            audio_id = video_id
            label_id = 1

        elif sample_type == "NEGATIVE_TEMPORAL_MISMATCH":
            audio_id = video_id
            label_id = 0
            # -- note that the audio window center will be shifted into the TalkNetDataset when training the model

        elif sample_type == "NEGATIVE_PARTIAL_SPEAKER_MISMATCH":
            audio_speaker_id = video_speaker_id

            if len(samples_per_speaker[audio_speaker_id]) > 1:
                audio_id = random.choice( [sample for sample in samples_per_speaker[audio_speaker_id] if sample != video_id] )
            else:
                audio_id = None

            label_id = 0

        elif sample_type == "NEGATIVE_COMPLETE_SPEAKER_MISMATCH":
            audio_speaker_id = random.choice( [speaker for speaker in speaker_ids if speaker != video_speaker_id] )
            audio_id = random.choice( samples_per_speaker[audio_speaker_id] )
            label_id = 0

        else:
            raise ValueError("Ups! It should not happen :S")

        # -- adding new sample to the dataset split
        new_sample = (video_id, audio_id, label, window_center, video_path, audio_path)
        if (new_sample not in dataset) and (new_sample[1] is not None)
            dataset.append(new_sample)
            repeat_type = False
            count = count + 1
        else:
            repeat_type = True

    # -- creating dataset split
    df_dataset = pd.DataFrame(dataset, columns=["video_id", "audio_id", "label", "video_path", "audio_path"])

    return df_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset splits for fine-tuning TalkNet-ASD to your language.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-samples", default=100_000, type=int, help="Number of samples for the training dataset. Validation and test will have a 30% of them")
    parser.add_argument("--face-crops-dir", required=True, type=str, help="Directory where extracted face crops were stored")
    parser.add_argument("--mfccs-dir", required=True, type=str, help="Directory where MFCCs were stored")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory where the resulting dataset split")
    args = parser.parse_args()

    datasets = sorted(os.listdir(args.face_crops_dir))
    for dataset in datasets:
        dataset_dir = os.path.join(args.face_crops_dir, dataset)
        samples_per_speaker = get_samples_per_speaker(dataset_dir, args.face_crops_dir, args.mfccs_dir)

        df_dataset = get_dataset_split(samples_per_speaker, args.n_samples)

        output_path = os.path.join(args.output_dir, f"{dataset}.csv")
        df_dataset.to_csv(output_path, index = False)
