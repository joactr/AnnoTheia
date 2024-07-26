import os
import cv2
import glob
import pickle
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm

def load_video(filename):
    """load_video.
        :param filename: str, the fileanme for a video sequence.
    """
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()

def read_pickle(filename):
    with open(filename, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data

if __name__ == '__main__':

    # -- command line arguments
    parser = argparse.ArgumentParser(description='DisVoice-based Prosody Feature Extraction', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--annotated-dir', type=str, default='./output/2018/SYG-2018/')
    parser.add_argument('--output-dir', required=True, type=str)
    args = parser.parse_args()

    # -- creating output directory
    os.makedirs(os.path.join(args.output_dir, 'MP4s'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'WAVs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'FACEs'), exist_ok=True)

    # -- reading annotations
    annotated_programs_paths = glob.glob(f'{args.annotated_dir}{os.path.sep}**{os.path.sep}*_annotated.csv')

    for program_idx, annotated_program_path in enumerate(tqdm(annotated_programs_paths, leave=False)):
        annotated_program = pd.read_csv(annotated_program_path)

        for sample_idx, sample in tqdm(annotated_program.iterrows(), total=annotated_program.shape[0]):
            video_source_path = sample['video']
            sample_start = sample['sample_start']
            sample_duration = sample['duration']

            # -- saving trimmed scene
            output_path = os.path.join(
                args.output_dir,
                'MP4s',
                f'{os.path.basename(video_source_path).split(".")[0]}_{sample_start}_{sample["sample_end"]}.mp4',
            )

            # subprocess.call([
            #     'ffmpeg',
            #     '-y',
            #     '-ss', str(sample_start),
            #     '-t', str(sample_duration),
            #     '-i', video_source_path,
            #     '-c:v',
            #     'libx264',
            #     '-c:a',
            #     'aac',
            #     '-r',
            #     '25',
            #     output_path,
            # ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


            # -- temporal face extraction
            face_output_path = os.path.join(
                args.output_dir,
                'FACEs',
                f'{os.path.basename(video_source_path).split(".")[0]}_{sample_start}_{sample["sample_end"]}.png',
            )

            spkr_id = sample['speaker']
            frame = load_video(output_path).__next__()

            start_frame = int( (sample_start - sample['scene_start']) * 25  )
            end_frame = int( (sample['sample_end'] - sample['scene_start']) * 25 )

            pickle_data = read_pickle(sample['pickle_path'])
            face_bounding = pickle_data['face_boundings'][spkr_id][start_frame]
            left, top, right, bottom = face_bounding
            x, y = left, top
            w, h = abs(right-left), abs(bottom-top)
            mh, mw = int((y+h)*0.05), int((x+w)*0.05)
            frame_face = frame[y-mh:y+h+mh, x-mw:x+w+mw]
            # frame_face = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

            cv2.imwrite(face_output_path, frame_face)

            # -- temporal wav extraction
            wav_output_path = os.path.join(
                args.output_dir,
                'WAVs',
                f'{os.path.basename(video_source_path).split(".")[0]}_{sample_start}_{sample["sample_end"]}.wav',
            )

            # subprocess.call([
            #     'ffmpeg',
            #     '-y',
            #     '-i', output_path,
            #     '-ar', '16000',
            #     '-ac', '1',
            #     '-acodec', 'pcm_s16le',
            #     wav_output_path,
            # ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


