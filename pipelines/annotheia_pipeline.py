import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
from termcolor import cprint
from collections import defaultdict

import scipy.io.wavfile as scipy_wavfile

from modules.scene_detection.abs_scene_detector import AbsSceneDetector
from modules.face_detection.abs_face_detector import AbsFaceDetector
from modules.face_alignment.abs_face_aligner import AbsFaceAligner
from modules.active_speaker_detection.abs_active_speaker_detector import AbsASD
from modules.automatic_speech_recognition.abs_automatic_speech_recognizer import AbsASR

from pipelines.abs_pipeline import AbsPipeline

from utils.face_detection import detect_multiple_faces
from utils.audio_processing import extract_wav_from_video
from utils.video_processing import convert_video_to_target_fps, save_scene
from utils.scene_detection import check_video_duration, get_suitable_scenes
from utils.pipeline import non_overlap_sliding_strategy, get_speaking

class AnnoTheiaPipeline(AbsPipeline):

    def __init__(
        self,
        scene_detection: AbsSceneDetector,
        face_detection: AbsFaceDetector,
        face_alignment: Optional[AbsFaceAligner],
        active_speaker_detection: AbsASD,
        automatic_speech_recognition: AbsASR,
        min_length: int = 12,
        threshold: int = 0.04,
        window_size: int = 25,
        smoothing_window_size: int = 11,
        face_max_frame: int = 10,
        min_face_size: int = 32,
        max_face_distance_thr = 50,
        method: str = "no_overlap+smoothing",
        align_margin: float = 0.3,
        save_scenes: bool = False,
    ):

        # -- fixed settings
        self.target_fps = 25

        # -- pipeline hyperparameters
        self.min_length = min_length
        self.threshold = threshold
        self.window_size = window_size
        self.smoothing_window_size = smoothing_window_size
        self.face_max_frame = face_max_frame
        self.min_face_size = min_face_size
        self.max_face_distance_thr = max_face_distance_thr
        self.method = method
        self.align_margin = align_margin
        self.save_scenes = save_scenes

        # -- pipeline modules
        self.scene_detection = scene_detection
        self.face_detection = face_detection
        self.face_alignment = face_alignment
        self.active_speaker_detection = active_speaker_detection
        self.automatic_speech_recognition = automatic_speech_recognition

    def process_video(self, video_path, output_dir, video_df_output_path, last_processed_scene_id):
        """Process a video to detect the candidate scenes to compile a new audio-visual database.
        Args:
            video_path (str): path where the video clip is stored.
            output_dir (str): directory where all the useful information will be stored.
            video_df_output_path (str): path where the trimmed candidate scenes will be saved as CSV.
            last_processed_scene_id (list): integer id of the last already processed scene.
        """

        # 1. Scene detection
        # 1.1. Splitting the video into scenes [(scene_path, start_timestamp, end_timestamp)]
        scenes_list = self.scene_detection.split_video_into_scenes(video_path)

        # 1.2. Filtering out the non-suitable scenes according to different criteria
        scenes_list = get_suitable_scenes(scenes_list, self.face_detection, self.face_max_frame)

        # -- for each detected scene
        for i, (scene_path, start_timestamp, end_timestamp) in enumerate(scenes_list):
            scene_id = int( scene_path.split("-")[-1].split(".")[0] )

            # -- discarding already processed scenes
            if scene_id > last_processed_scene_id:
                cprint(f"\n\t(Pipeline) Processing scene {str(i).zfill(4)} of a total of {str(len(scenes_list)).zfill(4)} ...\n", "light_grey", attrs=["bold","reverse"])

                # 1.3. Converting each scene to 25 fps
                scene_duration = check_video_duration(scene_path)
                convert_video_to_target_fps(scene_path, self.target_fps, self.scene_detection.temp_dir)
                scene_frames = int(scene_duration * self.target_fps)

                # 1.4. Extracting audio waveforms from scene video clip
                waveform_path = extract_wav_from_video(scene_path, self.scene_detection.temp_dir)
                _, audio_waveform = scipy_wavfile.read(waveform_path)

                # 2. Face detection + Face Alignment
                face_crops, face_boundings, face_landmarks, face_frames = detect_multiple_faces(scene_path, self.face_detection, self.face_alignment, self.min_face_size, self.max_face_distance_thr)

                # 3. Active speaker detection
                # 3.1. Applying the pipeline sliding strategy when obtaining the frame-wise ASD scores
                if self.method == "no_overlap+smoothing":
                    asd_scores = non_overlap_sliding_strategy(
                        audio_waveform,
                        face_crops,
                        face_frames,
                        scene_frames,
                        self.window_size,
                        self.smoothing_window_size,
                        self.active_speaker_detection,
                    )
                else:
                    raise ValueError(f"Unknow pipeline sliding strategy: {self.method}. There is only implementation for the 'non-overlap' strategy.")

                # 3.2. Applying decision threshold
                asd_labels = defaultdict(list)
                side_window_size = int( (self.window_size - 1) / 2 )

                for actual_speaker in face_crops.keys():
                    thresholding_window_length = len(face_crops[actual_speaker]) + side_window_size
                    asd_scores[actual_speaker] = asd_scores[actual_speaker][side_window_size:thresholding_window_length]

                    thr_scores = (np.array(asd_scores[actual_speaker]) > self.threshold).astype(np.int64)
                    asd_labels[actual_speaker] += thr_scores.tolist()

                # 4. Automatic speech recognition
                transcription = self.automatic_speech_recognition.get_transcription(waveform_path)

                # -- saving staff
                norm_scene_path = os.path.normpath(scene_path)
                scene_id = os.path.splitext(str(norm_scene_path.split(os.sep)[-1]))[0]

                if self.save_scenes:
                    scene_output_path = os.path.join(output_dir, "scenes", scene_id+".mp4")
                    save_scene(scene_output_path, scene_path, waveform_path, face_boundings, face_frames, asd_scores, asd_labels)

                # -- pickle containing useful information for the future audiovisual database
                pickle_dict = {
                    "face_boundings": face_boundings,
                    "face_landmarks": face_landmarks,
                    "asd_labels": asd_labels,
                    "transcription": transcription,
                }

                pickle_output_path = os.path.join(output_dir, "pickles", scene_id+".pkl")
                with open(pickle_output_path, "wb") as f:
                    pickle.dump(pickle_dict, f)

                # 5. Aligning transcriptions for each scene
                words = []
                aligns = []

                # -- processing transcription alignment provided by the ASR module
                for segment in transcription["segments"]:
                    for word in segment["words"]:
                        words.append( word["word"] )
                        aligns.append( (word["start"], word["end"]) )

                # -- aligning transcription for each scene
                scenes_info = []
                for speaker_id in asd_scores.keys():
                    # -- obtaining valid scenes where a person is actually speaking
                    for (start, end) in get_speaking(asd_labels[speaker_id], self.min_length, self.target_fps):
                        start_w, end_w = 0, 0

                        # -- alleviating in case of alignment mistakes by the ASR module
                        for i, (word_start, word_end) in enumerate(aligns):
                            if start+self.align_margin >= word_start and start-self.align_margin <= word_end:
                                start_w = i
                            if word_start <= end:
                                end_w = i

                        # -- compiling useful information about the detected scenes
                        scene_info = {
                            "video": video_path,
                            "scene_start": start_timestamp,
                            "sample_start": start_timestamp + start,
                            "sample_end": start_timestamp + end,
                            "duration": round((start_timestamp + end) - (start_timestamp + start), 2),
                            "speaker": speaker_id,
                            "pickle_path": pickle_output_path,
                            "transcription": "".join(words[start_w:(end_w+1)]).strip(),
                            "scene_path": scene_path, # -- just to control in case of reanuding a video processing
                        }
                        scenes_info.append(scene_info)

                # -- saving information w.r.t. the detected candidate scenes
                video_df = pd.DataFrame(scenes_info, columns=["video", "start", "end", "duration", "speaker", "pickle_path", "transcription", "scene_path"])
                video_df.to_csv(video_df_output_path, mode="a", index=False, header=not os.path.exists(video_df_output_path))

            else:
                cprint(f"\n\t(Pipeline) Scene {scene_path} was already processed in previous executions. Skipping it!\n", "light_grey", attrs=["bold","reverse"])

