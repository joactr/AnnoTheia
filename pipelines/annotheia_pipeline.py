import os
from typing import Optional
from termcolor import cprint
from collections import defaultdict

from scipy.io.wavfile as scipy_wavfile

from modules.scene_detection.abs_scene_detector import AbsSceneDetector
from modules.face_detection.abs_face_detector import AbsFaceDetector
from modules.face_alignment.abs_face_aligner import AbsFaceAligner
from modules.active_speaker_detection.abs_active_speaker_detector import AbsASD
from modules.automatic_speech_recognition.abs_automatic_speech_recognizer import AbsASR

from pipelines.abs_pipeline import AbsPipeline

from utils.face_detection import detect_multiple_faces
from utils.audio_processing import extract_wav_from_video
from utils.video_processing import convert_video_to_target_fps
from utils.scene_detection import check_video_duration, get_suitable_scenes

class AnnoTheiaPipeline(AbsPipeline):

    def __init__(
        self,
        scene_detection: AbsSceneDetector,
        face_detection: AbsFaceDetector,
        face_alignment: Optional[AbsFaceAligner],
        active_speaker_detection: AbsASD,
        speech_recognition: AbsASR,
        min_length: int = 12,
        threshold: int = 0.04,
        window_size: int = 25,
        smoothing_window_size: int = 11,
        face_max_frame: int = 10,
        min_face_size: int = 32,
        max_face_distance_thr = 50,
        method: str = "no-overlap",
        output_scenes: bool = True,
        output_dir: str = "./outputs/",
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
        self.min_method = min_method
        self.output_scenes = output_scenes
        self.output_dir = output_dir

        # -- pipeline modules
        self.scene_detection = scene_detection,
        self.face_detection = face_detection,
        self.face_alignment = face_alignment,
        self.active_speaker_detection = active_speaker_detection,
        self.speech_recognition = speech_recognition,

    def process_video(self, video_path):
        """Process a video to detect the candidate scenes to compile a new audio-visual database.
        Args:
        Returns:
        """
        total_scores = defaultdict(list)
        side_window_size = int( (self.window_size - 1) / 2 )
        side_smoothing_window_size = int( (self.smoothing_window_size - 1) / 2 )

        # 1. Scene detection
        # 1.1. Splitting the video into scenes [(scene_path, start_timestamp, end_timestamp)]
        scenes_list = self.scene_detection.split_video_into_scenes(video_path)

        # 1.2. Filtering out the non-suitable scenes according to different criteria
        scenes_list = get_suitable_scenes(scenes_list, self.face_detector, self.face_max_frame)

        # -- for each detected scene
        for scene_path, start_timestamp, end_timestamp in scenes_list:

            # 1.3. Converting each scene to 25 fps
            scene_duration = check_video_duration(scene_path)
            convert_video_to_target_fps(scene_path, self.target_fps, self.scene_detection.temp_dir)
            scene_frames = int(scene_duration * self.target_fps)

            # 1.4. Extracting audio waveforms from scene video clip
            wav_path = extract_wav_from_video(scene_path, self.scene_detection.temp_dir)
            _, audio_waveform = scipy_wavfile.read(wav_path)

            # 2. Face detection
            face_crops, face_boundings, face_frames = detect_multiple_faces(scene_path, self.face_detection, self.min_face_size, self.max_face_distance_thr)

            # 3. Active speaker detection
            # -- for each detected person on the scene
            if self.method == "no-overlap":
                for actual_speaker in face_crops.keys():
                    cprint(f"Analyzing if person {actual_speaker} on scene the one who is actually speaking", "blue", attrs=["bold", "reverse"])

                    # -- depending on the video sliding strategy method chosen
                    scene_length = len(face_crops[actual_speaker])
                    extended_scene_length = scene_length + self.window_size + 1

                    # -- for each window sample
                    for window_idx in range(0, extended_scene_length, self.window_size):
                        window_center = face_frames[actual_speaker][0] + window_idx

                        # 3.1. Extracting preprocessed input data
                        acoustic_input, visual_input = self.active_speaker_detection.preprocess_input(
                            audio_waveform,
                            face_crops[actual_speaker],
                            window_center=window_center,
                            window_size=self.window_size,
                            total_video_frames=scene_frames,
                        )

                        # 3.2. Computing the frame-wise ASD scores
                        scores = self.active_speaker_detection.get_asd_scores(
                            acoustic_input,
                            visual_input,
                        )

                        # 3.3. Gathering the average score of the window sample
                        total_scores[actual_speaker].extend( scores[:, 1] )

                    # -- smoothing sliding window
                    score_length = len(total_scores[actual_speaker])
                    for frame_idx in range(0, score_length):
                        start = max(0, frame_idx - side_smoothing_window_size
                        end = frame_idx + side_smoothing_window_size + 1

                        total_scores[actual_speaker][i] = np.mean(
                            total_scores[actual_speaker][start:end]
                        )

           else:
                raise ValueError(f"Unknow video sliding strategy: {self.method}. There is only implementation for the 'non-overlap' strategy.")
