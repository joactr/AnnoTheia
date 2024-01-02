from typing import Optional

from modules.scene_detection.abs_scene_detector import AbsSceneDetector
from modules.face_detection.abs_face_detector import AbsFaceDetector
from modules.face_alignment.abs_face_aligner import AbsFaceAligner
from modules.active_speaker_detection.abs_active_speaker_detector import AbsASD
from modules.automatic_speech_recognition.abs_automatic_speech_recognizer import AbsASR

from pipelines.abs_pipeline import AbsPipeline

class AnnoTheiaPipeline(AbsPipeline):

    def __init__(
        self,
        video_dir,
        scene_detection: AbsSceneDetector,
        face_detection: AbsFaceDetector,
        face_alignment: Optional[AbsFaceAligner],
        active_speaker_detection: AbsASD,
        speech_recognition: AbsASR,
        min_length: int = 12,
        threshold: int = 0.04,
        window_size: int = 25,
        smoothing_window_size: int = 11,
        min_face_size: int = 32,
        min_method: bool = False,
        output_scenes: bool = True,
    ):

        self.scene_detection=scene_detection,
        self.face_detection=face_detection,
        self.face_alignment=face_alignment,
        self.active_speaker_detection=active_speaker_detection,
        self.speech_recognition=speech_recognition,
