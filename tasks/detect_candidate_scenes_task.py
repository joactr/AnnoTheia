import argparse
from utils.configs.class_choices import ClassChoices

from modules.scene_detection import PySceneDetector
from modules.scene_detection.abs_scene_detector import AbsSceneDetector

from modules.face_detection import DSFDFaceDetector
from modules.face_detection import RetinaFaceDetector
from modules.face_detection.abs_face_detector import AbsFaceDetector

from modules.face_alignment import FANAligner
from modules.face_alignment.abs_face_aligner import AbsFaceAligner

from modules.active_speaker_detection import TalkNetASD
from modules.active_speaker_detection.abs_active_speaker_detector import AbsASD

from modules.automatic_speech_recognition import WhisperASR
from modules.automatic_speech_recognition.abs_automatic_speech_recognizer import AbsASR

from pipelines import AnnoTheiaPipeline
from pipelines.abs_pipeline import AbsPipeline

scene_detection_choices = ClassChoices( name="scene_detection", classes=dict(
    pyscenedetect=PySceneDetector,
    ),
    type_check=AbsSceneDetector,
    default="pyscenedetect",
)

face_detection_choices = ClassChoices(
    name="face_detection",
    classes=dict(
        dsfd=DSFDFaceDetector,
        retina=RetinaFaceDetector,
    ),
    type_check=AbsFaceDetector,
    default="dsfd",
)

face_alignment_choices = ClassChoices(
    name="face_alignment",
    classes=dict(
        fan=FANAligner,
    ),
    type_check=AbsFaceAligner,
    default=None,
    optional=True
)

active_speaker_detection_choices = ClassChoices(
    name="active_speaker_detection",
    classes=dict(
        talknet=TalkNetASD,
    ),
    type_check=AbsASD,
    default="talknet",
)

automatic_speech_recognition_choices = ClassChoices(
    name="automatic_speech_recognition",
    classes=dict(
        whisper=WhisperASR,
    ),
    type_check=AbsASR,
    default="whisper",
)

pipeline_choices = ClassChoices(
    name="pipeline",
    classes=dict(
        annotheia=AnnoTheiaPipeline,
    ),
    type_check=AbsPipeline,
    default="annotheia",
)

class DetectCandidateScenesTask():

    @classmethod
    def build_pipeline(self, args: argparse.Namespace) -> AnnoTheiaPipeline:
        # 1. Scene Detection
        scene_detection_class = scene_detection_choices.get_class(args.scene_detection)
        scene_detection = scene_detection_class(**args.scene_detection_conf)

        # 2. Face Detection
        face_detection_class = face_detection_choices.get_class(args.face_detection)
        face_detection = face_detection_class(**args.face_detection_conf)

        # 3. Face Alignment
        if getattr(args, "face_alignment", None) is not None:
            face_alignment_class = face_alignment_choices.get_class(args.face_alignment)
            face_alignment = face_alignment_class(**args.face_alignment_conf)
        else:
            face_alignment = None

        # 4. Active Speaker Detection
        active_speaker_detection_class = active_speaker_detection_choices.get_class(args.active_speaker_detection)
        active_speaker_detection = active_speaker_detection_class(**args.active_speaker_detection_conf)

        # 5. Automatic Speech Recognition
        automatic_speech_recognition_class = automatic_speech_recognition_choices.get_class(args.automatic_speech_recognition)
        automatic_speech_recognition = automatic_speech_recognition_class(**args.automatic_speech_recognition_conf)

        # 6. Building AnnoTheia Pipeline
        try:
            pipeline_class = pipeline_choices.get_class(args.pipeline)
        except AttributeError:
            pipeline_class = pipeline_choices.get_class("annotheia")

        pipeline = pipeline_class(
            scene_detection=scene_detection,
            face_detection=face_detection,
            face_alignment=face_alignment,
            active_speaker_detection=active_speaker_detection,
            automatic_speech_recognition=automatic_speech_recognition,
            **args.pipeline_conf,
        )

        return pipeline
