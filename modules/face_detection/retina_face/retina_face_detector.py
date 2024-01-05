from termcolor import cprint
from ibug.face_detection import RetinaFacePredictor
from modules.face_detection.abs_face_detector import AbsFaceDetector

class RetinaFaceDetector(AbsFaceDetector):
    def __init__(self, model_id="resnet50", threshold=0.8, device="cpu"):

        self.face_detector = RetinaFacePredictor(
            device=device,
            threshold=threshold,
            model=RetinaFacePredictor.get_model(model_id),
        )
        cprint(f"\t(Face Detection) Retina Face Detector ({model_id}) initialized with a threshold of {threshold}", "yellow", attrs=["bold", "reverse"])

    def detect_faces(self, frame):
        """Detect each face appearing on the scene.
        Args:
            frame (np.ndarray): a frame read from the scene clip.

        Returns:
            [(left, top, right, bottom)]: list of bounding boxes representing the faces detected on the scene for each person.
        """

        return self.face_detector(frame, rgb=False)
