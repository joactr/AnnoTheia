from termcolor import cprint
from ibug.face_alignment import FANPredictor
from modules.face_alignment.abs_face_aligner import AbsFaceAligner

class FANAligner(AbsFaceAligner):
    def __init__(self, model_id="2dfan2_alt", device="cpu"):

        self.face_aligner = FANPredictor(
            device=device,
            model=FANPredictor.get_model(model_id),
        )
        cprint(f"\t(Face Alignment) FAN Predictor ({model_id}) initialized", "cyan", attrs=["bold", "reverse"])

    def detect_facial_landmarks(self, frame, face_bbs):
        """Detect the facial landmarks of each face appearing on the frame.
        Args:
            frame (np.ndarray): a frame read from the scene clip.
            face_bbs (np.ndarray): face bounding boxes provided by the face detector.
        Returns:
            np.ndarray: array containing the detected facial landmarks (N,L,2), where N refers to number of faces and L to the number of detected landmarks.
        """

        landmarks, _ = self.face_aligner(frame, face_bbs, rgb=False)
        return landmarks
