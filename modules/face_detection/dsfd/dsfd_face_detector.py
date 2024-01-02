import face_detection
from modules.face_detection.abs_face_detector import AbsFaceDetector

class DSFDFaceDetector(AbsFaceDetector):
    def __init__(self, confidence_threshold=0.35, nms_iou_threshold=0.5):

        self.face_detector = face_detection.build_detector(
            "DSFDDetector",
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=0.5,
        )

    def detect_faces(self, frame):
        """Detect each face appearing on the scene.
        Args:
            frame (np.array): a frame read from the scene clip.

        Returns:
            list[(top,bottom,right,left)]: list of bounding boxes representing the faces detected on the scene.
        """

        return self.face_detector.detect(frame)
