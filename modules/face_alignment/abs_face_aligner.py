from abc import ABC, abstractmethod

class AbsFaceAligner(ABC):
    """The common abstract class among face aligners.
    """

    @abstractmethod
    def detect_facial_landmarks(self, face_bb):
        """Detect the facial landmarks of each face appearing on the frame.
        Args:
            face_bb (np.ndarray): face bounding boxes provided by the face detector.
        Returns:
            np.ndarray: array containing the detected facial landmarks.
        """

        raise NotImplementedError
