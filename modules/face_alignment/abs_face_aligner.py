from abc import ABC, abstractmethod

class AbsFaceAligner(ABC):
    """The common abstract class among face aligners.
    """

    @abstractmethod
    def detect_facial_landmarks(self, frame, face_bbs):
        """Detect the facial landmarks of each face appearing on the frame.
        Args:
            frame (np.ndarray): a frame read from the scene clip.
            face_bbs (np.ndarray): face bounding boxes provided by the face detector.
        Returns:
            np.ndarray: array containing the detected facial landmarks (N,L,2), where N refers to number of faces and L to the number of detected landmarks.
        """

        raise NotImplementedError
