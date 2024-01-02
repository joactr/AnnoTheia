from abc import ABC, abstractmethod

class AbsFaceDetector(ABC):
    """The common abstract class among face detectors.
    """

    @abstractmethod
    def detect_faces(self, frame):
        """Detect each face appearing on the scene.
        Args:
            frame (np.ndarray): a frame read from the scene clip.
        Returns:
            np.ndarray: array containing the bounding boxes representing the faces detected on the scene.
        """

        raise NotImplementedError
