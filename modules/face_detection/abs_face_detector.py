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
            [(left, top, right, bottom)]: list of bounding boxes representing the faces detected on the scene for each person.
        """

        raise NotImplementedError
