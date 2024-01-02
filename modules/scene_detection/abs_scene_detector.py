from abc import ABC, abstractmethod

class AbsSceneDetector(ABC):
    """The common abstract class among scene detectors.
    """

    @abstractmethod
    def split_video_into_scenes(self, video_path):
        """Split a video into the different scenes composing it.
        Args:
            video_path (str): path specifying where the video is stored.
        Returns:
            [(scene_path, start_timestamp, end_timestamp): list of tuples representing the detected scenes.
        """

        raise NotImplementedError
