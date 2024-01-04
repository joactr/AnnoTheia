from abc import ABC, abstractmethod

class AbsASD(ABC):
    """The common abstract class among active speaker detectors.
    """

    @abstractmethod
    def preprocess_input(self, audio_waveform, face_crops, window_center, total_video_frames):
        """Prepares the input data streams to the audio-visual ASD model.
        Args:
            audio_waveform (np.ndarray): acoustic input data stream reprensenting the waveform.
            face_crops (np.ndarray): visual input data stream representing the cropped face images.
            window_center (int): index indicating the center of the window input stream data.
            window_size (int): integer indicating the size of the window input stream data.
            total_video_frames (int): total number of frames composing the scene as a reference for padding.
        Returns:
            acoustic_input (non-defined): the acoustic input your ASD model needs.
            visual_input (non-defined): the visual input your ASD model needs.
        """

        raise NotImplementedError

    @abstractmethod
    def get_asd_scores(self, acoustic_input, visual_input):
        """Obtains the frame-wise score predictions provided by the audio-visual ASD model.
        Args:
            acoustic_input (np.ndarray): acoustic input features.
            visual_input (np.ndarray): visual input features.
        Returns:
            scores (np.ndarray): matrix containing the frame-wise score predictions.
        """

        raise NotImplementedError
