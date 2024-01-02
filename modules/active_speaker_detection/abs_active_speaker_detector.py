from abc import ABC, abstractmethod

class AbsASD(ABC):
    """The common abstract class among active speaker detectors.
    """

    @abstractmethod
    def get_asd_scores(self, acoustic_input, visual_input):
        """Obtaining the frame-wise score predictions thanks to an ASD model.
        Args:
            acoustic_input (np.ndarray): acoustic input features.
            visual_input (np.ndarray): visual input features.
        Returns:
            scores (list): list containing the frame-wise score predictions.
        """

        raise NotImplementedError
