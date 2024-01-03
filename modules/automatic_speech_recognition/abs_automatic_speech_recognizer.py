from abc import ABC, abstractmethod

class AbsASR(ABC):
    """The common abstract class among active speaker detectors.
    """

    @abstractmethod
    def get_transcription(audio_path):
        """Obtaining the transcription from an audio waveform.
        Args:
            audio_path (str): path where the audio waveform to transcript is stored.
        Returns:
            transcription (list): list containing the transcriptions in the Whisper format. It have to include the word timestamps.
           Specifically, a 3-level nested dictionary like this: {'segments': [{'words': {'word': [str], 'start': float, 'end': float}}]}
        """

        raise NotImplementedError
