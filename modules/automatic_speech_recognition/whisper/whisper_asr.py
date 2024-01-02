import whisper
from modules.automatic_speech_recognition.abs_speech_recognizer import AbsASR

class WhisperASR(AbsASR):
    def __init__(self, model_size="small", lang="auto"):
         self.lang = lang
         self.whisper = whisper.load_model(model_size)

    def get_transcription(audio_path):
        """Obtaining the transcription from an audio waveform.
        Args:
            audio_path (str): path where the audio waveform to transcript is stored.
        Returns:
            transcription (list): list containing the transcriptions in the Whisper format. It have to include the word timestamps.
        """
        if self.lang == "auto":
            transcription = transcription_model.transcribe(
                audio_path,
                verbose=False,
                word_timestamps=True,
            )
        else:
            transcription = transcription_model.transcribe(
                audio_path,
                language=self.lang,
                verbose=False,
                word_timestamps=True,
            )

        return transcription
