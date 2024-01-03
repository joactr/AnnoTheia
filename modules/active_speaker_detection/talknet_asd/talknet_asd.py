import torch
import random

from modules.active_speaker_detection.abs_active_speaker_detector import AbsASD
from modules.active_speaker_detection.talknet_asd.model.talknet_model import TalkNetModel

class TalkNetASD(AbsASD):
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device
        self.talknet_asd = TalkNetModel(device=self.device)

        print(f"Loading TalkNet-ASD from checkpoint: {checkpoint_path}")
        self.talknet_asd.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    def preprocess_input(self, audio_waveform, face_crops, window_center, window_size, total_video_frames):
        """Prepares the input data streams to the audio-visual ASD model.
        Args:
            audio_waveform (np.ndarray): acoustic input data stream reprensenting the waveform.
            face_crops (np.ndarray): visual input data stream representing the cropped face images.
            window_center (int): index indicating the center of the window input stream data.
            window_size (int): integer indicating the size of the window input stream data.
            total_video_frames (int): total number of frames composing the scene as a reference for padding.
        Returns:
            acoustic_input (torch.tensor): acoustic input for the ASD model.
            visual_input (torch.tensor): visual input for the ASD model.
        """

        # -- extracting MFCCs assuming video data at 25 fps
        mfccs = python_speech_features.mfcc(
            audio_waveform,
            samplerate=16000,
            numcep=13,
            winlen=0.025,
            winstep=0.010,
        )

        # -- padding the acoustic input + adding batch dimension
        acoustic_input = self._padding_audio(
            mfccs,
            label=1,
            window_center=window_center,
            window_size=window_size,
            total_video_frames=total_video_frames,
        ).unsqueeze(0)

        visual_input = self._padding_video(
            face_crops,
            window_center=window_center,
            window_size=window_size,
        ).unsqueeze(0)

        return acoustic_input, visual_input

    def get_asd_scores(self, acoustic_input, visual_input):
        """Obtains the frame-wise score predictions provided by the audio-visual ASD model.
        Args:
            acoustic_input (np.ndarray): acoustic input features.
            visual_input (np.ndarray): visual input features.
        Returns:
            scores (list): list containing the frame-wise score predictions.
        """

        scores = self._forward(acoustic_input, visual_input)
        return scores.detach().cpu().numpy().tolist()

    def _forward(self, acoustic_input, visual_input):
        """Forward pass of TalkNet-ASD to obtain the frame-wise score predictions.
        Args:
            acoustic_input (np.ndarray): acoustic input features.
            visual_input (np.ndarray): visual input features.
        Returns:
            scores (torch.Tensor): tensor containing the frame-wise score predictions.
        """
        with torch.no_grad():
            acoustic_input = acoustic_input.to(self.device)
            visual_input = visual_input.to(self.device)

            audio_emb = self.talknet_asd.forward_audio_frontend(acoustic_input)
            video_emb = self.talknet_asd.forward_visual_frontend(visual_input)

            audio_emb, video_emb = self.talknet_asd.forward_cross_attention(audio_emb, video_emb)
            audiovisual_emb = self.talknet_asd.forward_audio_visual_backend(audio_emb, video_emb)

            scores, _ = self.lossAV.forward(audiovisual_emb)

        return scores

    def _padding_audio(audio, label, window_center, window_size, total_video_frames):
        # -- computing the maximum number of frames for the audio cues assuming video data at 25 fps
        max_audio_frames = total_video_frames * 4

        # -- if audio input is shorter, we pad the audio sequence
        if audio.shape[0] < max_audio_frames:
            pad_amount = max_audio_frames - audio.shape[0]
            audio = np.pad(audio, ((0, pad_amount), (0, 0)), 'wrap')
        # -- if audio input is longer, we discard last tokens
        audio = audio[:max_audio_frames, :]

        n_side_frames_video = int((window_size-1)/2)
        n_side_frames_audio = n_side_frames_video*4

        # -- convert to Torch tensor
        audio = torch.FloatTensor(audio)
        audio_frames = audio.shape[0]

        # -- if it is a positive sample
        if label == 1:
            window_center = window_center * 4
        # -- if it is a negative sample
        if label == 0:
            window_center = random.randint(0, len(audio))

        # -- setting window boundings
        ini = window_center - n_side_frames_audio
        fin = window_center + n_side_frames_audio + 4

        # -- padding at the beggining
        if window_center < n_side_frames_audio:
            pad_amount = n_side_frames_audio - window_center
            audio = F.pad(audio, (0, 0, pad_amount, 0), "constant", 0)

            # -- updating window boundings
            ini = 0
            fin = window_size*4

        # -- padding at the end
        if (window_center + n_side_frames_audio + 4) >= audio_frames:
            pad_amount = (n_side_frames_audio + window_center) - (audio_frames + 4)
            audio = F.pad(audio, (0, 0, 0, pad_amount), "constant", 0)

            # -- updating window boundings
            ini = len(audio) - (window_size * 4)

        # -- window sampling
        audio = audio[ini:fin]

        return audio  # (T, 13)

    def _padding_video(video, window_center, window_size):
        n_side_frames = int((window_size-1)/2)

        # -- convert to Torch tensor
        video = torch.FloatTensor(np.array(video))

        video_frames = video.shape[0]

        # -- setting window boundings
        ini = window_center-n_side_frames
        fin = window_center+n_side_frames+1

        # -- padding at the beggining
        if window_center < n_side_frames:
            pad_amount = n_side_frames - window_center
            video = F.pad(video, (0, 0, 0, 0, pad_amount, 0), "constant", 0)

            # -- updating window boundings
            ini = 0
            fin = window_size

        # -- padding at the end
        if window_center+n_side_frames >= video_frames:
            pad_amount = (n_side_frames + window_center) - (video_frames + 1)
            video = F.pad(video, (0, 0, 0, 0, 0, pad_amount), "constant", 0)

            # -- updating window boundings
            ini = len(video)-(window_size)

        # -- window sampling
        video = video[ini:fin]

        return video  # (T, 112, 112)