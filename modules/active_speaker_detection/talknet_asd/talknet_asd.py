import torch

from modules.active_speaker_detection.abs_active_speaker_detector import AbsASD
from modules.active_speaker_detection.talknet_asd.model.talknet_arch import TalkNetArch

class TalkNetASD(AbsASD):
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device
        self.talknet_asd = TalkNetArch(device=self.device)

        print(f"Loading TalkNet-ASD from checkpoint: {checkpoint_path}")
        self.talknet_asd.load_state_dict(checkpoint_path, map_location=self.device))

    def get_asd_scores(self, acoustic_input, visual_input):
        """Obtaining the frame-wise score predictions thanks to an ASD model.
        Args:
            acoustic_input (np.ndarray): acoustic input features.
            visual_input (np.ndarray): visual input features.
        Returns:
            scores (list): list containing the frame-wise score predictions.
        """

        scores = self._forward(acoustic_input, visual_input)
        return scores.detach().cpu().numpy().tolist())

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
