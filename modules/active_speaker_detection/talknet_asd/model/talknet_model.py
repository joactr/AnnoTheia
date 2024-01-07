import os
import sys
import time
import subprocess
import pandas as pd
from tqdm import tqdm
from colorama import Fore

import torch
import torch.nn as nn
import torch.nn.functional as F

# -- try-catch needed for the 'Fine-Tuning TalkNet-ASD' tutorial
try:
    from modules.active_speaker_detection.talknet_asd.layers.losses import LossAV, LossA, LossV
except ModuleNotFoundError:
    from layers.losses import LossAV, LossA, LossV

# -- try-catch needed for the 'Fine-Tuning TalkNet-ASD' tutorial
try:
    from modules.active_speaker_detection.talknet_asd.model.talknet_arch import TalkNetArch
except:
    from model.talknet_arch import TalkNetArch

class TalkNetModel(nn.Module):
    def __init__(self, learning_rate=0.0001, lr_decay=0.95, device="cpu", **kwargs):
        super(TalkNetModel, self).__init__()

        self.device = device

        # -- model architecture
        self.model = TalkNetArch().to(self.device)
        self.lossAV = LossAV().to(self.device)
        self.lossA = LossA().to(self.device)
        self.lossV = LossV().to(self.device)

        # -- optimizing + scheduler
        self.optim = torch.optim.Adam(self.parameters(), lr = learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lr_decay)

    def forward(self, x):
        """Forward pass once the model has been estimated.
        Args:
            x (tuple): tuple of two torch.Tensor representing the acoustic and visual input data streams
        Returns:
            scores (torch.Tensor): tensor containing the frame-wise ASD scores
            labels (torch.Tensor): tensor containing the frame-wise ASD labels
        """
        with torch.no_grad():
            # -- getting input data
            audio_features, visual_features = x

            # -- audio visual frontend
            audio_embedding = self.model.forward_audio_frontend(audio_features.to(self.device))
            visual_embedding = self.model.forward_visual_frontend(visual_features.to(self.device))

            # -- audio visual cross attention + backend
            audio_embedding, visual_embedding = self.model.forward_cross_attention(audio_embedding, visual_embedding)
            audiovisual_outs = self.model.forward_audio_visual_backend(audio_embedding, visual_embedding)

            # -- computation of scores and labels
            scores, labels = self.lossAV.forward(audiovisual_outs)

        return scores, labels

    def train_network(self, loader, epoch, **kwargs):
        """Training TalkNet-ASD model.
        Args:
            loader (torch.utils.data.DataLoader): training data loader containing the data for the estimation.
            epoch (int): integer indicating the epoch of the trainig where we are.
            kwargs (dict): the rest of input arguments useful for the training.
        Returns:
            loss (float): loss obtained in this training epoch.
            lr (float): learning rate of this training epoch.
        """
        self.train()

        index, top1, loss = 0, 0, 0
        self.scheduler.step(epoch - 1)
        lr = self.optim.param_groups[0]['lr']

        # -- training process
        for batch_idx, (audio_features, visual_features, labels) in enumerate(tqdm(loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.GREEN, Fore.RESET)), start=1):
            # -- resetting the optimizer
            self.zero_grad()

            # -- audio visual frontend
            audio_embedding = self.model.forward_audio_frontend( audio_features.to(self.device) )
            visual_embedding = self.model.forward_visual_frontend( visual_features.to(self.device) )
            audio_embedding, visual_embedding = self.model.forward_cross_attention(audio_embedding, visual_embedding)

            # -- audio visual backend
            audiovisual_outs= self.model.forward_audio_visual_backend(audio_embedding, visual_embedding)
            audio_outs = self.model.forward_audio_backend(audio_embedding)
            visual_outs = self.model.forward_visual_backend(visual_embedding)

            # -- loss computation
            labels = labels.reshape((-1)).to(self.device)

            audiovisual_loss, _, _, prec = self.lossAV.forward(audiovisual_outs, labels)
            audio_loss = self.lossA.forward(audio_outs, labels)
            visual_loss = self.lossV.forward(visual_outs, labels)

            nloss = audiovisual_loss + 0.4 * audio_loss + 0.4 * visual_loss

            # -- backward pass + updating parameters
            nloss.backward()
            self.optim.step()

            # -- log information
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            index += len(labels)

        return loss / batch_idx, lr

    def evaluate_network(self, loader, output_path, dataset, **kwargs):
        """Evaluate the estimated TalkNet model.
        Args:
            loader (torch.utils.data.DataLoader): data loader containing the data for the evaluation.
            output_path (str): path where to save the CSV containig the results on the evaluation dataset.
            dataset (str): string indicating the dataset we are evaluating.
            kwargs (dict): the rest of input arguments useful for the training.
        Returns:
            loss (float): loss obtained by the evaluated model.
            precision (float): precision obtained by the evaluated model.
            mAP (float): mean Average Precision obtained by the evaluated model.
        """
        self.eval()

        index, top1, loss = 0, 0, 0
        pred_scores, pred_labels = [], []
        window_size = kwargs["window_size"]

        for batch_idx, (audio_features, visual_features, labels) in enumerate(tqdm(loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.BLUE, Fore.RESET)), start=1):
            with torch.no_grad():

                # -- audio visual frontend
                audio_embedding = self.model.forward_audio_frontend(audio_features.to(self.device))
                visual_embedding = self.model.forward_visual_frontend(visual_features.to(self.device))
                audio_embedding, visual_embedding = self.model.forward_cross_attention(audio_embedding, visual_embedding)

                # -- audio visual backend
                audiovisual_outs= self.model.forward_audio_visual_backend(audio_embedding, visual_embedding)
                audio_outs = self.model.forward_audio_backend(audio_embedding)
                visual_outs = self.model.forward_visual_backend(visual_embedding)

                # -- loss computation
                labels = labels.reshape((-1)).to(self.device)

                audiovisual_loss, pred_score, pred_label, prec = self.lossAV.forward(audiovisual_outs, labels)
                audio_loss = self.lossA.forward(audio_outs, labels)
                visual_loss = self.lossV.forward(visual_outs, labels)
                nloss = audiovisual_loss + 0.4 * audio_loss + 0.4 * visual_loss

                # -- gathering metrics
                top1 += prec
                index += len(labels)
                loss += nloss.detach().cpu().numpy()
                pred_scores.extend(pred_score[:,1].detach().cpu().numpy())
                pred_labels.extend(pred_label.detach().cpu().numpy().astype(int))

        # -- saving evaluation results into a CSV
        precision_eval = 100 * (top1 / index)

        # -- we dont remember what we were doing here, but it is working :)
        df = pd.read_csv(kwargs[f"{dataset}_dataset"])
        df = df.loc[df.index.repeat(window_size)].reset_index(drop=True)

        # -- extending dataframe to incorporate both the predictions and scores
        df["pred"] = pred_labels
        df["score"] = pred_scores
        df.index.name = 'uid'

        # -- saving new dataset evaluated
        df.to_csv(output_path)

        # -- computing mean Average Precision (mAP)
        mAP = str(subprocess.check_output([
            "python",
            "-O",
            "./scripts/get_map.py",
            "-p",
            output_path,
        ])).split(' ')[2][:5]

        if mAP[-1] == "%": mAP = mAP[:-1]
        mAP = float(mAP)

        return loss / batch_idx, precision_eval, mAP

    def save_parameters(self, checkpoint_path):
        """Save model checkpoint.
        Args:
            checkpoint_path: path where to save the model checkpoint.
        """
        torch.save(self.state_dict(), checkpoint_path)
        print(f"Saving model checkpoint in: {checkpoint_path}")

    def load_parameters(self, checkpoint_path):
        """Load the model from a checkpoint.
        Args:
            checkpoint_path: path to the checkpoint from where to load the model.
        """
        self_state = self.state_dict()
        loaded_state = torch.load(checkpoint_path, map_location=torch.device(self.device))
        for name, param in loaded_state.items():
            orig_name = name;
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%orig_name)
                    continue
            if self_state[name].size() != loaded_state[orig_name].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(orig_name, self_state[name].size(), loaded_state[orig_name].size()))
                continue
            self_state[name].copy_(param)
