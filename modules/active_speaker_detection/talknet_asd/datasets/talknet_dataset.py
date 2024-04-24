import os
import random
from math import floor,ceil

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class TalkNetDataset(Dataset):

    def __init__(self, csv_path, nframes):
        """Dataset for the TalkNet-ASD training and evaluation.
        Args:
            csv_path: path to the CSV file containing the dataset samples
            nframes: number of frames composing the sample window, i.e., the window size
        """
        # -- video_id, audio_id, label, window_center, video_path, audio_path
        self.samples = pd.read_csv(csv_path)

        # -- window sample settings
        self.nframes = nframes
        self.n_side_frames = int((self.nframes-1)/2)
        # -- assuming audio at 100fps and video at 25fps
        self.audio_n_side_frames = self.n_side_frames * 4

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # -- model input data
        audio = self.__get_audio__(index)
        video = self.__get_video__(index)

        # -- target labels
        labels = self.__get_labels__(index)

        return audio, video, labels

    def __get_labels__(self, index):
        label = self.samples.iloc[index]["label"]

        # -- frame-wise label tensor
        if label == 1:
            labels = np.ones(self.nframes)
        else:
            labels = np.zeros(self.nframes)

        return torch.LongTensor(labels)

    def __get_video__(self, index):
        sample = self.samples.iloc[index]

        # -- sample information
        video_id = sample["video_id"]
        label = sample["label"]
        window_center = sample["video_window_center"]

        # -- input data path
        video_path = sample["video_path"]
        np_video = np.load(video_path)["data"]
        video = torch.FloatTensor(np_video)

        # -- obtaining window sample
        video_frames = video.shape[0]
        ini = window_center - self.n_side_frames
        fin = window_center + self.n_side_frames + 1

        # -- padding at the beginning
        if window_center < self.n_side_frames:
            pad_amount = self.n_side_frames - window_center
            video = F.pad(video, (0,0,0,0,pad_amount,0), "constant", 0)

            # -- updating window boundings
            ini = 0
            fin = self.nframes

        # -- padding at the end
        if window_center+self.n_side_frames >= video_frames:
            pad_amount = (self.n_side_frames + window_center) - video_frames + 1
            video = F.pad(video, (0,0,0,0,0,pad_amount), "constant", 0)

            # -- updating window boundings
            ini = len(video)-(self.nframes)

        # -- video window sampling
        video = video[ini:fin]

        return video # (T, 112, 112)

    def __get_audio__(self, index):
        sample = self.samples.iloc[index]

        # -- sample information
        audio_id = sample["audio_id"]
        label = sample["label"]
        window_center = sample["video_window_center"]
        video_id = sample["video_id"]

        # -- input data path
        audio_path = sample["audio_path"]
        np_audio = np.load(audio_path)["data"]
        audio = torch.FloatTensor(np_audio)

        # -- if the sample is positive ...
        if label == 1:
            # -- audio is correctly aligned to the video (assuming audio at 100fps and video at 25fps)
            window_center = window_center * 4

        # -- if the sample it is negative ...
        if label == 0:

            # -- and video and audio come from the same video clip ...
            if audio_id == video_id:
                # -- audio is randomly shifted to not be aligned to the video, checking that at most both windows overlap 50%
                window_center = self._random_audio_window_with_no_overlap(window_center, int(audio_frames/4), self.audio_n_side_frames, threshold=0.5)

            # -- and video and audio come from different video clips ...
            else:
                # -- audio is randomly sampled with no care because it is a different person or the same person in a different sample
                window_center = random.randint(0, len(audio))

        # -- obtaining window sample
        audio_frames = audio.shape[0]
        ini = window_center - self.audio_n_side_frames
        fin = window_center + self.audio_n_side_frames + 4

        # -- padding at the beginning
        if window_center < self.audio_n_side_frames:
            pad_amount = self.audio_n_side_frames - window_center
            audio = F.pad(audio, (0,0,pad_amount,0), "constant", 0)

            # -- updating window boundings
            ini = 0
            fin = self.nframes*4

        # -- padding at the end
        if window_center+self.audio_n_side_frames+4 >= audio_frames:
            pad_amount = (self.audio_n_side_frames + window_center) - audio_frames + 4
            audio = F.pad(audio,(0,0,0,pad_amount), "constant", 0)

            # -- updating window boundings
            ini = len(audio)-(self.nframes*4)

        # -- audio window sampling
        audio = audio[ini:fin]

        return audio # (T, 13)

    def _random_audio_window_with_no_overlap(self, video_center, video_frame_length, n_side_frames, threshold):
        """Looks for an index where the center of an audio window is not overlapping with the existing video window,
          returning, the index of the center of the window, or the minimum padding center in case there is no index in which they do not overlap.
        Args:
            video_center: index indicating the center of the window sample.
            video_frame_length: length of the video in terms of frames.
            n_side_frames (int): number of frames on each side of the window.
            threshold (float): value between 0 and 1, representing the maximum percentage of overlapping allowed.
        Returns:
            (int): index of the new audio window no overlapping with the video window
        """
        window_size = (n_side_frames * 2) + 1
        fits_start, fits_end = True, True

        overlapLeft = 1 - max(abs(video_center-0), 0) / window_size
        overlapRight = 1 - max(abs(video_center-video_frame_length), 0) / window_size

        if overlapLeft > treshold:
            fits_start = False
        if overlapRight > treshold:
            fits_end = False

        if not fits_start and not fits_end:
            min_padding_left = floor(video_center - (treshold*window_size))
            min_padding_right = ceil(video_center + (treshold*window_size))

            return random.choice([min_padding_left,min_padding_right])

        overlap = False
        while overlap:
            index = random.randint(0, video_frame_length)

            if (1 - max(abs(video_center-index), 0) / window_size) > treshold:
                overlap = True
            if not overlap:
                # -- assuming audio at 100fps and video at 25fps
                return index*4


