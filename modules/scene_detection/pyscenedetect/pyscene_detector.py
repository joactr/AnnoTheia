import os
import glob
from termcolor import cprint

from modules.scene_detection.abs_scene_detector import AbsSceneDetector
from scenedetect import detect, ContentDetector, split_video_ffmpeg

from utils.scene_detection import check_video_duration

class PySceneDetector(AbsSceneDetector):
    def __init__(self, fps=25, temp_dir = "./temp/"):

        self.fps = fps
        self.temp_dir = temp_dir
        cprint(f"\t(Scene Detection) PySceneDetect initialized", "green", attrs=["bold", "reverse"])

    def split_video_into_scenes(self, video_path):
        """Split a video into the different scenes composing it.
        Args:
            video_path (str): path specifying where the video is stored.
        Returns:
            [(scene_path, start_timestamp, end_timestamp): list of tuples representing the detected scenes.
        """

        # -- creating temporal directory
        os.makedirs(self.temp_dir, exist_ok=True)

        # -- detecting scenes of the video
        scenes = detect(video_path, ContentDetector())

        # -- split scenes longer than 60 seconds
        scenes = self._split_long_scenes(scenes)

        num_scenes_to_print = len(scenes) if len(scenes) > 0 else 1
        cprint(f"\n\n\t(Scene Detection) Splitting {video_path} into {num_scenes_to_print} scenes...", "green", attrs=["bold", "reverse"])

        # -- splitting the video into scenes and save them
        os.chdir(self.temp_dir)
        split_video_ffmpeg(f".{video_path}", scenes)
        os.chdir("..")
        cprint(f"\t(Scene Detection) Saving scene video clips in {self.temp_dir}", "green", attrs=["bold", "reverse"])

        # -- if no scenes were detected, return the original video path
        if len(scenes) == 0:
            scene_list = [(video_path, 0, check_video_duration(video_path))]
        else:
            video_list = sorted(glob.glob(self.temp_dir+"/*"))
            scene_list = [
                (scene_path, timestamp[0].get_seconds(), timestamp[1].get_seconds())
                for scene_path, timestamp in zip(video_list, scenes)
            ]

        return scene_list

    def _split_long_scenes(self, scenes, max_seconds=60):
        """Split those scenes that are longer than {max_seconds} seconds just for computational resources limitations.
        Args:
            [(start_timestamp, end_timestamp): list of tuples representing the detected scenes.
        Returns:
            [(start_timestamp, end_timestamp): list of detected scenes where long ones were split.
        """
        split_scenes = []
        max_frames = int(max_seconds * self.fps)
        for start_timestamp, end_timestamp in scenes:
             scene_frames = end_timestamp - start_timestamp

             if scene_frames > max_frames:

                 start_split = start_timestamp
                 while start_split < end_timestamp:
                     end_split = min(start_split+max_frames, end_timestamp)
                     split_scenes.append( (start_split, end_split)  )
                     start_split = end_split

             else:
                 split_scenes.append( (start_timestamp, end_timestamp) )

        return split_scenes
