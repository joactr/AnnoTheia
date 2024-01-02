from modules.scene_detector.abs_scene_detector import AbsSceneDetector
from scenedetect import detect, ContentDetector, split_video_ffmpeg

class PySceneDetector(AbsSceneDetector):
    def __init__(self, temp_dir = "./temp/"):

        self.temp_dir = temp_dir

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

        # -- splitting the video into scenes and save them
        print(f"Splitting {video_path} into scenes...")

        os.chdir(self.temp_dir)
        split_video_ffmpeg(f".{video_path}", scenes)
        os.chdir("..")

        # -- if no scenes were detected, return the original video path
        if len(scenes) == 0:
            scene_list = [(video_path, 0, self._check_video_duration(video_path))]
        else:
            video_list = glob.glob(self.temp_dir+"/*")
            scene_list = [
                (scene_path, timestamp[0].get_seconds(), timestamp[1].get_seconds())
                for scene_path, timestamp in zip(video_list, scenes)
            ]

        return scene_list

    def _check_video_duration(self, video_path):
        duration = subprocess.check_output([
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path
        ])

        return float(duration)

