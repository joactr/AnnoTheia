import os
import cv2
import subprocess
from tqdm import tqdm
from termcolor import cprint

def check_video_duration(video_path):
    """Computes the duration in seconds of a video clip
    Args:
        video_path: path to where the video clip is stored.
    Returns:
        float: duration in seconds of the video clip.
    """
    duration = subprocess.check_output([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ])

    return float(duration)

def get_suitable_scenes(scenes_list, face_detector, face_max_frame):
    """Filters a list of scenes to find the suitable ones where at least
      a face has appeared in the first {face_max_frame} frame.

    Args:
        scenes_list ([(scene_path, start_timestamp, end_timestamp)]): list of tuples representing scenes as tuples of three elements.
        face_detector (torch.nn.Module): face detector module.
        face_max_frame (int): the maximum frame where at least one face should have appeared on scene.

    Returns:
        [(scene_path, start, end)]: sublist of the input {scenes_list} containing only the scenes considered as suitable for the toolkit.
    """
    suitable_scenes = []

    cprint(f"SD: Filtering the detected scenes...", "green", attrs=["bold", "reverse"])
    for scene_path, start_timestamp, end_timestamp in tqdm(scenes_list):
        # -- setting boolean variable
        suitable = True

        # -- reading scene
        cap = cv2.VideoCapture(scene_path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # -- if the video scene does not satisfy the {face_max_frame} condition
        if frames < face_max_frame:
            suitable = False

        # -- reading frame at position {face_max_frame}
        cap.set(cv2.CAP_PROP_POS_FRAMES, face_max_frame - 1)
        ret, frame = cap.read()

        # -- if no face is detected
        if len(face_detector.detect_faces(frame)) == 0:
            suitable = False

        # -- in case it is a non-suitable scene, we remove it and continue with the next scene
        if not suitable:
            cap.release()
            os.remove(scene_path)
            continue

        suitable_scenes.append( (scene_path, start_timestamp, end_timestamp) )
        cap.release()
    cprint(f"SD: Discarding {len(scenes_list) - len(suitable_scenes)} non-suitable scenes.", "green", attrs=["bold", "reverse"])

    return suitable_scenes
