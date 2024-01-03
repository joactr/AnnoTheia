import os
import cv2
import subrprocess

def convert_video_to_target_fps(video_path, target_fps, temp_dir):
    """Converts a video clip into {target_fps} fps.
    Args:
        video_path: path to where the video clip is stored.
        target_fps: the desired fps to convert into.
        temp_dir: temporary directory where the scenes where stored.
    """
    # -- reading video clip
    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    # -- checking if it necessary to do the fps convertion
    if abs(actual_fps - target_fps) > 0.1:

        # -- system call to ffmpeg
        output_path = os.path.join(temp_dir, f"{os.path.basename(video_path)}_{target_fps}fps.mp4")
        subprocess.call([
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-r",
            output_path,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        # Remove original video and change name of the new one
        os.remove(video_path)
        os.rename(output_path, video_path)

    # -- releasing video capture
    cap.release()
