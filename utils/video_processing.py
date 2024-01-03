import os
import cv2
import subprocess

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

def save_scene(output_path, scene_path, waveform_path, face_boundings, face_frames, asd_scores, asd_labels):
    cap = cv2.VideoCapture(scene_path)
    video_frames = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        video_frames.append(frame)
        if ret == False:
            break

    for speaker_id in asd_scores.keys():
        # FOR EACH FRAME WHERE THE FACE IS SHOWING
        for i, fr in enumerate(face_frames[speaker_id]):
            try:
                frame = video_frames[fr]
                greenValue = int(255*asd_labels[speaker_id][i])
                redValue = 255-greenValue
                color = (0, greenValue, redValue)
                xmin, ymin, xmax, ymax = face_boundings[speaker_id][i]
                frame = cv2.rectangle(
                    frame, (xmin, ymin), (xmax, ymax), color, 1)
                cv2.putText(frame, "{:.2f}".format(
                    asd_scores[speaker_id][i]*100), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            except:
                pass
    cap.release()
    cv2.destroyAllWindows()

    create_video(output_path, video_frames, waveform_path, width, height)

def create_video(output_path, video_frames, waveform_path, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter("temp.mp4", fourcc, 25, (width, height))

    for frame in video_frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

    subprocess.call([
        "ffmpeg",
        "-y",
        "-i",
        os.getcwd()+f"/temp.mp4",
        "-i",
        waveform_path,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-shortest",
        output_path,
        "-loglevel",
        "quiet",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    os.remove(os.getcwd()+f"/temp.mp4")
