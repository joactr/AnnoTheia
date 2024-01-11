import numpy as np
from termcolor import cprint
from collections import defaultdict

def non_overlap_sliding_strategy(audio_waveform, face_crops, face_frames, scene_frames, window_size, smoothing_window_size, active_speaker_detection):
    """Applies the non-overlapping sliding strategy when computing the frame-wise ASD scores.
    Args:
        audio_waveform (np.ndarray): numpy array representating the audio waveform.
        face_crops (list): list including the face cropped images appearing on scene.
        face_frames (list): list indicating in which frames each person was appearing on scene.
        window_size (int): integer indicating the size of the window samples to apply the ASD module.
        smoothing_window_size (int): integer indicating the size of the smoothing window to apply over the ASD scores.
        active_speaker_detection (AbsASD): active speaker detector module.
    Returns:
        asd_scores (defaultdict): dictionary of lists gathering for each possible speaker the frame-wise ASD scores.
    """

    # -- for each detected person on the scene
    asd_scores = defaultdict(list)
    for actual_speaker in face_crops.keys():
        cprint(f"\n\t(Active Speaker Detection) Analyzing if person {actual_speaker} on scene the one who is actually speaking", "blue", attrs=["bold", "reverse"])

        # -- depending on the video sliding strategy method chosen
        scene_length = len(face_crops[actual_speaker])
        extended_scene_length = scene_length + window_size + 1
        window_loop_range = range(0, extended_scene_length, window_size)

        # -- for each window sample
        window_count = 1
        for window_idx in window_loop_range:
            print(f"\t\tProcessing window {str(window_count).zfill(4)} of {str(len(window_loop_range)).zfill(4)}", end="\r")

            # -- computing window center index
            window_center = face_frames[actual_speaker][0] + window_idx

            # 3.1. Extracting preprocessed input data
            acoustic_input, visual_input = active_speaker_detection.preprocess_input(
                audio_waveform,
                face_crops[actual_speaker],
                window_center=window_center,
                window_size=window_size,
                total_video_frames=scene_frames,
            )

            # 3.2. Computing the frame-wise ASD scores
            scores = active_speaker_detection.get_asd_scores(
                acoustic_input,
                visual_input,
            )

            # 3.3. Gathering the average score of the window sample
            asd_scores[actual_speaker].extend( scores[:, 1].tolist() )

            # -- updating counter
            window_count += 1

        # -- smoothing sliding window
        score_length = len(asd_scores[actual_speaker])
        side_smoothing_window_size = int( (smoothing_window_size - 1) / 2 )
        for frame_idx in range(0, score_length):
            start = max(0, frame_idx - side_smoothing_window_size)
            end = frame_idx + side_smoothing_window_size + 1

            asd_scores[actual_speaker][frame_idx] = np.mean(
                asd_scores[actual_speaker][start:end]
            )

    return asd_scores

def get_speaking(asd_labels, min_length, fps):
    """Obtains the indices where a person is actually speaking during more than {min_length} consecutive frames.
    Args:
      asd_labels (list): list containing for a specific speaker the thresholding over the frame-wise ASD scores of the scene.
      min_length (int): minimum length in terms of frames to accept a scene as valid.
      fps (int): frame per second rate to consider.

    Returns:
      idx_list (list): list containing the indeces where a person is actually speaking.
    """
    prev_idx = 0
    pos_frames = 0

    idx_list = []
    for i, num in enumerate(asd_labels):
        if num == 1:
            pos_frames += 1
            # -- check if it is the end of a sequence
            if i == len(asd_labels) - 1 or asd_labels[i+1] == 0:
                # -- we add +1 because the sequence includes the current index
                if i-prev_idx + 1 >= min_length:
                    idx_list.append((prev_idx/fps, (i+1)/fps))
                # -- move to the next index
                prev_idx = i + 1
        else:
            if i-prev_idx >= min_length:
                idx_list.append((prev_idx/fps, i/fps))
            # -- move to the next index
            prev_idx = i + 1
            pos_frames = 0

    return idx_list
