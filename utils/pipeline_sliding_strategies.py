from termcolor import cprint
from collections import defaultdict

def non_overlap_sliding_strategy(audio_waveform, face_crops, face_frames, window_size, smoothing_window_size, active_speaker_detection):
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
        cprint(f"ASD: Analyzing if person {actual_speaker} on scene the one who is actually speaking", "blue", attrs=["bold", "reverse"])

        # -- depending on the video sliding strategy method chosen
        scene_length = len(face_crops[actual_speaker])
        extended_scene_length = scene_length + window_size + 1

        # -- for each window sample
        for window_idx in range(0, extended_scene_length, window_size):
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
            asd_scores[actual_speaker].extend( scores[:, 1] )

        # -- smoothing sliding window
        score_length = len(asd_scores[actual_speaker])
        side_smoothing_window_size = int( (smoothing_window_size - 1) / 2 )
        for frame_idx in range(0, score_length):
            start = max(0, frame_idx - side_smoothing_window_size)
            end = frame_idx + side_smoothing_window_size + 1

            asd_scores[actual_speaker][i] = np.mean(
                asd_scores[actual_speaker][start:end]
            )

    return asd_scores
