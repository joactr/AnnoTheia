from tqdm import tqdm

def get_suitable_scenes(video_list, scene_list, face_detector, max_frame):
    """Filters a list of scenes to find the suitable ones where at least
      a face has appeared in the first {max_frame} frame.

    Args:
        scene_list ([(video_path, start_timestamp, end_timestamp)]): list of tuples representing scenes as tuples of three elements
        face_detector (torch.nn.Module): face detector module
        max_frame (int): the maximum frame where at least one face should have appeared on scene

    Returns:
        [(video_path, start, end)]: sublist of the input {scene_list} containing only the scenes considered as suitable for the toolkit
    """
    suitable_scenes = []

    print("Filtering suitable scenes...")
    for video_path, start_timestamp, end_timestamp in tqdm(scene_list):
        suitable = True

        # -- reading scene
        cap = cv2.VideoCapture(video_path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # -- if the video scene does not satisfy the {max_frame} condition
        if frames < max_frame:
            suitable = False

        # -- reading frame at position {max_frame}
        cap.set(cv2.CAP_PROP_POS_FRAMES, max_frame - 1)
        ret, frame = cap.read()

        # -- if no face is detected
        if len(face_detector.detect(frame)) == 0:
            suitable = False

        # -- in case it is a non-suitable scene, we remove it and continue with the next scene
        if not suitable:
            cap.release()
            os.remove(video_path)
            continue

        suitable_scenes.append( (video_path, start_timestamp, end_timestamp) )
        cap.release()

    return suitable_scenes
