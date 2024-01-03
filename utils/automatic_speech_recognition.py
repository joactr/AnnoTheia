def get_speaking(asd_labels, min_length, fps):
    """Obtains the indices where a person is actually speaking during more than {min_length} consecutive frames.
    Args:
      asd_labels (dict): dictionary containing for each possible speaker the thresholding over the frame-wise ASD scores for a specific scene.
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
                if i-prev_idx + 1 >= min_length:  # +1 because the sequence includes the current index
                    # +1 because the sequence includes the current index
                    idx_list.append((prev_idx/fps, (i+1)/fps))
                prev_idx = i + 1  # Move to the next index
        else:
            if i-prev_idx >= min_length:
                idx_list.append((prev_idx/fps, i/fps))
            prev_idx = i + 1  # Move to the next index
            pos_frames = 0

    return idx_list
