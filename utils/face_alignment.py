"""
Code used in the algorithm in charge of extracting the SOTA Regions of Interest was mainly based on the work carried out in:

https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/tree/master/dataloader

"""
import os
import cv2
import pickle
import numpy as np

def load_video(filename):
    """load_video.
        :param filename: str, the fileanme for a video sequence.
    """
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()

def landmarks_interpolate(landmarks):
    """landmarks_interpolate.
        :param landmarks: List, the raw landmark (in-place)
    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks

def linear_interpolate(landmarks, start_idx, stop_idx):
    """linear_interpolate.
        :param landmarks: ndarray, input landmarks to be interpolated.
        :param start_idx: int, the start index for linear interpolation.
        :param stop_idx: int, the stop for linear interpolation.
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

def affine_transform(
        frame,
        landmarks,
        reference,
        grayscale=False,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0
    ):
        """affine_transform.
            :param frame: numpy.array, the input sequence.
            :param landmarks: List, the tracked landmarks.
            :param reference: numpy.array, the neutral reference frame.
            :param grayscale: bool, save as grayscale if set as True.
            :param target_size: tuple, size of the output image.
            :param reference_size: tuple, size of the neural reference frame.
            :param stable_points: tuple, landmark idx for the stable points.
            :param interpolation: interpolation method to be used.
            :param border_mode: Pixel extrapolation method .
            :param border_value: Value used in case of a constant border. By default, it is 0.
        """
        # Prepare everything
        if grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stable_reference = np.vstack([reference[x] for x in stable_points])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0

        # Warp the face patch and the landmarks
        transform = cv2.estimateAffinePartial2D(np.vstack([landmarks[x] for x in stable_points]),
                                                stable_reference, method=cv2.LMEDS)[0]
        transformed_frame = cv2.warpAffine(
            frame,
            transform,
            dsize=(target_size[0], target_size[1]),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()

        return transformed_frame, transformed_landmarks

def cut_patch(img, landmarks, height, width, threshold=5):
    """cut_patch.
        :param img: ndarray, an input image.
        :param landmarks: ndarray, the corresponding landmarks for the input image.
        :param height: int, the distance from the centre to the side of of a bounding box.
        :param width: int, the distance from the centre to the side of of a bounding box.
        :param threshold: int, the threshold from the centre of a bounding box to the side of image.
    """
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception('too much bias in width')

    cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img

def rotate_frame(landmarks, frame):
    """rotate_frame.
        param: landmarks: list, the extracted 68 facial landmarks
        param: frame: ndarray, image of the current frame from the corresponding sample
    """
    # Locating landmarks related with the shape and corners of the speaker's mouth
    left_lip_corner = landmarks[0] # point 49
    right_lip_corner = landmarks[6] # point 55
    upper_lip_center = landmarks[3] # point 52
    lower_lip_center = landmarks[9] # point 58

    # Computing the angle between lip corners. In this way, we check if the speaker's face is tilted
    dY = right_lip_corner[1] - left_lip_corner[1]
    dX = right_lip_corner[0] - left_lip_corner[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Computing the mouth's center x,y-coordinates
    mouth_center = ((left_lip_corner[0] + right_lip_corner[0] + upper_lip_center[0] + lower_lip_center[0]) // 4,
        (left_lip_corner[1] + right_lip_corner[1] + upper_lip_center[1] + lower_lip_center[1]) // 4)

    # Computing  the Rotation Matrix in order to align the frame regarding the speaker's mouth
    M = cv2.getRotationMatrix2D(mouth_center, angle, 1)

    # Applying the affine transform
    dim = frame.shape[0:2]
    aligned_frame = cv2.warpAffine(frame, M, (dim[1], dim[1]), flags=cv2.INTER_CUBIC)

    return aligned_frame
