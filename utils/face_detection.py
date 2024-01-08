import cv2
import numpy as np
from termcolor import cprint
from collections import defaultdict

def detect_multiple_faces(video_path, face_detector, face_aligner=None, min_face_size=32, max_face_distance_thr=50):
    """Detects multiple faces on the scene controlling the identity of each person.
    Args:
        video_path (str): path where the video clip is stored.
        face_detector (AbsFaceDetector): face detector module.
        face_aligner (AbsFaceAligner): face aligner module.
        min_face_size (int): minimum size dimension to filter out non-suitable faces.
        max_face_distance_thr (float): threshold to determine if we found a new person on scene.
    Returns:
        multi_face_crops (defaultdict): dictionary of lists including the face crops prepared for the audio-visual ASD module.
        multi_face_boundings (defaultdict): dictionary of lists including the bounding boxes of each person appearing on scene.
        multi_face_landmarks (defaultdict): dictionary of lists including the facial landmarks of each person appearing on scene.
        multi_face_frames (defaultdict): dictionary of lists indicating in which frames each person was appearing on scene.
    """

    # -- capturing video clip
    cap = cv2.VideoCapture(video_path)
    total_nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cprint(f"\t(Face Detection) Detecting faces on the scene {video_path}...", "yellow", attrs=["bold", "reverse"])
    cprint(f"\t(Face Alignment) Facial Landmarks will also be extracted...", "cyan", attrs=["bold", "reverse"])

    # -- reading first frame
    ret, frame = cap.read()

    frame_count = 0
    multi_face_crops = defaultdict(list)
    multi_face_boundings = defaultdict(list)
    multi_face_landmarks = defaultdict(list)
    multi_face_frames = defaultdict(list)

    are_there_faces = False
    # -- while we do not reach the end of the video clip
    while ret:
        print(f"\t\tProcessing frame {str(frame_count+1).zfill(4)} of {str(total_nframes).zfill(4)}", end="\r")

        # -- -- detecting all faces on the frame alognside their corresponding bounding boxes
        face_crops, face_boundings, face_landmarks = _detect_suitable_faces(video_path, frame, face_detector, face_aligner, min_face_size)

        # -- -- if they are the first faces detected
        if not are_there_faces:
            if len(face_crops) > 0:
                are_there_faces = True
                for face_idx in range(len(face_crops)):
                    multi_face_crops[face_idx].append( face_crops[face_idx] )
                    multi_face_boundings[face_idx].append( face_boundings[face_idx] )
                    multi_face_landmarks[face_idx].append( face_landmarks[face_idx] )
                    multi_face_frames[face_idx] = [frame_count]
        else:
            # -- identifying different people based on the distance between bounding boxes and according to a magic threshold in case new faces
            if len(face_crops) > 0:
                for face_idx in range(len(face_crops)):
                    min_dist = 2**32
                    pred_face = -1

                    # -- comparing the recent detected face to the already saved faces from previous frames
                    for face_id in multi_face_boundings.keys():
                        dist = np.linalg.norm(np.asarray(
                            face_boundings[face_idx]) - np.asarray(multi_face_boundings[face_id][-1]
                        ))

                        if dist < min_dist:
                            pred_face = face_id
                            min_dist = dist

                       # -- it will be a new face if ...
                        if min_dist > max_face_distance_thr:
                            pred_face = len(multi_face_boundings.keys())

                    multi_face_crops[pred_face].append( face_crops[face_idx] )
                    multi_face_boundings[pred_face].append( face_boundings[face_idx] )
                    multi_face_landmarks[pred_face].append( face_landmarks[face_idx] )
                    multi_face_frames[pred_face].append( frame_count )

        ret, frame = cap.read()
        frame_count += 1
    print()

    return multi_face_crops, multi_face_boundings, multi_face_landmarks, multi_face_frames

def _detect_suitable_faces(video_path, image, face_detector, face_aligner, min_face_size=32):
    """Detects all faces in an image and return them alongside their corresponding bounding boxes.
    Args:
        video_path (str): path were the video clip is stored just to debug purposes.
        image (np.ndarray): numpy array representing the image frame.
        face_detector (AbsFaceDetector): face detection module.
        face_aligner (AbsFaceAligner): face aligner module.
        min_face_size (int): minimum size dimension to filter out non-suitable faces.
    Returns:
        [np.ndarray]: list of cropped face images.
        [np.ndarray]: list of face bounding boxes.
    """
    detections = face_detector.detect_faces(image)
    predictions = face_aligner.detect_facial_landmarks(image, detections)

    face_crops = []
    face_boundings = []
    face_landmarks = []

    if len(detections) > 0:
        for face_idx, bb in enumerate(detections):
            left, top, right, bottom = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])

            # -- only faces that do satisfy the size condition
            if right - left >= min_face_size and bottom - top >= min_face_size:

                # -- scaling faces to a specific dimension
                resized_face = cv2.resize(
                    image[max(top, 0):bottom, max(left, 0):right],
                    (112, 112),
                )
                # -- converting faces to grayscale images
                grayscale_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)

                # -- adding suitable detected faces
                face_crops.append( grayscale_face )
                face_boundings.append( (left, top, right, bottom) )
                face_landmarks.append( predictions[face_idx] )

    return face_crops, face_boundings, face_landmarks
