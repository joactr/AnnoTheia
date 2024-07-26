import os
import glob
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from deepface import DeepFace

def process_face(face_path):
    # -- we are sure there is only one face
    face_repr = DeepFace.represent(
        img_path=face_path,
        model_name='Facenet',
        enforce_detection=False,
    )
    face_embed = np.array(face_repr[0]['embedding'])

    output_path = os.path.join(args.output_dir, os.path.basename(face_path).replace('.png', '.npz'))
    np.savez_compressed(output_path, data=face_embed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ArcFace-based Face ID Embeddings')
    parser.add_argument("--face-dir", default="./data/FACEs/", type=str)
    parser.add_argument("--output-dir", default="./data/face_embeddings/", type=str)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    face_paths = glob.glob(f'{args.face_dir}{os.path.sep}*.png')


    loop = tqdm(face_paths)
    joblib.Parallel(n_jobs=8)(
        joblib.delayed(process_face)(face_path) for face_path in loop
    )
