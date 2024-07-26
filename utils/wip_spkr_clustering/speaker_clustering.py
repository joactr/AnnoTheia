import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import sentence_transformers
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Speaker Clustering')
    parser.add_argument("--embeds-dir", default="./data/face_embeddings/", type=str)
    parser.add_argument("--faces-dir", default="./data/FACEs/", type=str)
    parser.add_argument("--output-dir", default="./data/clusters/", type=str)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)



    embedding_paths = sorted(glob.glob(f'{args.embeds_dir}{os.path.sep}*.npz'))
    embeddings = np.array([np.load(embedding_path)['data'] for embedding_path in embedding_paths])
    embeddings = StandardScaler().fit_transform(embeddings)

    # voice_embedding_paths = sorted(glob.glob(f'{args.embeds_dir.replace("face_", "voice_")}{os.path.sep}*.npz'))
    # voice_embeddings = np.array([np.load(voice_embedding_path)['data'] for voice_embedding_path in voice_embedding_paths])
    # voice_embeddings = StandardScaler().fit_transform(voice_embeddings)
    # embeddings = np.hstack((face_embeddings, voice_embeddings))

    face_paths = sorted(glob.glob(f'{args.faces_dir}{os.path.sep}*.png'))

    # embeddings = face_embeddings
    clusters = sentence_transformers.util.community_detection(embeddings, threshold=0.8)
    for cluster_idx, cluster in enumerate(tqdm(clusters, leave=False)):
        cluster_path = os.path.join(args.output_dir, f'speaker_{str(cluster_idx).zfill(4)}')
        os.makedirs(cluster_path, exist_ok=True)

        for sample_idx in tqdm(cluster):
            src_path = face_paths[sample_idx]
            dst_path = os.path.join(cluster_path, os.path.basename(src_path))

            os.system(f'cp {src_path} {dst_path}')
