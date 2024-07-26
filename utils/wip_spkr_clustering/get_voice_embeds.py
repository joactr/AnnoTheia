import os
import glob
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Wav2Vec Speaker ID Embeddings')
    parser.add_argument("--wavs-dir", default="./WAVs/", type=str)
    parser.add_argument("--output-dir", default="./voice_embeddings/", type=str)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    wav_paths = glob.glob(f'{args.wavs_dir}{os.path.sep}*.wav')

    model = Wav2Vec2ForSequenceClassification.from_pretrained('superb/wav2vec2-base-superb-sid')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('superb/wav2vec2-base-superb-sid')

    for wav_path in tqdm(wav_paths):
        speech, _ = librosa.load(wav_path, sr=16000, mono=True)
        inputs = feature_extractor(speech, sampling_rate=16000, padding=True, return_tensors='pt')

        avg_last_hidden_state = model(**inputs).hidden_states[-1].squeeze(0).mean(dim=0)

        output_path = os.path.join(args.output_dir, os.path.basename(wav_path).replace('.wav', '.npz'))
        np.savez_compressed(output_path, data=avg_last_hidden_state.detach().cpu().numpy())
