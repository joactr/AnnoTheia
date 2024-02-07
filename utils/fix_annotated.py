import sys
import pandas as pd

if __name__ == "__main__":
    ref_path = sys.argv[1]
    annotated_path = sys.argv[2]

    ref_df = pd.read_csv(ref_path)
    annotated_df = pd.read_csv(annotated_path)

    new_df = []
    for i, row_tuple in enumerate(annotated_df.itertuples(index=False, name=None)):
        video, sample_start, sample_end, duration, speaker, pickle_path, transcription, scene_path, _ = row_tuple

        # -- obtaining first sample of the scene
        scene_samples = ref_df[ref_df["pickle_path"] == pickle_path]
        scene_start = scene_samples.iloc[0]["scene_start"]

        new_df.append( (video, scene_start, sample_start, sample_end, duration, speaker, pickle_path, transcription, scene_path) )

    new_df = pd.DataFrame(new_df, columns=["video", "scene_start", "sample_start", "sample_end", "duration", "speaker", "pickle_path", "transcription", "scene_path"])
    new_df.to_csv(annotated_path.replace(".csv", "_fixed.csv"))

