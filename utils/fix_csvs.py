import os
import sys
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

    root_dir = sys.argv[1]

    program_ids = sorted( os.listdir(root_dir) )
    for program_id in tqdm(program_ids):
        if program_id in ["LM-20170116","LM-20170119","LM-20170125"]:
            continue
        program_csv_path = os.path.join(root_dir, program_id, program_id+".csv")

        new_df = []
        df_program = pd.read_csv(program_csv_path)
        for i, row_tuple in enumerate(df_program.itertuples(index=False, name=None)):
            video, sample_start, sample_end, duration, speaker, pickle_path, transcription, scene_path = row_tuple

            # -- obtaining first sample of the scene
            scene_samples = df_program[df_program["scene_path"] == scene_path]
            scene_start = scene_samples.iloc[0]["start"]

            new_df.append( (video, scene_start, sample_start, sample_end, duration, speaker, pickle_path, transcription, scene_path) )

        new_df = pd.DataFrame(new_df, columns=["video", "scene_start", "sample_start", "sample_end", "duration", "speaker", "pickle_path", "transcription", "scene_path"])
        new_df.to_csv(program_csv_path.replace(".csv", "_fixed.csv"))
