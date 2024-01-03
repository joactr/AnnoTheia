import os
import yaml
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from tasks import DetectCandidateScenesTask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for detecting candidate scenes from long videos to compile a new audio-visual database",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video-dir", required=True, type=str, help="Directory where the videos to be processed are stored.")
    parser.add_argument("--config-file", required=True, type=str, help="Path to a configuration file specifying the details to build the AnnoTheia's candidate scene detection pipeline.")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory where the output provided by the toolkit will be stored.")

    args = parser.parse_args()

    # -- reading configuration file
    config_file = Path(args.config_file)
    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = argparse.Namespace(**config)

    # -- obtaining temporary directory
    temp_dir = config.scene_detection_conf["temp_dir"]

    # -- building the toolkit's pipeline
    pipeline = DetectCandidateScenesTask.build_pipeline(config)

    videos_to_process = sorted( os.listdir(args.video_dir) )
    for video_filename in tqdm(videos_to_process, leave=False):
        # -- creating output directories
        video_output_dir = os.path.join(args.output_dir, video_filename.split(".")[0])
        if config.pipeline_conf["save_scenes"]:
            scenes_output_dir = os.path.join(video_output_dir, "scenes")
            os.makedirs(scenes_output_dir, exist_ok=True)

        pickles_output_dir = os.path.join(video_output_dir, "pickles")
        os.makedirs(pickles_output_dir, exist_ok=True)

        # -- removing temporary files from the previous video that was processed
        temp_files_to_remove = glob.glob(f"{temp_dir}/*")
        for temp_filename_to_remove in temp_files_to_remove:
            os.remove(temp_filename_to_remove)

        # -- processing video using the AnnoTheia's pipeline
        video_path = os.path.join(args.video_dir, video_filename)
        scenes_info = pipeline.process_video(video_path, video_output_dir)

        # -- saving information w.r.t. the detected candidate scenes
        video_df_output_path = os.path.join(video_output_dir, "scenes_info.csv")
        video_df = pd.DataFrame(scenes_info, columns=["video", "scene_start", "ini", "end", "speaker", "pickle_path", "transcription"])
        video_df.to_csv(video_df_output_path, index=False)
