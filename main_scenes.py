import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import glob
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from termcolor import cprint

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
    cprint(f"\n(Pipeline) Building AnnoTheia's Pipeline...", "light_grey", attrs=["bold","reverse"])
    pipeline = DetectCandidateScenesTask.build_pipeline(config)

    videos_to_process = sorted( os.listdir(args.video_dir) )
    cprint(f"\n(Pipeline) Proccessing the {len(videos_to_process)} videos included in {args.output_dir}", "light_grey", attrs=["bold","reverse"])
    for video_filename in tqdm(videos_to_process, desc="Videos", leave=False):
        # -- getting a video ID
        video_id = os.path.splitext(video_filename)[0]

        # -- creating output directories
        video_output_dir = os.path.join(args.output_dir, video_id)

        if config.pipeline_conf["save_scenes"]:
            scenes_output_dir = os.path.join(video_output_dir, "scenes")
            os.makedirs(scenes_output_dir, exist_ok=True)

        pickles_output_dir = os.path.join(video_output_dir, "pickles")
        os.makedirs(pickles_output_dir, exist_ok=True)

        # -- removing temporary files from the previous video that was processed
        shutil.rmtree(temp_dir, ignore_errors=True)

        # -- processing video using the AnnoTheia's pipeline
        video_path = os.path.join(args.video_dir, video_filename)
        video_df_output_path = os.path.join(video_output_dir, f"{video_id}.csv")

        # -- checking last scene processed in case you are reanuding the video processing
        last_processed_scene_id = -1
        if os.path.exists(video_df_output_path):
            last_processed_scene_id = int( pd.read_csv(video_df_output_path)["scene_path"].tolist()[-1].split(".")[0][-4:] )

        pipeline.process_video(video_path, video_output_dir, video_df_output_path, last_processed_scene_id)

        cprint(f"\n\t(Pipeline) {video_filename} has been processed. What are you waiting for? Come on, you can annotate it!", "light_grey", attrs=["bold","reverse"])
        cprint(f"\t(Pipeline) Check the candidate scenes in {video_df_output_path}\n", "light_grey", attrs=["bold","reverse"])

    # -- removing temporary files from the previous video that was processed
    shutil.rmtree(temp_dir, ignore_errors=True)
