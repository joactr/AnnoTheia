import yaml
import argparse
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

    # -- building the toolkit's pipeline
    pipeline = DetectCandidateScenesTask.build_pipeline(config)

    # -- creating output directories
    videoclips_output_dir = os.path.join(config.output_dir, "videos")
    os.makedirs(videoclips_output_dir, exist_ok=True)

    pickles_output_dir = os.path.join(config.output_dir, "pickles")
    os.makedirs(pickles_output_dir, exist_ok=True)


    # -- removing temporary files from the previous video that was processed
