import yaml
import argparse
from pathlib import Path

from tasks import DetectCandidateScenesTask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for detecting candidate scenes from long videos to compile a new audio-visual database",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config-file", required=True, type=str, help="Path to a configuration file specifying the details to build the AnnoTheia's candidate scene detection pipeline")

    args = parser.parse_args()

    # -- reading configuration file
    config_file = Path(args.config_file)
    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = argparse.Namespace(**config)

    pipeline = DetectCandidateScenesTask.build_pipeline(config)
    print(pipeline)
