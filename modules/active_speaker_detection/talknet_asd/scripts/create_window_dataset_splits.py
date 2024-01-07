import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracting the 13-component MFCCs TalkNet-ASD is expecting.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--video-dir", required=True, type=str)
    parser.add_argument("--face-crops-dir", required=True, type=str)
    parser.add_argument("--mfccs-output-dir", required=True, type=str)
    args = parser.parse_args()

    # TODO
