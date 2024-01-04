# Active Speaker Detection and Transcription

AnnoTheia is a data annotation tool that takes in a video and uses Active Speaker Detection (ASD) techniques to extract the fragments where a person is speaking and obtain the transcription. It also has a GUI where you can manually check and validate the outputted samples.

## Features:

- Automatic ASD and transcription
- Manual review and correction
- Support for multiple video formats
- Customizable ASD parameters
- You can use your own ASD model
- Output of classified video with scores and bounding boxes (optional)

## Usage:

`python main_scenes.py --video-dir ${PATH_TO_VIDEO_DIR} --config-file ${PATH_TO_CONFIG_FILE} --output-dir ${PATH_TO_OUTPUT_DIR}`

### Arguments:

The execution arguments are read from the [config.json](/config.json) file, where you can modify the following settings:

- inputVideo: Path to video file to analyze
- outputVideo: Outputs a classified video with the scores and bounding boxes for each speaker and frame (default: False)
- minLength: Minimum length of detected speech segment (in frames) to be considered (default: 12)
- threshold: Classification threshold, lower values result in more detections but higher false positive rates (default: 0.04)
- windowSize: Context window size that the model analyzes (default: 51)
- whisperLang: Whisper audio-to-text language code (default: auto)
- minMethod: Analyzes the video with a sliding window taking into account only the central frame (very slow) (default: False)
- smoothWindowSize: Postprocessing sliding window size (in frames) for score smoothing (default: 11)

### Requirements:
- PYTHON 3.10
- FFMPEG
- VLC MEDIA PLAYER
- Python libraries found in requirements.txt
    - Can be installed with `pip install -r requirements.txt`

## GUI:
![Graphical user interface of AnnoTheia](./doc/image/interface.png)

### To launch the GUI, simply run the following command:

`python main_gui.py`

This will open a window where you can load a video and view the output of the ASD model. You can also manually review and correct the transcription.
