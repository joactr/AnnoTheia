import numpy as np
import face_detection
import os, python_speech_features
import scipy.io.wavfile as wav
import pandas as pd
from talkNet import talkNet
import torch
import json
import pickle
import tools
from collections import defaultdict
import whisper

config_file = open('config.json')
config = json.load(config_file)

# Models
detector = face_detection.build_detector(
"DSFDDetector", confidence_threshold=.3, nms_iou_threshold=.5) #DSFDDetector
model = talkNet()
checkpoint_name = 'model51_0004.model'
model.load_state_dict(torch.load(os.path.join('weights',checkpoint_name)))
transcription_model = whisper.load_model("small")

windowSize = config.get("windowSize", 51)
totalScores = defaultdict(list)
sideWindowSize = int((windowSize-1)/2)
sequential = config.get("minMethod", "False") == "False"
meanWSize = config.get("smoothWindowSize")
meanSideSize = int((meanWSize-1)/2)
thr = config.get("threshold", 0.04)

videoPath = config.get("inputVideo")

os.makedirs("outputs/videos", exist_ok=True)
os.makedirs("outputs/pickles", exist_ok=True)


output_samples = []

video_list, scene_list = tools.getVideoScenes(videoPath)

# ANALYZE EACH SEPARATE VIDEO
for inputVideo, (video_start,video_end) in zip(video_list,scene_list):
    videoDuration = tools.checkVideoDuration(inputVideo)
    #######
    # CONVERT VIDEO TO 25FPS AND 100HZ OF AUDIO SAMPLING (FFMPEG)
    #######
    videoFrames = int(videoDuration*25)
    fps = 25
    # Load and process video
    res, facePos, faceFrames = tools.saveMultiFace(inputVideo,detector,50)
    # AUDIO PROCESSING
    audioPath = tools.convert_video_to_audio_ffmpeg(inputVideo)
    _,sig = wav.read(audioPath)
    audio = python_speech_features.mfcc(sig, 16000, numcep = 13, winlen = 0.025, winstep = 0.010) # Assumes video at 25 fps and audio at 100 hz



    if sequential:
        # SECUENCIAL
        for actualSpeaker in res.keys():
            print("ANALYZING SPEAKER", actualSpeaker)
            for i in range(0,len(res[actualSpeaker])+windowSize+1,windowSize):
                center = faceFrames[actualSpeaker][0]+i
                iAudio = tools.padAudio(audio,1,center,windowSize,videoFrames).unsqueeze(0)
                iVideo = tools.padVideo(res[actualSpeaker],center-faceFrames[actualSpeaker][0],windowSize).unsqueeze(0)
                scores, labels = model((iAudio,iVideo))
                totalScores[actualSpeaker].extend(scores[:,1].detach().cpu().numpy().tolist())
            # MEAN SLIDING WINDOW
            for i in range(len(totalScores[actualSpeaker])):
                ini = max(0,i-meanSideSize) # No negative values
                end = i+meanSideSize+1 # +1 as python does not take into account last value
                totalScores[actualSpeaker][i] = np.mean(totalScores[actualSpeaker][ini:end])
    else:
        # MINIMO
        for actualSpeaker in res.keys():
            print("ANALYZING SPEAKER", actualSpeaker)
            for i in range(len(res[actualSpeaker])):
                center = faceFrames[actualSpeaker][0]+i
                iAudio = tools.padAudio(audio,1,center,windowSize,videoFrames).unsqueeze(0)
                iVideo = tools.padVideo(res[actualSpeaker],center-faceFrames[actualSpeaker][0],windowSize).unsqueeze(0)
                scores, labels = model((iAudio,iVideo))
                totalScores[actualSpeaker].append(scores[sideWindowSize][1].detach().cpu().numpy())
            
            # MEAN SLIDING WINDOW
            for i in range(len(totalScores[actualSpeaker])):
                ini = max(0,i-meanSideSize) # No negative values
                end = i+meanSideSize+1 # +1 as python does not take into account last value
                totalScores[actualSpeaker][i] = np.mean(totalScores[actualSpeaker][ini:end])

    # SMOOTHING
    predArray = defaultdict(list)

    # APPLY THRESHOLD
    for actualSpeaker in res.keys():
        # Delete < 0 frame predictions (initial)
        if sequential:
            totalScores[actualSpeaker] = totalScores[actualSpeaker][sideWindowSize:len(res[actualSpeaker])+sideWindowSize]
        # Classify as speaking if score higher than threshold
        for sc in totalScores[actualSpeaker]:
            predLabel = 1 if sc > thr else 0
            predArray[actualSpeaker].append(predLabel)

    # BOUNDING BOXES AND CONFIDENCE
    path = os.path.normpath(inputVideo)
    videoName = str(path.split(os.sep)[-1])

    if config.get("outputVideo", "False") == "True": 
        tools.saveFullVideo(videoName, inputVideo, audioPath, totalScores, faceFrames, predArray, facePos)

    # SAVE VIDEO DATA
    if config.get("whisperLang", "auto") == "auto":
        transcription = transcription_model.transcribe(audioPath, verbose=False, word_timestamps=True)
    else:
        transcription = transcription_model.transcribe(audioPath, language=config.get("whisperLang"), verbose=False, word_timestamps=True)
    pkDict = {"facePos":facePos,"faceFrames":faceFrames,"preds":predArray,"transcription":transcription}
    with open("outputs/npz/"+videoName+'.pkl', 'wb') as f:  # open a text file
        pickle.dump(pkDict, f) # serialize the list

    # EXTRACT APPROPIATE FRAGMENT OF TRANSCRIPTION
    wordArr = []
    alignArr = []
    for seg in transcription["segments"]:
        for w in seg["words"]:
            wordArr.append(w["word"])
            alignArr.append((w["start"],w["end"]))
    
    # EXTRACT VALID SCENES AND ALIGN TRANSCRIPTION
    align_margin = 0.1
    for speakerN in totalScores.keys():
        for (ini,end) in tools.getSpeaking(predArray[speakerN],config.get("minLength", 12),25):
            iniW, endW = -1, -1
            for i, (alignIni, alignEnd) in enumerate(alignArr):
                if ini-align_margin <= alignEnd and ini >= alignIni:
                    iniW = i
                if alignEnd+align_margin <= end:
                    endW = i
            if iniW > -1 and endW > -1:
                newRow = {'video':videoPath, 'speaker':speakerN, 'ini': video_start+ini, 'end':video_start+end,
                            'dataPath': "outputs/npz/"+videoName+".pkl", 'transcription':''.join(wordArr[iniW:endW+1])}
                output_samples.append(newRow)
    
    os.remove(inputVideo)

df = pd.DataFrame(output_samples,columns=['video', 'speaker', 'ini', 'end', 'dataPath', 'transcription'])
df.to_csv(r"outputs/res.csv")