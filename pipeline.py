import numpy as np
import face_detection
import argparse
import os, python_speech_features
import scipy.io.wavfile as wav
import pandas as pd
from talkNet import talkNet
import torch
import cv2
import pickle
import tools
import subprocess
from collections import defaultdict
import whisper

# if len(sys.argv) != 2:
#     print("""
#         This script detects active speech in videos

#         Usage:  pipeline.py videoFile/directory
#         """)
#     sys.exit(0)

parser = argparse.ArgumentParser()
optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('-i', '--inputVideo',  metavar="inputVideo", help="Video file or directory directly containing video files", required=True, type=str)
parser.add_argument('-m', '--minLength', metavar="minLength", help="Minimum length of detected speech segment (in frames) to be considered",type=int, default=12)
parser.add_argument('-t', '--threshold', metavar="threshold", help="Classification threshold, lower values result in more detections but higher false positive rates",type=float, default=0.04)
parser.add_argument('-o', '--outputVideo', action='store_true', help="Outputs a classified video with the scores and bounding boxes for each speaker and frame")
#parser.add_argument('-m', '--model', metavar="model", help="Boolean that indicates if a classified video is given as output",type=bool, default=False)
parser.add_argument('-w', '--windowSize', metavar="windowSize", help="Context window size that the model analyzes",type=int, default=51)
parser.add_argument('-l', '--whisperLang', metavar="whisperLang", help="Whisper audio-to-text language code (es,en,zh,de...)", type=str, default="auto")
parser.add_argument('-n', '--minMethod', action='store_true', help="Analyzes the video with a sliding window taking into account only the central frame (very slow)")
parser.add_argument('-s', '--smoothWindowSize', metavar="smoothWindowSize", help="Postprocessing sliding window size for score smoothing",type=int, default=11)

parser._action_groups.append(optional)
parsed_args = parser.parse_args()
args = vars(parsed_args)


detector = face_detection.build_detector(
"DSFDDetector", confidence_threshold=.3, nms_iou_threshold=.5) #DSFDDetector

inputVideo = args['inputVideo']
videoDuration = tools.checkVideoDuration(inputVideo)
videoFrames = int(videoDuration*25)
fps = 25

#Carga vídeo
res, facePos, faceFrames = tools.saveMultiFace(inputVideo,detector,50)
# AUDIO PROCESSING
audioPath = tools.convert_video_to_audio_ffmpeg(inputVideo)
_,sig = wav.read(audioPath)
audio = python_speech_features.mfcc(sig, 16000, numcep = 13, winlen = 0.025, winstep = 0.010) #ASUME VIDEO A 25 Y AUDIO A 100, MODIFICAR

model = talkNet()
model.load_state_dict(torch.load("./weights/model51_0004.model"))
windowSize = args["windowSize"]

totalScores = defaultdict(list)
sideWindowSize = int((windowSize-1)/2)
sequential = not args["minMethod"]
meanWSize = args["smoothWindowSize"]
meanSideSize = int((meanWSize-1)/2)
thr = args["threshold"]

if sequential:
    # SECUENCIAL
    for actualSpeaker in res.keys():
        print("ANALYZING SPEAKER", actualSpeaker)
        for i in range(0,len(res[actualSpeaker])+windowSize+1,windowSize):
            #center = i
            center = faceFrames[actualSpeaker][0]+i
            #print("FRAME Nº ",center)
            iAudio = tools.padAudio(audio,1,center,windowSize,videoFrames).unsqueeze(0)
            iVideo = tools.padVideo(res[actualSpeaker],center-faceFrames[actualSpeaker][0],windowSize).unsqueeze(0)
            #print(iAudio[0][0][0:5],iVideo[0][0][0][0:5])
            scores,labels= model((iAudio,iVideo))
            totalScores[actualSpeaker].extend(scores[:,1].detach().cpu().numpy().tolist())
            #predArray[actualSpeaker].extend(labels.detach().cpu().numpy().tolist())

        # MEAN SLIDING WINDOW
        for i in range(len(totalScores[actualSpeaker])):
            ini = max(0,i-meanSideSize) # No negative values
            end = i+meanSideSize+1 # +1 as python does not take into account last value
            #print(ini,end)
            totalScores[actualSpeaker][i] = np.mean(totalScores[actualSpeaker][ini:end])
else:
    # MINIMO
    for actualSpeaker in res.keys():
        print("ANALYZING SPEAKER", actualSpeaker)
        for i in range(len(res[actualSpeaker])):
            #center = i
            center = faceFrames[actualSpeaker][0]+i
            iAudio = tools.padAudio(audio,1,center,windowSize,videoFrames).unsqueeze(0)
            iVideo = tools.padVideo(res[actualSpeaker],center-faceFrames[actualSpeaker][0],windowSize).unsqueeze(0)
            #print(iAudio[0][0][0:5],iVideo[0][0][0][0:5])
            scores,labels= model((iAudio,iVideo))
            totalScores[actualSpeaker].append(scores[sideWindowSize][1].detach().cpu().numpy())
        
        # MEAN SLIDING WINDOW
        for i in range(len(totalScores[actualSpeaker])):
            ini = max(0,i-meanSideSize) # No negative values
            end = i+meanSideSize+1 # +1 as python does not take into account last value
            #print(ini,end)
            totalScores[actualSpeaker][i] = np.mean(totalScores[actualSpeaker][ini:end])

# SMOOTHING
predArray = defaultdict(list)

# APPLY THRESHOLD
for actualSpeaker in res.keys():
    # Delete < 0 frame predictions (initial)
    #Solo para metodo secuencial ->
    if sequential:
        totalScores[actualSpeaker] = totalScores[actualSpeaker][sideWindowSize:len(res[actualSpeaker])+sideWindowSize]
    # Classify as speaking if score higher than threshold
    for sc in totalScores[actualSpeaker]:
        predLabel = 1 if sc > thr else 0
        predArray[actualSpeaker].append(predLabel)


def createVideo(videoName,imgFrames,audioPath,width,height):
    video = cv2.VideoWriter("output.mp4", 0, 25, (width,height))

    for image in imgFrames:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
    subprocess.call(["ffmpeg","-y","-i",os.getcwd()+f"/output.mp4","-i",audioPath,"-map","0:v","-map","1:a", "-c:v", "copy", "-shortest", os.getcwd()+f"/outputs/videos/{videoName}"])
    os.remove(os.getcwd()+f"/output.mp4")
    
# Returns a list of [start,end] timestamps in seconds of the parts where a speaker is speaking with minimum length of minLength frames
def getSpeaking(arr,minLength,fps):
    prev_idx = 0
    posFrames = 0
    idx_list = []
    for i,num in enumerate(arr):
        if num == 1:
            posFrames +=1
        else:
            if i-prev_idx >= minLength:
                idx_list.append((prev_idx/fps,i/fps))
            prev_idx = i
            posFrames = 0
    return idx_list

def saveFullVideo(videoName):
    cap = cv2.VideoCapture(inputVideo)
    videoImages = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, image = cap.read()
        videoImages.append(image)
        if ret == False:
            break


    for speakerN in totalScores.keys():
        for i, fr in enumerate(faceFrames[speakerN]): #Para cada frame en el que aparezca la cara
            try:
                image = videoImages[fr]
                greenValue = int(255*predArray[speakerN][i])
                redValue = 255-greenValue
                color = (0,greenValue,redValue)
                #print(color)
                xmin,ymin,xmax,ymax = facePos[speakerN][i]
                image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax),color , 1)
                cv2.putText(image, "{:.2f}".format(totalScores[speakerN][i]*100), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            except:
                pass
    cap.release()
    cv2.destroyAllWindows()
    
    
    createVideo(videoName,videoImages,audioPath,width,height)

#Asignación de bounding boxes y probabilidades al video de output
path = os.path.normpath(inputVideo)
videoName = str(path.split(os.sep)[-1])
if args["outputVideo"]:
    os.makedirs("outputs/videos", exist_ok=True)
    saveFullVideo(videoName)

# GUARDAR DATOS 
# COGER SOLO FRAGMENTO DE AUDIO EN EL QUE SE DETECTA HABLANTE
os.makedirs("outputs/pickles", exist_ok=True)
model = whisper.load_model("small")


if args["whisperLang"] == "auto":
    transcription = model.transcribe(audioPath, verbose=False, word_timestamps=True)
else:
    transcription = model.transcribe(audioPath, language=args["whisperLang"], verbose=False, word_timestamps=True)
pkDict = {"facePos":facePos,"faceFrames":faceFrames,"preds":predArray,"transcription":transcription}
with open("outputs/npz/"+videoName+'.pkl', 'wb') as f:  # open a text file
    pickle.dump(pkDict, f) # serialize the list

# EXTRAER PARTE APROPIADA DE LA TRANSCRIPCIÓN
wordArr = []
alignArr = []
for seg in transcription["segments"]:
    for w in seg["words"]:
        wordArr.append(w["word"])
        alignArr.append((w["start"],w["end"]))
pdRows = []

for speakerN in totalScores.keys():
    for (ini,end) in getSpeaking(predArray[speakerN],args["minLength"],25):
        iniW, endW = -1, -1
        for i, (alignIni, alignEnd) in enumerate(alignArr):
            if iniW == -1 and ini <= alignEnd and ini >= alignIni:
                iniW = i
            if endW == -1 and end <= alignEnd and end >= alignIni:
                endW = i
        if iniW > -1 and endW > -1:
            newRow = {'video':inputVideo, 'speaker':speakerN, 'ini': ini, 'end':end, 'dataPath': "outputs/npz/"+videoName+".pkl", 'transcription':''.join(wordArr[iniW:endW+1])}
            pdRows.append(newRow)

df = pd.DataFrame(pdRows,columns=['video', 'speaker', 'ini', 'end', 'dataPath', 'transcription'])
df.to_csv(r"outputs/res.csv")