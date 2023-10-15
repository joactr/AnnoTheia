import subprocess
import os
import cv2
from math import floor,ceil
import torch
import torch.nn.functional as F
import random
import numpy as np
from collections import defaultdict


def convert_video_to_audio_ffmpeg(video_file, output_ext="wav"):
    """Extracts audio from video to .wav format, returns path of the resulting audio"""
    #filename, ext = os.path.splitext(video_file)
    filename = os.path.basename(os.path.realpath(video_file))
    filename = filename.split(r'/|\\')[-1]
    print(filename)
    #Gets working directory and extracts audio
    subprocess.call(["ffmpeg","-y","-i",video_file,"-vn","-ac","1","-ar","16000","-acodec", "pcm_s16le",  os.getcwd()+f"/{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    return os.getcwd()+f"/{filename}.{output_ext}" #Nombre del audio

def extractBiggestFace(img,detector):
    """
    Detecta todas las caras de una imagen y devuelve la más grande recortada y reescalada a 112x112
    """
    detections = detector.detect(img)
    idx_max = -1
    area_max = -1
    for i,cntr in enumerate(detections):
        xmin,ymin,xmax,ymax = int(cntr[0]),int(cntr[1]),int(cntr[2]),int(cntr[3]) #Guardamos bounding box
        area = (xmax-xmin)*(ymax-ymin)
        if area > area_max: #Comprobamos si la cara es la más grande
            idx_max = i
            area_max = area
            #print(area,idx_max)
        #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    cntr = detections[idx_max]
    try:
        xmin,ymin,xmax,ymax = int(cntr[0]),int(cntr[1]),int(cntr[2]),int(cntr[3])
        resImage = cv2.resize(img[max(ymin,0):ymax, xmin:xmax], (112, 112)) #Cara detectada, reescalamos
        resImage = cv2.cvtColor(resImage, cv2.COLOR_BGR2GRAY)
        return resImage, (xmin,ymin,xmax,ymax)
    except:
        print(cntr)
        cv2.imshow('image',img)
        cv2.waitKey(0)

def extractFaces(img,detector):
    """
    Detecta todas las caras de una imagen y devuelve la más grande recortada y reescalada a 112x112
    """
    detections = detector.detect(img)
    idx_max = -1
    area_max = -1
    faces = []
    faceCoords = []
    try:
        for i,cntr in enumerate(detections):
            xmin,ymin,xmax,ymax = int(cntr[0]),int(cntr[1]),int(cntr[2]),int(cntr[3]) #Guardamos bounding box
            resImage = cv2.resize(img[max(ymin,0):ymax, xmin:xmax], (112, 112)) #Cara detectada, reescalamos
            resImage = cv2.cvtColor(resImage, cv2.COLOR_BGR2GRAY)
            faces.append(resImage)
            faceCoords.append((xmin,ymin,xmax,ymax))
        return faces, faceCoords
    except:
        print(cntr)
        cv2.imshow('image',img)
        cv2.waitKey(0)

def saveFaceCrops(videoPath,detector):
    vidcap = cv2.VideoCapture(videoPath)
    success,image = vidcap.read()
    count = 0
    faceArray = []
    facePos = []
    while success:
        resImage, (xmin,ymin,xmax,ymax) = extractBiggestFace(image,detector)
        facePos.append((xmin,ymin,xmax,ymax))
        faceArray.append(resImage) #DE MOMENTO SIEMPRE HAY CARA
        success,image = vidcap.read()
        count += 1
    return faceArray,facePos #Devuelve número de frames

def saveMultiFace(videoPath,detector,maxDistance):
    vidcap = cv2.VideoCapture(videoPath)
    success,image = vidcap.read()
    count = 0
    faceArray = defaultdict(list)
    facePos = defaultdict(list)
    frames = defaultdict(list)
    while success:
        faces, faceCoords = extractFaces(image,detector)
        if count == 0: #No hay caras aún
            for d in range(len(faces)):
                faceArray[d].append(faces[d])
                facePos[d].append(faceCoords[d])
                frames[d] = [0]
        else: #Vamos a comparar a qué cara pertenece según posición (implementar threshold por si nueva cara)
            for f in range(len(faces)):
                minDist = 2**32
                predFace = -1
                for key in facePos.keys():
                    res = np.linalg.norm(np.asarray(faceCoords[f])-np.asarray(facePos[key][-1]))
                    #print(facePos[f],facePos[key][-1])
                    #print(f"{f} = {key}",res)
                    if res < minDist:
                        predFace = key
                        minDist = res
                    if minDist > maxDistance: 
                        predFace = len(facePos.keys())
                faceArray[predFace].append(faces[f])
                facePos[predFace].append(faceCoords[f])
                frames[predFace].append(count)
                

        success,image = vidcap.read()
        count += 1
    return faceArray,facePos,frames #Devuelve número de frames

def checkVideoDuration(videoPath):
    res = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", videoPath])
    return float(res)


def splitVideo(videoPath,splitDuration,videoDuration, outputDir="splitVideos"):
    nSplits = ceil(videoDuration/splitDuration)
    splitIni = 0
    os.makedirs("./splitVideos", exist_ok=True)
    outputPath = ""
    if outputDir[0] == ".":
        outputPath = os.getcwd()+f"/{outputDir}"
    else:
        outputPath = outputDir
    for i in range(nSplits):
        subprocess.call(["ffmpeg","-y","-i",videoPath,"-ss",str(splitDuration*i), "-t",str(splitDuration),  f"{outputPath}/{i}.mp4"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
        splitIni += splitDuration


def randomNoOverlap(videoCenter, videoLen, treshold, nSideFrames):
    """
    Busca un índice donde poder meter el centro de una ventana de audio
    sin solapar con la de vídeo existente, devuelve el índice del centro
    de la ventana, o centro de padding mínimo en caso de no existir índice 
    en el que no se solapen.
    Params:
        videoCenter: Índice del array del centro de la ventana de la muestra
        videoLen: Longitud del vídeo
        Threshold: Valor entre 0 y 1, porcentaje maximo de solapamiento permitido
        nSideFrames: Número de frames pertenecientes a cada lado de la ventana
    Returns:
        index: Centro de la ventana
    """
    windowSize = nSideFrames*2+1
    fitsStart,fitsEnd = True, True
    
    overlapLeft = 1-max(abs(videoCenter-0),0)/windowSize
    overlapRight = 1-max(abs(videoCenter-videoLen),0)/windowSize
    if overlapLeft > treshold:
        fitsStart = False
    if overlapRight > treshold:
        fitsEnd = False
        
    if not fitsStart and not fitsEnd:
        minPaddingLeft = floor(videoCenter-(treshold*(windowSize)))
        minPaddingRight = ceil(videoCenter+(treshold*(windowSize)))
        return random.choice([minPaddingLeft,minPaddingRight])
    
    overlap = True
    while overlap:
        index = random.randint(0, videoLen)
        overlap = False
        if 1-max(abs(videoCenter-index),0)/windowSize > treshold:
            overlap = True
        if not overlap:
           return index*4 #Para audio se alinea cuatro frames por cada uno de video

def padVideo(video, center, nframes):
    nSideFrames = int((nframes-1)/2)
    video = torch.FloatTensor(video)
    videoFrames = video.shape[0]
    #print(videoFrames)
    ini = center-nSideFrames
    fin = center+nSideFrames+1
    if center < nSideFrames: #Necesitamos hacer padding por la izquierda
        padAmount = nSideFrames - center
        video = F.pad(video,(0,0,0,0,padAmount,0), "constant", 0) #Padding al principio
        ini = 0
        fin = nframes
    if center+nSideFrames >= videoFrames: #Necesitamos hacer padding al final
        padAmount = (nSideFrames+center) - videoFrames+1
        video = F.pad(video,(0,0,0,0,0,padAmount), "constant", 0) #Padding al final
        ini = len(video)-(nframes)
    video = video[ini:fin]
    return video # (T,96,96)

def padAudio(audio, label, center,nframes,videoFrames):
    maxAudio = videoFrames*4
    if audio.shape[0] < maxAudio: #Si es un poco más corto hacemos padding
        shortage = maxAudio - audio.shape[0]
        audio = np.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:maxAudio,:] #Se recorta

    nSideFrames = int((nframes-1)/2)
    nSideFramesAudio = nSideFrames*4
    audio = torch.FloatTensor(audio)
    audioFrames = audio.shape[0]
    #print(audioFrames)
    if label == 1: #Muestra positiva
        center = center*4
    if label == 0: #Muestra negativa
            center = random.randint(0,len(audio))
    ini = center-nSideFramesAudio
    fin = center+nSideFramesAudio+4
    if center < nSideFramesAudio: #Necesitamos hacer padding por la izquierda
        padAmount = nSideFramesAudio - center
        audio = F.pad(audio,(0,0,padAmount,0), "constant", 0) #Padding al principio
        ini = 0
        fin = nframes*4
    if center+nSideFramesAudio+4 >= audioFrames: #Necesitamos hacer padding al final
        padAmount = (nSideFramesAudio+center) - audioFrames+4
        audio = F.pad(audio,(0,0,0,padAmount), "constant", 0) #Padding al final
        ini = len(audio)-(nframes*4)
        
    audio = audio[ini:fin]

    return audio # (T,96,96)