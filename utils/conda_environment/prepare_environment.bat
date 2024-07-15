@echo off
REM -- installing Git LFS
call conda install conda-forge::git-lfs -y
git lfs install

REM -- installing submodules
git submodule init
git submodule update

REM -- -- installing the iBug face detection repo
cd .\modules\face_detection\ibug_facedetector\
git lfs pull
pip install -e .
cd ..\..\..

REM -- -- installing the iBug face alignment repo
cd .\modules\face_alignment\ibug_facealigner\
pip install -e .
cd ..\..\..

REM -- installing package to play audios in Windows 
conda install conda-forge::mpg123 -y

REM -- installing ffmpeg package
call conda install -c conda-forge ffmpeg -y

REM -- installing the rest of needed packages
pip install -r requirements.txt