#!/bin/bash

# -- installing Git LFS
conda install conda-forge::git-lfs -y
git lfs install

# -- installing submodules
git submodule init
git submodule update

# -- -- installing the iBug face detection repo
cd ./modules/face_detection/ibug_facedetector/
git lfs pull
pip install -e .
cd ../../..

# -- -- installing the iBug face alignment repo
cd ./modules/face_alignment/ibug_facealigner/
pip install -e .
cd ../../..

# -- installing package to play audios in linux
conda install conda-forge::mpg123

# -- installing the rest of needed packages
pip install -r requirements.txt

