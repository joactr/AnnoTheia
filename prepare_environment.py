import os

if __name__ == "__main__":

    # -- installing Git LFS
    os.system("conda install conda-forge::git-lfs -y")
    os.system("git lfs install")

    # -- installing submodules
    os.system("git submodule init")
    os.system("git submodule update")

    # -- -- installing the iBug face detection repo

    os.system("cd ./modules/face_detection/ibug_facedetector/")
    os.system("git lfs pull")
    os.system("pip install -e .")
    os.system("cd ../../..")

    # -- -- installing the iBug face alignment repo

    os.system("cd ./modules/face_alignment/ibug_facealigner/")
    os.system("pip install -e .")
    os.system("cd ../../..")

    # -- installing the rest of needed packages
    os.system("pip install -r requirements.txt")
