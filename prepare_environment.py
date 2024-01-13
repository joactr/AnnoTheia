import os
import platform

if __name__ == "__main__":

    if platform.system() == "Windows":
        os.system("utils\\conda_environment\\prepare_environment.bat")
    else:
        os.system("bash ./utils/conda_environment/prepare_environment.bash")

