import os
import vlc
import cv2
import pickle
import platform
import argparse
import subprocess
import pandas as pd
import tkinter as tk
import customtkinter

# Modes: "System" (standard), "Dark", "Light"
customtkinter.set_appearance_mode("System")
# Themes: "blue" (standard), "green", "dark-blue"
customtkinter.set_default_color_theme("blue")

class Screen(tk.Frame):
    '''
    Screen widget: Embedded video player from local or youtube
    '''

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, bg='black')
        self.parent = parent
        # Creating VLC player
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
        self.media = None
        self.os_platform = platform.system()

    def GetHandle(self):
        # Getting frame ID
        return self.winfo_id()

    def _play_depending_on_platform(self):
        if self.os_platform == "Windows":
            self.player.set_hwnd(self.winfo_id())
        elif self.os_platform == "Darwin":
            ns = _GetNSView(self.winfo_id())
            if ns:
                self.player.set_nsobject(ns)
            else:
                self.player.set_xwindow(self.winfo_id())
        else:
            self.player.set_xwindow(self.winfo_id())

    def play(self, _source):
        # Function to start player from given source
        self.media = self.instance.media_new(_source)
        self.media.get_mrl()
        self.player.set_media(self.media)

        self._play_depending_on_platform()

        self.player.play()

    def stop(self):
        self.player.stop()

    def playpause(self):
        self.player.pause()

        if self.player.get_length() - 350 < self.player.get_time():
            self.player.set_media(self.media)
            self._play_depending_on_platform()
            self.player.play()

    def forward(self, seconds):
        # Maxes out at the video length
        newTime = min(self.player.get_time() + seconds *
                      1000, self.media.get_duration())
        self.player.set_time(newTime)

    def backward(self, _source, seconds):
        # If the video has ended a reset is needed, uses one second of margin
        if self.player.get_time()+1000 >= self.media.get_duration():
            # Reset the video
            newTime = max(self.media.get_duration() - seconds*1000, 0)
            self.media = self.instance.media_new(_source)
            self.media.get_mrl()
            self.player.set_media(self.media)

            self._play_depending_on_platform()

            self.player.play()
            self.player.set_time(newTime)

        else:
            # Can't go back past the start
            newTime = max(self.player.get_time() - seconds*1000, 0)
            self.player.set_time(newTime)

class Loader():
    def __init__(self, scenes_info_path):
        self.df = pd.read_csv(scenes_info_path)
        self.index = 0
        self.temp_dir = "./gui_temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.createVideo()

    def createVideo(self):
        row = self.df.iloc[self.index]
        speakerN = row["speaker"]

        with open(row["pickle_path"], 'rb') as f:  # open a text file
            loaded = pickle.load(f)  # serialize the list
        facePos = loaded["face_boundings"][speakerN]

        # Cut appropiate segment for sample and convert to 25fps
        segment_path = os.path.join(self.temp_dir, f'{self.index}_{row["ini"]}_{row["end"]}.mp4')

        subprocess.call([
            "ffmpeg",
            "-y",
            "-i",
            str(row["video"]),
            "-ss",
            str(row["ini"]),
            "-to",
            str(row["end"]),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-r",
            "25",
            segment_path,
            "-loglevel",
            "quiet",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        cap = cv2.VideoCapture(segment_path)
        videoImages = []
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        iniFrame = int((row["ini"]-row["scene_start"])*25)

        frameN = iniFrame
        finalFrame = int(row["end"]*25)

        while frameN < finalFrame:
            try:
                ret, image = cap.read()
                color = (0, 255, 0)
                xmin, ymin, xmax, ymax = facePos[frameN]
                videoImages.append(cv2.rectangle(
                    image, (xmin, ymin), (xmax, ymax), color, 1))
                frameN += 1
                if ret == False:
                    break
            except Exception as e:
                print("ERROR--------------", frameN, len(facePos))
                break

        self.saveTrimmedVideo(videoImages, width, height, segment_path)

    def saveTrimmedVideo(self, imgFrames, width, height, segment_path):
        video = cv2.VideoWriter(
            f"{self.temp_dir}/temp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
        for image in imgFrames:
            video.write(image)
        cv2.destroyAllWindows()
        video.release()

        subprocess.call([
            "ffmpeg",
            "-y",
            "-i",
            f"{self.temp_dir}/temp.mp4",
            "-i",
            segment_path,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "./temp2.mp4",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class App(customtkinter.CTk):
    def __init__(self, scenes_info_path):
        super().__init__()
        self.loader = Loader(scenes_info_path)

        # configure window
        self.title("AnnoTheia - Processing scene " + str(self.loader.index) + " of " + str(len(self.loader.df)))
        self.geometry(f"{1200}x{720}")
        # configure grid layout (4x4)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure((1, 2, 3, 4), weight=1)
        self.grid_columnconfigure(5, weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Create videoFrame holder
        self.videoHolder = tk.Frame(self)
        self.videoHolder.grid(row=0, column=0, padx=(20, 40), pady=(
            10, 10), sticky="nsew", rowspan=4, columnspan=5)
        # Init vlc player
        self.player = Screen(self.videoHolder)
        self.player.place(relx=0.0005, rely=0, relwidth=0.999, relheight=1)
        self.player.play('temp2.mp4')
        # Prev and next buttons
        self.prev_button = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=(
            "gray10", "#DCE4EE"), text="Prev", command=self.prevSample)
        self.prev_button.grid(row=7, column=2, padx=(
            20, 20), pady=(20, 20), sticky="e")
        self.next_button = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=(
            "gray10", "#DCE4EE"), text="Next", command=self.nextSample)
        self.next_button.grid(row=7, column=3, padx=(
            20, 20), pady=(20, 20), sticky="w")
        # Create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=4, column=0, padx=(20, 20),
                          pady=(20, 20), columnspan=5, sticky="nsew")
        # Accept and incorrect buttons
        self.accept_button = customtkinter.CTkButton(master=self, fg_color="#adedbe", border_width=2, text_color=(
            "gray10", "#DCE4EE"), text="Accept", command=self.saveSample)
        self.accept_button.grid(row=0, column=5, padx=(20, 20))
        self.incorrect_button = customtkinter.CTkButton(master=self, fg_color="#edadad", border_width=2, text_color=(
            "gray10", "#DCE4EE"), text="Incorrect", command=self.deleteSample)
        self.incorrect_button.grid(row=1, column=5, padx=(20, 20))

        self.label = customtkinter.CTkLabel(
            master=self, text="F1 - Play/Pause \nF2 - Rewind 5s \nF3 - Forward 5s", justify="left")
        self.label.grid(row=4, column=5, padx=(
            20, 20), pady=(20, 0), sticky="w")
        # Add value to textbox
        self.textbox.insert(
            "0.0", self.loader.df.iloc[self.loader.index]["transcription"])

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def prevSample(self):
        if self.loader.index > 0:
            self.loader.index -= 1
            self.playVideo()

    def nextSample(self):
        if self.loader.index < len(self.loader.df)-1:
            self.loader.index += 1
            self.playVideo()

    def deleteSample(self):
        # Delete sample from dataframe
        self.loader.df = self.loader.df.drop(labels=self.loader.index, axis=0)
        # If there are still videos in dataframe, play the next one
        if self.loader.index < len(self.loader.df):
            self.playVideo()

    def saveSample(self):
        # REHACER METODO
        pass

    def playVideo(self):
        self.player.stop()
        self.textbox.delete("0.0", "end")
        self.loader.createVideo()
        self.textbox.insert(
            "0.0", self.loader.df.iloc[self.loader.index]["transcription"])
        self.player.play('temp2.mp4')
        app.title("AnnoTheia - PROCESSING SCENE " + str(self.loader.index) + " OF " + str(len(self.loader.df)))

    def fun(self, event):
        if event.keysym == 'F1':
            self.player.playpause()
        if event.keysym == 'F2':
            self.player.backward('temp2.mp4', 1)
        if event.keysym == 'F3':
            self.player.forward(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for supervising and annotating the candidate scenes provided by the AnnoTheia's Pipeline",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--scenes-info-path", required=True, type=str, help="Path to a CSV file where we can find the information w.r.t. the candidate scenes of a specific video.")

    args = parser.parse_args()

    app = App(args.scenes_info_path)
    app.bind("<KeyPress>", app.fun)
    app.mainloop()

    os.system("rm ./temp2.mp4")
    os.system(f"rm -rf {app.loader.temp_dir}")
