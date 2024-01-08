import os
import vlc
import cv2
import shutil
import pickle
import platform
import argparse
import subprocess
import pandas as pd

import customtkinter
import tkinter as tk
from CTkMessagebox import CTkMessagebox

from ibug.face_alignment.utils import plot_landmarks

# -- modes: "System" (standard), "Dark", "Light"
customtkinter.set_appearance_mode("System")
# -- themes: "blue" (standard), "green", "dark-blue"
customtkinter.set_default_color_theme("blue")

class Screen(tk.Frame):
    """ Screen widget: Embedded video player from local or youtube
    """

    def __init__(self, parent, *args, **kwargs):

        self.media = None
        self.parent = parent
        self.os_platform = platform.system()
        tk.Frame.__init__(self, parent, bg='black')

        # -- creating VLC player
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()

    def get_handle(self):
        return self.winfo_id()

    def play(self, _source):
        # -- function to start player from given source
        self.media = self.instance.media_new(_source)
        self.media.get_mrl()
        self.player.set_media(self.media)

        # -- play video depending on the operating system
        self._play_depending_on_platform()
        self.player.play()

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

    def stop(self):
        self.player.stop()

    def play_pause(self):
        self.player.pause()

        if self.player.get_length() - 350 < self.player.get_time():
            # -- play video depending on the operating system
            self.player.set_media(self.media)
            self._play_depending_on_platform()
            self.player.play()

    def forward(self, seconds):
        # -- maxes out at the video length
        newTime = min(self.player.get_time() + seconds * 1000, self.media.get_duration())
        self.player.set_time(newTime)

    def backward(self, _source, seconds):
        # -- if the video has ended, a reset is needed using one second of margin
        if self.player.get_time()+1000 >= self.media.get_duration():
            # -- reset the video
            newTime = max(self.media.get_duration() - seconds*1000, 0)
            self.media = self.instance.media_new(_source)
            self.media.get_mrl()
            self.player.set_media(self.media)

            # -- play video depending on the operating system
            self._play_depending_on_platform()
            self.player.play()
            self.player.set_time(newTime)

        else:
            # -- can't go back past the start
            newTime = max(self.player.get_time() - seconds*1000, 0)
            self.player.set_time(newTime)

class Loader():
    def __init__(self, scenes_info_path, temp_dir, final_video_clip_path):

        self.index = 0
        self.temp_dir = temp_dir
        self.final_video_clip_path = final_video_clip_path

        # -- creating temporary directory
        self.df = pd.read_csv(scenes_info_path)
        os.makedirs(self.temp_dir, exist_ok=True)

        # -- displaying the video clip
        self.create_video()

    def create_video(self):
        row = self.df.iloc[self.index]

        speaker_id = row["speaker"]
        # -- loading pickle containing useful information from the pipeline
        with open(row["pickle_path"], 'rb') as f:
            loaded = pickle.load(f)

        # -- getting face bounding boxes + face landmarks
        face_boundings = loaded["face_boundings"][speaker_id]
        face_landmarks = loaded["face_landmarks"][speaker_id]

        # -- trimming appropiate segment of the sample and converting it to 25fps
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

        # -- reading the trimmed segment
        cap = cv2.VideoCapture(segment_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ini_frame = int( (row["ini"] - row["scene_start"]) * 25)

        video_frames = []
        n_frame = ini_frame
        bb_color = (0, 255, 0)
        final_frame = int( row["end"] * 25 )

        # -- reading frame by frame
        while n_frame < final_frame:
            ret, image = cap.read()

            # -- drawing face bounding box
            left, top, right, bottom = face_boundings[n_frame]
            frame_to_video = cv2.rectangle(image, (left, top), (right, bottom), bb_color, 1)

            # -- drawing face landmarks
            if len(face_landmarks) > 0:
                landmarks = face_landmarks[n_frame]
                plot_landmarks(frame_to_video, landmarks)

            # -- gathering frames to create the video clip
            video_frames.append(frame_to_video)

            # -- updating frame counter
            n_frame += 1

            # -- sanity checking
            if ret == False:
                break

        self.save_trimmed_video(video_frames, frame_width, frame_height, segment_path)

    def save_trimmed_video(self, video_frames, frame_width, frame_height, segment_path):
        # -- creating temporary video clip from the segment
        video = cv2.VideoWriter(f"{self.temp_dir}/temp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width, frame_height))
        # -- gathering all the video frame after drawings the bounding box and face landmarks
        for image in video_frames:
            video.write(image)
        cv2.destroyAllWindows()
        video.release()

        # -- adding sound to the silent video clip just created
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
            self.final_video_clip_path,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class App(customtkinter.CTk):
    def __init__(self, scenes_info_path, output_file_path):
        super().__init__()

        self.video_id = scenes_info_path.split(os.sep)[-2]

        ## -- defining different settings
        self.temp_dir = "./temp_gui"
        self.final_video_clip_path = os.path.join(self.temp_dir, "temp2.mp4")
        self.loader = Loader(scenes_info_path, temp_dir=self.temp_dir, final_video_clip_path=self.final_video_clip_path)
        self.output_file_path = output_file_path

        # -- configuring window
        self.geometry(f"{1200}x{720}")
        self.title(f"AnnoTheia - Processing scene {self.loader.index+1} of {len(self.loader.df)}")

        # -- configuring grid layout (4x4)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure((1, 2, 3, 4), weight=1)
        self.grid_columnconfigure(5, weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # -- creating video frame holder
        self.video_holder = tk.Frame(self)
        self.video_holder.grid(row=0, column=0, padx=(20, 40), pady=(10, 10), sticky="nsew", rowspan=4, columnspan=5)

        # -- initiating VLC player
        self.player = Screen(self.video_holder)
        self.player.place(relx=0.0005, rely=0, relwidth=0.999, relheight=1)
        self.player.play(self.final_video_clip_path)

        # -- defining 'prev' and 'next' buttons
        self.prev_button = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), text="Prev", command=self.prev_sample)
        self.prev_button.grid(row=7, column=2, padx=(20, 20), pady=(20, 20), sticky="e")

        self.next_button = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), text="Next", command=self.next_sample)
        self.next_button.grid(row=7, column=3, padx=(20, 20), pady=(20, 20), sticky="w")

        # -- creating textbox
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=4, column=0, padx=(20, 20), pady=(20, 20), columnspan=5, sticky="nsew")

        # -- defining 'accept' and 'incorrect' buttons
        self.accept_button = customtkinter.CTkButton(master=self, fg_color="#adedbe", border_width=2, text_color=("gray10", "#DCE4EE"), text="Accept", command=self.save_sample)
        self.accept_button.grid(row=0, column=5, padx=(20, 20))

        self.incorrect_button = customtkinter.CTkButton(master=self, fg_color="#edadad", border_width=2, text_color=("gray10", "#DCE4EE"), text="Incorrect", command=self.delete_sample)
        self.incorrect_button.grid(row=1, column=5, padx=(20, 20))

        # -- drawing helping legend
        self.legend = customtkinter.CTkLabel(master=self, text="F1 - Play/Pause \nF2 - Rewind 5s \nF3 - Forward 5s", justify="left")
        self.legend.grid(row=4, column=5, padx=(20, 20), pady=(20, 0), sticky="w")

        # -- adding transcription to the textbox
        self.textbox.insert("0.0", self.loader.df.iloc[self.loader.index]["transcription"])

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def prev_sample(self):
        self.loader.index = min(max(0, self.loader.index - 1), len(self.loader.df) - 1)
        self.play_video()

    def next_sample(self):
        # -- if the user reached the last candidate sample
        if (self.loader.index + 1) >= len(self.loader.df):
            self.loader.index = len(self.loader.df) - 1
            CTkMessagebox(title=f"Congratulations!!", message=f"Video {self.video_id} has been annotated :) Please, close the GUI unless you want to check your decisions",)()
        else:
            self.loader.index += 1
            self.play_video()

    def save_sample(self):
        # -- if file exists, append to it. If not, create new file
        if os.path.exists(self.output_file_path):
            # -- getting accepted sample
            accepted_sample = pd.DataFrame([self.loader.df.iloc[self.loader.index]])
            # -- just in case, updating the supervised transcription
            accepted_sample["transcription"] = self.textbox.get("0.0", "end").strip()

            # -- updating annotated and supervised samples
            annotated_df = pd.read_csv(self.output_file_path)
            annotated_df = pd.concat([annotated_df, accepted_sample], ignore_index=True).drop_duplicates()
            annotated_df.to_csv(self.output_file_path, index=False)

            # -- updating dataframe into memory just in case the user come back to discard the sample :S
            self.loader.df.iloc[self.loader.index]["transcription"] = self.textbox.get("0.0", "end").strip()

        else:
            # -- getting accepted sample
            first_accepted_sample = pd.DataFrame([self.loader.df.iloc[self.loader.index]])
            # -- just in case, updating the supervised transcription
            first_accepted_sample["transcription"] = self.textbox.get("0.0", "end").strip()
            # -- creating annotated dataframe
            first_accepted_sample.to_csv(self.output_file_path, index=False)

        # -- get the next candidate scene
        self.next_sample()

    def delete_sample(self):
        # -- perhaps this sample was previously accepted, so it has to be removed from the annotated dataframe
        annotated_df = pd.read_csv(self.output_file_path)
        sample_to_remove = self.loader.df.iloc[self.loader.index]
        annotated_df = annotated_df.drop(index = annotated_df[
            (annotated_df["video"] == sample_to_remove["video"])
            & (annotated_df["scene_start"] == sample_to_remove["scene_start"])
            & (annotated_df["ini"] == sample_to_remove["ini"])
            & (annotated_df["end"] == sample_to_remove["end"])
            & (annotated_df["speaker"] == sample_to_remove["speaker"])
            & (annotated_df["pickle_path"] == sample_to_remove["pickle_path"])
            & (annotated_df["transcription"] == sample_to_remove["transcription"])
        ].index)
        annotated_df.to_csv(self.output_file_path, index=False)

        # -- removing the sample from the dataframe into memory
        self.loader.df = self.loader.df.drop(labels=self.loader.index, axis=0)
        self.loader.index -= 1
        self.play_video()

    def play_video(self):
        # -- updating screen displaying a new sample
        self.player.stop()

        self.textbox.delete("0.0", "end")
        self.loader.create_video()
        self.textbox.insert("0.0", self.loader.df.iloc[self.loader.index]["transcription"])
        self.player.play(self.final_video_clip_path)

        app.title(f"AnnoTheia - Processing scene {self.loader.index+1} of {len(self.loader.df)}")

    def keyword_func(self, event):
        # -- keyword shortages functionaly
        if event.keysym == 'F1':
            self.player.play_pause()
        if event.keysym == 'F2':
            self.player.backward(self.final_video_clip_path, 1)
        if event.keysym == 'F3':
            self.player.forward(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for supervising and annotating the candidate scenes provided by the AnnoTheia's Pipeline",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--scenes-info-path", required=True, type=str,
                        help="Path to a CSV file where we can find the information w.r.t. the candidate scenes of a specific video.")

    args = parser.parse_args()

    # -- creating an annotated copy of the scene's info CSV
    extension_index = args.scenes_info_path.rfind('.csv')
    output_csv = args.scenes_info_path[:extension_index] + '_annotated.csv'

    # -- starting the user interface
    app = App(args.scenes_info_path, output_csv)
    app.bind("<KeyPress>", app.keyword_func)
    app.mainloop()

    # -- removing temporary files
    shutil.rmtree(app.loader.temp_dir)
