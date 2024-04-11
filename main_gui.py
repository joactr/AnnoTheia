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

from utils.gui import play_sound_threaded

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
    def __init__(self, scenes_info_path, annotated_output_path, temp_dir, final_video_clip_path):

        self.index = 0
        self.temp_dir = temp_dir
        self.scenes_info_path = scenes_info_path
        self.annotated_output_path = annotated_output_path
        self.final_video_clip_path = final_video_clip_path

        # -- reading candidate scenes
        self.df = pd.read_csv(self.scenes_info_path)

        # -- creating temporary directory
        os.makedirs(self.temp_dir, exist_ok=True)

        # -- creating the annotated version
        if not os.path.exists(self.annotated_output_path):
            self.annotated_df = pd.DataFrame([], columns=["video", "scene_start", "sample_start", "sample_end", "duration", "speaker", "pickle_path", "transcription"])
        else:
            self.annotated_df = pd.read_csv(self.annotated_output_path)
            self.annotated_df = self.annotated_df.loc[:, ~self.annotated_df.columns.str.contains('^Unnamed')]

        # -- displaying the video clip
        self.create_video()

    def create_video(self):
        row = self.df.iloc[self.index]

        speaker_id = row["speaker"]
        # -- loading pickle containing useful information from the pipeline
        with open(row["pickle_path"], 'rb') as f:
            loaded = pickle.load(f)

        # -- getting face bounding boxes + face landmarks
        asd_scores = [0.8] * 100000
        face_boundings = loaded["face_boundings"][speaker_id]
        face_landmarks = loaded["face_landmarks"][speaker_id]

        # -- trimming appropiate segment of the sample and converting it to 25fps
        segment_path = os.path.join(self.temp_dir, f'{self.index}_{row["scene_start"]}_{row["sample_end"]}.mp4')

        subprocess.call([
            "ffmpeg",
            "-y",
            "-ss",
            str(row["scene_start"]),
            "-to",
            str(row["sample_end"]),
            "-i",
            str(row["video"]),
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

        video_frames = []
        bb_color = (0, 255, 0)
        conf_color = (139, 0, 0)
        start_frame = int( (row["sample_start"] - row["scene_start"]) * 25 )
        final_frame = int( (row["sample_end"] - row["scene_start"]) * 25 )

        # -- reading frame by frame
        n_frame = start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while(n_frame <= final_frame):
            ret, image = cap.read()

            if ret == True:
                # -- drawing face bounding box
                left, top, right, bottom = face_boundings[n_frame]
                frame_to_video = cv2.rectangle(image, (left, top), (right, bottom), bb_color, 1)

                # -- displaying frame-level confidence
                if "asd_scores" in loaded.keys():
                    cv2.putText(image, "{:.2f}".format(loaded["asd_scores"][n_frame]*100), (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, conf_color, 1)

                # -- drawing face landmarks
                if len(face_landmarks) > 0:
                    landmarks = face_landmarks[n_frame]
                    plot_landmarks(frame_to_video, landmarks)

                # -- gathering frames to create the video clip
                video_frames.append(frame_to_video)

                # -- updating frame counter
                n_frame += 1

            else:
                break

        # -- extracting adequately sampled audio stream
        subprocess.call([
            "ffmpeg",
            "-y",
            "-ss",
            str(start_frame/25),
            "-t",
            str(row["duration"]),
            "-i",
            segment_path,
            "-q:a",
            "0",
            "-map",
            "a",
            segment_path.replace(".mp4", ".aac"),
            "-loglevel",
            "quiet",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


        self.save_trimmed_video(video_frames, frame_width, frame_height, segment_path.replace(".mp4", ".aac"))

    def save_trimmed_video(self, video_frames, frame_width, frame_height, audio_segment_path):
        temp_path = f"{self.temp_dir}/temp.mp4"

        # -- remove temporary video clips just in case
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # -- creating temporary video clip from the segment
        video = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width, frame_height))
        # -- gathering all the video frame after drawings the bounding box and face landmarks
        for image in video_frames:
            video.write(image)
        cv2.destroyAllWindows()
        video.release()

        # -- remove temporary video clips just in case
        if os.path.exists(self.final_video_clip_path):
            os.remove(self.final_video_clip_path)

        # -- adding sound to the silent video clip just created
        subprocess.call([
            "ffmpeg",
            "-y",
            "-i",
            temp_path,
            "-i",
            audio_segment_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            self.final_video_clip_path,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class App(customtkinter.CTk):
    def __init__(self, scenes_info_path, output_file_path, max_history_len):
        super().__init__()

        # -- setting up
        self.os_platform = platform.system()
        self.video_id = scenes_info_path.split(os.sep)[-2]
        self.output_file_path = output_file_path
        self.scenes_info_path = scenes_info_path

        # -- history management
        self.history = []
        self.max_history_len = max_history_len

        ## -- defining different settings
        self.temp_dir = "./temp_gui2"
        self.final_video_clip_path = os.path.join(self.temp_dir, "temp2.mp4")
        self.loader = Loader(scenes_info_path, annotated_output_path=self.output_file_path, temp_dir=self.temp_dir, final_video_clip_path=self.final_video_clip_path)

        # -- configuring window
        self.geometry(f"{1200}x{720}")
        self.title(f"AnnoTheia - Processing scene {self.loader.index+1} of {len(self.loader.df)} from {self.scenes_info_path}")

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
        self.jump_button = customtkinter.CTkButton(master=self, fg_color="#f7a80a", border_width=2, text_color=("gray10", "#DCE4EE"), text="Jump to:", command=self.jump_to)
        self.jump_button.grid(row=7, column=0, padx=(20, 0), pady=(20, 20), sticky="w")

        self.jumpto_textbox = customtkinter.CTkTextbox(self, width=100, height=25)
        self.jumpto_textbox.grid(row=7, column=1, padx=(0, 0), pady=(20, 20), columnspan=1, sticky="w")

        self.prev_button = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), text="Prev", command=self.prev_sample)
        self.prev_button.grid(row=7, column=2, padx=(0, 20), pady=(20, 20), sticky="e")

        self.next_button = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), text="Next", command=self.next_sample)
        self.next_button.grid(row=7, column=3, padx=(20, 20), pady=(20, 20), sticky="w")

        self.save_button = customtkinter.CTkButton(master=self, fg_color="#adedbe", border_width=2, text_color=("gray10", "#DCE4EE"), text="Save", command=self.save_df)
        self.save_button.grid(row=7, column=5, padx=(20, 20), pady=(20, 20), sticky="w")

        # -- creating textbox
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=4, column=0, padx=(20, 20), pady=(20, 20), columnspan=5, sticky="nsew")

        # -- defining 'accept' and 'incorrect' buttons
        self.accept_button = customtkinter.CTkButton(master=self, fg_color="#adedbe", border_width=2, text_color=("gray10", "#DCE4EE"), text="Accept", command=self.save_sample)
        self.accept_button.grid(row=0, column=5, padx=(20, 20))

        self.incorrect_button = customtkinter.CTkButton(master=self, fg_color="#edadad", border_width=2, text_color=("gray10", "#DCE4EE"), text="Incorrect", command=self.delete_sample)
        self.incorrect_button.grid(row=1, column=5, padx=(20, 20))

        # -- defining 'undo' buttons
        self.undo_button = customtkinter.CTkButton(master=self, fg_color="#5ea6ff", border_width=2, text_color=("gray10", "#DCE4EE"), text="Oops! Undo", command=self.undo)
        self.undo_button.grid(row=2, column=5, padx=(20, 20))
        self.undo_button._state = tk.DISABLED

        # -- drawing helping legend
        self.legend = customtkinter.CTkLabel(master=self, text="F1 - Play/Pause \nF2 - Rewind 5s \nF3 - Forward 5s", justify="left")
        self.legend.grid(row=4, column=5, padx=(20, 20), pady=(20, 0), sticky="w")

        # -- adding transcription to the textbox
        self.textbox.insert("0.0", self.loader.df.iloc[self.loader.index]["transcription"])

    def _play_sound_depending_on_platform(self, sound_filename):
        sound_path = os.path.join("doc", "sounds", f"{sound_filename}.mp3")
        if self.os_platform == "Windows":
            play_sound_threaded(sound)
        else:
            os.system(f"mpg123 -q {sound_path}")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def jump_to(self):
        # -- minus one because the index in dataframes starts with zero
        text_from_jumpto = self.jumpto_textbox.get("0.0", "end").strip()
        if len(text_from_jumpto) > 0 and text_from_jumpto.isdigit():
            user_idx = int(text_from_jumpto) - 1
            self.loader.index = max(0, min(user_idx, len(self.loader.df)-1))
            self.play_video()

    def prev_sample(self):
        self.loader.index = max(0, self.loader.index - 1)
        self.play_video()

    def next_sample(self):
        # -- if the user reached the last candidate sample
        if (self.loader.index + 1) >= len(self.loader.df):
            self.loader.index = len(self.loader.df) - 1
            CTkMessagebox(title=f"Congratulations!!", message=f"Video {self.video_id} has been annotated :) Please, close the GUI unless you want to check your decisions",)()
        else:
            self.loader.index += 1
            self.play_video()

    def save_df(self):
        self.loader.df = self.loader.df.loc[:, ~self.loader.df.columns.str.contains('^Unnamed')]
        self.loader.df.to_csv(self.loader.scenes_info_path.replace(".csv", "_saved.csv"))
        CTkMessagebox(title=f"Success!!", message=f"Your annotation progress has been saved in {self.loader.scenes_info_path.replace('.csv', '_saved.csv')}!",)()

    def save_sample(self):
        # -- acoustic user feedback
        self._play_sound_depending_on_platform("correct")

        # -- add a new supervised sample to the annotated dataframe
        accepted_sample = self.loader.df.iloc[self.loader.index]
        accepted_sample = accepted_sample.drop(labels=["scene_path"]) # -- we do not need it

        # -- updating transcription with the probable corrected one by the user
        old_transcription = accepted_sample["transcription"]
        accepted_sample["transcription"] = self.textbox.get("0.0", "end").strip()

        # -- appending the new accepted sample to the annotated dataframe
        self.loader.annotated_df = pd.concat([
            self.loader.annotated_df,
            accepted_sample.to_frame().T,
        ], ignore_index=True)

        # -- updating the stored CSV file
        self.loader.annotated_df = self.loader.annotated_df.loc[:, ~self.loader.annotated_df.columns.str.contains('^Unnamed')]
        self.loader.annotated_df.to_csv(self.output_file_path, index=False)

        # -- updating non-annotated dataframe into memory just in case the user come back to discard the sample :S
        self.loader.df.at[self.loader.index, "transcription"] = self.textbox.get("0.0", "end").strip()

        # -- updating the history to allow the user to come back to a previous stage
        self.history.append( ("accepted_sample", accepted_sample, old_transcription, self.loader.index, None, None) )

        # -- in case the button was disabled
        self.undo_button._state = tk.NORMAL

        # -- controlling its length just in case memory issues
        if len(self.history) > self.max_history_len:
            self.history.pop(0)

        # -- get the next candidate scene
        self.next_sample()

    def delete_sample(self):
        self._play_sound_depending_on_platform("incorrect")

        # -- perhaps this sample was previously accepted, so it has to be removed from the annotated dataframe
        sample_to_remove = self.loader.df.iloc[self.loader.index]

        previous_annotated_len = len(self.loader.annotated_df)
        annotated_remove_idx = self.loader.annotated_df[
            (self.loader.annotated_df["video"] == sample_to_remove["video"])
            & (self.loader.annotated_df["scene_start"] == sample_to_remove["scene_start"])
            & (self.loader.annotated_df["sample_start"] == sample_to_remove["sample_start"])
            & (self.loader.annotated_df["sample_end"] == sample_to_remove["sample_end"])
            & (self.loader.annotated_df["duration"] == sample_to_remove["duration"])
            & (self.loader.annotated_df["speaker"] == sample_to_remove["speaker"])
            & (self.loader.annotated_df["pickle_path"] == sample_to_remove["pickle_path"])
            & (self.loader.annotated_df["transcription"] == sample_to_remove["transcription"])].index

        self.loader.annotated_df = self.loader.annotated_df.drop(index=annotated_remove_idx)
        self.loader.annotated_df = self.loader.annotated_df.reset_index(drop=True)

        # -- necessary to recover a previous stage because of the 'undo' button
        was_removed_from_annotated_df = len(self.loader.annotated_df) != previous_annotated_len

        # -- updating the stored CSV file
        self.loader.annotated_df = self.loader.annotated_df.loc[:, ~self.loader.annotated_df.columns.str.contains('^Unnamed')]
        self.loader.annotated_df.to_csv(self.output_file_path, index=False)

        # -- removing the sample from the original dataframe into memory
        self.loader.df = self.loader.df.drop(labels=self.loader.index, axis=0)
        self.loader.df = self.loader.df.reset_index(drop=True)

        # -- updating the history to allow the user to come back to a previous stage
        self.history.append( ("deleted_sample", sample_to_remove, None, self.loader.index, was_removed_from_annotated_df, annotated_remove_idx) )

        # -- in case the button was disabled
        self.undo_button._state = tk.NORMAL

        # -- controlling its length just in case memory issues
        if len(self.history) > self.max_history_len:
            self.history.pop(0)

        # -- it plays the next video without need for increasing the index
        self.loader.index = min(self.loader.index, len(self.loader.df))
        if len(self.loader.df) > 0:
            self.play_video()
        else:
            CTkMessagebox(title=f"Congratulations!!", message=f"Video {self.video_id} has been annotated :) Please, close the GUI.",)()

    def undo(self):
        if len(self.history) > 0:
            # -- taking the last user decision
            decision_type, undo_sample, old_transcription, undo_loader_idx, was_removed_from_annotated_df, annotated_remove_idx = self.history.pop(-1)

            # -- disable button if it the case
            if len(self.history) == 0:
                self.undo_button._state = tk.DISABLED

            if decision_type == "accepted_sample":
                # -- for the original dataframe into memory, it is just getting the old transcription
                self.loader.df.at[undo_loader_idx, "transcription"] = old_transcription

                # -- for the annotated dataframe, we have to remove the sample that we add
                self.loader.annotated_df = self.loader.annotated_df.drop(index = self.loader.annotated_df[
                        (self.loader.annotated_df["video"] == undo_sample["video"])
                        & (self.loader.annotated_df["scene_start"] == undo_sample["scene_start"])
                        & (self.loader.annotated_df["sample_start"] == undo_sample["sample_start"])
                        & (self.loader.annotated_df["sample_end"] == undo_sample["sample_end"])
                        & (self.loader.annotated_df["duration"] == undo_sample["duration"])
                        & (self.loader.annotated_df["speaker"] == undo_sample["speaker"])
                        & (self.loader.annotated_df["pickle_path"] == undo_sample["pickle_path"])
                        & (self.loader.annotated_df["transcription"] == undo_sample["transcription"])
                ].index)
                self.loader.annotated_df = self.loader.annotated_df.reset_index(drop=True)

                # -- and update the stored CSV file
                self.loader.annotated_df = self.loader.annotated_df.loc[:, ~self.loader.annotated_df.columns.str.contains('^Unnamed')]
                self.loader.annotated_df.to_csv(self.output_file_path, index=False)

            elif decision_type == "deleted_sample":
                # -- for the original dataframe into memory, we have to add the sample that was removed
                if undo_loader_idx > 0:
                    self.loader.df = pd.concat([
                        self.loader.df.loc[0:(undo_loader_idx-1)],
                        undo_sample.to_frame().T,
                        self.loader.df.loc[undo_loader_idx:],
                    ], ignore_index=True)
                else:
                    self.loader.df = pd.concat([
                        undo_sample.to_frame().T,
                        self.loader.df,
                    ], ignore_index=True)

                # -- for the annotated dataframe, in case the sample was previosly accepted and consequently removed, we have to add it
                if was_removed_from_annotated_df:
                    annotated_remove_idx = annotated_remove_idx.values[0]
                    if annotated_remove_idx > 0:
                        self.loader.annotated_df = pd.concat([
                            self.loader.annotated_df.loc[0:(annotated_remove_idx-1)],
                            undo_sample.to_frame().T,
                            self.loader.annotated_df.loc[annotated_remove_idx:],
                        ], ignore_index=True)
                    else:
                        self.loader.annotated_df = pd.concat([
                            undo_sample.to_frame().T,
                            self.loader.annotated_df,
                        ])

                    # -- and update the stored CSV file
                    self.loader.annotated_df = self.loader.annotated_df.loc[:, ~self.loader.annotated_df.columns.str.contains('^Unnamed')]
                    self.loader.annotated_df.to_csv(self.output_file_path, index=False)

            # -- in both cases
            # -- we have to update the loader dataframe index
            self.loader.index = undo_loader_idx

            # -- and play a video clip
            self.play_video()

    def play_video(self):
        # -- updating screen displaying a new sample
        self.player.stop()

        self.textbox.delete("0.0", "end")
        self.loader.create_video()
        self.textbox.insert("0.0", self.loader.df.iloc[self.loader.index]["transcription"])
        self.player.play(self.final_video_clip_path)

        app.title(f"AnnoTheia - Processing scene {self.loader.index+1} of {len(self.loader.df)} from {self.scenes_info_path}")

    def keyword_func(self, event):
        # -- keyword shortages functionaly
        if event.keysym == 'F1':
            self.player.play_pause()
        if event.keysym == 'F2':
            self.player.backward(self.final_video_clip_path, 1)
        if event.keysym == 'F3':
            self.player.forward(1)

        # if event.keysym == "Return":
        #     text_from_jumpto = self.jumpto_textbox.get("0.0", "end").strip()
        #     if len(text_from_jumpto) > 0 and text_from_jumpto.isdigit():
        #         self.jump_to()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for supervising and annotating the candidate scenes provided by the AnnoTheia's Pipeline",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--scenes-info-path", required=True, type=str, help="Path to a CSV file where we can find the information w.r.t. the candidate scenes of a specific video.")
    parser.add_argument("--max-history-len", default=100, type=int, help="Integer representing the user history annotation in order to allow the user to come back to a previous stage")

    args = parser.parse_args()

    # -- creating an annotated copy of the scene's info CSV
    extension_index = args.scenes_info_path.rfind('.csv')
    output_csv = args.scenes_info_path[:extension_index].replace("_saved", "") + '_annotated.csv'

    # -- starting the user interface
    app = App(args.scenes_info_path, output_csv, args.max_history_len)
    app.bind("<KeyPress>", app.keyword_func)
    app.mainloop()

    # -- removing temporary files
    shutil.rmtree(app.loader.temp_dir)
