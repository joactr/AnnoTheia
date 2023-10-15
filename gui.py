import tkinter as tk
import tkinter.messagebox
import customtkinter
from tkvideo import tkvideo
import vlc
import pandas as pd
import cv2
import pickle
import subprocess

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

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
        
        

    def GetHandle(self):
        # Getting frame ID
        return self.winfo_id()

    def play(self, _source):
        # Function to start player from given source
        self.media = self.instance.media_new(_source)
        self.media.get_mrl()
        self.player.set_media(self.media)

        self.player.set_hwnd(self.winfo_id())
        self.player.play()
    
    def stop(self):
        self.player.stop()

    def playpause(self):
        print(self.player.is_playing())
        #print(self.player.get_position())
        self.player.pause()

        if self.player.get_length() - 350 < self.player.get_time():
            self.player.set_media(self.media)
            self.player.set_hwnd(self.winfo_id())
            self.player.play()
        

    def forward(self, seconds):
        #Maxes out at the video length
        newTime = min(self.player.get_time() + seconds*1000, self.media.get_duration())
        self.player.set_time(newTime)

    def backward(self, _source, seconds):
        # If the video has ended a reset is needed, uses one second of margin
        if self.player.get_time()+1000 >= self.media.get_duration(): 
            # Reset the video
            newTime = max(self.media.get_duration() - seconds*1000, 0)
            self.media = self.instance.media_new(_source)
            self.media.get_mrl()
            self.player.set_media(self.media)
            self.player.set_hwnd(self.winfo_id())
            self.player.play()
            
            self.player.set_time(newTime)
        else:
            # Can't go back past the start
            newTime = max(self.player.get_time() - seconds*1000, 0)
            self.player.set_time(newTime)
        



class Loader():
    def __init__(self):
        self.df = pd.read_csv("outputs/res.csv")
        self.index = 0
        self.createVideo()

    def createVideo(self):
        row = self.df.iloc[self.index]
        speakerN = row["speaker"]
        with open(row["dataPath"], 'rb') as f:  # open a text file
            loaded = pickle.load(f) # serialize the list
        facePos = loaded["facePos"][speakerN]
        videoPath = row["video"]

        cap = cv2.VideoCapture(videoPath)
        videoImages = []
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        iniFrame = int(row["ini"]*25)
        frameN = iniFrame
        finalFrame = int(row["end"]*25)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameN-1)

        print(frameN, finalFrame)
        while frameN < finalFrame:
            ret, image = cap.read()
            videoImages.append(image)
            frameN+=1
            if ret == False:
                break

        for i, fr in enumerate(range(iniFrame, finalFrame)): #Para cada frame en el que aparezca la cara
            try:
                color = (0,255,0)
                xmin,ymin,xmax,ymax = facePos[fr]
                videoImages[i] = cv2.rectangle(videoImages[i], (xmin,ymin), (xmax,ymax),color , 1)     
            except Exception as e :
                pass

        command = ["ffmpeg","-y","-i",str(videoPath),"-ss",str(row["ini"]),"-to",str(row["end"]),"-q:a","0","-map", "a", "temp.wav"]
        subprocess.call(command)

        self.saveTrimmedVideo(videoImages,width,height)

    def saveTrimmedVideo(self,imgFrames,width,height):
        print(len(imgFrames))
        print("si")
        video = cv2.VideoWriter("temp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, (width,height))
        print("si")
        for image in imgFrames:
            video.write(image)
        cv2.destroyAllWindows()
        video.release()
        res = subprocess.call(["ffmpeg","-y","-i","temp.mp4","-i","temp.wav","-map","0:v","-map","1:a", "-c:v", "copy", "-shortest","temp2.mp4"])


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.loader = Loader()

        # configure window
        self.title("AnnoTheia")
        self.geometry(f"{1200}x{720}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure((1, 2, 3,4), weight=1)
        self.grid_columnconfigure(5, weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Create videoFrame holder
        self.videoHolder = tk.Frame(self)
        self.videoHolder.grid(row=0, column=0, padx=(20,40), pady=(10, 10),sticky="nsew",rowspan=4, columnspan=5)
        # Init vlc player
        self.player = Screen(self.videoHolder)
        
        self.player.place(relx=0.0005, rely=0, relwidth=0.999, relheight=1)
        self.player.play('temp2.mp4')
        
        

        self.prev_button = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), text="Prev",command=self.saveSample)
        self.prev_button.grid(row=7, column=2, padx=(20, 20), pady=(20, 20), sticky="e")

        self.next_button = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), text="Next",command=self.saveSample)
        self.next_button.grid(row=7, column=3, padx=(20, 20), pady=(20, 20), sticky="w")


        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=4, column=0, padx=(20, 20), pady=(20, 20), columnspan=5, sticky="nsew")

        #Right side

        self.accept_button = customtkinter.CTkButton(master=self, fg_color="#adedbe", border_width=2, text_color=("gray10", "#DCE4EE"), text="Accept",command=self.saveSample)
        self.accept_button.grid(row=0, column=5, padx=(20, 20))

        self.incorrect_button = customtkinter.CTkButton(master=self, fg_color="#edadad", border_width=2, text_color=("gray10", "#DCE4EE"), text="Incorrect",command=self.saveSample)
        self.incorrect_button.grid(row=1, column=5, padx=(20, 20))

        self.label = customtkinter.CTkLabel(master=self, text="F1 - Play/Pause \nF2 - Rewind 5s \nF3 - Forward 5s", justify="left")
        self.label.grid(row=4, column=5, padx=(20, 20), pady=(20, 0), sticky="w")

        # set default values
        #self.appearance_mode_optionemenu.set("Dark")
        #self.textbox.insert("0.0", "---Video transcription---")
        self.textbox.insert("0.0", self.loader.df.iloc[self.loader.index]["transcription"])

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def saveSample(self):
        self.player.stop()
        print(self.textbox.get("0.0","end"))
        self.textbox.delete("0.0","end")
        self.loader.index += 1
        self.loader.createVideo()
        self.textbox.insert("0.0", self.loader.df.iloc[self.loader.index]["transcription"])
        self.player.play('temp2.mp4')

    def fun(self, event):
        if event.keysym=='F1':
            self.player.playpause()
        if event.keysym=='F2':
            self.player.backward('temp2.mp4',1)
        if event.keysym=='F3':
            self.player.forward(1)




if __name__ == "__main__":
    app = App()
    app.bind("<KeyPress>", app.fun)
    app.mainloop()
