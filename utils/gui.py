from threading import Thread
from playsound import playsound


def play_sound_threaded(path):
    """
    Play sound file in a separate thread
    (don't block current thread)
    """
    def play_thread_function():
        playsound(path)

    play_thread = Thread(target=play_thread_function)
    play_thread.start()