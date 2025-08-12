# utils/sound_alert.py
from playsound import playsound
import threading

def play_alert(sound_path):
    threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()
