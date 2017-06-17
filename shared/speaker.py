#!/usr/bin/env python
import os

class Speaker:
    def __init__(self):
        self.voice = 'en-us'

    def set_voice(self, voice):
        self.voice = voice

    def speak(self, words):
        os.system("espeak -v " + self.voice + " --stdout '" + words + "' | paplay")

if __name__ == '__main__':
    s = Speaker()
    s.speak("This is a test")
