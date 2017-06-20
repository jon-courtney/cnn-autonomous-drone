#!/usr/bin/env python
import subprocess

class Speaker:
    def __init__(self):
        self.voice = 'en-us'
        self.words = None

    def set_voice(self, voice):
        self.voice = voice

    def speak(self, words):
        subprocess.Popen(["espeak", "-v", self.voice, words])

    def speak_new(self, words):
        if words != self.words:
            self.words = words
            self.speak(words)

if __name__ == '__main__':
    s = Speaker()
    s.speak("This is a test")
