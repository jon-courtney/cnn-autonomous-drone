# ImageWindow.py
from PIL import Image, ImageDraw, ImageTk
try:
    import Tkinter as tkinter
except ImportError:
    import tkinter

# Thanks to the following examples for the Tk code...
# http://code.activestate.com/recipes/521918-pil-and-tkinter-to-display-images/
# https://stackoverflow.com/questions/19895877/tkinter-cant-bind-arrow-key-events

class ImageWindow:
    def __init__(self, width, height, root=None):
        if root==None:
            self.root = tkinter.Tk()
            # self.root = tkinter.Toplevel()
        else:
            self.root = root

        self.width = width
        self.height = height
        self.key = ''
        self.num_annotated = 0

        self.root.geometry('%dx%d' % (width, height))
        self.root.bind("<Key>", lambda e: self.key_handler(e, self))
        # self.root.bind("<Left>", self.arrow_handler)
        # self.root.bind("<Right>", self.arrow_handler)
        # self.root.bind("<Up>", self.arrow_handler)
        # self.root.bind("<Down>", self.arrow_handler)

    @staticmethod
    def key_handler(event, obj):
        obj.set_key(event.keysym)
        event.widget.quit()

    def set_key(self, key):
        self.key = key

    def get_key(self):
        return self.key

    def show_image(self, image):
        self.tki = ImageTk.PhotoImage(image)
        self.label = tkinter.Label(self.root, image=self.tki)
        self.label.place(x=0, y=0, width=self.width, height=self.height)
        return self

    def force_focus(self):
        self.root.focus_force()

    def wait(self):
        self.root.mainloop()

    def update(self):
        self.root.update()

    def close(self):
        self.root.destroy()
