# Action.py

# No enums in Python 2  :-(
class Action:
    def __init__(self):
        self.SCAN = 0
        self.TARGET = 1
        self.TARGET_LEFT = 2
        self.TARGET_RIGHT = 3
        self.TARGET_UP = 4
        self.TARGET_DOWN = 5

        self._names = ['SCAN', 'TARGET', 'TARGET_LEFT', 'TARGET_RIGHT', 'TARGET_UP', 'TARGET_DOWN']

    def name(self, i):
        return self._names[i]

    def value(self, n):
        return self._names.index(n)
