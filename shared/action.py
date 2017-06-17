# Action.py

# No enums in Python 2  :-(
class Action:
    SCAN = 0
    TARGET = 1
    TARGET_LEFT = 2
    TARGET_RIGHT = 3
    TARGET_UP = 4
    TARGET_DOWN = 5

    names = ['SCAN', 'TARGET', 'TARGET_LEFT', 'TARGET_RIGHT', 'TARGET_UP', 'TARGET_DOWN']

    def name(i):
        return Action.names[i]

    def value(n):
        return Action.names.index(n)
