from dronecmds import *
import traceback

def replay_best_episode():
    createRoom('(0 0, 500 0, 500 1000, 0 1000, 0 0)', 300)
    locate(495.03882940497266, 147.72527326343504, 0)
    createTargetIn(498.66136109842984, 204.9880918867376, 263.49233051885295, 500.66136109842984, 206.9880918867376, 265.49233051885295)
    takeOff()
    forward(22)
    forward(22)
    goUp(30)
    goUp(8)
    goUp(8)
    goUp(8)
    goUp(32)
    goUp(32)
    forward(21)
    forward(21)
    land()

if __name__ == '__main__':
    print("**** TEST n°6 : mouvements de base, en passant par une fonction dans VIEWER_TKMPL.")
    createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode())
