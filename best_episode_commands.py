from dronecmds import *

raw_commands =[
    #backward(68),
    #goLeft(42),
    #goLeft(22),
    #forward(2),
    #goLeft(22),
    #goDown(40),
    #goUp(12),
    #forward(6),
    #goRight(3),
    #goRight(3),
    #forward(4),
]
def replay_best_episode():
    locate(99, 99, 90)
    takeOff()
    goLeft(80)
    backward(56)
    goDown(28)
    land()
createRoom('(0 0, 101 0, 101 101, 0 101, 0 0)', 100)
createTargetIn(22, 42, 53, 24, 44, 55)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
