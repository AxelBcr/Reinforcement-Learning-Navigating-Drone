from dronecmds import *

raw_commands =[
    #goLeft(29),
    #goLeft(28),
    #backward(66),
    #backward(47),
    #goLeft(36),
    #backward(15),
    #goLeft(51),
    #backward(1),
    #backward(1),
    #backward(99),
]
def replay_best_episode():
    locate(490, 490, 90)
    takeOff()
    backward(229)
    goLeft(144)
    land()
createRoom('(0 0, 499 0, 499 499, 0 499, 0 0)', 500)
createTargetIn(9, 9, 9, 11, 11, 11)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
