from dronecmds import *

raw_commands =[
    #backward(79),
    #goLeft(23),
    #goDown(72),
    #goLeft(65),
    #backward(72),
    #forward(12),
    #goLeft(78),
    #goLeft(95),
    #goLeft(78),
    #forward(24),
    #backward(57),
    #goLeft(89),
    #backward(92),
    #backward(57),
    #backward(33),
    #goLeft(10),
    #goLeft(10),
    #backward(70),
    #backward(25),
    #backward(23),
    #goLeft(16),
    #goLeft(16),
    #backward(9),
]
def replay_best_episode():
    locate(490, 490, 90)
    takeOff()
    backward(481)
    goLeft(480)
    goDown(72)
    land()
createRoom('(0 0, 499 0, 499 499, 0 499, 0 0)', 500)
createTargetIn(9, 9, 9, 11, 11, 11)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
