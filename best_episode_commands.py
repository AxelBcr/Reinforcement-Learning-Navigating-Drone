from dronecmds import *

raw_commands =[
    #goRight(80),
    #goUp(67),
    #forward(79),
    #forward(91),
    #goUp(95),
    #forward(27),
    #forward(96),
    #goUp(80),
    #goRight(26),
    #forward(16),
    #goUp(53),
    #goRight(91),
    #goRight(62),
    #forward(4),
    #forward(4),
    #forward(4),
    #forward(4),
    #forward(7),
    #forward(7),
    #forward(7),
    #forward(7),
    #goRight(90),
    #goRight(75),
    #forward(28),
    #goUp(29),
    #forward(50),
    #goLeft(58),
    #backward(21),
    #goRight(18),
    #goRight(13),
    #backward(6),
]
def replay_best_episode():
    locate(10, 10, 90)
    takeOff()
    forward(391)
    goRight(390)
    goUp(319)
    land()
createRoom('(0 0, 499 0, 499 499, 0 499, 0 0)', 500)
createTargetIn(399, 399, 399, 401, 401, 401)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
