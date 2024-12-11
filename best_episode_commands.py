from dronecmds import *

raw_commands =[
    #backward(50),
    #goLeft(90),
    #goLeft(36),
    #goLeft(59),
    #backward(68),
    #goDown(27),
    #backward(14),
    #goLeft(72),
    #backward(39),
    #goDown(45),
    #goLeft(54),
    #backward(16),
    #backward(16),
    #backward(2),
    #backward(2),
    #backward(2),
    #backward(2),
    #backward(2),
    #backward(2),
    #backward(2),
    #backward(2),
    #backward(2),
    #goLeft(38),
    #backward(81),
    #forward(25),
    #backward(85),
    #goLeft(92),
    #forward(28),
    #forward(7),
    #forward(7),
    #goLeft(42),
    #backward(8),
    #backward(95),
    #backward(10),
    #backward(10),
    #backward(10),
    #backward(9),
    #backward(9),
    #goLeft(4),
    #backward(17),
]
def replay_best_episode():
    locate(495, 495, 90)
    takeOff()
    backward(488)
    goLeft(487)
    goDown(72)
    land()
createRoom('(0 0, 499 0, 499 499, 0 499, 0 0)', 500)
createTargetIn(4, 4, 4, 6, 6, 6)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
