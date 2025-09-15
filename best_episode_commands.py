from dronecmds import *

raw_commands =[
    #goUp(91),
    #forward(98),
    #goRight(47),
    #forward(51),
    #goUp(70),
    #goUp(41),
    #forward(73),
    #forward(74),
    #forward(65),
    #goRight(95),
    #forward(17),
    #goRight(83),
    #backward(28),
    #backward(69),
    #forward(8),
    #forward(8),
    #forward(8),
    #goRight(59),
    #forward(45),
    #goRight(92),
    #backward(67),
    #backward(27),
    #backward(35),
    #goRight(43),
    #forward(38),
    #goUp(96),
    #backward(21),
    #backward(55),
    #goDown(21),
    #forward(28),
    #goRight(9),
]
def replay_best_episode():
    locate(20, 25, 90)
    takeOff()
    goRight(428)
    goUp(277)
    forward(211)
    land()
createRoom('(0 0, 499 0, 499 499, 0 499, 0 0)', 500)
createTargetIn(449, 233, 353, 451, 235, 355)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
