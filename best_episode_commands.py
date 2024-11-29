from dronecmds import *

raw_commands =[
    #forward(94),
    #forward(99),
    #forward(53),
    #goUp(66),
    #goUp(84),
    #goRight(78),
    #goRight(45),
    #goRight(73),
    #goUp(31),
    #goRight(55),
    #forward(8),
    #forward(8),
    #forward(41),
    #forward(41),
    #forward(30),
    #forward(68),
    #goRight(74),
    #forward(71),
    #forward(9),
    #forward(9),
    #goRight(98),
    #forward(63),
    #forward(61),
    #forward(9),
    #forward(9),
    #forward(9),
    #forward(9),
    #forward(9),
    #forward(11),
    #forward(11),
    #forward(11),
    #forward(11),
    #forward(11),
    #forward(12),
    #forward(12),
    #forward(12),
    #forward(12),
    #forward(12),
    #forward(74),
    #forward(31),
    #forward(55),
    #goRight(38),
    #goRight(19),
    #goUp(15),
    #goUp(9),
]
def replay_best_episode():
    locate(10, 10, 90)
    takeOff()
    forward(300)
    forward(300)
    forward(300)
    forward(47)
    goUp(201)
    goRight(300)
    goRight(174)
    land()
createRoom('(0 0, 500 0, 500 1000, 0 1000, 0 0)', 300)
createTargetIn(479, 949, 277, 481, 951, 279)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
