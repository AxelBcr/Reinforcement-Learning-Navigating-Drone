from dronecmds import *

raw_commands =[
    #forward(39),
    #forward(48),
    #forward(35),
    #forward(35),
    #goRight(42),
    #goRight(42),
    #goRight(47),
    #forward(23),
    #forward(23),
    #goRight(36),
    #forward(33),
    #forward(33),
    #forward(47),
    #forward(30),
    #forward(30),
    #goRight(24),
    #goRight(24),
    #forward(22),
    #forward(22),
    #goRight(17),
    #goRight(17),
    #goRight(17),
    #goRight(33),
    #goRight(33),
    #forward(36),
    #goDown(35),
    #forward(36),
    #goRight(35),
    #goRight(35),
    #forward(30),
    #goRight(19),
    #goRight(19),
    #goRight(17),
    #goRight(17),
    #goRight(17),
    #goRight(37),
    #goRight(37),
    #forward(25),
    #forward(25),
    #forward(34),
    #forward(34),
    #goRight(21),
    #goRight(21),
    #forward(45),
    #forward(23),
    #forward(23),
    #forward(23),
    #goRight(22),
    #goRight(22),
    #goRight(19),
    #goRight(19),
    #goRight(19),
    #forward(20),
    #forward(20),
    #forward(43),
    #forward(43),
    #goRight(20),
    #goRight(20),
    #goRight(20),
    #forward(9),
    #forward(9),
]
def replay_best_episode():
    locate(10, 80, 90)
    takeOff()
    forward(451.0)
    forward(419.0)
    goRight(451.0)
    goRight(289.0)
    goDown(35)
    land()
createRoom('(0 0, 1000 0, 1000 1000, 0 1000, 0 0)', 1000)
createTargetIn(749, 949, 49, 751, 951, 51)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
