from dronecmds import *

raw_commands =[
    #forward(50),
    #forward(48),
    #goRight(48),
    #goRight(46),
    #goRight(46),
    #forward(47),
    #forward(47),
    #forward(26),
    #goRight(43),
    #goRight(45),
    #goRight(46),
    #forward(50),
    #forward(33),
    #forward(33),
    #forward(50),
    #goRight(44),
    #forward(45),
    #goRight(44),
    #goRight(44),
    #goRight(42),
    #forward(38),
    #forward(38),
    #goRight(47),
    #forward(34),
    #goRight(50),
    #forward(49),
    #forward(25),
    #forward(25),
    #forward(50),
    #forward(50),
    #forward(41),
    #forward(24),
    #forward(30),
    #forward(30),
    #goRight(30),
    #goRight(30),
    #goRight(37),
    #forward(32),
    #forward(32),
    #forward(14),
    #forward(14),
    #forward(14),
    #goRight(46),
    #goRight(46),
    #goRight(28),
    #goDown(11),
    #goDown(11),
    #goDown(11),
]
def replay_best_episode():
    locate(10, 10, 90)
    takeOff()
    forward(451.0)
    forward(451.0)
    forward(43.0)
    goRight(451.0)
    goRight(295.0)
    goDown(31)
    land()
createRoom('(0 0, 1000 0, 1000 1000, 0 1000, 0 0)', 1000)
createTargetIn(749, 949, 49, 751, 951, 51)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
