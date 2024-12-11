from dronecmds import *

raw_commands =[
    #forward(77),
    #goUp(93),
    #goUp(93),
    #goRight(64),
    #goRight(42),
    #goUp(40),
    #forward(49),
    #goUp(81),
    #forward(19),
    #forward(2),
    #forward(2),
    #forward(2),
    #forward(2),
    #forward(2),
    #forward(2),
    #forward(2),
    #forward(2),
    #forward(2),
    #forward(2),
    #forward(3),
    #forward(3),
    #forward(3),
    #forward(3),
    #forward(3),
    #forward(3),
    #forward(3),
    #forward(3),
    #forward(3),
    #goUp(83),
    #forward(4),
    #forward(4),
    #forward(4),
    #goDown(50),
    #goUp(43),
    #forward(4),
    #forward(4),
    #forward(4),
    #forward(4),
    #forward(4),
    #forward(4),
    #forward(4),
    #forward(4),
    #forward(4),
    #forward(5),
    #forward(5),
    #forward(5),
    #forward(5),
    #forward(5),
    #forward(72),
    #forward(11),
    #forward(20),
    #goRight(22),
    #goUp(25),
    #forward(26),
    #goRight(12),
]
def replay_best_episode():
    locate(10, 10, 90)
    takeOff()
    goUp(408)
    forward(394)
    goRight(140)
    land()
createRoom('(0 0, 499 0, 499 499, 0 499, 0 0)', 500)
createTargetIn(149, 399, 489, 151, 401, 491)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
