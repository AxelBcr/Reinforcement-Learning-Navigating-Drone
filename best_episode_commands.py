from dronecmds import *

raw_commands =[
    #goUp(47),
    #goUp(46),
    #goUp(46),
    #goUp(46),
    #forward(38),
    #goRight(46),
    #goUp(11),
    #goUp(11),
    #goUp(11),
    #goUp(11),
    #goUp(11),
    #goUp(11),
    #goUp(11),
    #goUp(11),
    #goRight(40),
    #goRight(40),
    #forward(42),
    #forward(39),
    #goRight(46),
    #forward(41),
    #forward(41),
    #goUp(49),
    #goUp(49),
    #goUp(49),
    #forward(43),
    #goRight(43),
    #forward(20),
    #forward(20),
    #forward(47),
    #goRight(30),
    #goRight(30),
    #forward(41),
    #forward(41),
    #goUp(16),
]
def replay_best_episode():
    locate(230, 100, 90)
    takeOff()
    goUp(421)
    forward(403)
    goRight(269)
    land()
createRoom('(0 0, 1000 0, 1000 1000, 0 1000, 0 0)', 1000)
createTargetIn(499, 499, 499, 501, 501, 501)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
