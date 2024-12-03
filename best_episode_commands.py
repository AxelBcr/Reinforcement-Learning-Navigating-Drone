from dronecmds import *

raw_commands =[
    #forward(99),
    #goRight(70),
    #goUp(72),
    #goRight(41),
    #goUp(58),
    #forward(74),
    #forward(85),
    #forward(47),
    #forward(49),
    #goRight(27),
    #goUp(42),
    #goRight(92),
    #forward(8),
    #forward(8),
    #forward(8),
    #goRight(72),
    #goUp(76),
    #goRight(82),
    #forward(25),
    #forward(25),
    #forward(88),
    #goUp(38),
    #goRight(37),
    #goRight(71),
    #forward(71),
    #goUp(70),
    #forward(74),
    #goUp(35),
    #goUp(66),
    #forward(16),
    #goRight(41),
    #goUp(52),
    #forward(44),
    #goUp(29),
    #backward(10),
    #goRight(74),
    #forward(33),
    #goRight(91),
    #forward(34),
    #goUp(88),
    #backward(49),
    #backward(18),
    #goUp(65),
    #goDown(95),
    #goUp(36),
    #backward(6),
]
def replay_best_episode():
    locate(10, 10, 90)
    takeOff()
    forward(491)
    forward(282)
    goRight(491)
    goRight(198)
    goUp(491)
    goUp(225)
    goDown(95)
    backward(80)
    land()
createRoom('(0 0, 800 0, 800 800, 0 800, 0 0)', 800)
createTargetIn(699, 699, 699, 701, 701, 701)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
