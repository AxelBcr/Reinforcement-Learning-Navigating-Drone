from dronecmds import *

raw_commands =[
    #goDown(29),
    #goDown(11),
]
def replay_best_episode():
    locate(400, 400, 90)
    takeOff()
    goDown(38)
    land()
createRoom('(0 0, 499 0, 499 499, 0 499, 0 0)', 500)
createTargetIn(99, 99, 99, 101, 101, 101)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
