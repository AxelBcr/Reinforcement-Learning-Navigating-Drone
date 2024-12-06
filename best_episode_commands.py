from dronecmds import *

raw_commands =[
]
def replay_best_episode():
    locate(10, 10, 90)
    takeOff()
    land()
createRoom('(0 0, 499 0, 499 499, 0 499, 0 0)', 500)
createTargetIn(399, 399, 399, 401, 401, 401)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
