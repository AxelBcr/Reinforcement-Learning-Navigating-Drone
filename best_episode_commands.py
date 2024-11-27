from dronecmds import *

def replay_best_episode():
    locate(23.114051563202064, 281.7521876452558, 180)
    takeOff()
    goUp(31.913364402639147)
    goUp(39)
    backward(49)
    backward(30)
    goRight(40)
    land()
createRoom('(0 0, 500 0, 500 1000, 0 1000, 0 0)', 300)
createDefineTarget(197.53548092834959, 64.73725710995492, 112.58453123770555)
createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)
