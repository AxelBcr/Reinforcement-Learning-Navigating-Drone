from dronecmds import *

if __name__ == '__main__':
    print("**** TEST n°1 : mouvements de base, vus dans la console.")
    try :
        createRoom("(0 0, 500 0, 500 1000, 0 1000, 0 0)", 300)
        createTarget()
        createDrone(DRONE_VIRTUAL, VIEWER_CONSOLE)
        locate(200,200,90)
        takeOff()
        forward(100)
        goUp(50)
        rotateLeft(90)
        forward(100)
        land()
    except Exception as err:
        print(err)
        display()
