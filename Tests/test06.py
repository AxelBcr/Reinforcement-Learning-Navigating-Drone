from dronecmds import *
import traceback

def monprogramme() :
    """
    Fonction qui décrit le programme du drone.
    """
    try :
        locate(100,100,90)
        takeOff()
        forward(100)
        goUp(50)
        rotateLeft(90)
        forward(50)
        forward(200)
        land()
    except Exception as err:
        print(err)
        display()
        traceback.print_exc()

if __name__ == '__main__':
    print("**** TEST n°6 : mouvements de base, en passant par une fonction dans VIEWER_TKMPL.")
    createRoom("(0 0, 500 0, 500 1000, 0 1000, 0 0)", 300)
    createTargetIn(200, 200, 150, 300, 300, 200)
    createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=monprogramme)
