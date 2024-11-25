from dronecmds import *

if __name__ == '__main__':
    print("**** TEST n°3 : comme Test n°2 mais avec une pièce plus complexe.")
    try :
        createRoom("(0 0, 1000 0, 1000 600, 500 600, 500 1200, 0 1200, 0 0)", 300)
        createTargetIn(200, 200, 50, 300, 300, 150)
        createDrone(DRONE_VIRTUAL, VIEWER_BASICMPL)
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
