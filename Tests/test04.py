from dronecmds import *

if __name__ == '__main__':
    print("**** TEST n°4 : choix du visualiseur et erreur de manoeuvre.")
    visu=""
    while visu!="1" and visu!="2" :
        visu=input("Choisissez votre visualiseur : 1 pour VIEWER_CONSOLE ou 2 pour VIEWER_BASICMPL ")
    if visu=="1" :
        visu = VIEWER_CONSOLE
    else :
        visu = VIEWER_BASICMPL
    try :
        createRoom("(0 0, 500 0, 500 1000, 0 1000, 0 0)", 300)
        createTarget()
        createDrone(DRONE_VIRTUAL, visu)
        locate(100,100,90)
        takeOff()
        forward(50)
        goUp(50)
        backward(150) # recule dans le mur !
        # Les 3 instructions suivantes ne sont pas exécutées...
        rotateLeft(90)
        forward(50)
        land()
    except Exception as err:
        print(err)
        display()
