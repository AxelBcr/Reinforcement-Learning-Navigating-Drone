"""
Constantes et procédures permettant d'écrire - sans connaissance en POO -
le programme de navigation d'un drone dans une pièce (son environnement).
Ce drone est virtuel, mais toutes les procédures représentent des commandes
qui existent pour un drone de type Tello Edu.
"""

from dronecore.roomshply import RoomShp
from dronecore.dronevirt import *
from viewermpl import ViewerBasicMPL
from viewertk import ViewerTkMPL
from gym import spaces
import numpy as np

#################
#   Constantes  #
#################

DRONE_VIRTUAL = "DroneVirtual"
"""
Constante qui identifie un drone virtuel (simulé) associé à la classe DroneVirtual.
"""

DRONE_TELLO = "DroneTello"
"""
Constante qui identifie un drone réel de type Tello Edu, associé à la classe DroneTello.
"""

VIEWER_CONSOLE = "ViewerConsole"
"""
Constante qui identifie une vue associée à la classe ViewerConsole.
"""

VIEWER_BASICMPL = "ViewerBasicMPL"
"""
Constante qui identifie une vue associée à la classe ViewerChartDir.
"""

VIEWER_TKMPL = "ViewerTkMPL"
"""
Constante qui identifie une vue associée à la classe ViewerTkMPL.
"""

#####################################
# Objets de base pour la simulation #
#####################################

room:ARoom = None
"""
La pièce à explorer.
"""

target:Position = None
"""
La cible à détecter dans la pièce.
"""

drone:ADrone = None
"""
Le drone à commander.
"""

viewer:AViewer = None
"""
L'interface de visualisation à utiliser.
"""

###########################################
# Commandes à utiliser pour la simulation #
###########################################
def display() :
    drone.display()
    #if viewer is not None :
    #    viewer.display()

def createRoom(description: str | tuple, height: int):
    """
    Create the room environment.
    This must be the first function called to set up the room.

    :param description: A string or tuple defining the shape of the room.
                        For example: "(0 0, 500 0, 500 1000, 0 1000, 0 0)"
    :param height: Height of the room in cm.
    """
    global room
    try:
        print(f"Creating room with description: {description} and height: {height}")
        room = RoomShp(description, height)  # Initializes the room shape
        print(f"Room successfully created: {room}")
    except Exception as e:
        print(f"Error while creating room: {e}")
        room = None  # Ensure room is None if initialization fails
        raise e


def createDrone(droneId: str, viewerId: str, progfunc=None):
    global drone, viewer
    print(f"Creating drone with ID: {droneId}, Viewer ID: {viewerId}")

    if droneId == DRONE_VIRTUAL:
        print("Initializing DroneVirtual...")
        drone = DroneVirtual()
        drone.target = target
        print(f"Drone initialized: {drone}")

        if viewerId == VIEWER_TKMPL:
            print("Initializing ViewerTkMPL...")
            viewer = ViewerTkMPL(drone, room, target, progfunc=progfunc)
            drone.viewer = viewer
            print(f"Viewer successfully initialized and linked to drone: {viewer}")
    else:
        raise Exception("Drone Identifier Unknown!")

    print(f"Final Drone: {drone}, Viewer: {drone.viewer}")
    return drone


def createTarget() :
    """
    Création de la cible à une position aléatoire dans la pièce à explorer. Ce doit
    être la seconde instruction à appeler si on veut ajouter une cible à chercher.
    Créer une cible n'est pas obligatoire.
    """
    global target
    target=room.getRandomPosition()
    print("Target : x={} y={} z={}".format(target.x, target.y, target.y))

def createTargetIn(x1:float, y1:float, z1:float, x2:float, y2:float, z2:float) :
    """
    Création de la cible à une position aléatoire dans le cube défini par (x1,y1,z1)
    et (x2,y2,z2). Ce doit être la seconde instruction à appeler si on veut ajouter
    une cible à chercher.
	:param x1 abscisse inférieure gauche du cube
	:param y1 ordonnée inférieure gauche du cube
	:param z1 hauteur inférieure gauche du cube
	:param x2 abscisse supérieure droite du cube
	:param y2 ordonnée supérieure droite du cube
	:param z2 hauteur supérieure droite du cube
    """
    global target
    target=room.getRandomPosition(Position(x1,y1,z1), Position(x2,y2,z2))
    print("Target : x={} y={} z={}".format(target.x, target.y, target.z))


def locate(x, y, heading) :
    """
    Positionne le drone sur le sol (z=0) par rapport au repaire de la pièce.
    :param x: abscisse du drone
    :param y: ordonnée du drone
    :param heading: cap du drone en degrés
    """
    drone.locate(x, y, heading, room)
    drone.display()

def takeOff() :
    """
    Le drone décolle et va se positionner en vol stationnaire (à environ 80cm du sol dans le cas
	du drone tello edu). Si le drone est déjà en vol, la commande est sans effet.
    """
    drone.takeOff()
    drone.display()

def land() :
    """
    Le drone se pose droit sous lui. Sans effet si le drone est déjà posé.
    """
    drone.land()
    drone.display()

def forward(n:int) :
    """
    Le drone avance droit devant lui de n cm. Sans effet si le drone n’a pas décollé.
    :param n: nombre de cm
    """
    drone.forward(n)
    drone.display()

def backward(n:int) :
    """
    Le drone recule de n cm (sens opposé à forward), sans changer de cap.
    Sans effet si le drone n’a pas décollé.
    :param n: nombre de cm
    """
    drone.backward(n)
    drone.display()

def goUp(n:int) :
    """
    Le drone s’élève de n cm par rapport à son altitude courante.
    Sans effet si le drone n’a pas décollé.
    :param n: nombre de cm
    """
    drone.goUp(n)
    drone.display()

def goDown(n:int) :
    """
    Le drone descend de n cm par rapport à son altitude courante.
    Sans effet si le drone n’a pas décollé.
    :param n: nombre de cm
    """
    drone.goDown(n)
    drone.display()

def goLeft(n:int) :
    """
    Le drone se déplace latéralement de n cm sur sa gauche.
    Sans effet si le drone n’a pas décollé.
    :param n: nombre de cm
    """
    drone.goLeft(n)
    drone.display()

def goRight(n:int) :
    """
    Le drone se déplace latéralement de n cm sur sa droite.
    Sans effet si le drone n’a pas décollé.
    :param n: nombre de cm
    """
    drone.goRight(n)
    drone.display()

def rotateLeft(n:int) :
    """
    Le drone pivote de n degrés vers sa gauche, sans modifier sa position.
    Sans effet si le drone n’a pas décollé.
    :param n: nombre de degrés
    """
    drone.rotateLeft(n)
    drone.display()

def rotateRight(n:int) :
    """
    Le drone pivote de n degrés vers sa droite, sans modifier sa position.
    Sans effet si le drone n’a pas décollé.
    :param n: nombre de degrés
    """
    drone.rotateRight(n)
    drone.display()

def isTargetDetected() -> bool :
    """
    Interroge le drone pour savoir s'il a trouvé la cible.
    :return: True si la cible est détectée, False sinon
    """
    return drone.isTargetDetected()

def getPosition() -> Position :
    """
    Fonction qui retourne la position courante du drone.
    :return: la position
    """
    return drone.position

def getHeight() -> int :
    """
    Fonction qui retourne l’altitude du drone (approximative pour un drone réel), en cm.
    :return: l'altitude courante en cm
    """
    return drone.getHeight()

def getHeading(unit:str="radian") -> float|int :
    """
    Récupère le cap (angle par rapport à l'axe des abscisses), en radian ou en degrés.
    :param unit:
    :return: le cap
    """
    if unit=="radian" :
        return drone.getHeading()
    else :
        return round(drone.getHeading()*180/pi)
