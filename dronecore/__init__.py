"""
Ensemble de classes qui permettent de simuler la programmation d'un drone devant
naviguer dans un environnement clos (une pièce).
"""

import time
from abc import ABC, abstractmethod
from enum import Enum
from dronecore.envgeo import ARoom, Position
from math import *

class DroneState(Enum) :
    KO = 0
    ONGROUND = 1
    INFLIGHT = 2

class CommandResult(Enum):
    RES_OK="OK"
    RES_NO="NO"
    RES_BREAK="BREAK"

class CommandType(Enum):
    CMD_NONE=("no command", None)
    CMD_LOCATE=("locates", None)
    CMD_TAKEOFF=("takes off", None)
    CMD_LAND=("lands", None)
    CMD_FORWARD=("moves forward", "cm")
    CMD_BACKWARD=("moves back", "cm")
    CMD_GOUP=("rises", "cm")
    CMD_GODOWN=("descends", "cm")
    CMD_GORIGHT=("flies right", "cm")
    CMD_GOLEFT=("flies left", "cm")
    CMD_ROTATERIGHT=("rotates right", "degree")
    CMD_ROTATELEFT=("rotates left", "degree")

    def __new__(cls, value, unit:str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.unit = unit
        return obj

class Command :
    def __init__(self, ctype:CommandType=CommandType.CMD_NONE, amount:int=-1, response:bool=True, result:CommandResult=CommandResult.RES_OK):
        self.ctype = ctype # the command type
        self.amount = amount # the amount provided to the command (a number of centimeters or of degrees)
        self.response = response # the response : true if valid, false if not valid
        self.result = result # the result of the command on the drone

    def __str__(self):
        if self.amount>0 :
            return "Command {} : amount={}{} response={} result={}".format(self.ctype.name, self.amount,
                                                                           self.ctype.unit, self.response, self.result.name)
        else :
            return "Command {} : response={} result={}".format(self.ctype.name, self.response, self.result.name)

class ADrone(ABC) :

    def __init__(self):
        self.state = DroneState.ONGROUND # the current state of the drone
        self.command = Command() # the command sends to the drone
        self.viewer:AViewer|None = None # the viewer used to interact with the drone
        self.previous = Position()  # the previous position of the drone
        self.position = Position()  # the current position of the drone


    def display(self, message:str=None):
        if self.viewer is not None :
            self.viewer.display(message)
        else :
            print(message)

    @abstractmethod
    def setFlightParameters(self, filename : str) :
        """
        Charge les paramètres du drone à partir d'un fichier de propriétés. Ces paramètres
        sont par exemple l'altitude du drone au décollage, le nombre min et max de cm dont
        il peut se déplacer avec un ordre forward, etc.
        :param filename: le nom du fichier de propriétés.
        """
        pass

    @abstractmethod
    def locate(self, x:float, y:float, heading:int, room : ARoom)  :
        """
        Positionne le drone par rapport au repaire.
        :param x: abscisse du drone.
        :param y: ordonnée du drone.
        :param heading: cap du drone en degrés par rapport à l'axe des abscisses.
        :param room: salle dans laquelle le drone est placé.
        """
        pass

    @abstractmethod
    def takeOff(self) :
        """
        Le drone décolle et va se positionner en vol stationnaire à environ 80cm du sol ;
        si le drone est déjà en vol, la commande est sans effet.
        """
        pass

    @abstractmethod
    def land(self) :
        """
        Le drone se pose droit sous lui. Sans effet si le drone est déjà posé.
        """
        pass

    @abstractmethod
    def forward(self, n:int) :
        """
        Le drone avance droit devant lui de n cm. Sans effet si le drone n’a pas décollé.
        :param n: le nombre de centimètres.
        """
        pass

    @abstractmethod
    def backward(self, n:int) :
        """
        Le drone recule de n cm (sens opposé à forward). Sans effet si le drone n’a pas décollé.
        :param n: le nombre de centimètres.
        """
        pass

    @abstractmethod
    def goUp(self, n:int) :
        """
        Le drone s’élève de n cm par rapport à son altitude courante. Sans effet si le drone n’a pas décollé.
        :param n: le nombre de centimètres.
        """
        pass

    @abstractmethod
    def goDown(self, n:int) :
        """
        Le drone descend de n cm par rapport à son altitude actuelle. Sans effet si le drone n’a pas décollé.
        :param n: le nombre de centimètres.
        """
        pass

    @abstractmethod
    def goLeft(self, n:int) :
        """
        Le drone se déplace latéralement de n cm sur sa gauche. Sans effet si le drone n’a pas décollé.
        :param n: le nombre de centimètres.
        """
        pass

    @abstractmethod
    def goRight(self, n:int) :
        """
        Le drone se déplace latéralement de n cm sur sa droite. Sans effet si le drone n’a pas décollé.
        :param n: le nombre de centimètres.
        """
        pass

    @abstractmethod
    def rotateLeft(self, n:int) :
        """
        Le drone pivote de n degrés vers sa gauche, sans modifier sa position. Sans effet si le drone n’a pas décollé.
        :param n: un angle en degrés.
        """
        pass

    @abstractmethod
    def rotateRight(self, n:int) :
        """
        Le drone pivote de n degrés vers sa droite, sans modifier sa position. Sans effet si le drone n’a pas décollé.
        :param n: un angle en degrés.
        """
        pass

    @abstractmethod
    def isTargetDetected(self) -> bool :
        """
        Interroge le drone pour savoir s'il a détecté la cible.
        :return: True si la cible est détectée, False sinon
        """
        pass

    @abstractmethod
    def getHeight(self) -> int :
        """
        Fonction qui retourne l’altitude (approximative) du drone, en cm.
        :return: altitude par rapport au plancher de la pièce en cm.
        """
        pass

    @abstractmethod
    def getHeading(self) -> float :
        """
        Fonction qui retourne le cap du drone en radians.
        :return: cap en radians.
        """
        pass

    def savePosition(self):
        self.previous.x = self.position.x
        self.previous.y = self.position.y
        self.previous.z = self.position.z
        self.previous.heading = self.position.heading

    def getPreviousPosition(self) -> Position :
        """
        Fonction qui retourne la position précédente du drone dans le repère de la salle.
        :return: position du drone
        """
        return self.previous

    def getCurrentPosition(self) -> Position :
        """
        Fonction qui retourne la position courante du drone dans le repère de la salle.
        :return: position du drone
        """
        return self.position

    def getState(self) -> DroneState :
        """
        Fonction qui retourne l'état courant du drone (en vol, au sol, ou KO).
        :return: l'état du drone
        """
        return self.state

    def getCommand(self) -> Command :
        """
        Fonction qui retourne la dernière commande reçue par le drone.
        :return: la dernière commande reçue par le drone
        """
        return self.command

    def getAmount(self) -> int :
        """
        Fonction qui retourne la quantité transmise avec la dernière commande
        (nombre de cm pour une distance, ou de degrés pour un angle).
        :return: la quantité (ou -1 si sans objet)
        """
        return self.command.amount

class AViewer(ABC) :
    """
    Ancestor of drone visualisation.
    """

    def __init__(self, drone : ADrone, room : ARoom, target : Position = None, showDegrees : bool = True) :
        self.drone = drone
        self.room = room
        self.target = target
        self.showDegrees = showDegrees

    @abstractmethod
    def display(self, message:str=None):
        """
        Display the drone and its environment
        :param message: a message to show
        """
        pass

    def getStateString(self) -> str :
        p=self.drone.getCurrentPosition()
        s="State={} | Position X={} Y={} Z={}".format(self.drone.getState().name,p.x, p.y, p.z)
        if self.showDegrees :
            s+=" Heading={}°)".format(round(180*p.heading/pi))
        else:
            s += " (Heading={} rad)".format(p.heading)
        return s

class ViewerConsole(AViewer) :
    """
    Basic implementation of AViewer : visualization of the drone's state
    and sent command in the console.
    """

    def display(self, message:str=None):
        if message is None :
            if self.drone.getState()==DroneState.KO :
                print(">>>>>> Drone is KO - Program stopped <<<<<<")
            print(self.getStateString()," [", self.drone.getCommand(),"]")
            time.sleep(0.5)
        else :
            print(">>>",message,"<<<")
