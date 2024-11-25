"""
Implémentation de la classe abstraite ADrone. Le drone virtuel respecte les contraintes
et les commandes disponibles pour un drone de type Tello Edu.
"""
from dronecore import *
from math import *

class DroneVirtual(ADrone) :

    def __init__(self):
        super().__init__()
        self.minMove = 20  # minimum movement (in cm)
        self.maxMove = 500  # maximum movement (in cm)
        self.minRotation = 1  # minimum rotation (in degree)
        self.maxRotation = 360  # maximum rotation (in degree)
        self.takeoffAltitude = 80  # altitude when take off (in cm)
        self.minSecAltitude = 10  # minimum security altitude (in cm)
        self.radiusDetection = 50  # radius detection of a target (in cm)
        self.room = None  # the room where the drone is located
        self.target = None  # the position of the target in the room
        self.detected = False  # the detection value (true = target detected)

    def __str__(self):
        return "Virtual drone (class DroneVirtual) - state = {}".format(self.state)

    def detectTarget(self) :
        if self.target is not None:
            self.detected = (self.position.distance(self.target) < self.radiusDetection)

    def setFlightParameters(self, filename: str) :
        pass

    def locate(self, x: float, y: float, heading: int, room: ARoom) :
        self.command.ctype = CommandType.CMD_LOCATE
        self.command.amount = -1
        if self.state==DroneState.ONGROUND :
            self.command.response=True
            self.room=room
            self.position.setCoord(x, y, 0, pi*heading/180)
            self.previous.setCoord(x, y, 0, self.position.heading)
        else :
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display('WARNING : command "locate" can be used only before the drone takes off')

    def takeOff(self) :
        self.command.ctype = CommandType.CMD_TAKEOFF
        self.command.amount = -1
        if self.state == DroneState.ONGROUND :
            self.savePosition()
            self.position.z += self.takeoffAltitude
            self.command.response = True
            self.command.result = CommandResult.RES_OK
            self.state = DroneState.INFLIGHT
            self.detectTarget()
        else :
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display("Cannot takes off : drone flying or not ready")

    def land(self) :
        self.command.ctype = CommandType.CMD_LAND
        self.command.amount = -1
        if self.state == DroneState.INFLIGHT :
            self.savePosition()
            self.position.z = 0
            self.command.response = True
            self.command.result = CommandResult.RES_OK
            self.state = DroneState.ONGROUND
            self.detectTarget()
        else :
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display("Cannot lands : drone is on the ground")

    def _move(self, direction:str, n: int) :
        if self.state != DroneState.INFLIGHT :
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display("Cannot moves {} : drone is not flying".format(direction))
        elif n < self.minMove :
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display("Warning : {} movement inapplicable : minimum movement is {} cm".format(direction, self.minMove))
        elif n > self.maxMove :
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display("Warning : {} movement inapplicable : maximum movement is {} cm".format(direction, self.maxMove))
        else :
            self.command.response = True
            self.savePosition()
            tmp = Position()
            if direction=="forward" :
                tmp.x = round(self.position.x + n * cos(self.position.heading))
                tmp.y = round(self.position.y + n * sin(self.position.heading))
            elif direction=="backward" :
                tmp.x = round(self.position.x - n * cos(self.position.heading))
                tmp.y = round(self.position.y - n * sin(self.position.heading))
            crash = self.room.intersectWall2(tmp, self.position)
            if crash is not None :
                self.position.x = crash.x
                self.position.y = crash.y
                self.command.result = CommandResult.RES_BREAK
                self.state=DroneState.KO
                raise Exception("Crash ! drone hits a wall")
            self.position.x = tmp.x
            self.position.y = tmp.y
            self.command.result = CommandResult.RES_OK
            self.detectTarget()

    def forward(self, n: int) :
        self.command.ctype = CommandType.CMD_FORWARD
        self.command.amount = n
        self._move("forward", n)

    def backward(self, n: int) :
        self.command.ctype = CommandType.CMD_BACKWARD
        self.command.amount = n
        self._move("backward", n)

    def goUp(self, n: int) :
        self.command.ctype = CommandType.CMD_GOUP
        self.command.amount = n
        if self.state == DroneState.INFLIGHT :
            self.command.response = True
            self.savePosition()
            if self.position.z + n >= self.room.getHeight() :
                self.position.z = self.room.getHeight()
                self.command.result = CommandResult.RES_BREAK
                self.state=DroneState.KO
                raise Exception("Crash ! drone hits room ceiling")
            self.position.z += n
            self.command.result = CommandResult.RES_OK
            self.detectTarget()
        else :
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display("Cannot moves up : drone is not flying")

    def goDown(self, n: int) :
        self.command.ctype = CommandType.CMD_GODOWN
        self.command.amount = n
        if self.state == DroneState.INFLIGHT :
            self.command.response = True
            self.savePosition()
            if self.position.z - n <= self.minSecAltitude :
                self.position.z = self.minSecAltitude
                self.display("WARNING : Minimum altitude reached - altitude safety engaged")
            else :
                self.position.z -= n
            self.command.result = CommandResult.RES_OK
            self.detectTarget()
        else :
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display("Cannot moves down : drone is not flying")

    def _goLateral(self, direction, n):
        if self.state != DroneState.INFLIGHT :
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display("Cannot moves {} : drone is not flying".format(direction))
        elif n<self.minMove :
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display("Warning : {} movement inapplicable : minimum movement is {} cm".format(direction, self.minMove))
        elif n > self.maxMove:
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display("Warning : {} movement inapplicable : maximum movement is {} cm".format(direction, self.maxMove))
        else :
            self.command.response = True
            self.savePosition()
            tmp = Position()
            if direction=="left" :
                tmp.x = round(self.position.x - n * sin(self.position.heading))
                tmp.y = round(self.position.y + n * cos(self.position.heading))
            elif direction=="right" :
                tmp.x = round(self.position.x + n * sin(self.position.heading))
                tmp.y = round(self.position.y - n * cos(self.position.heading))
            crash = self.room.intersectWall2(tmp, self.position)
            if crash is not None:
                self.position.x = crash.x
                self.position.y = crash.y
                self.command.result = CommandResult.RES_BREAK
                self.state = DroneState.KO
                raise Exception("Crash ! drone hits a wall")
            self.position.x = tmp.x
            self.position.y = tmp.y
            self.command.result = CommandResult.RES_OK
            self.detectTarget()

    def goLeft(self, n: int) :
        self.command.ctype = CommandType.CMD_GOLEFT
        self.command.amount = n
        self._goLateral("left",n)

    def goRight(self, n: int) :
        self.command.ctype = CommandType.CMD_GORIGHT
        self.command.amount = n
        self._goLateral("right", n)

    def _rotate(self, direction:str, n:int):
        if self.state != DroneState.INFLIGHT :
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display("Cannot rotate {} : drone is not flying".format(direction))
        elif n < self.minRotation :
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display("Rotation inapplicable : minimum angle is {} degrees".format(self.minRotation))
        elif n > self.maxRotation :
            self.command.response = False
            self.command.result = CommandResult.RES_NO
            self.display("Rotation inapplicable : maximum angle is {} degrees".format(self.maxRotation))
        else :
            self.savePosition()
            if direction=="left" :
                self.position.heading += (pi * n) / 180.0
            else :
                self.position.heading -= (pi * n) / 180.0
            self.command.result = CommandResult.RES_OK
            self.detectTarget()

    def rotateLeft(self, n: int) :
        self.command.ctype = CommandType.CMD_ROTATELEFT
        self.command.amount = n
        self._rotate("left", n)

    def rotateRight(self, n: int) :
        self.command.ctype = CommandType.CMD_ROTATERIGHT
        self.command.amount = n
        self._rotate("right", n)

    def isTargetDetected(self) -> bool:
        return (self.target is not None) and self.position.distance(self.target)<self.radiusDetection

    def getHeight(self) -> int:
        return round(self.position.z)

    def getHeading(self) -> float:
        return self.position.heading
