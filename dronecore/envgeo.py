from abc import ABC, abstractmethod
from math import sqrt, pi

class Position :

    def __init__(self, x:float=0, y:float=0, z:float=0, heading:float=0) :
        self.x=x
        self.y=y
        self.z=z
        self.heading=heading

    def setCoord(self, x:float, y:float, z:float, heading:float):
        self.x=x
        self.y=y
        self.z=z
        self.heading=heading

    def distance(self, other) -> float :
        return sqrt((self.x - other.x) * (self.x - other.x) + (self.y - other.y) * (self.y - other.y) + (self.z - other.z) * (self.z - other.z));

    def getHeadingRadian(self):
        return self.heading*pi/180

    def __str__(self):
        """ Représentation de la position sous forme d'une chaine de caractères """
        return "Position [X={}, Y={}, Z={}, Heading={}]".format(self.x, self.y, self.z, self.heading)

class Segment2D :

    def __init__(self, x1:float, y1:float, x2:float, y2:float) :
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2

    def __str__(self):
        """ Représentation d'un segment sous forme d'une chaine de caractères """
        return "x1={}, y1={}, x2={}, y2={}".format(self.x1, self.y1, self.x2, self.y2)

    def __repr__(self):
        """ Représentation d'un segment sous forme d'une chaine de caractères """
        return "Segment2D [x1={}, y1={}, x2={}, y2={}]".format(self.x1, self.y1, self.x2, self.y2)

class ARoom(ABC) :

    def __init__(self, height:int=250):
        self.__height = height

    def getHeight(self) -> int :
        """
        Get the height of the room.
        :return: the height in cm
        """
        return self.__height

    def setHeight(self, height:int):
        """
        Set the height of the room (must be >0).
        :param height: the height to set
        """
        if height>0 :
            self.__height = height

    @abstractmethod
    def getLengthX(self) -> int :
        """
        Get the length of the room along the x-axis.
        :return: the length (in cm)
        """
        pass

    @abstractmethod
    def getLengthY(self) -> int :
        """
        Get the length of the room along the y-axis
        :return: the length (in cm)
        """
        pass

    @abstractmethod
    def intersectWall(self, p1:Position, p2:Position) -> bool :
        """
        Determines if the segment between positions p1 and p2 has an intersection with a wall of the room.
        :param p1: the first position
        :param p2: the second position
        :return: true if there is an intersection, false otherwise
        """
        pass

    @abstractmethod
    def intersectWall2(self, p1:Position, p2:Position) -> Position :
        """
        Get the position of the intersection between segment defined by positions p1 and p2, and a wall of the room. If there is no intersection, it returns None.
        :param p1: the first position
        :param p2: the second position
        :return: the position where the segment intersects a wall (None if no intersection)
        """
        pass

    @abstractmethod
    def getRandomPosition(self, p1:Position=None, p2:Position=None) -> Position :
        """
        Get a random position :
         - inside the polygon defined by the room if p1 and p2 are None;
         - inside the cuboid defined by positions p1 and p2, and inside the room.
        :return: a position
        """
        pass

    @abstractmethod
    def getWalls2D(self, h:int=0) -> []:
        """
        Get the array of walls coordinates in (x,y) space.
        :return: the walls as an array of Segment2D objects
        """
        pass