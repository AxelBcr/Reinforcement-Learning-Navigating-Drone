"""
Implémentation de la classe abstraite ARoom basée sur le package shapely. Une pièce
est définie comme un polygone (classe shapely.Polygon).
"""
from dronecore.envgeo import *
from shapely import Polygon, LineString, intersects, intersection, Point
from shapely.wkt import loads
from random import uniform

class RoomShp(ARoom) :

    def __init__(self, coords, height:int=250):
        super().__init__(height)
        self.geometry = None
        if type(coords) == tuple :
            self.geometry = Polygon(coords)
        elif type(coords) == str :
            coords="POLYGON("+coords+")"
            self.geometry = loads(coords)
        else :
            raise Exception("The coordinates that describe the perimeter of the room are incorrectly formatted.")
        self.__lengthX = self.geometry.bounds[2]
        self.__lengthY = self.geometry.bounds[3]

    def __str__(self):
        if self.geometry is None :
            return "Undefined Room"
        else :
            return "Room "+str(self.geometry)

    def getLengthX(self) -> int:
        return self.__lengthX

    def getLengthY(self) -> int:
        return self.__lengthY

    def intersectWall(self, p1: Position, p2: Position) -> bool:
        return intersects(LineString([(p1.x, p1.y), (p2.x, p2.y)]),
                          self.geometry.exterior)

    def intersectWall2(self, p1: Position, p2: Position) -> Position|None:
        g=intersection(LineString([(p1.x, p1.y), (p2.x, p2.y)]),
                       self.geometry.exterior)
        if len(g.coords)==0 :
            return None
        else :
            return Position(x=g.coords[0][0], y=g.coords[0][1], z=p1.z)

    def getRandomPosition(self, p1: Position = None, p2: Position = None) -> Position:
        [xmin, ymin, xmax, ymax] = self.geometry.bounds
        if p1 is None and p2 is None :
            pt = Point(uniform(xmin, xmax), uniform(ymin, ymax))
            while not self.geometry.contains(pt):
                pt = Point(uniform(xmin, xmax), uniform(ymin, ymax))
            return Position(pt.x, pt.y, uniform(0,self.getHeight()))
        elif p1 is not None and ((p2 is None) or (p2 is not None and p1.distance(p2)<0.1)) :
            pt = Point(p1.x, p1.y)
            if self.geometry.contains(pt) and p1.z<self.getHeight() :
                return Position(p1.x, p1.y, p1.z)
            else :
                raise Exception("The given point is not inside the room")
        elif p1 is None :
            pt = Point(p2.x, p2.y)
            if self.geometry.contains(pt) and p2.z<self.getHeight() :
                return Position(p2.x, p2.y, p2.z)
            else:
                raise Exception("The given point is not inside the room")
        else :
            pt = Point(p1.x, p1.y)
            if not self.geometry.contains(pt) or p1.z>self.getHeight() or p1.z<0 :
                raise Exception("The first point is not inside the room")
            pt = Point(p2.x, p2.y)
            if not self.geometry.contains(pt) or p2.z>self.getHeight() or p2.z<0 :
                raise Exception("The second point is not inside the room")
            return Position(uniform(p1.x, p2.x), uniform(p1.y, p2.y),uniform(p1.z,p2.z))

    def getWalls2D(self, h:int=0) :
        coords=list(self.geometry.exterior.coords)
        x=[]
        y=[]
        z=[]
        for c in coords :
            x.append(c[0])
            y.append(c[1])
            z.append(h)
        return x, y, z

if __name__ == '__main__':
    room = RoomShp("(0 0, 500 0, 500 500, 0 500, 0 0)")
    print(room)

    coords=((0,0), (500,0), (500,500), (0,500), (0,0))
    print(type(coords))
    room = RoomShp( coords )
    print(room)
    print(type(room.geometry))
    print(room.getLengthX(), room.getLengthY())
    print(room.geometry.exterior)

    line = LineString([(250, 250), (250, 750)])
    print(intersects(line, room.geometry.exterior))
    g=intersection(line, room.geometry.exterior)
    print(type(g))
    print(g.coords[0])

    print("Test intersectWall2 :")
    print(room.intersectWall2(Position(250,250), Position(250, 450)))

    line = LineString([(250, 250), (250, 450)])
    print(intersects(line, room.geometry.exterior))
    print(intersection(line, room.geometry.exterior))

    line = LineString([(0, 0), (250, 0)])
    print(intersects(line, room.geometry.exterior))
    g=intersection(line, room.geometry.exterior)
    print(type(g))
    print(g.coords[0])
    print(g.coords[0][0], g.coords[0][1])

    print("Generate position 1")
    print(room.getRandomPosition())
    print("Generate position 2")
    print(room.getRandomPosition(Position(250,250)))
    print("Generate position 3")
    print(room.getRandomPosition(Position(200,200), Position(300,300)))

    print(room.getWalls2D())
