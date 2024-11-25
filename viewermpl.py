from dronecore import *
import matplotlib.pyplot as plt
import mplext


class ViewerBasicMPL(AViewer) :
    """
    A drone viewer based uniquely on matplotlib.
    """

    def __init__(self, drone : ADrone, room : ARoom, target : Position = None, showDegrees : bool = True):
        super().__init__(drone, room, target, showDegrees)
        self.fig, self.ax = plt.subplots(1, 1, subplot_kw={"projection":"3d"}, figsize=(8,8))
        self.fig.tight_layout()
        self.ax.view_init(20, 20)
        self.ax.set_xlim(-0.5, self.room.getLengthX()+0.5)
        self.ax.set_ylim(-0.5, self.room.getLengthY()+0.5)
        self.ax.set_zlim(-0.5, self.room.getHeight()+0.5)
        x, y, z = room.getWalls2D()
        self.ax.plot(x, y, z, color=(.8, .4, .4), linewidth=5)
        if target is not None :
            self.ax.scatter(target.x, target.y, target.z, color=(.4, .2, .8), edgecolors='red', marker="D", s=80)
        plt.pause(1.0)

    def display(self, message:str=None):
        if message is None :
            if self.drone.getState()==DroneState.KO :
                print(">>>>>> Drone is KO - Program stopped <<<<<<")
            print(self.getStateString()," [", self.drone.getCommand(),"]")
            p1=self.drone.getPreviousPosition()
            p2=self.drone.getCurrentPosition()
            # draw the current position to ground segment
            if p1.x!=p2.x or p1.y!=p2.y :
                x_values = [p2.x, p2.x]
                y_values = [p2.y, p2.y]
                z_values = [p2.z, 0]
                self.ax.plot(x_values, y_values, z_values, color=(.8, .8, .8), linewidth=1)
            # draw the previous to current position segment
            x_values = [p1.x, p2.x]
            y_values = [p1.y, p2.y]
            z_values = [p1.z, p2.z]
            self.ax.plot(x_values, y_values, z_values, color=(.2, .5, .2), alpha=0.8, linestyle="--", linewidth=2)
            # draw the drone
            if self.drone.getState() != DroneState.KO :
                #self.ax.scatter(p2.x, p2.y, p2.z, color=(.2, .7, .2), edgecolors='green', marker=">", s=60)
                dx=50.0*cos(p2.heading)
                dy=50.0*sin(p2.heading)
                self.ax.arrow3D(p2.x-dx/2, p2.y-dy/2, p2.z,
                                dx, dy, 0,
                                mutation_scale=10, ec ='green', fc=(.2, .7, .2))
            else :
                self.ax.scatter(p2.x, p2.y, p2.z, color=(.9, .4, .3), marker="X", s=80)
            plt.pause(1.0)
        else :
            print(">>>",message,"<<<")

