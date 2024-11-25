import os
import tkinter as tk
import tkinter.ttk as ttk
from idlelib.tooltip import Hovertip
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import dronecmds
from dronecore import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import pathlib


class ViewerTkMPL(AViewer):
    """
    An implementation of AViewer based on tkinter and matplotlib.
    """

    BASE_DIR = pathlib.Path(__file__).parent
    """
    The directory that holds this viewer module.
    """

    def __init__(self, drone: ADrone, room: ARoom, target: Position = None, progfunc=None, showDegrees: bool = True):
        super().__init__(drone, room, target, showDegrees)
        self.progfunc = progfunc
        self.running = False
        # Creation of the main window
        self.window = tk.Tk()
        self.window.tk.call('tk', 'scaling', 3)
        self.window.title("Drone Viewer")
        self.window.geometry("1400x1400")
        # Creation of the main toolbar
        self.wintoolbar = ttk.Frame(self.window, relief=tk.RAISED)
        target_img = self.createImgTk("target_64px.png")
        self.targetButton = ttk.Button(self.wintoolbar, image=target_img, command=self.reset_target)
        self.targetButton.image = target_img
        self.targetButton.pack(side=tk.LEFT, padx=2, pady=2)
        Hovertip(self.targetButton, 'Reset the target position')
        run_img = self.createImgTk("run_64px.png")
        self.runButton = ttk.Button(self.wintoolbar, image=run_img, command=self.run)
        self.runButton.image = run_img
        self.runButton.pack(side=tk.LEFT, padx=2, pady=2)
        Hovertip(self.runButton, 'Run your drone program')
        exit_img = self.createImgTk("exit_64px.png")
        self.exitButton = ttk.Button(self.wintoolbar, image=exit_img, command=self.quit)
        self.exitButton.image = exit_img
        self.exitButton.pack(side=tk.LEFT, padx=2, pady=2)
        Hovertip(self.exitButton, 'Close the window and exit')
        ttk.Separator(self.wintoolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, ipadx=4, padx=2, pady=2,
                                                                fill=tk.Y)
        labelDelay = ttk.Label(self.wintoolbar, text="Delay (sec) :")
        labelDelay.pack(side=tk.LEFT, padx=2, pady=2)
        self.delay = tk.DoubleVar()
        self.delay.set(1.0)
        self.delayScale = tk.Scale(self.wintoolbar, variable=self.delay, from_=0.0, to=2.0,
                                   resolution=0.1, orient=tk.HORIZONTAL, length=300)
        self.delayScale.pack(side=tk.LEFT, padx=2, pady=2)
        Hovertip(self.delayScale, 'Change the simulation speed')
        # Creation of the commands viewer label
        self.infolbl=tk.Label(self.window, text=self.getStateString(), bg='gray15', fg='white', relief=tk.RAISED)
        # Creation of the Matplolib chart
        self.fig, self.ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, dpi=70, figsize=(4, 4))
        self._drawFigure()
        self.fig.tight_layout()
        # Get the tkinter canvas and toolbar from the Matplotlib chart
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.mpltoolbar = NavToolbarTk(self.canvas, self.window)
        self.mpltoolbar.update()
        # Pack all the different components
        self.wintoolbar.pack(side=tk.TOP, fill=tk.X)
        self.infolbl.pack(side=tk.TOP, expand=True, fill=tk.X)
        self.mpltoolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        # Just to force redraw !!!!
        frame = tk.Frame(self.window)
        frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        drone.viewer = self
        if self.progfunc is not None:
            tk.mainloop()
        else:
            self._setRunning(True)
            self.window.update()

    def _drawFigure(self):
        """
        Draw the matplotlib figure with the axis, the walls and the target.
        """
        self.ax.clear()
        self.ax.view_init(20, 20)
        self.ax.set_xlim(-0.5, self.room.getLengthX() + 0.5)
        self.ax.set_ylim(-0.5, self.room.getLengthY() + 0.5)
        self.ax.set_zlim(-0.5, self.room.getHeight() + 0.5)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        # Draw walls
        x, y, z = self.room.getWalls2D()
        self.ax.plot(x, y, z, color=(.8, .4, .4), linewidth=5)
        x, y, z = self.room.getWalls2D(h=self.room.getHeight())
        self.ax.plot(x, y, z, color=(.8, .4, .4), linewidth=4)
        # Draw target if any
        if self.target is not None:
            self.ax.scatter(self.target.x, self.target.y, self.target.z,
                            color=(.4, .2, .8), edgecolors='red', marker="D", s=80)
            x = [self.target.x, self.target.x]
            y = [self.target.y, self.target.y]
            z = [self.target.z, 0]
            self.ax.plot(x, y, z, color='red', linestyle="--", linewidth=1)
            x = [0, 0]
            y = [self.target.y, self.target.y]
            z = [self.target.z, 0]
            self.ax.plot(x, y, z, color=(.8, .8, .8), linestyle="--", linewidth=1)
            self.ax.scatter(0, self.target.y, self.target.z,
                            color=(.8, .8, .8), marker="D", s=80)
            x = [self.target.x, self.target.x]
            y = [0, 0]
            z = [self.target.z, 0]
            self.ax.plot(x, y, z, color=(.8, .8, .8), linestyle="--", linewidth=1)
            self.ax.scatter(self.target.x, 0, self.target.z,
                            color=(.8, .8, .8), marker="D", s=80)

    def createImgTk(self, name):
        """
        Load an image from the 'images' subdirectory, that can be used by tkinter widgets.
        :param name: the name of the image
        :return: the compatible tkinter image
        """
        img = Image.open(ViewerTkMPL.BASE_DIR / 'images' / name)
        img = ImageTk.PhotoImage(img)
        return img

    def reset_target(self):
        """
        Reset the target's position.
        """
        if not self.running :
            dronecmds.createTarget()
            self.target=dronecmds.target
            self._drawFigure()
            self.canvas.draw()

    def run(self):
        """
        Run the function self.progfunc if it exists.
        """
        if self.progfunc is not None:
            self._drawFigure()
            self.canvas.draw()
            self._setRunning(True)
            self.drone.state = DroneState.ONGROUND
            self.progfunc()
            self._setRunning(False)

    def quit(self):
        """
        Close the main window and leave the application.
        """
        self.window.quit()

    def _setRunning(self, value:bool):
        """
        Set the drone's simulation state (set attribute 'self.running' and change button's state).
        :param value: True if the simulation is launched, False otherwise
        """
        self.running = value
        if value :
            self.targetButton.config(state=tk.DISABLED)
            self.runButton.config(state=tk.DISABLED)
        else :
            self.targetButton.config(state=tk.NORMAL)
            self.runButton.config(state=tk.NORMAL)

    def display(self, message: str = None):
        """
        Draw the last movement of the drone, or display a message if any.
        :param message: the message to display.
        """
        if message is None:
            if self.drone.getState() == DroneState.KO:
                print(">>>>>> Drone is KO - Program stopped <<<<<<")
            print(self.getStateString(), " [", self.drone.getCommand(), "]")
            self.infolbl.config(text=self.getStateString()+"\n["+str(self.drone.getCommand())+"]")
            p1 = self.drone.getPreviousPosition()
            p2 = self.drone.getCurrentPosition()
            # draw the current position to ground segment
            if p1.x != p2.x or p1.y != p2.y:
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
            if self.drone.getState() != DroneState.KO:
                dx = 50.0 * cos(p2.heading)
                dy = 50.0 * sin(p2.heading)
                self.ax.arrow3D(p2.x - dx / 2, p2.y - dy / 2, p2.z,
                                dx, dy, 0,
                                mutation_scale=15, ec='green', fc=(.2, .7, .2))
            else:
                self.ax.scatter(p2.x, p2.y, p2.z, color=(.9, .4, .3), marker="X", s=80)
                self.ax.annotate3D('Crash !', (p2.x, p2.y, p2.z),
                              xytext=(30, -30),
                              textcoords='offset points',
                              bbox=dict(boxstyle="round", fc="tomato"),
                              arrowprops=dict(arrowstyle="-|>", ec='tomato', fc='black', lw=5))
            self.canvas.draw()
            self._pause_update(5)
        else:
            print(">>>", message, "<<<")

    def _pause_update(self, n: int):
        """
        Performs a loop 'update - sleep' n times to get better responsiveness of the window.
        :param n: divide the delay by n
        """
        d = self.delay.get() / n
        for _ in range(n):
            self.window.update()
            time.sleep(d)

    def render(self):
        """
        Render the current state of the viewer.
        """
        try:
            # Call existing methods to update the visualization (example names)
            self.update_canvas()  # Ensure the canvas reflects current state
            self.window.update()  # Refresh the Tkinter window
        except AttributeError:
            print("ViewerTkMPL does not support rendering. Skipping visualization.")


class NavToolbarTk(NavigationToolbar2Tk):
    """
    An enriched navigation toolbar associated to matplotlib chart.
    Can be vertical or horizontal.
    """

    def __init__(self, canvas, window, vertical=False):
        self.vertical = vertical
        super().__init__(canvas, window, pack_toolbar=False)

    # override _Button()
    def _Button(self, text, image_file, toggle, command):
        #b = super()._Button(text, image_file, toggle, command)
        if image_file.find("_large") == -1:
            try:
                (name, ext) = image_file.rsplit(".", 1)  # Use rsplit to handle multiple periods
            except ValueError:
                raise ValueError(f"Invalid image file format: {image_file}")
            image_file = name + "_large." + ext
        img_file = os.path.join(mpl.get_data_path(), 'images', image_file)
        im = tk.PhotoImage(master=self, file=img_file)
        #im = im.zoom(2, 2)
        b = tk.Button(master=self, text=text, padx=2, pady=2, image=im, command=command)
        b._ntimage = im
        if self.vertical:
            b.pack(side=tk.TOP)  # re-pack button in vertical direction
        else:
            b.pack(side=tk.LEFT)  # re-pack button in horizontal direction
        return b

    # override _Spacer() to create vertical separator
    def _Spacer(self):
        if self.vertical:
            s = tk.Frame(self, width=26, relief=tk.RIDGE, bg="DarkGray", padx=2)
            s.pack(side=tk.TOP, pady=5)  # pack in vertical direction
            return s
        else:
            return super()._Spacer()

    # disable showing mouse position in toolbar
    def set_message(self, s):
        pass
