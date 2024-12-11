# Drone Target Finder

This project is an advanced algorithm designed to navigate a drone within a simulated room to locate a target. 
It uses a reinforcement learning approach combined with an interactive environment. 
The system is built to simulate commands for a Tello Edu drone, with features for re-training the model to adapt to new target positions.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Modules Overview](#modules-overview)
- [Acknowledgements](#acknowledgements)
- [Contacts](#contacts)

---

## Features

- **Room Simulation:** Customizable 3D room environment for drone navigation.
- **Target Detection:** Intelligent algorithm to locate a target in the simulated room.
- **Reinforcement Learning:** Implements Q-Learning for trajectory optimization.
- **Visualization:** Real-time 3D trajectory plotting for training and performance monitoring.
- **Dynamic Updates:** Allows reconfiguration of the target's location with retraining capabilities.
- **Replay Mechanism:** Replays the best navigation trajectory using generated commands.

---

## Architecture

**Reinforcement_Learning_Navigating_Drone/**

│   
├── **dronecmds.py**                
├── **FunctionsLib.py**             
├── **MainDrone.py**                
├── **best_episode_commands.py**           
├── **ChangingTarget.py**              
└── **README.md**                   

### Key Modules

1. **dronecmds.py**:
   Provides fundamental drone operations such as movement, positioning, and target detection.

2. **FunctionsLib.py**:
   Includes the primary classes and functions for training, reward computation, and command smoothing.

3. **MainDrone.py**:
   Serves as the entry point for executing the training and generating optimal commands.

4. **best_episode_commands.py**:
   Stores and replays the best episode commands after training.

5. **ChangingTarget.py**:
   Allows users to update the target position dynamically and retrain the model.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Warukho/Reinforcement-Learning-Navigating-Drone.git
   cd Reinforcement-Learning-Navigating-Drone

---

## Usage

**1. Initial Training**

To train the drone to find the target in the room 
(you won't be able to modify the coords of the Target here, it is only to test on a single run):

      python MainDrone.py
      
**2. Change Target and Retrain**

To dynamically update the target position and retrain the model:

      python ChangingTarget.py

**3. Replay the Best Episode**

To replay the optimal trajectory after training:

      python best_episode_commands.py

---

## Modules Overview

**dronecmds.py**

Core module for drone simulation commands:
      
      createRoom(description, height): Sets up the environment.
      createDrone(droneId, viewerId, progfunc): Initializes the drone and its visualization.
      locate(x, y, heading): Positions the drone.
      takeOff(), land(): Controls the drone's flight.
      Movement commands like forward(n), backward(n), goUp(n), goDown(n), etc.
      isTargetDetected(): Checks if the drone detects the target.
      
**FunctionsLib.py**

Utilities for training, environment setup, and command smoothing:

      initialize_settings(): Configures environment and training parameters.
      DroneVirtual: Simulates the drone’s behavior in the room.
      training_loop(env_with_viewer, num_episodes, max_steps_per_episode): Conducts the reinforcement learning loop.
      get_training_results(env_with_viewer): Fetches the best trajectory and commands.
      smooth_commands(commands): Optimizes command sequences for efficiency.
      MainDrone.py
      
**The MainDrone execution script for:**

      Running the training loop.
      Visualizing results in 3D.
      Saving the best trajectory.

      
**best_episode_commands.py**

Handles:

      Storing the optimal trajectory commands.
      Replaying the trained navigation strategy.
      ChangingTarget.py
      
Facilitates:

      Dynamic updates to the target position.
      Retraining the drone for the new position.
      Updating the replay commands for the new trajectory.

## Acknowledgements
Chauvet Pierre : Developed the dronecmds, mplext, viewermpl, viewertk modules for drone operations, and dronecore, images, Tests files.
Bouchaud--Roche Axel : Worked on reinforcement learning and dynamic environment adaptation.

## Contacts
For questions or contributions, please contact:
Chauvet Pierre :   
[Github](https://github.com/pechauvet)
[Email](pierre.chauvet@uco.fr)

Bouchaud--Roche Axel :   
[Github](https://github.com/Warukho)   
[Email](axelbouchaudroche@gmail.com)
