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

---

## Features

- Simulates drone navigation in a room of customizable dimensions.
- Finds targets using a reinforcement learning-based training loop.
- Supports real-time updates for new target positions and retraining.
- Generates optimal navigation commands after training.
- Includes a 3D visualization of the drone’s trajectory.

---

## Architecture

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
   git clone https://github.com/Warukho/Drone_Bouchaud_Roche_Axel.git
   cd Drone_Bouchaud_Roche_Axel
