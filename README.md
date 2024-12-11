# Navigating Drone Using Reinforcement Learning

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
- [Entire Lore For Interested People](#lore)

---

# Features

- **Room Simulation:** Customizable 3D room environment for drone navigation.
- **Target Detection:** Intelligent algorithm to locate a target in the simulated room.
- **Reinforcement Learning:** Implements Q-Learning for trajectory optimization.
- **Visualization:** Real-time 3D trajectory plotting for training and performance monitoring.
- **Dynamic Updates:** Allows reconfiguration of the target's location with retraining capabilities.
- **Replay Mechanism:** Replays the best navigation trajectory using generated commands.

---

# Architecture

**Reinforcement_Learning_Navigating_Drone/**

│   
├── **dronecmds.py**                
├── **FunctionsLib.py**             
├── **MainDrone.py**                
├── **best_episode_commands.py**           
├── **ChangingTarget.py**              
└── **README.md**                   

## Key Modules

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

# Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Warukho/Reinforcement-Learning-Navigating-Drone.git
   cd Reinforcement-Learning-Navigating-Drone

---

# Usage

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

# Modules Overview

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

# Acknowledgements
Chauvet Pierre : Developed the dronecmds, mplext, viewermpl, viewertk modules for drone operations, and dronecore, images, Tests files.   
Bouchaud--Roche Axel : Worked on reinforcement learning and dynamic environment adaptation.

---

# Contacts
For questions or contributions, please contact:  

Chauvet Pierre :   
[Github](https://github.com/pechauvet)   
[Email](pierre.chauvet@uco.fr)

Bouchaud--Roche Axel :   
[Github](https://github.com/Warukho)   
[Email](axelbouchaudroche@gmail.com)

---

# Lore

Here lays a complete and detailed explanation of the process, make sure you have some time before beginning the reading:   

## Reinforcement Learning Navigating Drone Project :   

The **Reinforcement Learning Navigating Drone** project was developed as part of a class assignment for the "Algorithmic and Programming 1" course at IMA (Angers, France). The task was to create an algorithm to control a drone in a 3D room and locate a target. While the project was initially designed to be solved with simple loops and logic, I opted to take an advanced approach by integrating **Q-learning**, a reinforcement learning algorithm, leveraging my prior experience with Python programming.

---

### Project Overview

The original assignment required:
1. Designing a straightforward exploration algorithm using basic drone commands such as `forward`, `rotateLeft`, and `goUp`.
2. Implementing and testing the algorithm in Python within a simulated environment.
3. Creating a generalized version of the algorithm to adapt to rooms of varying sizes.

Instead of following the conventional approach, I implemented Q-learning to enable the drone to learn optimal navigation strategies autonomously. This decision was driven by my two years of Python experience during high school, which provided the confidence to explore advanced techniques.

---

### Challenges and Solutions

#### Compatibility Issues
Initially, the project environment provided by Pierre Chauvet was incompatible with my setup. To address this, I utilized **ChatGPT+**, but only for debugging purposes. This tool helped me identify and resolve technical issues efficiently, allowing me to focus on algorithm development without compromising learning objectives.

---

### Why Q-learning?

I chose Q-learning for its:
- **Efficiency**: The drone could learn from its actions and improve its ability to locate the target.
- **Scalability**: The algorithm generalized well to different room sizes and layouts without requiring additional hardcoding.
- **Advanced Learning**: This approach offered an opportunity to deepen my understanding of reinforcement learning while exceeding the assignment’s expectations.
- **Improving My Knowledge**: Lerning how to use Q-learning was a whole journey through using new Python's libs, mathematical concepts.

---

### Technical Highlights

1. **Simulated Environment**:
   - Implemented a 3D coordinate system to define the room and target locations.
   - Utilized basic drone commands such as `takeOff`, `land`, and directional movements (`forward(n)`, `rotateLeft(n)`).

2. **State and Action Representation**:
   - **States**: Represented as the drone's position in the room (x, y, z).
   - **Actions**: Included movement commands like forward, rotate, and ascend.

3. **Reward System**:
   - Positive rewards for reducing the distance to the target.
   - High rewards for detecting the target (within 50 cm).
   - Penalties for inefficiency, such as revisiting previous states.

4. **Dynamic Adaptation**:
   - Retraining the model for different room dimensions or target locations required no major code modifications.

---

### Results and Learning

#### Key Outcomes
- The drone successfully navigated simulated rooms and located targets with high efficiency.
- The algorithm proved adaptable to varying room configurations and target positions.
- The project demonstrated the practical application of reinforcement learning, particularly in balancing exploration vs. exploitation and managing state discretization.

#### Reflections
This project went beyond the original requirements and provided an opportunity to explore advanced methodologies in reinforcement learning. While the assignment's scope was introductory, integrating Q-learning allowed me to:
- Push my technical boundaries and apply advanced methods to a real-world problem. By using Q-learning, I transformed a basic task into a meaningful exploration of reinforcement learning.
- Reinforced the importance of persistence in overcoming technical challenges. The use of ChatGPT+ for debugging highlighted the value of leveraging tools responsibly to complement problem-solving and learning.   

Overall, this project showcased my ability to adapt and innovate while meeting the course requirements. It was an enriching journey that strengthened both my programming expertise and my understanding of machine learning principles.



---

### Special Thanks

Special thanks to **Pierre Chauvet** for the course framework and project guidance. The project's evolution was driven by both foundational principles and the flexibility to explore advanced concepts.
