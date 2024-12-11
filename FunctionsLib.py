import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from dronecmds import createRoom, createDrone
from tqdm import tqdm

def initialize_settings():
    """
    Initialise les paramètres de la salle, du drone, et des cibles.
    """
    global settings

    # Saisie des paramètres de la salle, drone, et cible
    room_x = int(input("Enter the depth of the room: "))
    room_y = int(input("Enter the width of the room: "))
    room_height = int(input("Enter the height of the room: "))


    target_x = int(input("Enter the x coordinate of the target: "))
    while target_x < 0 or target_x >= room_x:
        target_x = int(input("Invalid x coordinate. Please enter a value between 0 and the depth of the room: "))

    target_y = int(input("Enter the y coordinate of the target: "))
    while target_y < 0 or target_y >= room_y:
        target_y = int(input("Invalid y coordinate. Please enter a value between 0 and the width of the room: "))

    target_z = int(input("Enter the z coordinate of the target: "))
    while target_z < 0 or target_z >= room_height:
        target_z = int(input("Invalid z coordinate. Please enter a value between 0 and the height of the room: "))

    drone_x = int(input("Enter the x coordinate of the drone: "))
    while drone_x < 0 or drone_x >= room_x:
        drone_x = int(input("Invalid x coordinate. Please enter a value between 0 and the depth of the room: "))

    drone_y = int(input("Enter the y coordinate of the drone: "))
    while drone_y < 0 or drone_y >= room_y:
        drone_y = int(input("Invalid y coordinate. Please enter a value between 0 and the width of the room: "))

    num_episodes = int(input("Enter the number of episodes: "))
    max_steps_per_episode = int(input("Enter the maximum number of steps per episode: "))

    settings = {
        "room_x": room_x,
        "room_y": room_y,
        "room_height": room_height,
        "target_x": target_x,
        "target_y": target_y,
        "target_z": target_z,
        "drone_x": drone_x,
        "drone_y": drone_y,
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps_per_episode
    }

    return settings

#%% Innit Settings
settings = initialize_settings()

#%% DroneVirtual, Compute_Reward, Step, Reset, Target, Room
class DroneVirtual:
    def __init__(self, drone, room, room_size=(settings["room_x"], settings["room_y"], settings["room_height"]), max_steps= settings["max_steps_per_episode"]):
        # Initializing Variables
        self.drone = drone
        self.room = room
        self.room_depth, self.room_width, self.room_height = room_size
        self.max_distance = min(self.room_width, self.room_depth, self.room_height)-1
        self.target_position = self.create_target()
        self.radius_detection = 50
        self.max_steps = max_steps
        self.step_count = 0

        # Discretize observation space
        self.state_bins = [
            np.linspace(0, self.room_width, round(5 + (self.room_width ** 0.45))),
            np.linspace(0, self.room_depth, round(5 + (self.room_depth ** 0.45))),
            np.linspace(0, self.room_height, round(5 + (self.room_height ** 0.45))),
        ]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.room_depth, self.room_width, self.room_height]),
            dtype=np.float32,
        )
        # Action space: (direction, distance)
        self.action_space = spaces.Tuple((
            spaces.Discrete(6),  # 6 directions
            spaces.Discrete(self.max_distance)  # Steps
        ))
        # Initialize viewer
        self.viewer = drone.viewer
        self.state = None
        self.prev_distance = float('inf')

    def discretize_state(self, state):
        """
        Discretize the state into bins.
        Rounding values to avoid float points calculation errors
        """
        discretized = []
        for dim, bins in zip(state, self.state_bins):
            discretized.append(np.digitize(dim, bins) - 1)
        return tuple(discretized)

    def create_target(self):
        """
        Create a target within the room using user's inputs.
        """
        return np.array([settings["target_x"], settings["target_y"], settings["target_z"]])

    def reset(self):
        """
        Reset the drone to a fixed position. Target remains fixed.
        """
        # Fixed drone initial position
        self.state = np.array([settings["drone_x"], settings["drone_y"], 80])  # takeOff() initial position is at 80
        self.drone.locate(self.state[0], self.state[1], self.state[2], self.room)

        # Reset variables
        self.visited_states = set()
        self.prev_distance = np.linalg.norm(self.state - self.target_position)
        self.step_count = 0

        return self.discretize_state(self.state)

    def step(self, action):
        """
        Execute a step in the environment based on the chosen action.
        `action` is a tuple (direction, distance).
        """
        direction, distance = action

        x, y, z = self.state
        if direction == 0 and y + distance <= self.room_depth:  # Move Up
            y += distance
        elif direction == 1 and y - distance >= 0:  # Move Down
            y -= distance
        elif direction == 2 and x - distance >= 0:  # Move Left
            x -= distance
        elif direction == 3 and x + distance <= self.room_width:  # Move Right
            x += distance
        elif direction == 4 and z + distance <= self.room_height:  # Ascend
            z += distance
        elif direction == 5 and z - distance >= 0:  # Descend
            z -= distance

        self.state = np.array([x, y, z])
        self.drone.locate(self.state[0], self.state[1], self.state[2], self.room)

        # Increment step count
        self.step_count += 1

        # Compute reward
        reward = self.compute_reward(self.state, self.step_count, self.max_steps)
        done = reward >= 1000 or self.step_count >= self.max_steps
        return self.discretize_state(self.state), reward, done, {}

    def compute_reward(self, state, step_count, max_steps):
        """
        Compute the reward for the current state of the drone.
        """
        # Calculate the distance from the drone to the target
        distance = np.linalg.norm(state - self.target_position)
        reward = 0

        # Check if the drone has reached the target
        if distance <= 5:  # Within 5 cm of the target
            remaining_steps = max_steps - step_count
            reward = 1000 + remaining_steps * 10  # Large reward for reaching target quickly
            return reward  # Return immediately when the target is reached

        # Calculate the change in distance from the previous state
        delta_distance = self.prev_distance - distance

        # Reward or penalize based on the change in distance
        if delta_distance > 0:
            reward += delta_distance * 10  # Reward for reducing distance
        else:
            reward += delta_distance * 10  # Smaller penalty for moving away

        # Penalty for revisiting states (to discourage oscillation)
        discretized_state = tuple(map(int, state))  # Discretize state to avoid floating-point issues (round values)
        if discretized_state in self.visited_states:
            reward -= 100  # Penalty for revisiting a state
        self.visited_states.add(discretized_state)

        # Penalty for each step to encourage efficiency
        reward -= (0.5+(settings["max_steps_per_episode"]/settings["num_episodes"])) * step_count  # Scaled penalty based on step count

        # Update the previous distance for the next step
        self.prev_distance = distance

        return reward


def smooth_commands(commands):
    """
    Smooths and reorders commands based on proximity to walls, with cancellation of opposing commands.

    :param commands: List of tuples (direction, distance).
    :return: Smoothed and reordered list of commands.
    """

    smoothed_commands = []
    max_distance = 490

    # Opposing direction pairs (direction -> opposing direction)
    opposing_directions = {
        0: 1,  # forward -> backward
        1: 0,  # backward -> forward
        2: 3,  # goLeft -> goRight
        3: 2,  # goRight -> goLeft
        4: 5,  # goUp -> goDown
        5: 4,  # goDown -> goUp
    }

    # Aggregate movements by direction
    movement_totals = {}
    for direction, distance in commands:
        direction = int(direction)  # Ensure direction is a Python int
        distance = int(distance)    # Ensure distance is a Python int
        if direction in movement_totals:
            movement_totals[direction] += distance
        else:
            movement_totals[direction] = distance

    # Cancel out opposing movements with proper dominance handling
    for direction in list(movement_totals.keys()):
        opposing_direction = opposing_directions.get(direction)
        if opposing_direction in movement_totals:
            # Ensure opposing direction is present
            if direction not in movement_totals or opposing_direction not in movement_totals:
                continue
            # Handle dominant movement
            if movement_totals[direction] > movement_totals[opposing_direction]:
                movement_totals[direction] -= movement_totals[opposing_direction]
                del movement_totals[opposing_direction]
            elif movement_totals[direction] < movement_totals[opposing_direction]:
                movement_totals[opposing_direction] -= movement_totals[direction]
                del movement_totals[direction]
            else:
                # If they cancel each other out completely
                del movement_totals[direction]
                del movement_totals[opposing_direction]

    # Order commands by magnitude and proximity to walls
    sorted_directions = sorted(
        movement_totals.keys(),
        key=lambda d: abs(movement_totals[d]),  # Prioritize larger movements
        reverse=True
    )

    for direction in sorted_directions:
        distance = movement_totals[direction]
        while distance >= max_distance:
            smoothed_commands.append((direction, max_distance))
            distance -= max_distance
        if distance > 0:
            smoothed_commands.append((direction, distance))

    return smoothed_commands


#Training code
def training_loop(env_with_viewer, num_episodes, max_steps_per_episode):
    """
    Boucle d'entraînement pour entraîner le drone.
    """
    global best_episode_reward, best_episode_trajectory, best_episode_actions, trajectory, trajectory_actions

    # Training parameters
    alpha = 0.05  # Learning rate
    gamma = 0.995  # Importance of future rewards
    epsilon = 0.98  # Randomness rate
    epsilon_decay = 0.92  # Randomness decay rate
    epsilon_min = 0.01  # Minimum randomness rate

    # Rounded discretize observation states
    space_x: int = round(5 + (settings["room_x"] ** 0.45))
    space_y: int = round(5 + (settings["room_y"] ** 0.45))
    space_z: int = round(5 + (settings["room_height"] ** 0.45))

    q_table = np.zeros((space_x, space_y, space_z, 6, 100))  # Shape: (x_bins, y_bins, z_bins, directions, distances)
    best_episode_reward = -float('inf')

    for episode in tqdm(range(num_episodes), desc="Training", unit="episode"):
        state = env_with_viewer.reset()
        total_reward = 0
        trajectory = []
        trajectory_actions = []
        done = False

        for step in range(max_steps_per_episode):
            if env_with_viewer.state.shape == (3,):
                trajectory.append(env_with_viewer.state.copy())
            else:
                raise ValueError("State is not 3D.")

            if np.random.random() < epsilon:
                direction = np.random.choice(6)
                distance = np.random.choice(100)
                action = (direction, distance)
            else:
                action = np.unravel_index(np.argmax(q_table[state]), q_table[state].shape)

            next_state, reward, done, _ = env_with_viewer.step(action)
            old_value = q_table[state][action[0]][action[1]]
            next_max = np.max(q_table[next_state])
            q_table[state][action[0]][action[1]] = old_value + alpha * (reward + gamma * next_max - old_value)

            state = next_state
            total_reward += reward
            trajectory_actions.append(action)

            if done:
                break

        if total_reward > best_episode_reward:
            if trajectory:
                best_episode_reward = total_reward
                best_episode_trajectory = trajectory.copy()
                best_episode_actions = trajectory_actions.copy()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if not best_episode_trajectory:
        print("No trajectory exceeded the best reward. Using the last recorded trajectory as the default.")
        best_episode_trajectory = trajectory.copy()
        best_episode_actions = trajectory_actions

    return best_episode_actions, best_episode_trajectory


def get_training_results(env_with_viewer):
    """
    Return the results of the training, including the best trajectory and actions.
    """

    best_episode_actions, best_episode_trajectory = training_loop(env_with_viewer, settings["num_episodes"], settings["max_steps_per_episode"])

    if not best_episode_actions:
        print("Warning: No best episode actions recorded. Returning empty list.")
        best_episode_actions = []
    if not best_episode_trajectory:
        print("Warning: No best episode trajectory recorded. Returning empty list.")
        best_episode_trajectory = []
    return best_episode_actions, best_episode_trajectory, settings

# Save commands to a Python file
def writing_commands(best_episode_actions, room_x, room_y, room_height, drone_x, drone_y, target_x, target_y, target_z):

    #%% Convert actions to commands
    actions_to_commands = {
        0: "forward",
        1: "backward",
        2: "goLeft",
        3: "goRight",
        4: "goUp",
        5: "goDown"
    }

    room_description = f"(0 0, {room_x - 1} 0, {room_x - 1} {room_y - 1}, 0 {room_y - 1}, 0 0)"

    raw_commands = []
    for direction, distance in best_episode_actions:
        command = f"{actions_to_commands[direction]}({distance})"
        raw_commands.append(command)

    # Smoothing raw_commands
    smoothed_commands = smooth_commands(best_episode_actions)

    # Extract positions from the environment
    initial_heading = 90  # Modify based on actual logic

    with open("best_episode_commands.py", "w") as f:
        f.write("from dronecmds import *\n\n")

        f.write("raw_commands =[\n")
        for direction, distance in best_episode_actions:
            command = f"{actions_to_commands[direction]}({distance})"
            f.write(f"    #{command},\n")
        f.write("]\n")

        f.write("def replay_best_episode():\n")

        # Save initial drone position and heading
        f.write(f"    locate({drone_x}, {drone_y}, {initial_heading})\n")

        # Add movement commands
        f.write(f"    takeOff()\n")

        # Smoothing raw_commands
        for direction, distance in smoothed_commands:
            f.write(f"    {actions_to_commands[direction]}({distance})\n")

        # End of the flight
        f.write("    land()\n")

        # Room setup
        f.write(f"createRoom('{room_description}', {room_height})\n")

        # Target position
        f.write(f"createTargetIn({target_x - 1}, {target_y - 1}, {target_z - 1}, "
                f"{target_x + 1}, {target_y + 1}, {target_z + 1})\n")

        # Save drone creation
        f.write("createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)\n")


# Training parameters (resetting Q-table)
def initialize_q_table():
    space_x = round(5 + (settings["room_x"] ** 0.45))
    space_y = round(5 + (settings["room_y"] ** 0.45))
    space_z = round(5 + (settings["room_height"] ** 0.45))
    return np.zeros((space_x, space_y, space_z, 6, 100))

# Function to reset the environment and Q-table
def reset_environment(new_target_position,  env_with_viewer):
    global q_table

    # Update settings for the new target
    settings["target_x"] = new_target_position[0]
    settings["target_y"] = new_target_position[1]
    settings["target_z"] = new_target_position[2]

    # Reset the drone's position
    last_drone_position = (settings["drone_x"], settings["drone_y"])
    settings["drone_x"] = last_drone_position[0]
    settings["drone_y"] = last_drone_position[1]

    # Reset Q-table to avoid biases
    q_table = initialize_q_table()

    # Update the environment's target position
    env_with_viewer.target_position = np.array([settings["target_x"], settings["target_y"], settings["target_z"]])


# %% Room creation & Hyperparameters & Commands writing
room_description = f"(0 0, {settings["room_x"]-1} 0, {settings["room_x"]-1} {settings["room_y"]-1}, 0 {settings["room_y"]-1}, 0 0)"
room = createRoom(room_description, settings["room_height"]-1)
drone = createDrone("DroneVirtual", "ViewerTkMPL")
env_with_viewer = DroneVirtual(drone, room, room_size=(
settings["room_x"] - 1, settings["room_y"] - 1, settings["room_height"] - 1))

# Rounded discretize observation states
space_x: int = round(5 + (settings["room_x"] ** 0.45))
space_y: int = round(5 + (settings["room_y"] ** 0.45))
space_z: int = round(5 + (settings["room_height"] ** 0.45))

# Training parameters
alpha = 0.05  # Learning rate
gamma = 0.995  # Importance of future rewards
epsilon = 0.98  # Randomness rate
epsilon_decay = 0.92  # Randomness decay rate
epsilon_min = 0.01  # Minimum randomness rate