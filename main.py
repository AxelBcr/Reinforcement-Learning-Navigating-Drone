#%% Imports
from dronecmds import *
from typing import Callable
from dronecore.roomshply import RoomShp
from dronecore.dronevirt import *
from viewermpl import ViewerBasicMPL
from viewertk import ViewerTkMPL
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D visualization

class DroneVirtualGymWithViewer:
    def __init__(self, drone, room, room_size=(500, 1000), target_position=(200, 200)):
        """
        Initialize the Drone Virtual Gym environment with Viewer.
        :param drone: The drone object.
        :param room: The room object.
        :param room_size: Tuple indicating room width and height.
        :param target_position: Tuple indicating the target position in the room.
        """
        self.drone = drone
        self.room = room
        self.room_width, self.room_height = room_size
        self.target_position = np.array(target_position)
        self.radius_detection = 50  # Detection radius for the target (cm)

        # Discretize observation space
        self.state_bins = [
            np.linspace(0, self.room_width, 20),
            np.linspace(0, self.room_height, 20),
            np.linspace(0, 300, 10),  # Altitude discretization (10 bins)
        ]

        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.room_width, self.room_height]),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(6)  # Actions: Up, Down, Left, Right, Ascend, Descend
        self.viewer = drone.viewer
        self.state = None

    def discretize_state(self, state):
        return tuple(
            np.digitize(state[i], self.state_bins[i]) - 1 for i in range(len(state))
        )

    def reset(self):
        """
        Reset the drone to a random position in the room, including altitude.
        """
        self.state = np.array([
            random.uniform(0, self.room_width),
            random.uniform(0, self.room_height),
            random.uniform(0, 300),  # Random altitude (0 to 300 cm)
        ])
        self.drone.locate(self.state[0], self.state[1], self.state[2], self.room)

        # Reset visited states at the start of each episode
        self.visited_states = set()

        return self.discretize_state(self.state)

    def step(self, action):
        """
        Execute a step in the environment based on the chosen action.
        :param action: The action to take (0: Up, 1: Down, 2: Left, 3: Right, 4: Ascend, 5: Descend).
        :return: next_state, reward, done, info
        """
        x, y, z = self.state
        if action == 0 and y < self.room_height:  # Up
            y += 20
        elif action == 1 and y > 0:  # Down
            y -= 20
        elif action == 2 and x > 0:  # Left
            x -= 20
        elif action == 3 and x < self.room_width:  # Right
            x += 20
        elif action == 4 and z < 300:  # Ascend (max altitude = 300 cm)
            z += 20
        elif action == 5 and z > 0:  # Descend (min altitude = 0 cm)
            z -= 20

        self.state = np.array([x, y, z])
        self.drone.locate(self.state[0], self.state[1], self.state[2], self.room)

        reward = self.compute_reward(self.state)
        done = reward >= 500  # End if the target is reached
        return self.discretize_state(self.state), reward, done, {}

    def compute_reward(self, state):
        """
        Compute the reward for the current state of the drone.
        """
        distance = np.linalg.norm(state - self.target_position)

        # Check for a crash (outside room boundaries or altitude limits)
        if state[0] < 0 or state[0] > self.room_width or state[1] < 0 or state[1] > self.room_height or state[2] < 0 or \
                state[2] > 300:
            return -200  # Crash penalty

        # Reward for reaching the exact location of the target
        if np.allclose(state, self.target_position, atol=1e-2):
            return 500  # High reward for reaching the target

        # Reward for detecting the target
        if self.drone.isTargetDetected():
            return 100 - distance  # Scaled reward for getting closer

        # Penalize repetitive states
        if tuple(state) in self.visited_states:
            return -50  # Penalty for oscillating

        # Penalize small movement steps or lack of progress
        step_penalty = -0.5

        self.visited_states.add(tuple(state))  # Add the current state to visited states
        return step_penalty


def create_environment_with_viewer():
    """
    Create the room, target, and drone, and ensure they are properly linked.
    """
    createRoom("(0 0, 500 0, 500 1000, 0 1000, 0 0)", 300)
    createTargetIn(200, 200, 150, 300, 300, 200)

    global drone
    drone = createDrone(DRONE_VIRTUAL, VIEWER_TKMPL)
    return drone


def plot_trajectory(trajectory, room_width, room_height):
    """
    Plot the drone trajectory in physical room dimensions.
    """
    positions = [state for state, _, _ in trajectory]
    x, y = zip(*positions)  # Extract x and y positions

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', label="Trajectory")
    plt.title("Drone Trajectory - Best Episode")
    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    plt.xlim(0, room_width)
    plt.ylim(0, room_height)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_3d_trajectory(trajectory, room_width, room_height, room_depth=300):
    """
    Plot the drone trajectory in 3D physical room dimensions.
    """
    positions = [state for state, _, _ in trajectory]
    x, y, z = zip(*positions)  # Extract x, y, and z positions

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, marker='o', label="Trajectory")
    ax.set_title("3D Drone Trajectory - Best Episode")
    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.set_zlabel("Z Position (cm)")
    ax.set_xlim(0, room_width)
    ax.set_ylim(0, room_height)
    ax.set_zlim(0, room_depth)
    ax.grid(True)
    ax.legend()
    plt.show()



# Initialize the environment
drone_with_viewer = create_environment_with_viewer()
env_with_viewer = DroneVirtualGymWithViewer(drone=drone_with_viewer, room=room)

# Q-learning parameters
alpha = 0.9
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1000
max_steps_per_episode = 200

# Initialize Q-table
q_table = np.zeros((20, 20, env_with_viewer.action_space.n))

# Track the best episode
best_episode_trajectory = []
highest_reward = float('-inf')

# Training loop
episode_rewards_with_viewer = []

for episode in range(num_episodes):
    state = env_with_viewer.reset()
    total_reward = 0
    done = False
    trajectory = []  # Track the trajectory of the episode

    for step in range(max_steps_per_episode):
        action = np.argmax(q_table[state]) if random.uniform(0, 1) > epsilon else env_with_viewer.action_space.sample()
        next_state, reward, done, _ = env_with_viewer.step(action)

        trajectory.append((env_with_viewer.state.copy(), action, reward))

        # Update Q-table
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])
        q_table[state][action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state
        total_reward += reward

        if done:
            break

    # Update the best episode
    if total_reward > highest_reward:
        highest_reward = total_reward
        best_episode_trajectory = trajectory

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    episode_rewards_with_viewer.append(total_reward)

# Visualize the rewards
plt.figure()
plt.plot(range(len(episode_rewards_with_viewer)), episode_rewards_with_viewer)
plt.title("Rewards per Episode (with Viewer)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

# Plot the trajectory of the best episode
plot_trajectory(best_episode_trajectory, env_with_viewer.room_width, env_with_viewer.room_height)
plot_3d_trajectory(best_episode_trajectory, env_with_viewer.room_width, env_with_viewer.room_height, room_depth=300)
