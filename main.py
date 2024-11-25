import numpy as np
import random
import matplotlib.pyplot as plt
from gym import spaces
from dronecmds import createRoom, createDrone, createTarget
from mpl_toolkits.mplot3d import Axes3D


def create_random_target():
    """
    Create a random target using the `createTarget()` function.
    Fallback mechanism: Place target randomly if `createTarget()` fails.
    """
    # Fallback: Manually place the target at a random position
    print("Fallback: Manually placing target at a random valid location.")
    x = random.uniform(0, 500)
    y = random.uniform(0, 1000)
    z = random.uniform(0, 300)
    print(f"Fallback Target Placed: x={x}, y={y}, z={z}")
    return np.array([x, y, z])



class DroneVirtualGymWithViewer:
    def __init__(self, drone, room, room_size=(500, 1000, 300), max_steps=200):
        # Initialize as before
        self.drone = drone
        self.room = room
        self.room_width, self.room_height, self.room_depth = room_size
        self.target_position = create_random_target()  # Set target once here
        self.radius_detection = 50
        self.max_steps = max_steps
        self.step_count = 0

        # Discretize observation space
        self.state_bins = [
            np.linspace(0, self.room_width, 20),
            np.linspace(0, self.room_height, 20),
            np.linspace(0, self.room_depth, 10),
        ]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.room_width, self.room_height, self.room_depth]),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(6)
        self.viewer = drone.viewer
        self.state = None
        self.prev_distance = float('inf')

    def reset(self):
        """
        Reset the drone to a random position. Target remains fixed.
        """
        # Reset drone position
        self.state = np.array([
            random.uniform(0, self.room_width),
            random.uniform(0, self.room_height),
            random.uniform(0, self.room_depth),
        ])
        self.drone.locate(self.state[0], self.state[1], self.state[2], self.room)

        # Reset variables
        self.visited_states = set()
        self.prev_distance = np.linalg.norm(self.state - self.target_position)
        self.step_count = 0  # Reset step count

        return self.discretize_state(self.state)


    def step(self, action):
        """
        Execute a step in the environment based on the chosen action.
        """
        x, y, z = self.state
        if action == 0 and y < self.room_height:  # Move Up
            y += 20
        elif action == 1 and y > 0:  # Move Down
            y -= 20
        elif action == 2 and x > 0:  # Move Left
            x -= 20
        elif action == 3 and x < self.room_width:  # Move Right
            x += 20
        elif action == 4 and z < self.room_depth:  # Ascend
            z += 20
        elif action == 5 and z > 0:  # Descend
            z -= 20

        self.state = np.array([x, y, z])
        self.drone.locate(self.state[0], self.state[1], self.state[2], self.room)

        # Increment step count
        self.step_count += 1

        # Compute reward
        reward = self.compute_reward(self.state, self.step_count, self.max_steps)
        done = reward >= 1000 or self.step_count >= self.max_steps  # End if target reached or max steps exceeded
        return self.discretize_state(self.state), reward, done, {}

    def compute_reward(self, state, step_count, max_steps):
        """
        Compute the reward for the current state of the drone.
        """
        distance = np.linalg.norm(state - self.target_position)
        reward = 0  # Base reward

        # Check if the drone crashes (goes out of bounds)
        if (
            state[0] < 0 or state[0] > self.room_width or
            state[1] < 0 or state[1] > self.room_height or
            state[2] < 0 or state[2] > self.room_depth
        ):
            return -1000  # Large penalty for crashing

        # Check if the drone has reached the target
        if np.linalg.norm(state - self.target_position) < 10:  # 10 cm tolerance
            # Add reward for reaching the target quickly
            remaining_steps = max_steps - step_count
            return 1000 + remaining_steps * 10  # Large reward for reaching quickly

        # Positive reward for moving closer to the target
        delta_distance = self.prev_distance - distance
        if delta_distance > 0:
            reward += delta_distance * 50  # Reward for getting closer
        else:
            reward -= 50  # Penalty for moving further away

        # Penalty for excessive steps
        if step_count > max_steps:
            reward -= 10  # Penalty for exceeding step limit

        # Penalty for oscillations or revisiting states
        if tuple(state) in self.visited_states:
            reward -= 100  # Penalize oscillating behavior

        self.visited_states.add(tuple(state))
        self.prev_distance = distance

        return reward

    def discretize_state(self, state):
        """
        Discretize the state into bins.
        """
        discretized = []
        for dim, bins in zip(state, self.state_bins):
            discretized.append(np.digitize(dim, bins) - 1)
        return tuple(discretized)



def plot_trajectory(trajectory, room_width, room_height, room_depth):
    """
    Plot the trajectory of the drone in 3D space.
    """
    x, y, z = zip(*trajectory)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, marker='o', label='Trajectory')
    ax.set_title("3D Drone Trajectory - Best Episode")
    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.set_zlabel("Z Position (cm)")
    ax.set_xlim([0, room_width])
    ax.set_ylim([0, room_height])
    ax.set_zlim([0, room_depth])
    ax.legend()
    plt.show()


# Main Code

# Initialize environment
room_description = "(0 0, 500 0, 500 1000, 0 1000, 0 0)"
room_height = 300
room = createRoom(room_description, room_height)
drone = createDrone("DroneVirtual", "ViewerTkMPL")
env_with_viewer = DroneVirtualGymWithViewer(drone, room, room_size=(500, 1000, 300))

# Training
num_episodes = 1000
max_steps_per_episode = 200
alpha = 0.1
gamma = 0.98
epsilon = 1.0
epsilon_decay = 0.92
epsilon_min = 0.01

q_table = np.zeros((20, 20, 10, 6))
episode_rewards = []
best_episode_trajectory = []
best_episode_reward = -float('inf')

for episode in range(num_episodes):
    state = env_with_viewer.reset()
    total_reward = 0
    trajectory = []
    done = False

    for step in range(max_steps_per_episode):
        if np.random.random() < epsilon:
            action = np.random.choice(env_with_viewer.action_space.n)
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _ = env_with_viewer.step(action)
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])

        q_table[state][action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state
        total_reward += reward
        trajectory.append(env_with_viewer.state)

        if done:
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    episode_rewards.append(total_reward)

    if total_reward > best_episode_reward:
        best_episode_reward = total_reward
        best_episode_trajectory = trajectory


# Plot episode rewards as a 2D plot
plt.figure()
plt.plot(episode_rewards)
plt.title("Rewards per Episode (with Viewer)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

# 3D trajectory visualization for the best episode
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Convert the best trajectory to a NumPy array
trajectory = np.array(best_episode_trajectory)

# Plot the trajectory
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Trajectory", color='blue')

# Highlight the start and end points
ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', label="Start Point", s=100)  # Start point
ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color='red', label="End Point", s=100)   # End point

# Add titles and labels
ax.set_title("3D Drone Trajectory - Best Episode")
ax.set_xlabel("X Position (cm)")
ax.set_ylabel("Y Position (cm)")
ax.set_zlabel("Z Position (cm)")
ax.legend()

# Show the plot
plt.show()

