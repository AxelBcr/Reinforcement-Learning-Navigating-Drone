import numpy as np
import random
import matplotlib.pyplot as plt
from gym import spaces
from dronecmds import createRoom, createDrone, createTarget
from mpl_toolkits.mplot3d import Axes3D

target_x = int(input("Enter the x coordinate of the target: "))
target_y = int(input("Enter the y coordinate of the target: "))
target_z = int(input("Enter the z coordinate of the target: "))

drone_x = int(input("Enter the x coordinate of the drone: "))
drone_y = int(input("Enter the y coordinate of the drone: "))

#%% DroneVirtual, Compute_Reward, Step, Reset, Target, Room
class DroneVirtual:
    def __init__(self, drone, room, room_size=(500, 1000, 300), max_steps=200):
        self.drone = drone
        self.room = room
        self.room_width, self.room_depth, self.room_height = room_size
        self.target_position = self.create_target()  # Set target once here
        self.radius_detection = 50
        self.max_steps = max_steps
        self.step_count = 0

        # Discretize observation space
        self.state_bins = [
            np.linspace(0, self.room_width, 20),
            np.linspace(0, self.room_depth, 20),
            np.linspace(0, self.room_height, 10),
        ]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.room_width, self.room_height, self.room_height]),
            dtype=np.float32,
        )
        # Action space: (direction, distance)
        self.action_space = spaces.Tuple((
            spaces.Discrete(6),  # 6 directions
            spaces.Discrete(50, start=20)  # Distance from 20 to 50 cm
        ))
        self.viewer = drone.viewer
        self.state = None
        self.prev_distance = float('inf')

    def create_target(self):
        """
        Create a random target within the room.
        """
        return np.array([target_x, target_y, target_z])

    def reset(self):
        """
        Reset the drone to a fixed position. Target remains fixed.
        """
        # Fixed drone initial position
        self.state = np.array([drone_x, drone_y, 80])  # Replace with your desired coordinates
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
        distance += 1  # Convert to 21-50 cm range for movement

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
        if distance <= 5:  # Within 2 cm of the target
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
        discretized_state = tuple(map(int, state // 10))  # Discretize state to avoid floating-point issues
        if discretized_state in self.visited_states:
            reward -= 100  # Penalty for revisiting a state
        self.visited_states.add(discretized_state)

        # Small penalty for each step to encourage efficiency
        reward -= 0.1 * step_count  # Scaled penalty based on step count

        # Update the previous distance for the next step
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


#%% Main code & Hyperparameters
room_description = "(0 0, 500 0, 500 1000, 0 1000, 0 0)"
room_height = 300
room = createRoom(room_description, room_height)
drone = createDrone("DroneVirtual", "ViewerTkMPL")
env_with_viewer = DroneVirtual(drone, room, room_size=(500, 1000, 300))

# Training parameters
num_episodes = 2000
max_steps_per_episode = 500
alpha = 0.09
gamma = 0.98
epsilon = 1.0
epsilon_decay = 0.96
epsilon_min = 0.01

#%% Initialize Q-table
q_table = np.zeros((20, 20, 10, 6, 50))  # Adding distance dimension
episode_rewards = []
best_episode_trajectory = []
best_episode_actions = []
best_episode_reward = -float('inf')

for episode in range(num_episodes):
    state = env_with_viewer.reset()
    total_reward = 0
    trajectory = []
    trajectory_actions = []
    done = False

    for step in range(max_steps_per_episode):
        if np.random.random() < epsilon:
            # Random exploration
            direction = np.random.choice(6)
            distance = np.random.choice(50)  # Valid range is 0-49 for Q-table
            action = (direction, distance)
        else:
            # Exploitation: choose the best action
            action = np.unravel_index(np.argmax(q_table[state]), q_table[state].shape)

        # Perform the action and observe the result
        next_state, reward, done, _ = env_with_viewer.step(action)
        old_value = q_table[state][action[0]][action[1]]
        next_max = np.max(q_table[next_state])

        # Update Q-value
        q_table[state][action[0]][action[1]] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state
        total_reward += reward
        trajectory.append(env_with_viewer.state)
        trajectory_actions.append(action)

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    episode_rewards.append(total_reward)

    # Track the best episode
    if total_reward > best_episode_reward:
        best_episode_reward = total_reward
        best_episode_trajectory = trajectory
        best_episode_actions = trajectory_actions

#%%Plot

# Plot the episode rewards
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label="Episode Reward")
plt.title("Training Rewards Per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.show()


# 3D trajectory visualization for the best episode
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Convert trajectory to a NumPy array for easier slicing
trajectory = np.array(best_episode_trajectory)

# Plot the trajectory
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Trajectory", color='blue')

# Highlight start and end points
ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', label="Start Point", s=100)
ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color='red', label="End Point", s=100)

# Plot the target position
target_x, target_y, target_z = env_with_viewer.target_position  # Replace with actual target position
ax.scatter(target_x, target_y, target_z, color='blue', label="Target", s=100)  # Add marker for target
# Annotate the target position
ax.text(target_x, target_y, target_z, 'Target', color='blue', fontsize=12)


# Fix axis labels and limits
ax.set_title("3D Drone Trajectory - Best Episode")
ax.set_xlabel("X Position (cm)")
ax.set_ylabel("Y Position (cm)")
ax.set_zlabel("Z Position (cm)")

# Ensure axis limits match the room dimensions
room_width = 1000  # Replace with your simulation room width
room_depth = 500  # Replace with your simulation room depth
room_height = 300   # Replace with your simulation room height
ax.set_ylim([room_width, 0])
ax.set_xlim([room_depth, 0])
ax.set_zlim([0, room_height])

# Add legend
ax.legend()

# Show the plot
plt.show()
#%% Convert actions to commands (wrong thing about coords)
actions_to_commands = {
    0: "forward",
    1: "backward",
    2: "goLeft",
    3: "goRight",
    4: "goUp",
    5: "goDown"
}

commands = []
for direction, distance in best_episode_actions:
    command = f"{actions_to_commands[direction]}({distance + 1})"
    commands.append(command)

# Extract positions from the environment
initial_heading = 90  # Modify based on actual logic

# Save commands to a Python file
with open("best_episode_commands.py", "w") as f:
    f.write("from dronecmds import *\n\n")
    f.write("def replay_best_episode():\n")

    # Save initial drone position and heading
    f.write(f"    locate({drone_x}, {drone_y}, {initial_heading})\n")

    # Add movement commands
    f.write(f"    takeOff()\n")
    for direction, distance in best_episode_actions:
        command = f"{actions_to_commands[direction]}({distance + 1})"
        f.write(f"    {command}\n")
    f.write("    land()\n")

    # Room setup
    room_description = "(0 0, 500 0, 500 1000, 0 1000, 0 0)"
    room_height = 300
    f.write(f"createRoom('{room_description}', {room_height})\n")

    # Target position
    f.write(f"createTargetIn({target_x - 1}, {target_y - 1}, {target_z - 1}, "
            f"{target_x + 1}, {target_y + 1}, {target_z + 1})\n")

    # Save drone creation
    f.write("createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)\n")

