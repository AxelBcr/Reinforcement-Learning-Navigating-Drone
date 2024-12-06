import subprocess
import sys
from main import get_training_results, settings, writing_commands, createRoom, createDrone, DroneVirtual, np, settings, \
    training_loop

# Inisialization of Best Episode Reward
best_episode_reward = -float('inf')

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

q_table = np.zeros((space_x, space_y, space_z, 6, 100))  # Shape: (x_bins, y_bins, z_bins, directions, distances)

# Training parameters
alpha = 0.05  # Learning rate
gamma = 0.995  # Importance of future rewards
epsilon = 0.98  # Randomness rate
epsilon_decay = 0.92  # Randomness decay rate
epsilon_min = 0.01  # Minimum randomness rate

# Get the results of the training, and write it to best_episode_commands.py
best_episode_actions, best_episode_trajectory, settings = get_training_results(env_with_viewer)
writing_commands(best_episode_actions, settings["room_x"], settings["room_y"], settings["room_height"],
                 settings["drone_x"], settings["drone_y"],
                 settings["target_x"], settings["target_y"], settings["target_z"])

print("Running updated best_episode_commands.py...")
subprocess.run([sys.executable, "best_episode_commands.py"])

def main():
    from main import settings

    #Gets  the last drone position
    last_drone_position = (settings["target_x"], settings["target_y"])

    #Asks for the new target position
    new_target_position = (
        int(input("Enter the x coordinate of the new target: ")),
        int(input("Enter the y coordinate of the new target: ")),
        int(input("Enter the z coordinate of the new target: "))
    )

    #Updates the settings
    settings["target_x"] = new_target_position[0]
    settings["target_y"] = new_target_position[1]
    settings["target_z"] = new_target_position[2]
    settings["drone_x"] = last_drone_position[0]
    settings["drone_y"] = last_drone_position[1]

    #Runs the training with new positions, and updates the commands
    training_loop(env_with_viewer, settings["num_episodes"], settings["max_steps_per_episode"])

    best_episode_actions, best_episode_trajectory, settings = get_training_results(env_with_viewer)

    writing_commands(best_episode_actions, settings["room_x"], settings["room_y"], settings["room_height"],
                     last_drone_position[0], last_drone_position[1],
                     new_target_position[0], new_target_position[1], new_target_position[2])



    print("Settings:", settings)
    print("Best Episode Actions:", best_episode_actions)
    print("Best Episode Trajectory:", best_episode_trajectory)


    print("Running updated best_episode_commands.py...")
    subprocess.run([sys.executable, "best_episode_commands.py"])


if __name__ == "__main__":
    main()
