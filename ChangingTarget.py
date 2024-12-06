import subprocess
from main import *

writing_commands()

# Initialisation des paramètres globaux
best_episode_reward = -float('inf')
best_episode_trajectory = []
best_episode_actions = []

# %% Main code & Hyperparameters
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

def update_best_episode_commands(
    drone_position, target_position, best_episode_actions, smoothed_commands, room_dimensions
):
    # %% Convert actions to commands
    actions_to_commands = {
        0: "forward",
        1: "backward",
        2: "goLeft",
        3: "goRight",
        4: "goUp",
        5: "goDown"
    }

    drone_x, drone_y, initial_heading = drone_position[0], drone_position[1], 90
    target_x, target_y, target_z = target_position
    room_x, room_y, room_height = room_dimensions

    with open("best_episode_commands.py", "w") as f:
        f.write("from dronecmds import *\n\n")
        f.write("raw_commands =[\n")
        for direction, distance in best_episode_actions:
            command = f"{actions_to_commands[direction]}({distance + 1})"
            f.write(f"    #{command},\n")
        f.write("]\n")

        f.write("def replay_best_episode():\n")
        f.write(f"    locate({drone_x}, {drone_y}, {initial_heading})\n")
        f.write(f"    takeOff()\n")
        for direction, distance in smoothed_commands:
            if distance > 20:
                f.write(f"    {actions_to_commands[direction]}({distance})\n")
        f.write("    land()\n")
        room_description = f"(0 0, {room_x} 0, {room_x} {room_y}, 0 {room_y}, 0 0)"
        f.write(f"createRoom('{room_description}', {room_height})\n")
        f.write(f"createTargetIn({target_x - 1}, {target_y - 1}, {target_z - 1}, "
                f"{target_x + 1}, {target_y + 1}, {target_z + 1})\n")
        f.write("createDrone(DRONE_VIRTUAL, VIEWER_TKMPL, progfunc=replay_best_episode)\n")


def get_smoothed_commands(best_episode_actions):
    smoothed_commands = []
    for action in best_episode_actions:
        direction, distance = action
        smoothed_commands.append((direction, distance))
    return smoothed_commands


def main():

    best_episode_actions, best_episode_trajectory, settings = get_training_results(env_with_viewer)
    print("Settings:", settings)
    print("Best Episode Actions:", best_episode_actions)
    print("Best Episode Trajectory:", best_episode_trajectory)

    room_dimensions = (settings["room_x"], settings["room_y"], settings["room_height"])
    last_drone_position = (settings["target_x"], settings["target_y"], settings["target_z"])

    new_target_position = (
        int(input("Enter the x coordinate of the new target: ")),
        int(input("Enter the y coordinate of the new target: ")),
        int(input("Enter the z coordinate of the new target: "))
    )

    smoothed_commands = get_smoothed_commands(best_episode_actions)

    update_best_episode_commands(
        last_drone_position,
        new_target_position,
        best_episode_actions,
        smoothed_commands,
        room_dimensions
    )

    print("Running updated best_episode_commands.py...")
    subprocess.run(["python3", "best_episode_commands.py"])

if __name__ == "__main__":
    main()
