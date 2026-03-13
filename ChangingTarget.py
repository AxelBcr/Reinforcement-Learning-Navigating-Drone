import subprocess
import sys

from FunctionsLib import (
    initialize_q_table, get_training_results, writing_commands,
    training_loop, reset_environment, env_with_viewer
)

q_table = initialize_q_table()

# Get the results of the training, and write it to best_episode_commands.py
best_episode_actions, best_episode_trajectory, settings = get_training_results(env_with_viewer)
writing_commands(best_episode_actions, settings["room_x"], settings["room_y"], settings["room_height"],
                 settings["drone_x"], settings["drone_y"],
                 settings["target_x"], settings["target_y"], settings["target_z"])

print("Running updated best_episode_commands.py...")
result = subprocess.run([sys.executable, "best_episode_commands.py"])
if result.returncode != 0:
    print(f"Warning: best_episode_commands.py exited with code {result.returncode}")


def main():
    global settings
    encore = "y"

    while encore == "y":
        # Gets the last drone position
        last_drone_position = (settings["target_x"], settings["target_y"])

        # Asks for the new target position
        new_target_x = int(input("Enter the x coordinate of the new target: "))
        while new_target_x <= 0 or new_target_x >= settings["room_x"] - 1:
            new_target_x = int(input(
                "Invalid x coordinate. Please enter a value strictly between 0 and {}: ".format(
                    settings["room_x"] - 1)))

        new_target_y = int(input("Enter the y coordinate of the new target: "))
        while new_target_y <= 0 or new_target_y >= settings["room_y"] - 1:
            new_target_y = int(input(
                "Invalid y coordinate. Please enter a value strictly between 0 and {}: ".format(
                    settings["room_y"] - 1)))

        new_target_z = int(input("Enter the z coordinate of the new target: "))
        while new_target_z <= 0 or new_target_z >= settings["room_height"] - 1:
            new_target_z = int(input(
                "Invalid z coordinate. Please enter a value strictly between 0 and {}: ".format(
                    settings["room_height"] - 1)))

        new_target_position = (new_target_x, new_target_y, new_target_z)

        # Updates the settings
        settings["target_x"] = new_target_position[0]
        settings["target_y"] = new_target_position[1]
        settings["target_z"] = new_target_position[2]

        settings["drone_x"] = last_drone_position[0]
        settings["drone_y"] = last_drone_position[1]

        # Reset the environment for the new target
        reset_environment(new_target_position, env_with_viewer)

        # Runs the training with new positions, and updates the commands
        try:
            best_episode_actions, best_episode_trajectory, settings = get_training_results(env_with_viewer)
        except Exception as e:
            print(f"Error during training: {e}")
            encore = str(input("Do you want to continue? (y/n) : ")).lower()
            continue

        writing_commands(best_episode_actions, settings["room_x"], settings["room_y"], settings["room_height"],
                         last_drone_position[0], last_drone_position[1],
                         new_target_position[0], new_target_position[1], new_target_position[2])

        print("Running updated best_episode_commands.py...")
        result = subprocess.run([sys.executable, "best_episode_commands.py"])
        if result.returncode != 0:
            print(f"Warning: best_episode_commands.py exited with code {result.returncode}")

        encore = str(input("Do you want to continue? (y/n) : ")).lower()


if __name__ == "__main__":
    main()
