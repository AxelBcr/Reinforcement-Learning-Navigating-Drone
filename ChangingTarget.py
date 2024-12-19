import subprocess
import sys
from FunctionsLib import *
from dronecmds import target

q_table = initialize_q_table()

# Get the results of the training, and write it to best_episode_commands.py
best_episode_actions, best_episode_trajectory, settings = get_training_results(env_with_viewer)
writing_commands(best_episode_actions, settings["room_x"], settings["room_y"], settings["room_height"],
                 settings["drone_x"], settings["drone_y"],
                 settings["target_x"], settings["target_y"], settings["target_z"])

print("Running updated best_episode_commands.py...")
subprocess.run([sys.executable, "best_episode_commands.py"])

def main():
    from FunctionsLib import settings

    while subprocess.Popen([sys.executable, "best_episode_commands.py"]):
        #Gets  the last drone position
        last_drone_position = (settings["target_x"], settings["target_y"])
        print(target)
        #Updates the settings
        new_target_position = target

        settings["drone_x"] = last_drone_position[0]
        settings["drone_y"] = last_drone_position[1]

        # Reset the environment for the new target
        reset_environment(new_target_position, env_with_viewer)

        #Runs the training with new positions, and updates the commands
        training_loop(env_with_viewer, settings["num_episodes"], settings["max_steps_per_episode"])

        best_episode_actions, best_episode_trajectory, settings = get_training_results(env_with_viewer)

        writing_commands(best_episode_actions, settings["room_x"], settings["room_y"], settings["room_height"],
                         last_drone_position[0], last_drone_position[1],
                         new_target_position[0], new_target_position[1], new_target_position[2])


        print("Running updated best_episode_commands.py...")
        subprocess.Popen([sys.executable, "best_episode_commands.py"])

        encore = str(input("Do you want to continue? (y/n) : "))


if __name__ == "__main__":
    main()