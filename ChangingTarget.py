import subprocess
from main import *

def update_best_episode_commands(
    drone_position, target_position, best_episode_actions, smoothed_commands, room_dimensions
):
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

def main():
    best_episode_actions, best_episode_trajectory, settings = get_training_results(best_episode_reward, best_episode_trajectory)

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

    smoothed_commands = [(0, 50), (3, 40)]  # Example: Replace with real data

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
