from FunctionsLib import *

#%% Main
if __name__ == "__main__":

    # Initialisation des paramètres globaux
    best_episode_reward = -float('inf')
    best_episode_trajectory = []
    best_episode_actions = []

    # %% Initialize Q-table
    q_table = np.zeros((space_x, space_y, space_z, 6, 100))  # Shape: (x_bins, y_bins, z_bins, directions, distances)
    episode_rewards = []

    training_loop(env_with_viewer, settings["num_episodes"], settings["max_steps_per_episode"])
    get_training_results(env_with_viewer)

    #%%Plot for visualization

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
    ax.set_ylim([settings["room_y"], 0])
    ax.set_xlim([settings["room_x"], 0])
    ax.set_zlim([0, settings["room_height"]])

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

    #Updating best_episode_commands.py
    writing_commands(best_episode_actions, settings["room_x"], settings["room_y"], settings["room_height"],
                     settings["drone_x"], settings["drone_y"],
                     settings["target_x"], settings["target_y"], settings["target_z"]
                     )