import time
import random

import numpy as np
import airsim

from models_tf import SimpleTFDQN
from util import ReplayMemory
from environment import ObstacleEnvironment


def action2motion(action, speed=2.0):
    """Turns an action from our fully connected net and turns it into movement.
    Assumes discrete actions space as an integer. Turns it into X, Y, Z
    velocities corresponding to our specific drone."""
    action_dict = {
        0: (0, 0, 0),
        1: (speed, 0, 0),
        2: (0, speed, 0),
        3: (-speed, 0, 0),
        4: (0, -speed, 0),
    }
    try:
        return action_dict[action]
    except KeyError:
        raise RuntimeError("Could not convert discrete action into movement.")


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


if __name__ == '__main__':
    a = airsim.MultirotorClient()
    a.confirmConnection()
    a.enableApiControl(True)
    a.armDisarm(True)
    env = ObstacleEnvironment(a)

    try:
        a.getLidarData()
    except Exception as err:
        print(str(err))
        raise RuntimeError("Couldn't access LIDAR Data. Most likely you haven't"
            "set up the lidar for our drone in settings.py. In this folder you'll"
            "there's a settings.json that you can copy into AirSim's. Usually "
            "settings.json can be found in ~/Documents/AirSim")
    
    # Create both network used for policy and one temporary copy
    policy = SimpleTFDQN(6, 5)

    num_episodes = 300
    episode_durations = []
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        a.reset()
        env.reset()
        a.confirmConnection()
        a.enableApiControl(True)
        a.armDisarm(True)


        reward = None  # initialize the reward to nothing at first
        action = None
        state = env.lidar_distances()
        COMMAND_DELAY = 0.05
        HOVER_POINT = 0
        last_command = time.time()
        start_time = time.time()
        while True:
            if time.time() - last_command > COMMAND_DELAY:
                reward = env.compute_reward()
                next_state = env.lidar_distances()
                # Don't record down the reward action pair unless we actually have
                # performed an action! Reward should be delayed from the action
                if action:
                    policy.memory[policy.num_steps % policy.memory_cap, :] = \
                        np.hstack((state, [action, reward], next_state))
                
                state = next_state
                action = policy.select_action(state, evaluation_mode=False)
                if policy.num_steps > 5000:
                    policy.update()

                # Actually move the drone here
                x, y, z = action2motion(action, speed=2.0)
                # Do a little bit of compensation if drone goes too high
                z_pos = a.getMultirotorState().kinematics_estimated.position.z_val
                correction_z = np.clip(HOVER_POINT - z_pos, -0.5, 0.5)
                a.moveByVelocityAsync(x, y, correction_z, 0.2)
                last_command = time.time()

            if env.moving_objects == 0:
                env.move_obstacle_at_drone(speed=0.8)
            env.tick()

            # Kill if we have collided and restart
            collision_info = a.simGetCollisionInfo()
            reward_too_low = reward and reward < -150
            done = reward_too_low or collision_info.has_collided 
            if done:
                episode_durations.append(time.time() - start_time)
                break
        policy.eps_start -= 0.00225
        policy.num_episodes += 1

        if i_episode % 20 == 0 and i_episode > 0:
            policy.saver.save(policy.sess, "dqn-model-tf", i_episode)