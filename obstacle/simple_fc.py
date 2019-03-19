import time

import numpy as np
import airsim
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from models_torch import SimpleFCDQN, ReplayMemory
from environment import ObstacleEnvironment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def action2motion(action, speed=0.25):
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

def compute_reward(env, collision_info):
    d = env.distance_from_nearest_object()
    # linearly interpolate negative reward for having the obstacle too close
    reward = 0
    CRASH_DISTANCE = 0.75
    CLOSE_DISTANCE = 1.5
    NEAR_DISTANCE = 5 
    if d < CRASH_DISTANCE:
        reward += -200
    elif d < CLOSE_DISTANCE:
        reward += -100
    elif d < NEAR_DISTANCE:
        reward += -100 + 100 * (d - CLOSE_DISTANCE) / (NEAR_DISTANCE - CLOSE_DISTANCE)
    
    return reward


if __name__ == '__main__':
    a = airsim.MultirotorClient()
    a.confirmConnection()
    a.enableApiControl(True)
    a.armDisarm(True)
    env = ObstacleEnvironment(a)

    try:
        a.getLidarData()
    except RPCError:
        raise RuntimeError("Couldn't access LIDAR Data. Most likely you haven't"
            "set up the lidar for our drone in settings.py. In this folder you'll"
            "there's a settings.json that you can copy into AirSim's. Usually "
            "settings.json can be found in ~/Documents/AirSim")
    
    for _ in range(2):
        a.reset()
        env.reset()
        a.confirmConnection()
        a.enableApiControl(True)
        a.armDisarm(True)
        
        COMMAND_DELAY = 0.2
        last_command = time.time()
        reward = 0
        for _ in range(10000):
            # Only perform a command after a delay, simulating the amount of time
            # it takes the drone to actually run control and send commands to the
            # motors. Also moving back and forth causes the drone to go vertical
            # a little bit so make sure it hovers at the same height.
            collision_info = a.simGetCollisionInfo()
            if time.time() - last_command > COMMAND_DELAY:
                last_command = time.time()
                x, y = np.random.uniform(-2, 2), np.random.uniform(-2, 2) 
                height = a.getMultirotorState().kinematics_estimated.position.z_val
                STABLE_HEIGHT = -0.5
                z_comp = np.clip(STABLE_HEIGHT - height, -0.5, 0.5)
                print(height)
                a.moveByVelocityAsync(x, y, z_comp, 0.1)
                reward = compute_reward(env, collision_info)
                print("Current reward: %f" % reward)
        
            if env.moving_objects == 0:
                env.move_obstacle_at_drone()
            
            env.tick()
            
            if reward < -150 or collision_info.has_collided:
                # Reset the environment, we've gotten BOOMED
                break
            
            
    env.reset()
    a.reset()

