import time
import random

import numpy as np
import airsim
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from models_torch import SimpleFCDQN, ReplayMemory, Transition
from environment import ObstacleEnvironment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def optimize_model():
    print('Optimizing.')
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state).float()
    action_batch = torch.cat(batch.action).long()
    reward_batch = torch.cat(batch.reward).float()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def select_action(state, steps_done=1):
    """Turns an action from our fully connected net and turns it into movement,
    with some probability of exporation with a random action"""
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    np.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.tensor(state).float().to(device)
            return int(policy_net(state).argmax().cpu().numpy())
    else:
        return int(np.random.choice(range(5)))


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
    policy_net = SimpleFCDQN(6, 10, 5).to(device)
    target_net = SimpleFCDQN(6, 10, 5).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)
    steps_done = 0


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
        COMMAND_DELAY = 0.2
        HOVER_POINT = 0
        last_command = time.time()
        start_time = time.time()
        while True:
            if time.time() - last_command > COMMAND_DELAY:
                reward = env.compute_reward()
                reward = torch.tensor([reward], device=device).float()
                next_state = env.lidar_distances()

                # Don't record down the reward action pair unless we actually have
                # performed an action! Reward should be delayed from the action
                if action:
                    state_torch = torch.tensor(state).float().to(device).view(1, 6)
                    next_state_torch = torch.tensor(state).float().to(device).view(1, 6)
                    action_torch = torch.tensor(action).float().view(1, 1).to(device)
                    memory.push(state_torch, action_torch, next_state_torch, reward)
                
                state = next_state
                action = select_action(state, steps_done=steps_done)
                x, y, z = action2motion(action, speed=2.0)
                # Do a little bit of compensation if drone goes too high
                z_pos = a.getMultirotorState().kinematics_estimated.position.z_val
                correction_z = np.clip(HOVER_POINT - z_pos, -0.5, 0.5)
                a.moveByVelocityAsync(x, y, correction_z, 0.2)


                # Perform one step of the optimization (on the target network)
                # with some probability
                if random.random() < 0.1:
                    optimize_model()
                    steps_done += 1

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

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save the model every 10 epochs
        if i_episode % 10 == 0:
            torch.save(policy_net, 'simple_dqn_6_10_5-{}.pth'.format(i_episode))
