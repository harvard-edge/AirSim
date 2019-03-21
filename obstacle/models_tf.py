
from __future__ import print_function
import gym
import tensorflow as tf
import numpy as np
from itertools import count
import random
import os
import argparse
from util import ReplayMemory


SEED = 420
random.seed(420)
np.random.seed(420)


class SimpleTFDQN(object):
    def __init__(self, input_size, output_size):

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # A few starter hyperparameters
        self.batch_size = 32
        self.gamma = 0.99
        self.learning_rate = 0.0005
        self.memory_cap = 10000
        # If using e-greedy exploration
        self.eps_start = 0.95
        self.epsilon = self.eps_start
        self.eps_end = 0.05
        self.eps_decay = 40000 # in episodes
        # If using a target network
        self.clone_steps = 500
        self.input_size = input_size 
        # memory
        self.replay_memory = ReplayMemory(100000)
        # Perhaps you want to have some samples in the memory before starting to train?
        self.min_replay_size = 10000
        self.memory = np.zeros((self.memory_cap, self.input_size * 2 + 2))
        self.actions = output_size 

        # Make sure to have a training and target network so we can calculate
        # approximate Q values
        self.observation_input = tf.placeholder(tf.float32, [None, self.input_size])
        self.observation_input_target = tf.placeholder(tf.float32, [None, self.input_size])
        self.q_target = tf.placeholder(tf.float32, [None, self.actions], name='Q_target')  # for calculating loss
        self.train_network = self.build_model(self.observation_input)
        self.target_network = self.build_model(self.observation_input_target,'target')

        t_params = tf.get_collection('target_params')
        e_params = tf.get_collection('train_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # define your update operations here...
        self.loss = tf.reduce_mean(tf.losses.huber_loss(self.q_target, self.train_network))
        self.reducer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


        self.num_episodes = 0
        self.num_steps = 0
        self.cost_his = []

        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, observation_input, scope='train'):
        """Make a simple fully connected model"""
        FC1_SIZE = 20
        FC2_SIZE = 20
        layer1_nodes, layer2_nodes = FC1_SIZE, FC2_SIZE

        with tf.variable_scope(scope):
            namespace = [scope+'_params', tf.GraphKeys.GLOBAL_VARIABLES]
            weights = tf.random_normal_initializer(0., 0.3)
            biases = tf.constant_initializer(0.1)

            # first hidden layer
            w1 = tf.get_variable('w1', [self.input_size, layer1_nodes], initializer=weights, collections=namespace)
            b1 = tf.get_variable('b1', [1, layer1_nodes], initializer=biases, collections=namespace)
            l1 = tf.nn.relu(tf.matmul(observation_input, w1) + b1)

            # second hidden layer
            w2 = tf.get_variable('w2', [layer1_nodes, layer2_nodes], initializer=weights, collections=namespace)
            b2 = tf.get_variable('b2', [1, layer2_nodes], initializer=biases, collections=namespace)
            l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # output layer
            w3 = tf.get_variable('w3', [layer2_nodes, self.actions], initializer=weights, collections=namespace)
            b3 = tf.get_variable('b3', [1, self.actions], initializer=biases, collections=namespace)

        return tf.matmul(l2, w3) + b3


    def select_action(self, obs, evaluation_mode=False):
        """
        TODO: Select an action given an observation using your model. This
        should include any exploration strategy you wish to implement
        If evaluation_mode=True, then this function should behave as if training is
        finished. This may be reducing exploration, etc.
        Currently returns a random action.
        """
        if np.random.uniform() > self.eps_start or evaluation_mode:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.train_network, feed_dict={self.observation_input: obs[np.newaxis, :]})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.actions)

        return action


    def update(self):
        """Updates the network using the replay buffer"""

        if self.num_steps % self.clone_steps == 0:
            self.sess.run(self.replace_target_op)

        if self.num_steps > self.memory_cap:
            sample_index = np.random.choice(self.memory_cap, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.num_steps, size=self.batch_size)
        batch_data = self.memory[sample_index, :]

        # Feed through both target and training network
        target_q, train_q = self.sess.run(
            [self.target_network, self.train_network],
            feed_dict={
                self.observation_input_target: batch_data[:, -self.input_size:],  # fixed params
                self.observation_input: batch_data[:, :self.input_size],  # newest params
            })

        fixed_target = train_q.copy()

        i_train = batch_data[:, self.input_size].astype(int)
        reward = batch_data[:, self.input_size + 1]

        fixed_target[np.arange(self.batch_size, dtype=np.int32), i_train] = reward + self.gamma * np.max(target_q, axis=1)

        _,cost = self.sess.run([self.reducer, self.loss],
                      feed_dict={self.observation_input: batch_data[:, :self.input_size],
                                 self.q_target: fixed_target})
        self.cost_his.append(cost)
