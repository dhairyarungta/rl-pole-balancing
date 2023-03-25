"""
implementation of the DDQN for cartpole balacing 
N1 network for estimating the predicted values, weights changed every batch update
N2 network for estimating the Q value used for finding the target value
we use two separate networks to prevent the cat chasing it's own tail sitatuion
which happens when we are using the same neural network for finding the predicted 
and target values. This occcurs because every weight update for the loss 
of target and predicted value, we shift the goal post for the target again .
This might lead to instability in the network.

Instead we keep the network weights used find the Q values in the target value same 
for a certain number of iterations of batch update and then copy the weights
formt the network used to predict the values.

N1 network is  called the policy network
N2 network is called the target network

"""
import sys 

import random
import gym
import numpy as np


import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tesorflow.keras import Model 
from collections import deque

from IPython.display import clear_output


#creating the class for the two network 
#using inheritance from keras Model

class Network(Model):
    def __init__(self, state_size: int, action_size : int,):
        super(Network, self).__init__()

        self.layer1 = tf.keras.layers.Dense(hidden_size, activation = 'relu')
        self.layer2 = tf.keras.layers.Dense(hidden_size, activation ='relu')
        self.value = tf.keras.layers.Dense(action_size)



class DDQNAgent:
    def __init__(
            self, 
            env: gym.Env,
            batch_size: int
            target_update:int
            ):

        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.batch_size = batch_size
        self.lr = 0.001
        self.target_update = target_update
        self.gamma = 0.99
        self.dqn_policy = Network (self.state_size, self.action_size)
        self.dqn_target = Network(self.state_size, self.action_size)
        self._target_hard_update()
    
    def get_action(self, state, epsilon):
        if np.random.rand()<=epsilon:
            action = np.random.choice(self.action_size)
        else :
            q_value = self.dqn_policy(tf.convert_to_tensor([state], dtype = tf.float32))[0]
            action = np.argmax(q_value)
        
        return action
    
    def _target_hard_update(self):
        self.dqn_target.set_weights(self.dqn.get_weights())
    
    def append_sample(self, state, action , reward, next_state, done):
        self.memory.append (state,action, reward, next_state, done)
    

        
    def train_step(self):
        mini_batch  = random.sample(self.memory, self.batch_size)
        #memory deque has as each entry (state, action , reward, next_state, done)
        states = [i[0] for i in mini_batch]
        actions = [i[1] for i in mini_batch]
        rewards = [i[2] for i in mini_batch]
        next_states = [i[3] for i in mini_batch]
        dones = [i[4] for i in mini_batch]

        dqn_variable = self.dqn_policy.trainable_variables 
        with tf.GradientTape as tape :
            states = tf.convert_to_tensor(np.vstack(states), dtype = tf.float32)
            actions = tf.convert_to_tensor(actions, dtype = tf.int32)
            rewards  = tf.convert_to_tensor(rewards, dtype = tf.float32)
            next_states = tf.convert_to_tensor(np.vstack(next_states), dtype = tf.float32)
            dones = tf.convert_to_tensor(dones, dtype = tf.float32)


            cur_Qs = self.dqn_policy(states)
            main_value = tf.reduce_sum(tf.one_hot(actions,self.action_size)*cur_Qs, axis = 1)

            next_Q_targs = self.dqn_target(next_states)
            next_action = tf.argmax(next_Q_targs, axis = 1)


            mask = 1 - dones
            target_value = rewards + self.gama
