import sys
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


import os
import tensorflow as tf
from tensorflow.python.client import device_lib
# disable deprecated warnings
from tensorflow.python.util import deprecation


EPISODES = 300

class DQNAgent : 
    def __init__(self, state_size, action_size, ddqn_flag = True):
        self.render = False
        self.load_model = False

        self.state_size = state_size
        self.actino_size = action_size
        self.discount_factor = 0.9
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.batch_size = 24
        self.train_start= 1000

        self.memory = deque(maxlen=2000)
        self.model = self.build_model()
        self.target_model =self.build_model()

        self.ddqn = ddqn_flag 
        

        self.target_model.set_weights (self.model.get_weights())

    def build_model(self):
        model = Sequential()
        model.add (Dense(24, input_dim = self.state_size, activation = 'relu', kernel_initializer='he_uniform'))
        model.add (Dense(24, input_dim = self.state_size, activation = 'relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation = 'linear',kernel_initializer='he_uniform' ))
        model.summary()
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
        return model
    
    def update_target_model (self, tau = 0.01):
        self.target_model.set_weights(self.model.get_weights())

    
    def save_weights(self,filename):
        self.model.save_weights(filename)
    
    def load_weights(self, filename):
        self.model.load_weights(filename)
    
    def get_action(self,state):
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.actino_size)
        
        else :
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state,action, reward, next_state,done))
    
    def get_target_q_value(self, next_state):
        if self.ddqn:
            action = np.argmax(self.model.predict(next_state)[0])
            max_q_value = self.target_model.predict(next_state)[0][action]
        
        else:
            max_q_value = 0
        return max_q_value
    

    def experience_replay (self):
        if len(self.memory)<self.train_start:
            return 
        batch_size = min(self.batch_size, len (self.memory))
        mini_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [],[]

        for state,action , reward,next_state, done in mini_batch:
            q_values_cs = self.model.predict(state)
            max_q_value_ns = self.get_target_q_value(next_state)
            if done :
                q_values_cs[0][action]= reward
            else :
                q_values_cs[0][action] =reward + (self.discount_factor*max_q_value_ns)
                state_batch.append(state[0])
                q_values_batch.append(q_values_cs[0])
        
        self.model.fit(np.array(state_batch),np.array(q_values_batch),batch_size=batch_size,epochs = 1, verbose = 0)
        self.update_epsilon()
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *=self.epsilon_decay
        
    

            
        


if __name__ == "__main__":
    env = gym.make('CartPole-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    last100Scores = deque(maxlen=100)
    stepCounter = 0
    max_steps =1
    frames  =[]
    Scores, episode = [],[]
    avg100Scores = []
    avgScores =[]
    for e in range (EPISODES):
        done  = False
        t = 0
        state = env.reset()
        state = np.reshape(state, [1,state_size])
        while not done :
            action = agent.get_action(state)
            next_state , reward, done , info = env.step(action)
            next_state = np.reshape(next_state, [1,state_size])
            reward= reward if not done else -100

            agent.append_sample(state, action, reward, next_state,done )
            agent.experience_replay()

            t+=1
            stepCounter+=1
            state = next_state

            if done :
                agent.update_target_model()
                Scores.append(t)
                avgScores.append(np.mean(Scores))
                last100Scores.append(t)
                avg100Scores.append(np.mean(last100Scores))

                episode.append(e)
                print("episode: {}, score: {}, memory length: {}, epsilon: {}".format(e, t, len(agent.memory), agent.epsilon))

                # with open()

                break
            
    env.close()

