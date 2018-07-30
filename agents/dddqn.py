################################################################################
#
# Author: Asher Elmquist
#
# This is a double dueling deep q network for reinforcement learning
#
#
################################################################################

import time
import random
import sys
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD, Adam


#hyper parameters
num_filters = 8
filter_size = (8,8)
learning_rate = 0.001
num_dense = 64
mem_size = 200000
max_epsilon = 1.0
min_epsilon = 0.01
annealing_steps = 1000
num_epochs = 1
batch_size = 50
discout_rate = .99

class DDDQNAgent:
    #constructor
    def __init__(self, state_shape, action_size, train=True):
        self.state_shape=state_shape
        self.action_size=action_size
        self.train = train

        self.memory = deque(maxlen=mem_size)
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = (max_epsilon-min_epsilon)/annealing_steps
        self.gamma = discout_rate

        self.main_model = self.network()
        self.target_model = self.network()

    #network
    def network(self):
        model = Sequential();
        model.add(Conv2D(num_filters, filter_size, padding='same', activation='relu', input_shape=self.state_shape))
        model.add(Conv2D(num_filters, filter_size, padding='same', activation='relu'))
        # model.add(Conv2D(num_filters, filter_size, padding='same', activation='relu'))
        # model.add(Conv2D(num_filters, filter_size, padding='same', activation='relu'))

        #this is where it should split

        #continue on with no split
        model.add(Flatten())
        model.add(Dense(num_dense, activation='relu'))
        model.add(Dense(num_dense, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=SGD(lr=learning_rate))

        return model

    #load models
    def load(self, main_name, target_name=""):
        self.main_model.load_weights(main_name)
        if(target_name != ""):
            self.target_model.load_weights(main_name)

    #save the model weights
    def save(self, main_name, target_name=""):
        self.main_model.save_weights(main_name)
        if(target_name != ""):
            self.target_model.save_weights(main_name)

    #choose an action
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            act_value = random.randrange(self.action_size)
            return act_value
        state = np.reshape(state, [1,self.state_shape[0],self.state_shape[1],self.state_shape[2]])
        act_value = np.argmax(self.main_model.predict(state)[0])
        return act_value

    #replay based on samples from memory
    def replay(self):
        if(len(self.memory) < batch_size):
            return
        #sample the memory
        minibatch = random.sample(self.memory, batch_size)

        for sample in minibatch:
            state, action, reward, next_state, done = sample
            #assume the target is exactly what we think for now
            target = self.main_model.predict(state)

            #if we are done, we know exactly the reward so modify reward output
            if(done):
                target[0][action] = reward

            #if sample is not from the end, use a predicted, discounted future reward
            else:
                future_reward = np.amax(self.target_model.predict(next_state)[0])
                target[0][action] = reward + self.gamma*future_reward

            #fit the main model to the new reward
            self.main_model.fit(state, target, epochs=num_epochs, verbose=0)

            #bring the target_model a bit closer to the main model
            self.target_model.fit(state, target, epochs=num_epochs, verbose=0)

        #update epsilon
        self.epsilon -= self.epsilon_decay
        if(self.epsilon < self.min_epsilon): self.epsilon=self.min_epsilon

    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1,self.state_shape[0],self.state_shape[1],self.state_shape[2]])
        next_state = np.reshape(next_state, [1,self.state_shape[0],self.state_shape[1],self.state_shape[2]])
        experience = (state, action, reward, next_state, done)
        if len(self.memory) + 1 >= mem_size:
            self.memory.pop()
        self.memory.appendleft(experience)












#END OF FILE
