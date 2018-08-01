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
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Conv2D, MaxPool2D, Flatten, Lambda, Concatenate
from keras.optimizers import SGD, Adam
from keras import losses
from keras import backend as K


#hyper parameters
num_filters = 32
filter_size = 4
learning_rate = 0.0001
num_dense = 256
mem_size = 100000
max_epsilon = 1.0
min_epsilon = 0.1
annealing_steps = 10000
batch_size = 200
num_batches = 1
discout_rate = .99
target_step = 0.001

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
        # self.update_target()

    #network
    def network(self):
        #sequential model
        # model = Sequential();
        # model.add(Conv2D(num_filters, filter_size, padding='same', activation='relu', input_shape=self.state_shape))
        # model.add(Conv2D(num_filters, filter_size, padding='same', activation='relu'))
        # model.add(Conv2D(num_filters, filter_size, padding='same', activation='relu'))
        # model.add(Conv2D(num_filters, filter_size, padding='same', activation='relu'))
        #
        # #this is where it should split
        #
        # #continue on with no split
        # model.add(Flatten())
        # model.add(Dense(num_dense, activation='relu'))
        # model.add(Dense(num_dense, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=SGD(lr=learning_rate))
        # model.compile(loss=self.huber_loss, optimizer=Adam(lr=learning_rate))

        #dueling model
        input = Input(shape=self.state_shape)
        conv0 = Conv2D(num_filters, 2*filter_size, strides=(4,4), padding='valid', activation='relu')(input)
        conv1 = Conv2D(2*num_filters, filter_size, strides=(2,2), padding='valid', activation='relu')(conv0)
        conv2 = Conv2D(2*num_filters, filter_size, strides=(1,1), padding='valid', activation='relu')(conv1)
        flatten = Flatten()(conv2)
        dense0 = Dense(num_dense, activation='relu')(flatten)
        dense1 = Dense(num_dense, activation='relu')(dense0)

        out = Dense(self.action_size + 1,activation='linear')(dense1) # add 1 more unit for value

        # out = value + avantage - mean(advantage)
        output = Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True, axis=1))(out)

        model = Model(input, output)

        model.summary()

        # model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
        model.compile(loss=losses.mean_absolute_error, optimizer=Adam(lr=learning_rate))

        return model

    def huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def update_target(self):
        main_weights = self.main_model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(main_weights)):
            target_weights[i] = target_step*main_weights[i] + (1-target_step)*target_weights[i]
        self.target_model.set_weights(target_weights)

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
        loss = -1
        if(len(self.memory) >= batch_size):
            #sample the memory
            minibatch = random.sample(self.memory, batch_size)
            # minibatch_list = minibatch
            # minibatch = np.asarray(minibatch)

            # print("minibatch shape",minibatch.shape)
            # print("minibatch type",type(minibatch))
            # print("shape of states:",minibatch[:,0].shape)
            states = []
            actions = np.array([]).astype(int)
            rewards = np.array([])
            next_states = []
            dones = np.array([]).astype(int)
            for m in minibatch:
                states.append(m[0])
                actions = np.append(actions,m[1])
                rewards = np.append(rewards,m[2])
                next_states.append(m[3])
                dones = np.append(dones,m[4])

            states = np.asarray(states)
            next_states = np.asarray(next_states)

            # print("States shape:",np.asarray(states).shape)
            # print("actions shape:",actions.shape)
            # print("rewards shape:",rewards.shape)
            # print("next_states shape:",np.asarray(next_states).shape)
            # print("dones shape:",dones.shape)

            # exit(1)
            # #do batch training
            # states = np.reshape(minibatch[:,0],[len(minibatch_list), minibatch[0,0].shape[0],minibatch[0,0].shape[1],minibatch[0,0].shape[2]])
            # print("State shape:",states.shape)

            # exit(1)
            #
            # actions = minibatch[:,1]
            # rewards = minibatch[:,2]
            # next_states = minibatch[:,3]
            # dones = minibatch[:,4]

            # print("next states shape:",next_states.shape)
            # print("next states:",next_states)

            # actions_newstate = np.argmax(self.main_model.predict(next_states),axis=1)
            actions_newstate = np.argmax(self.main_model.predict(next_states),axis=1)
            target_qvalues_newstate = self.target_model.predict(next_states)
            double_q = target_qvalues_newstate[range(target_qvalues_newstate.shape[0]), actions_newstate]

            done_multiplier = 1-dones   #0 if done, 1 if not
            target_q = rewards + self.gamma*double_q+done_multiplier


            q_values = self.main_model.predict(states)
            for i in range(q_values.shape[0]):
                q_values[i, int(actions[i])] = target_q[i]

            loss = self.main_model.train_on_batch(states,q_values)



            # for sample in minibatch:
            #     state, action, reward, next_state, done = sample
            #     #assume the target is exactly what we think for now
            #     target = self.main_model.predict(state)
            #
            #     #if we are done, we know exactly the reward so modify reward output
            #     if(done):
            #         target[0][action] = reward
            #
            #     #if sample is not from the end, use a predicted, discounted future reward
            #     else:
            #         future_reward = np.amax(self.target_model.predict(next_state)[0])
            #         target[0][action] = reward + self.gamma*future_reward
            #
            #     #fit the main model to the new reward
            #     self.main_model.fit(state, target, epochs=num_epochs, verbose=0)
            #
            # #bring the target_model a bit closer to the main model
            # self.update_target()
            # # self.target_model.fit(state, target, epochs=num_epochs, verbose=0)

        #update epsilon
        self.epsilon -= self.epsilon_decay
        if(self.epsilon < self.min_epsilon): self.epsilon=self.min_epsilon

        self.update_target()

        return loss

    def remember(self, state, action, reward, next_state, done):
        # state = np.reshape(np.asarray(state), [1,self.state_shape[0],self.state_shape[1],self.state_shape[2]])
        # next_state = np.reshape(np.asarray(next_state), [1,self.state_shape[0],self.state_shape[1],self.state_shape[2]])
        # experience = (state, action, reward, next_state, done)

        # if len(self.memory) + 1 >= mem_size:
        #     self.memory.pop()
        if(done):
            done = 1
        else:
            done = 0
        self.memory.append((state, action, reward, next_state, done))












#END OF FILE
