from environments.gridworld import gameEnv
from agents.dddqn import DDDQNAgent

from scipy import misc
import matplotlib.pyplot as plt
import time
import numpy as np

game_size = 5   #size of the square game grid
num_episodes = 10000  #run through this many episodes
pre_train_episodes = 10000    #number of random episodes to play before training
max_num_turns = 20  #only allow the agent to take up to this many turns
render = False

if __name__ == "__main__":
    env = gameEnv(partial=False,size=game_size)

    state = env.reset()

    # print("Shape",state.shape)
    #create the learning agent
    agent = DDDQNAgent(state.shape, env.actions)

    reward_list = []
    time_total_list = []
    time_train_list = []
    time_step_list = []
    time_act_list = []
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        turns = 0
        accumulated_reward = 0  #reward over each step

        time_total = 0
        time_train = 0
        time_step = 0
        time_act = 0
        start = time.time()
        while(turns < max_num_turns and not done):
            pt0 = time.time()
            #have the agent select an action
            action = agent.act(state)
            pt1 = time.time()
            time_act += pt1-pt0
            #find the result of that action
            next_state, reward, done = env.step(action)
            if(np.array_equal(state,next_state)):
                reward -= 0.2
            # print("reward:",reward)
            pt2 = time.time()
            time_step = pt2-pt1

            #update accumulated_reward
            accumulated_reward += reward


            #remember that action
            agent.remember(state, action, reward, next_state, done)
            # agent.remember(state, action, accumulated_reward, next_state, done)

            #update state and turn number
            state = next_state
            turns += 1

            if(render):
                env.render()

        if(ep > pre_train_episodes):
            #after each episode, have the agent replay from its memory
            loss = agent.replay()

            #print out the episode stats
            if(ep%10==0):
                print("Episode:",ep-pre_train_episodes,"| epsilon:","{0:.3f}".format(agent.epsilon), \
                "| total_reward:","{0:.1f}".format(accumulated_reward)," | loss:","{0:.3f}".format(loss))
        elif(ep%1000==0):
            print("Episode:",ep)
        reward_list.append((ep, accumulated_reward))

        pt3 = time.time()

        end = time.time()
        time_train = end-pt3
        time_total = end-start

        #update the times
        time_total_list.append((ep,time_total))
        time_train_list.append((ep,time_train))
        time_step_list.append((ep,time_step))
        time_act_list.append((ep,time_act))

    agent.save("save/dqn_weights.h5")
    # print(reward_list)
    #plot the results
    plt.figure(0)
    plt.plot(*zip(*reward_list))
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.show()

    plt.figure(1)
    plt.plot(*zip(*time_total_list))
    plt.plot(*zip(*time_train_list))
    plt.plot(*zip(*time_step_list))
    plt.plot(*zip(*time_act_list))
    plt.legend(['total time','training time','step time',"act time"])
    plt.xlabel('episode')
    plt.ylabel('time')
    plt.show()

    # play the results
    agent.epsilon = 0

    for ep in range(10):
        state = env.reset()
        done = False
        turns = 0
        accumulated_reward = 0  #reward over each step


        while(turns < max_num_turns and not done):

            #have the agent select an action
            action = agent.act(state)

            #find the result of that action
            next_state, reward, done = env.step(action)

            print("Action:",action)

            #update totals
            accumulated_reward += reward
            state = next_state
            turns += 1

            env.render()
        print("Total reward:",accumulated_reward)
