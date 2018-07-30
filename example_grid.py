from environments.gridworld import gameEnv
from agents.dddqn import DDDQNAgent

from scipy import misc
import matplotlib.pyplot as plt
import time

game_size = 5   #size of the square game grid
num_episodes = 10   #run through this many episodes
max_num_turns = 20  #only allow the agent to take up to this many turns
render = True

if __name__ == "__main__":
    env = gameEnv(partial=False,size=game_size)

    state = env.reset()

    # print("Shape",state.shape)
    #create the learning agent
    agent = DDDQNAgent(state.shape, env.actions)


    for ep in range(num_episodes):
        state = env.reset()
        done = False
        turns = 0
        accumulated_reward = 0  #reward over each step


        while(turns < max_num_turns and not done):

            #have the agent select an action
            action = agent.act(state)

            #find the result of that action
            next_state, reward, done = env.step(action)

            #remember that action
            agent.remember(state, action, reward, next_state, done)

            #update totals
            accumulated_reward += reward
            state = next_state
            turns += 1

            print("Action:",action)

            if(render): env.renderEnv()



        #print out the episode stats
        print("Episode:",ep,"| epsilon:",agent.epsilon, "| total_reward:",accumulated_reward)

        #after each episode, have the agent replay from its memory
        agent.replay()


    env.renderEnv()
