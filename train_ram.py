import numpy as np
import gym
from agent import *
from config import *


if __name__ == '__main__':
    env = gym.make(RAM_ENV_NAME)

    rewards = np.zeros((N, L))
    average_rewards = np.zeros((N, L))
    for i in range(N):
        print('{}/{}'.format(i+1, N))
        agent = ddpg_agent(env=env, 
                    lr1=LR1, 
                    lr2=LR2, 
                    tau=TAU, 
                    speed1=SPEED1, 
                    speed2=SPEED2, 
                    step=STEP, 
                    learning_time=LEARNING_TIME,
                    batch_size=BATCH_SIZE, 
                    OUN_noise=OUN, 
                    batchnorm=BN, 
                    clip=CLIP,
                    initialize=INIT,
                    hidden=HIDDEN)
        rewards[i, :], average_rewards[i, :] = agent.train(L, env)


    np.save('./rewards_{}_{}_{}_{}_{}_{}.npy'.format(agent.OUN_noise, agent.batchnorm, agent.clip, agent.initialize, agent.hidden[0], agent.hidden[1]), rewards)
    np.save('./average_rewards_{}_{}_{}_{}_{}_{}.npy'.format(agent.OUN_noise, agent.batchnorm, agent.clip, agent.initialize, agent.hidden[0], agent.hidden[1]), average_rewards)