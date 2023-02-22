
from abc import ABCMeta,abstractmethod
import numpy as np

class agent_action(metaclass=ABCMeta):
    @abstractmethod
    def act(self):
        pass
    
    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def soft_update(self):
        pass

    def train(self, n_episodes, reset):
        """run the RL frame work

        Args:
            n_episodes: int the epoch to.
            reset:  bool if to reset the noise.
        
        returns:
            reward: [] list .
            average_rewards: [] list.
        """
        rewards = []
        average_rewards = []

        for i in range(1, n_episodes+1):
            episodic_reward = 0
            state = self.env.reset()
            if reset:
                self.noise.reset()
            action = self.act(state, i)
            cont_env_step = 0

            while True:
                next_state, reward, done, _ = self.env.step(action)
                episodic_reward += reward
                self.memory.add(state, action, reward, next_state, done)
                cont_env_step += 1

                if len(self.memory.memory)>self.batch_size:
                    if cont_env_step % self.step == 0:
                        for _ in range(self.learning_time):
                            self.learn()
                
                if done:
                    break
                
                state = next_state.copy()
                action = self.act(state, i)
            
            rewards.append(episodic_reward)
            average_rewards.append(np.mean(rewards[-100:]))

            print('\rEpisode {}. Total score for this episode: {:.3f}, average score {:.3f}'.format(i, rewards[-1], average_rewards[-1]), end='')
            
            if i % 100 == 0:
                print('---------------')

        return rewards, average_rewards
