from agent_action import *
from networks import *
from config import DEVICE as device
import torch.optim as optim
from utils import *

class agent_args:
    
    def __init__(self, env, lr1=0.0001, lr2=0.001, tau=0.001, speed1=1, speed2=1, step=1, learning_time=1, batch_size=64, OUN_noise=True, batchnorm=True, clip=True, initialize=True, hidden=[256, 256]):
        
        #Initialize environment
        state_size, action_size = env.observation_space.shape[0], env.action_space.shape[0]
        self.env = env

        #Initialize some hyper parameters of agent
        self.lr1 = lr1
        self.lr2 = lr2
        self.tau = tau
        self.speed1 = speed1
        self.speed2 = speed2
        self.learning_time = learning_time
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = 0.99
        self.step = step
        self.OUN_noise = OUN_noise
        self.batchnorm = batchnorm
        self.clip = clip
        self.initialize = initialize
        self.hidden = hidden

        #Initialize agent (networks, replyabuffer and noise)

        
class agent(agent_args):
    def __init__(self, env, lr1=0.0001, lr2=0.001, tau=0.001, speed1=1, speed2=1, step=1, learning_time=1, batch_size=64, OUN_noise=True, batchnorm=True, clip=True, initialize=True, hidden=[256, 256]):
        super().__init__(env, lr1, lr2, tau, speed1, speed2, step, learning_time, batch_size, OUN_noise, batchnorm, clip, initialize, hidden)
                #Initialize agent (networks, replyabuffer and noise)
        self.actor_local = Actor(self.state_size, self.action_size, self.batchnorm, initialize, hidden).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.batchnorm, initialize, hidden).to(device)
        self.critic_local = Critic(self.state_size, self.action_size, self.batchnorm, initialize, hidden).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, self.batchnorm, initialize, hidden).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr1)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr2)
        
        self.memory = ReplayBuffer(self.action_size, buffer_size=1000000, batch_size=self.batch_size)

class ddpg_agent(agent_action,agent):
    def __init__(self, env, lr1=0.0001, lr2=0.001, tau=0.001, speed1=1, speed2=1, step=1, learning_time=1, batch_size=64, OUN_noise=True, batchnorm=True, clip=True, initialize=True, hidden=[256, 256]):
        super().__init__(env, lr1, lr2, tau, speed1, speed2, step, learning_time, batch_size, OUN_noise, batchnorm, clip, initialize, hidden)
        
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)

        self.noise = OUNoise(action_size=self.action_size)

    def act(self, state, i):
        state = torch.tensor(state, dtype=torch.float).to(device).view(1, -1)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).detach().view(-1).cpu().numpy()
        self.actor_local.train()
        if self.OUN_noise:
            noise = self.noise.sample()
        else:
            noise = self.noise.sigma * np.random.standard_normal(self.action_size)
        action += noise/math.sqrt(i)
        action = np.clip(action, -1, 1)
        return action
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for local_layer, target_layer in zip(local_model.modules(), target_model.modules()):
            for local_parameter, target_parameter in zip(local_layer.parameters(), target_layer.parameters()):
                target_parameter.data.copy_(tau*local_parameter.data + (1-tau)*target_parameter.data)
            try:
                target_layer.running_mean = tau * local_layer.running_mean + (1 - tau) * target_layer.running_mean
                target_layer.running_var = tau * local_layer.running_var + (1 - tau) * target_layer.running_var
            except:
                None
            
    def learn(self):
        
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = experiences
        
        with torch.no_grad():
            expected_rewards = rewards + (1-dones)*self.gamma*self.critic_target(next_states, self.actor_target(next_states))
        
        for _ in range(self.speed1):
            observed_rewards = self.critic_local(states, actions)
            L = (expected_rewards - observed_rewards).pow(2).mean()
            self.critic_optimizer.zero_grad()
            L.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()
        
        for _ in range(self.speed2):
            L = - self.critic_local(states, self.actor_local(states)).mean()
            self.actor_optimizer.zero_grad()
            L.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
            self.actor_optimizer.step()
        
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        