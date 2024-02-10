import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str = "relu", output_type: str = 'single'):
        super(NeuralNetwork, self).__init__()
        
        self.output_type = output_type
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("Activation function doesn't exist")
            
        layers = [nn.Linear(input_dim, hidden_size), self.activation]
        
        for _ in range(hidden_layers-1):
            layers += [nn.Linear(hidden_size, hidden_size), self.activation]
            
        self.layers = nn.Sequential(*layers)
        
        if self.output_type == 'single':
            self.output = nn.Linear(hidden_size, output_dim)
        else:
            self.output_mean = nn.Linear(hidden_size, output_dim)
            self.output_log_std = nn.Linear(hidden_size, output_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        network_outputs = self.layers(s)
        
        if self.output_type == 'single':
            return self.output(network_outputs)
        else:
            mean = self.output_mean(network_outputs)
            log_std = self.output_log_std(network_outputs)
            
            return mean, log_std                    
    
class Actor():
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        self.network = NeuralNetwork(self.state_dim, self.action_dim, self.hidden_size, 
                                  self.hidden_layers, 'relu', output_type='dual').to(self.device)
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.actor_lr)


    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        
        mean, log_std = self.network(state)
        log_std = self.clamp_log_std(log_std)
        
        normal = torch.distributions.Independent(torch.distributions.Normal(loc=mean, scale=torch.exp(log_std)), reinterpreted_batch_ndims=0)
        
        if deterministic:
            sampled_action = normal.sample()
        else:
            sampled_action = normal.rsample()

        action = torch.tanh(sampled_action)
        
        # We need to make sure to adjust the log_probability as well, since we squash the action into [-1,1] using tanh
        log_prob = normal.log_prob(sampled_action) - torch.log(1 - action.pow(2))
        
        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob

class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        input_dim = self.state_dim + self.action_dim
        output_dim = 1

        self.network = NeuralNetwork(input_dim, output_dim, self.hidden_size, 
                                     self.hidden_layers, activation='relu', output_type='single').to(self.device)
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.critic_lr)


class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param

class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        
        self.actor = Actor(hidden_size=64, hidden_layers=5, actor_lr=1e-3, 
                           state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)
        
        self.critic1 = Critic(hidden_size=64, hidden_layers=5, critic_lr=1e-3, 
                              state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)
        self.critic2 = Critic(hidden_size=64, hidden_layers=5, critic_lr=1e-3, 
                              state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)
        
        self.target_critic1 = Critic(hidden_size=64, hidden_layers=5, critic_lr=1e-3, 
                                     state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)
        self.target_critic2 = Critic(hidden_size=64, hidden_layers=5, critic_lr=1e-3, 
                                     state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)
        
        # Initially we want to copy the weights from the critics to the target critics
        self.critic_target_update(self.critic1.network, self.target_critic1.network, tau=0.0, soft_update=False)
        self.critic_target_update(self.critic2.network, self.target_critic2.network, tau=0.0, soft_update=False)
        
        # Parameter that decides how strongly we explore
        self.temperature = TrainableParameter(init_param=0.3, lr_param=1e-3, train_param=True, device=self.device)
        self.gamma = 0.99
        

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        state = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.get_action_and_log_prob(state, deterministic=not train)
            action = action.cpu().numpy().squeeze(0)

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch
        
        # --- Critic update ---
        with torch.no_grad():
            next_actions, log_probs = self.actor.get_action_and_log_prob(s_prime_batch, deterministic=True)
            target1 = self.target_critic1.network(torch.cat([s_prime_batch, next_actions], dim=1))
            target2 = self.target_critic2.network(torch.cat([s_prime_batch, next_actions], dim=1))
            
            targets = r_batch + self.gamma * (torch.min(target1, target2) - self.temperature.get_param() * log_probs)

        q1_pred = self.critic1.network(torch.cat([s_batch, a_batch], dim=1))
        q2_pred = self.critic2.network(torch.cat([s_batch, a_batch], dim=1))

        # Critic loss - Mean Squared Error between current Q-values and target Q-values
        critic_loss1 = nn.MSELoss()(q1_pred, targets)
        critic_loss2 = nn.MSELoss()(q2_pred, targets)
        
        self.run_gradient_update_step(self.critic1, critic_loss1)
        self.run_gradient_update_step(self.critic2, critic_loss2)
        
        # --- Actor update ---
        for param in self.critic1.network.parameters():
            param.requires_grad = False
        for param in self.critic2.network.parameters():
            param.requires_grad = False

        actions, log_probs = self.actor.get_action_and_log_prob(s_batch, deterministic=False)
        
        current_q1 = self.critic1.network(torch.cat([s_batch, actions], dim=1))
        current_q2 = self.critic2.network(torch.cat([s_batch, actions], dim=1))
        current_q_min = torch.min(current_q1, current_q2)
        
        actor_loss = -torch.mean(current_q_min - self.temperature.get_param() * log_probs)        
        
        self.run_gradient_update_step(self.actor, actor_loss)
        
        for param in self.critic1.network.parameters():
            param.requires_grad = True
        for param in self.critic2.network.parameters():
            param.requires_grad = True
        
        with torch.no_grad():
            self.critic_target_update(self.critic1.network, self.target_critic1.network, tau=0.005, soft_update=True)
            self.critic_target_update(self.critic2.network, self.target_critic2.network, tau=0.005, soft_update=True)
            
        # --- Temperature update ---
        _, temperature_log_probs = self.actor.get_action_and_log_prob(s_batch, deterministic=False)
        temperature_loss = -(self.temperature.get_log_param()) * (temperature_log_probs - 1)
        self.temperature.optimizer.zero_grad()
        temperature_loss.mean().backward()
        self.temperature.optimizer.step()

# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
        
