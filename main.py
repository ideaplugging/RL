import gymnasium as gym
from Method.DQN import DQNAgent, QNetwork
from Method.actor_critic import ActorNet, CriticNet, ActorCriticAgent
from Method.D3QN import DuelingQNetwork, D3QNAgent
from buffer import ReplayBuffer

import torch
import torch.optim as optim

from utils import learning_trend

# Hyperparameters
learning_rate = 0.0075
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
batch_size = 256
replay_size = 100000
num_episodes = 3000
target_update_interval = 10
hidden_sizes = [64, 64] # DQN

# Environment setup
env = gym.make('LunarLander-v2')
env.action_space.seed(0)
n_inputs = env.observation_space.shape[0]
n_actions = env.action_space.n
print(f'states_space: {n_inputs}, action_space: {n_actions}')

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cpu')

# User input for choosing method
# print("Choose training method:")
# print("1: DQN")
# print("2: D3QN")
# print("3: Actor-Critic")
#
# # print("4: All")
# choice = input("Enter your choice from 1 to 3: ")
#
# if choice == '1':
q_network = QNetwork(n_inputs, n_actions, hidden_sizes).to(device)
target_network = QNetwork(n_inputs, n_actions, hidden_sizes).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.AdamW(q_network.parameters(), lr=learning_rate)
agent = DQNAgent(n_actions,
                 q_network,
                 target_network,
                 optimizer,
                 ReplayBuffer(replay_size, device),
                 gamma,
                 epsilon,
                 epsilon_decay,
                 epsilon_end=0.01,
                 device=device)
agent.train(env, num_episodes, batch_size, target_update_interval)
env.close()
#
# elif choice == '3':
#     actor = ActorNet(n_inputs, n_actions, hidden_sizes).to(device)
#     critic = CriticNet(n_inputs, hidden_sizes).to(device)
#     actor_optimizer = optim.AdamW(actor.parameters(), lr=learning_rate)
#     critic_optimizer = optim.AdamW(critic.parameters(), lr=learning_rate)
#     agent = ActorCriticAgent(n_actions,
#                              actor,
#                              critic,
#                              actor_optimizer,
#                              critic_optimizer,
#                              ReplayBuffer(replay_size, device),
#                              gamma,
#                              epsilon,
#                              epsilon_decay,
#                              epsilon_end=0.01,
#                              device=device)
#     agent.train(env, num_episodes, batch_size)
#     env.close()
#
# elif choice == '2':
#     dueling_q_network = DuelingQNetwork(n_inputs, n_actions, hidden_sizes).to(device)
#     target_dueling_network = DuelingQNetwork(n_inputs, n_actions, hidden_sizes).to(device)
#     target_dueling_network.load_state_dict(dueling_q_network.state_dict())
#
#     # Optimizer for Dueling Q-Network
#     optimizer = optim.AdamW(dueling_q_network.parameters(), lr=learning_rate)
#
#     # Instantiate the D3QN Agent
#     d3qn_agent = D3QNAgent(n_actions,
#                            dueling_q_network,
#                            target_dueling_network,
#                            optimizer,
#                            ReplayBuffer(replay_size, device),
#                            gamma,
#                            epsilon,
#                            epsilon_decay,
#                            epsilon_end=0.01,
#                            device=device)
#
#     # Train the D3QN Agent
#     d3qn_agent.train(env, num_episodes, batch_size, target_update_interval)
#     env.close()
#
# else:
#     print("Invalid choice. Please enter from 1 to 3.")