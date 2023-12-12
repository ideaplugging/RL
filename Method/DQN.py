from collections import deque
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import save_model, save_rewards_to_tsv

class QNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden_sizes, dropout_rate=0.2, seed=0):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.layers = nn.ModuleList()
        prev_size = n_inputs
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            # self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        self.layers.append(nn.Linear(prev_size, n_actions))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Modified Agent Class
class DQNAgent:
    def __init__(self, n_actions, q_network, target_network, optimizer, memory, gamma, epsilon, epsilon_decay, epsilon_end, device, seed=0):
        self.n_actions = n_actions
        self.q_network = q_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.memory = memory
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.device = device
        self.seed = random.seed(seed)

    def select_action(self, state):
        device = self.device
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)  # Add batch dimension
        self.q_network.train()

        if random.random() < self.epsilon:
            return random.choice(np.arange(self.n_actions))  # Choose a random action
        else:
            return q_values.max(1)[1].item()  # Choose the action with the highest Q-value

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.memory.sample(batch_size)

        q_values = self.q_network(batch_states).gather(1, batch_actions)
        max_next_q_values = self.target_network(batch_next_states).detach().max(1)[0].unsqueeze(1)
        expected_q_values = batch_rewards + self.gamma * max_next_q_values * (1 - batch_dones.float())

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, env, num_episodes, batch_size, target_update_interval):
        all_episode_rewards = []
        all_avg_rewards = []
        scores_window = deque(maxlen=100) # For calculating the moving average

        for episode in range(num_episodes):
            state, _ = env.reset(seed=0)
            episode_reward = 0

            while True:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = (terminated or truncated)

                self.memory.push(state, action, reward, next_state, done)

                episode_reward += reward

                # Check the buffer size before calling learn
                if len(self.memory) > batch_size:
                    self.learn(batch_size)

                # Update the target network
                if episode % target_update_interval == 0:
                    self.update_target_network()

                state = next_state

                if done:
                    break

            self.decay_epsilon()

            all_episode_rewards.append(episode_reward)
            scores_window.append(episode_reward)  # Save most recent score
            avg_reward = np.mean(scores_window)  # Calculate average of last 100 episodes
            all_avg_rewards.append(avg_reward)
            print(f'\rMethod: DQN, Episode: {episode}\t Average Reward: {avg_reward:.2f}', end='')

            if episode % 100 == 0:
                print(f'\rMethod: DQN, Episode: {episode}\t Average Reward: {avg_reward:.2f}')

            if avg_reward >= 200.0:
                print(f'\rMethod: DQN, Episode: {episode}\t Average Reward: {avg_reward:.2f}')
                save_model(self.q_network.state_dict(), f'./saved_models/DQN_episode_{episode}.pth')
                print(f'\rBest model saved with average reward: {avg_reward} at episode {episode}')
                break
            save_rewards_to_tsv(all_episode_rewards, filename='./data/DQN_rewards.tsv')
            save_rewards_to_tsv(all_avg_rewards, filename='./data/DQN_avg_rewards.tsv')
