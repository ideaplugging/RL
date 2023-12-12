import torch
import numpy as np
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Experience Replay
class ReplayBuffer:
    def __init__(self, capacity, device, seed=0):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        self.seed = random.seed(seed)

    def push(self, *args):
        """Save a transition"""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        device = self.device
        mini_batch = random.sample(self.buffer, batch_size)
        state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, done_mask = transition
            state_lst.append(np.array(state))  # 상태를 numpy 배열로 변환
            action_lst.append([action])
            reward_lst.append([reward])
            next_state_lst.append(np.array(next_state))  # 다음 상태를 numpy 배열로 변환
            done_mask_lst.append([done_mask])

        # 리스트를 numpy 배열로 변환한 후 torch tensor로 변환
        return torch.tensor(np.array(state_lst), dtype=torch.float32).to(device), \
            torch.tensor(np.array(action_lst)).to(device), \
            torch.tensor(np.array(reward_lst), dtype=torch.float32).to(device), \
            torch.tensor(np.array(next_state_lst), dtype=torch.float32).to(device), \
            torch.tensor(np.array(done_mask_lst)).to(device)

    def __len__(self):
        return len(self.buffer)