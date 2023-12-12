import os
import csv
import pandas as pd
import torch
import matplotlib.pyplot as plt

def save_rewards_to_tsv(rewards, columns=['Episode', 'Reward'], filename=None):

    # Check if the directory exists; if not, create it
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(columns)
        for episode, reward in enumerate(list(rewards), start=1):
            writer.writerow([episode, reward])

def save_model(model, save_path):
    # 경로에서 폴더 이름을 분리합니다. 예: 'path/to/folder/model.pth' -> 'path/to/folder'
    directory = os.path.dirname(save_path)

    # 지정된 경로에 폴더가 존재하지 않으면 생성합니다.
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 모델을 지정된 경로에 저장합니다.
    torch.save(model, save_path)

def learning_trend(reward_data, avg_data, title, color):
    plt.figure(figsize=(12, 6))

    if not isinstance(reward_data, pd.DataFrame) or \
            not isinstance(avg_data, pd.DataFrame):

        reward = pd.read_csv(reward_data, sep='\t')
        avg = pd.read_csv(avg_data, sep='\t')
        reward['Episode'] = pd.to_numeric(reward['Episode'])
        avg['Episode'] = pd.to_numeric(avg['Episode'])

    plt.plot(reward.iloc[:,0], reward.iloc[:,1], label='Reward per Episode', color=color[0])
    plt.plot(avg.iloc[:, 0], avg.iloc[:, 1], label='Average reward per Episode', color=color[1])

    # Fitting a quadratic polynomial
    plt.axhline(y=200, color='r', linestyle='--', label='Solved Requirement')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    learning_trend('./DQN_data/DQN_rewards.tsv',
                    './DQN_data/DQN_avg_rewards.tsv',
                   'DQN Training for LunarLander (Quadratic Trend)',
                   ['blue', 'orange'])