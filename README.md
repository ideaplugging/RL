# Lunar Lander v2
본 프로젝트는 gymnasium에서 제공하는 Lunar Lander v2 문제를 DQN 방법으로 해결하는 것을 목표로 한다.

# 라이브러리 버전 정보
- gymnasium 0.28.1
- matplotlib 3.7.2
- numpy 1.26.2
- pandas 2.1.4
- torch 2.1.1

# 사용한 파라미터 정보
| Parameter               | Experiment 1 | Experiment 2 | Experiment 3 | Experiment 4 | Experiment 5 |
|-------------------------|--------------|--------------|--------------|--------------|--------------|
| learning_rate           | 0.0005       | 0.0075       | 0.0001       | 0.00075      | 0.0005       |
| gamma                   | 0.99         | 0.99         | 0.99         | 0.99         | 0.99         |
| epsilon                 | 1.0          | 1.0          | 1.0          | 1.0          | 1.0          |
| epsilon_decay           | 0.995        | 0.995        | 0.995        | 0.995        | 0.995        |
| batch_size              | 128          | 256          | 256          | 256          | 256          |
| replay_size             | 100000       | 100000       | 100000       | 100000       | 100000       |
| num_episodes            | 3000         | 3000         | 3000         | 3000         | 3000         |
| target_update_interval  | 5            | 10           | 10           | 10           | 10           |
| hidden_sizes            | [16, 16]     | [64, 64]     | [128, 64]    | [16, 16]     | [128, 64]    |




# 실행방법
- 파라미터 변경 시, main.py 파일 내에서 변경 필요하고, 실행은 main.py 파일 직접 실행<br>
```python main.py```

# 실험결과
<img src="https://github.com/ideaplugging/RL/assets/119948425/b1721c50-f622-453e-88c2-fe9b8bb47dc7" width="500">
<img src="https://github.com/ideaplugging/RL/assets/119948425/5c278436-2257-4844-8986-374255c60c45" width="500">
<img src="https://github.com/ideaplugging/RL/assets/119948425/895f9cf7-9452-4435-af24-e1412d272d5f" width="500">
<img src="https://github.com/ideaplugging/RL/assets/119948425/8013dd33-f3b0-4cbe-8110-4add8069e994" width="500">
<img src="https://github.com/ideaplugging/RL/assets/119948425/68483f42-2825-426a-aab9-591ab3c35a70" width="500">

# 디렉토리
- main.py<br>
- buffer.py ## Replybuffer<br>
- Method <br>
  - DQN.py<br>
- utils.py<br>





