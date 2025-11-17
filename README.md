# 🐼 Panda-Gym Reinforcement Learning: A2C / PPO / SAC / TD3 / DDPG

This project implements and compares several reinforcement learning algorithms on the `PandaReachDense-v3` robotic control task from Panda-Gym, using Stable-Baselines3 in Google Colab.

The goal is to train the Panda robot arm to reach a target position without trajectory planning, relying purely on continuous-control RL.

---

## Project Highlights

- Multiple RL algorithms implemented:
  - A2C (Advantage Actor-Critic)
  - PPO (Proximal Policy Optimization)
  - DDPG (Deep Deterministic Policy Gradient)
  - TD3 (Twin Delayed DDPG)
  - SAC (Soft Actor-Critic)
- Goal-based Dict observation using `MultiInputPolicy`
- Parallel vectorized environments (`n_envs = 4`)
- Observation and reward normalization via `VecNormalize`
- Automatic video recording of trained policies
- TensorBoard visualization for success rates, rewards, losses, entropy
- Episode evaluation utilities

---

## Directory Structure

```
.
├── notebooks/
│   └── PandaReach.ipynb          # Main training notebook
├── models/
│   ├── a2c.zip
│   ├── ppo.zip
│   ├── sac.zip
│   ├── td3.zip
│   └── ddpg.zip
│   └── *_vecnorm.pkl             # VecNormalize statistics
├── videos/
│   ├── a2c_slow.mp4
│   ├── ppo_slow.mp4
│   ├── sac_slow.mp4
│   ├── td3_slow.mp4
│   └── ddpg_slow.mp4
├── logs/
│   └── tensorboard/
└── README.md
```

---

## Task: PandaReachDense-v3

This task requires the panda robot’s end-effector (EEF) to reach a randomly sampled 3D target.

### Observation Space (Dict)

| Key | Description |
|-----|-------------|
| `observation` | 6D vector: (x, y, z, vx, vy, vz) |
| `achieved_goal` | Current EE position |
| `desired_goal` | Target position |

### Reward (dense)

```
reward = - || achieved_goal - desired_goal ||
```

### Success Criterion

```
distance < 0.05
```

Modify threshold:

```python
env.unwrapped.task.distance_threshold = 0.005
```

---

## Training Environment Setup

```python
def make_train_env(env_id: str, n_envs: int = 4):
    env = make_vec_env(env_id, n_envs=n_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env
```

For SAC, disable reward normalization:

```python
env = VecNormalize(env, norm_obs=True, norm_reward=False)
```

---

## Training Example (A2C)

```python
from stable_baselines3 import A2C

env = make_train_env("PandaReachDense-v3", n_envs=4)

model = A2C(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/a2c/"
)

model.learn(total_timesteps=200000)
model.save("models/a2c")
env.save("models/a2c_vecnorm.pkl")
```

---

## Evaluation Script

```python
def run_episodes(algo_name, n_episodes=5, max_steps_per_ep=100):
    eval_env = make_eval_env(env_id, vecnorm_path)
    model = ALGORITHMS[algo_name].load(model_path)

    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0

        for t in range(max_steps_per_ep):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward[0]
            steps += 1

            if done[0]:
                break

        print(f"Episode {ep}: steps={steps}, reward={total_reward:.3f}, success={info[0]['is_success']}")
```

---

## Recording Videos

```python
record_video("a2c", "videos/a2c_slow.mp4", max_steps=300)
record_video("ppo", "videos/ppo_slow.mp4", max_steps=300)
record_video("sac", "videos/sac_slow.mp4", max_steps=300)
record_video("td3", "videos/td3_slow.mp4", max_steps=300)
record_video("ddpg", "videos/ddpg_slow.mp4", max_steps=300)
```

---

## TensorBoard

```
%load_ext tensorboard
%tensorboard --logdir ./logs/
```

Metrics include:

- rollout/success_rate  
- rollout/ep_rew_mean  
- train/value_loss  
- train/policy_loss  
- train/actor_loss  
- train/critic_loss  
- train/ent_coef (for SAC)  
- time/total_timesteps  

---

## Algorithm Comparison Summary (Based on Actual Training Results)

| Algorithm | Success Rate | Avg Steps to Reach (ep_len_mean) | Learning Speed | Stability | Notes |
|----------|--------------|----------------------------------|----------------|-----------|-------|
| **DDPG** | ⭐⭐⭐⭐⭐ (1.00) | ⭐⭐⭐⭐⭐ (2.64 steps) | ⭐⭐⭐⭐⭐ (Very Fast) | ⭐⭐⭐⭐ | Fastest convergence, surprisingly stable in this task |
| **TD3**  | ⭐⭐⭐⭐⭐ (1.00) | ⭐⭐⭐⭐☆ (2.79 steps) | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ | Strong deterministic baseline, slightly slower than DDPG but more stable |
| **A2C**  | ⭐⭐⭐⭐⭐ (1.00) | ⭐⭐⭐ (6.44 steps) | ⭐⭐⭐ (Mid-speed) | ⭐⭐⭐⭐ | Good on-policy performance; moderate sample efficiency |
| **PPO**  | ⭐⭐⭐⭐⭐ (1.00) | ⭐⭐☆ (5.56 steps) | ⭐⭐⭐ (Mid-speed) | ⭐⭐⭐⭐ | Stable but requires more samples; slower improvement |
| **SAC**  | ⭐⭐⭐⭐☆ (0.98) | ⭐ (8.01 steps) | ⭐⭐ (Slow) | ⭐⭐⭐ | Soft actor-critic underperforms with dense reward and default auto-entropy settings |


---

## Dependencies

```
panda-gym
stable-baselines3
gymnasium
imageio
tensorboard
numpy
```

Install:

```
pip install panda-gym stable-baselines3[extra] imageio
```

---

## License

MIT License

---

## Acknowledgements

- Panda-Gym: https://github.com/qgallouedec/panda-gym  
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3  
- Developed in Google Colab
