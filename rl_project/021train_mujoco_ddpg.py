import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from collections import deque

class RealisticMotorWrapper(gym.Wrapper):
    def __init__(self, env, delay_steps=1, alpha=0.6, force_penalty=0.01, jerk_penalty=0.02):
        super().__init__(env)
        self.delay_steps = delay_steps      
        self.alpha = alpha                  
        self.force_penalty = force_penalty  
        self.jerk_penalty = jerk_penalty    
        
        self.action_queue = deque(maxlen=delay_steps + 1)
        self.last_raw_action = np.zeros(self.action_space.shape)
        self.last_real_action = np.zeros(self.action_space.shape)

        # 🌟 核心修復：擴增觀測空間 (Observation Space)
        # 原本只有 [位置, 角度, 速度, 角速度]，我們現在要把 [馬達真實出力] 也加進去！
        orig_obs_space = env.observation_space
        new_shape = (orig_obs_space.shape[0] + self.action_space.shape[0], )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=new_shape, dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.delay_steps + 1):
            self.action_queue.append(np.zeros(self.action_space.shape))
        self.last_raw_action = np.zeros(self.action_space.shape)
        self.last_real_action = np.zeros(self.action_space.shape)
        
        # 🌟 重置時，也要把馬達出力 (0) 綁在狀態後面
        augmented_obs = np.concatenate((obs, self.last_real_action))
        return augmented_obs, info

    def step(self, action):
        # 低通濾波與延遲
        smoothed_action = self.alpha * action + (1 - self.alpha) * self.last_real_action
        self.last_real_action = smoothed_action
        self.action_queue.append(smoothed_action)
        delayed_action = self.action_queue.popleft() 

        # 執行動作
        obs, reward, terminated, truncated, info = self.env.step(delayed_action)

        # 計算扣分 (稍微調降扣分力度，不要太嚴苛)
        force_cost = self.force_penalty * np.sum(np.square(delayed_action))
        jerk_cost = self.jerk_penalty * np.sum(np.square(action - self.last_raw_action))
        modified_reward = max(0.1, reward - force_cost - jerk_cost)
        
        self.last_raw_action = action

        # 🌟 核心修復：把滑車當前的「真實推力」告訴 AI 大腦
        augmented_obs = np.concatenate((obs, self.last_real_action))

        return augmented_obs, modified_reward, terminated, truncated, info

# ==========================================
# 重新建立訓練管線
# ==========================================
base_env = gym.make('InvertedPendulum-v4', max_episode_steps=100000)
env = RealisticMotorWrapper(base_env, delay_steps=1, alpha=0.6)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# 🌟 統一 TensorBoard 儲存路徑
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="rl_project/tensorboard_logs/")

print("🤖 裝上短期記憶後，重新開始訓練...")
# 這次跑 10 萬步試試看！
model.learn(total_timesteps=100000, tb_log_name="021_Realistic_Motor")
model.save("021")
print("✅ 訓練完成！")