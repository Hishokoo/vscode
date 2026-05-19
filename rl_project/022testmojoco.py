import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from collections import deque
import os

# ==========================================
# 1. 搬回我們特製的「真實馬達」外套
# (測試環境必須跟訓練環境一模一樣，大腦才看得懂)
# ==========================================
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
        
        augmented_obs = np.concatenate((obs, self.last_real_action))
        return augmented_obs, info

    def step(self, action):
        smoothed_action = self.alpha * action + (1 - self.alpha) * self.last_real_action
        self.last_real_action = smoothed_action
        self.action_queue.append(smoothed_action)
        delayed_action = self.action_queue.popleft() 

        obs, reward, terminated, truncated, info = self.env.step(delayed_action)
        
        # 測試階段其實用不到 reward，但為了維持 Wrapper 完整性保留計算
        force_cost = self.force_penalty * np.sum(np.square(delayed_action))
        jerk_cost = self.jerk_penalty * np.sum(np.square(action - self.last_raw_action))
        modified_reward = max(0.1, reward - force_cost - jerk_cost)
        
        self.last_raw_action = action
        augmented_obs = np.concatenate((obs, self.last_real_action))

        return augmented_obs, modified_reward, terminated, truncated, info

# ==========================================
# 2. 測試主程式開始
# ==========================================
print("🌍 建立測試環境 (開啟動畫)...")
# 不設定 max_episode_steps，讓它無限期跑下去
base_env = gym.make('InvertedPendulum-v4',max_episode_steps=100000, render_mode="human")
# 套上 Wrapper，參數必須跟訓練時一模一樣 (delay=1, alpha=0.6)
env = RealisticMotorWrapper(base_env, delay_steps=1, alpha=0.6)

print("🧠 載入具備記憶的真實馬達大腦 (v2)...")
model = DDPG.load(os.path.join(os.path.dirname(__file__), "021"))

print("🎥 開始測試！(終端機按 Ctrl+C 可強制關閉)")
obs, info = env.reset()

while True:
    # deterministic=True 代表關閉探索雜訊，拿出 100% 真本事
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 印出最終經過濾波、實際輸出給馬達的平滑力道
    print(f"馬達平滑出力: {env.last_real_action[0]:.3f} N")
    
    if terminated or truncated:
        print("⚠️ 失去平衡，自動重新發球！")
        obs, info = env.reset()