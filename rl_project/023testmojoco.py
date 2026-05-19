import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from collections import deque
import time
import os 
class RealisticMotorWrapper(gym.Wrapper):
    def __init__(self, env, delay_steps=1, alpha=0.6):
        super().__init__(env)
        self.delay_steps = delay_steps      
        self.alpha = alpha                  
        
        self.action_queue = deque(maxlen=delay_steps + 1)
        self.last_real_action = np.zeros(self.action_space.shape)

        orig_obs_space = env.observation_space
        new_shape = (orig_obs_space.shape[0] + self.action_space.shape[0], )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=new_shape, dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # ==========================================
        # 🌟 任務二：上帝模式 (加入初始隨機擾動)
        # ==========================================
        qpos = self.env.unwrapped.data.qpos.copy()
        qvel = self.env.unwrapped.data.qvel.copy()
        
        # 產生隨機傾角與角速度 (每次開局都像被推了一把)
        random_angle = np.random.uniform(-0.15, 0.15)
        random_angular_vel = np.random.uniform(-0.5, 0.5)
        
        qpos[1] = random_angle
        qvel[1] = random_angular_vel
        
        self.env.unwrapped.set_state(qpos, qvel)
        obs = self.env.unwrapped._get_obs()
        # ==========================================

        for _ in range(self.delay_steps + 1):
            self.action_queue.append(np.zeros(self.action_space.shape))
        self.last_real_action = np.zeros(self.action_space.shape)
        
        augmented_obs = np.concatenate((obs, self.last_real_action))
        return augmented_obs, info

    def step(self, action):
        smoothed_action = self.alpha * action + (1 - self.alpha) * self.last_real_action
        self.last_real_action = smoothed_action
        self.action_queue.append(smoothed_action)
        delayed_action = self.action_queue.popleft() 

        obs, reward, terminated, truncated, info = self.env.step(delayed_action)

        augmented_obs = np.concatenate((obs, self.last_real_action))
        return augmented_obs, reward, terminated, truncated, info

if __name__ == "__main__":
    print("🌍 建立展示環境...")
    base_env = gym.make('InvertedPendulum-v4', max_episode_steps=100000, render_mode="human")
    env = RealisticMotorWrapper(base_env, delay_steps=1, alpha=0.6)

    print("🧠 載入 021 偷懶大腦...")
    model = DDPG.load(os.path.join(os.path.dirname(__file__), "021"))

    obs, info = env.reset()
    step_count = 0
    episode_count = 1

    print(f"🎬 開始第 {episode_count} 回合展示 (觀察它如何靠牆偷懶)...")
    

    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        
        # 1. 執行動作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 2. 步數加 1
        step_count += 1
        time.sleep(0.01) 
        
        if step_count % 10 == 0:
            print(f"📍 位置: {obs[0]:>6.3f} m | ⚙️ 實際輸出: {action[0]:>6.3f} N")
            
        # 3. 強制煞車機制
        if step_count >= 100:
            print("⏱️ 達到 100 步上限，觸發強制重啟！")
            truncated = True 
            
        # 4. 結算與重置
        if terminated or truncated:
            print(f"⚠️ 第 {episode_count} 回合結束 (總步數: {step_count})。準備重新發球！")
            time.sleep(1.5) 
            
            obs, info = env.reset()
            step_count = 0        # 歸零
            episode_count += 1    # 回合數加一
            print("-" * 30)