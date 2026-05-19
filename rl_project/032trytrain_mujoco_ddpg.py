import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from collections import deque
import time
import os

# ==========================================
# 1. 穿上跟訓練時「完全一模一樣」的特製外套
# ==========================================
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
        # 1. 呼叫原始環境的 reset
        obs, info = self.env.reset(**kwargs)
        
        # ==========================================
        # 🌟 上帝模式：手動注入隨機初始擾動
        # ==========================================
        qpos = self.env.unwrapped.data.qpos.copy()
        qvel = self.env.unwrapped.data.qvel.copy()
        
        # 給予隨機的初始位置、角度與推力
        qpos[0] = np.random.uniform(-0.2, 0.2)
        qpos[1] = np.random.uniform(-0.05, 0.05)
        qvel[0] = np.random.uniform(-0.5, 0.5)
        
        # 將擾動寫入 MuJoCo 物理引擎
        self.env.unwrapped.set_state(qpos, qvel)
        
        # 🌟 【修復閃退關鍵】：
        # 絕對不要呼叫 self.env.unwrapped._get_obs()
        # 直接用 numpy 把修改後的數值手動拼裝成觀測值！
        obs = np.concatenate([qpos, qvel]).astype(np.float32)
        # ==========================================
        
        # 初始化馬達延遲佇列
        for _ in range(self.delay_steps + 1):
            self.action_queue.append(np.zeros(self.action_space.shape))
        self.last_real_action = np.zeros(self.action_space.shape)
        
        # 回傳擴增後的觀測值
        augmented_obs = np.concatenate((obs, self.last_real_action))
        return augmented_obs, info

    def step(self, action):
        # 物理慣性與延遲依然存在
        smoothed_action = self.alpha * action + (1 - self.alpha) * self.last_real_action
        self.last_real_action = smoothed_action
        self.action_queue.append(smoothed_action)
        delayed_action = self.action_queue.popleft() 

        obs, reward, terminated, truncated, info = self.env.step(delayed_action)
        
        # 加上出界死刑 (假設超過 0.8 就判定失敗重來)
        if abs(obs[0]) > 0.8:
            terminated = True

        augmented_obs = np.concatenate((obs, self.last_real_action))
        return augmented_obs, reward, terminated, truncated, info

# ==========================================
# 2. 測試主程式與動畫展示
# ==========================================
if __name__ == "__main__":
    print("🌍 建立視覺化測試環境...")
    # 🌟 【修復 2000 步限制關鍵】：加入 max_episode_steps=100000
    base_env = gym.make('InvertedPendulum-v4', render_mode="human", max_episode_steps=100000)
    env = RealisticMotorWrapper(base_env)

    print("🧠 載入 LQR 靜態穩定大腦...")
    # 載入你剛剛訓練好的模型 (請確認這裡的檔名跟你訓練出來的一致)
    model = DDPG.load(os.path.join(os.path.dirname(__file__), "031"))

    obs, info = env.reset()
    step_count = 0

    print("🎥 測試開始！請欣賞它結合機器學習與 LQR 控制的極致平衡...")
    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        original_action = action[0]
        
        # 實體馬達的死區 (Deadband)，消除微小抖動
        if abs(action[0]) < 0.05:
            action[0] = 0.0
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        time.sleep(0.01) 
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"📍 位置: {obs[0]:>6.3f} m | 🚀 大腦指令: {original_action:>6.3f} N | ⚙️ 實際輸出: {action[0]:>6.3f} N")
        
        if terminated or truncated:
            print("⚠️ 失去平衡或碰觸邊界，重新發球！")
            time.sleep(1)
            obs, info = env.reset()
