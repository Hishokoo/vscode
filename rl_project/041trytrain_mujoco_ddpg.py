import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from collections import deque
import os

# ==========================================
# 1. 建立客製化環境 Wrapper (機電整合與 LQR 獎勵)
# ==========================================
class RealisticMotorWrapper(gym.Wrapper):
    def __init__(self, env, delay_steps=1, alpha=0.6, force_penalty=0.005, jerk_penalty=0.02, position_penalty=0.5, velocity_penalty=0.1):
        super().__init__(env)
        # --- 物理硬體參數 ---
        self.delay_steps = delay_steps      
        self.alpha = alpha                  
        
        # --- LQR 獎勵權重參數 ---
        self.force_penalty = force_penalty  
        self.jerk_penalty = jerk_penalty    
        self.position_penalty = position_penalty 
        self.velocity_penalty = velocity_penalty 
        
        # --- 狀態記憶體 ---
        self.action_queue = deque(maxlen=delay_steps + 1)
        self.last_raw_action = np.zeros(self.action_space.shape)
        self.last_real_action = np.zeros(self.action_space.shape)

        # 擴增觀測空間：把馬達真實出力加入 AI 的大腦輸入中
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
        # ... (前面低通濾波與延遲照舊) ...
        smoothed_action = self.alpha * action + (1 - self.alpha) * self.last_real_action
        self.last_real_action = smoothed_action
        self.action_queue.append(smoothed_action)
        delayed_action = self.action_queue.popleft() 

        obs, reward, terminated, truncated, info = self.env.step(delayed_action)
        
        # 取得系統狀態 (針對 InvertedPendulum-v4)
        cart_pos = obs[0]
        pole_angle = obs[1]
        cart_vel = obs[2]
        pole_vel = obs[3]

        # ==========================================
        # 🌟 1. LQR 二次型懲罰 (對應 ISE 指標)
        # ==========================================
        # 角度與位置只要偏離中心，就給予強烈的平方扣分
        q_angle = 10.0 * (pole_angle ** 2)
        q_pos = 1.0 * (cart_pos ** 2)
        
        # 角速度與車速的平方扣分 (相當於增加阻尼 Ratio，消除震盪)
        q_pole_vel = 0.5 * (pole_vel ** 2)
        q_cart_vel = 0.1 * (cart_vel ** 2)

        # 動作扣分 (保護馬達)
        r_action = self.force_penalty * np.sum(np.square(delayed_action))
        r_jerk = self.jerk_penalty * np.sum(np.square(action - self.last_raw_action))

        # 基本存活分數扣掉所有 LQR 成本
        modified_reward = 1.0 - (q_angle + q_pos + q_pole_vel + q_cart_vel + r_action + r_jerk)

        # ==========================================
        # 🌟 2. 穩態死區超級獎勵 (對應 Settling Time 達標)
        # ==========================================
        # 如果角度誤差小於 0.02 弧度 (約1度)，且角速度極小 (代表它完美停住了)
        if abs(pole_angle) < 0.02 and abs(pole_vel) < 0.05:
            # 給予額外的高分獎勵！這會徹底改變 AI 的行為，它會渴望進入這個「完美靜止帶」
            modified_reward += 2.0 
            
            # 如果連車子都停在正中央，再加碼
            if abs(cart_pos) < 0.05 and abs(cart_vel) < 0.05:
                modified_reward += 5.0

        # 防呆機制：讓獎勵不要變成負太多導致梯度爆炸
        modified_reward = np.clip(modified_reward, -5.0, 10.0)

        # ... (後續狀態更新與回傳照舊) ...
        self.last_raw_action = action
        augmented_obs = np.concatenate((obs, self.last_real_action))

        return augmented_obs, modified_reward, terminated, truncated, info

# ==========================================
# 2. 訓練主程式與 TensorBoard 設定
# ==========================================
if __name__ == "__main__":
    # 建立環境並解除 1000 步限制
    base_env = gym.make('InvertedPendulum-v4', max_episode_steps=100000)
    env = RealisticMotorWrapper(base_env)

    # 設定探索雜訊
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    print("🚀 啟動 LQR 靜態穩定訓練計畫...")
    
    # 建立 DDPG 模型 (🌟 統一 TensorBoard 儲存路徑)
    log_dir = "rl_project/tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    model = DDPG(
        "MlpPolicy", 
        env, 
        action_noise=action_noise, 
        verbose=1,
        tensorboard_log=log_dir  # 📍 開啟 TensorBoard 紀錄
    )

    # 開始訓練 (因為任務變複雜了，建議步數拉長到 15 萬步)
    model.learn(total_timesteps=150000, tb_log_name="0412_Advanced_Control")

    # 儲存模型
    model.save("0412")
    print("✅ 訓練完成！模型已儲存。")