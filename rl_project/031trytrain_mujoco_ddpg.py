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
        # 1. 計算真實馬達物理行為 (低通濾波與延遲)
        smoothed_action = self.alpha * action + (1 - self.alpha) * self.last_real_action
        self.last_real_action = smoothed_action
        self.action_queue.append(smoothed_action)
        delayed_action = self.action_queue.popleft() 

        # 2. 與虛擬物理引擎互動
        obs, reward, terminated, truncated, info = self.env.step(delayed_action)

        # 3. 擷取狀態 (MuJoCo InvertedPendulum: obs[0] 是位置, obs[2] 是速度)
        cart_position = obs[0] 
        cart_velocity = obs[2] 
        
        # 4. 計算 LQR 靜態穩定懲罰
        position_cost = self.position_penalty * (cart_position ** 2)
        velocity_cost = self.velocity_penalty * (cart_velocity ** 2) 
        force_cost = self.force_penalty * np.sum(np.square(delayed_action))
        jerk_cost = self.jerk_penalty * np.sum(np.square(action - self.last_raw_action))
        
        # 5. 計算最終 Reward (確保有一個微小的生存保底分數)
        modified_reward = max(0.05, reward - force_cost - jerk_cost - position_cost - velocity_cost)
        
        # 更新狀態並回傳
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
    model.learn(total_timesteps=150000, tb_log_name="031_Static_Stability")

    # 儲存模型
    model.save("031")
    print("✅ 訓練完成！模型已儲存。")