import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# ==========================================
# 1. 建立 MuJoCo 連續滑車倒立擺環境
# ==========================================
env = gym.make('InvertedPendulum-v4')

# ==========================================
# 2. 設定 DDPG 專屬的「探索雜訊」
# ==========================================
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# ==========================================
# 3. 建立並訓練 DDPG 大腦 (🌟 加上 TensorBoard)
# ==========================================
print("🤖 開始使用 DDPG 訓練連續滑車倒立擺 (請耐心等候幾分鐘)...")

# 🌟 統一 TensorBoard 儲存路徑
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="rl_project/tensorboard_logs/")

# 🌟 在 learn 裡面加上 tb_log_name，給予清楚的名稱以利比較
model.learn(total_timesteps=50000, tb_log_name="011_Basic_DDPG")

# 把訓練好的大腦存檔
model.save("011")
print("✅ 訓練完成！模型已儲存為 '011.zip'")

env.close()

# ==========================================
# 4. 成果發表：開啟畫面觀看 AI 的細膩操作
# ==========================================
print("🎥 開啟動畫視窗，檢驗連續控制成果...")

env = gym.make('InvertedPendulum-v4', render_mode="human")
observation, info = env.reset()

for i in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"當下施加於滑車的推力: {action[0]:.3f} N")
    
    if terminated or truncated:
        print("⚠️ 失去平衡，重新啟動！")
        observation, info = env.reset()

env.close()