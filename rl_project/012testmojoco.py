import gymnasium as gym
import time  # 🌟 記得在檔案最上方或是這裡 import time
from stable_baselines3 import DDPG
import os

# ==========================================
# 1. 建立環境 (開啟動畫模式)
# ==========================================
env = gym.make('InvertedPendulum-v4', render_mode="human")

# ==========================================
# 2. 喚醒我們剛剛存檔的 AI 大腦！
# ==========================================
print("🧠 正在載入訓練好的 DDPG 大腦...")
# 直接讀取 .zip 檔，瞬間載入！
model = DDPG.load(os.path.join(os.path.dirname(__file__), "011"))

# ==========================================
# 3. 無限期表演迴圈 (直到你手動關閉)
# ==========================================
observation, info = env.reset()
print("🎥 開始無限期測試，如果你想結束，請在終端機按下 Ctrl + C")

# 把原本的 for 迴圈改成 while True (無限迴圈)
while True:
    # 讓大腦預測動作 (deterministic=True 拿出真本事)
    action, _states = model.predict(observation, deterministic=True)
    
    # 執行動作
    observation, reward, terminated, truncated, info = env.step(action)
    
    # 如果倒了或是超時，就重置環境繼續跑
    if terminated or truncated:
        print("⚠️ 失去平衡，自動重新發球！")
        observation, info = env.reset()

print("🎥 開啟動畫視窗，檢驗連續控制成果...")
env = gym.make('InvertedPendulum-v4', render_mode="human")
observation, info = env.reset()

for i in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"當下推力: {action[0]:.3f} N")
    
    # 🌟 關鍵煞車：強迫程式每跑一步就暫停 0.02 秒 (相當於 50 FPS)
    time.sleep(0.02) 
    
    if terminated or truncated:
        print("⚠️ 失去平衡，重新啟動！")
        # 為了看清楚跌倒，跌倒時可以暫停久一點點再重置
        time.sleep(0.5) 
        observation, info = env.reset()

env.close()