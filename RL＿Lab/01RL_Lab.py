# 執行 `pip install "gymnasium[classic-control]"` 來安裝此範例所需的套件。
import gymnasium as gym

# 建立我們的訓練環境 - 一個需要平衡上方桿子的台車 (CartPole)
# 這裡使用 render_mode="human"，執行時會彈出一個視窗顯示即時動畫
env = gym.make("CartPole-v1", render_mode="human")

# 重置環境，初始化系統以開始一個新的回合 (episode)
observation, info = env.reset()

# observation: 智能體 (agent) 能「看」到的觀測值 - 包含台車位置、速度、桿子角度等。
# info: 額外的除錯資訊 (在基礎的演算法學習中通常不需要用到)
print(f"初始觀測值: {observation}")
# 範例輸出: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# 對應意義: [台車位置, 台車速度, 桿子角度, 桿子角速度]

episode_over = False
total_reward = 0

# 馬可夫決策過程 (MDP) 的主迴圈，直到回合結束為止
while not episode_over:
    # 選擇一個動作 (Action): 0 = 向左推台車, 1 = 向右推台車
    # 目前我們只是隨機抽樣動作 - 未來你訓練的智能體會根據神經網路來做出更聰明的決策！
    action = env.action_space.sample()  

    # 執行該動作，並讓物理引擎計算下一幀發生的事情
    observation, reward, terminated, truncated, info = env.step(action)

    # reward (獎勵): 桿子每保持直立一個時間步 (step)，系統就會給予 +1 的獎勵
    # terminated (終止): 如果桿子傾斜角度過大，或台車跑出邊界 (代表智能體失敗)，則為 True
    # truncated (截斷): 如果我們達到了這個環境的時間上限 (通常是 500 步)，則為 True

    total_reward += reward
    # 只要滿足失敗(terminated)或超時(truncated)其中一項，回合就宣告結束
    episode_over = terminated or truncated

print(f"回合結束！總獎勵: {total_reward}")

# 關閉環境與彈出的顯示視窗
env.close()