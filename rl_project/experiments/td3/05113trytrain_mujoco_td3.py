# 051trytrain_mujoco_td3.py
# 相較於 041 的改進：
# - 演算法升級：DDPG → TD3（雙 Critic、Target Policy Smoothing、延遲 Actor 更新）
# - AdvancedMotorWrapper：新增感測器雜訊、二階差分（Snap）懲罰、移動平均偏差懲罰
# - 控制工程指標監控：ISE、IAE、ITAE、超調量、穩定時間、穩態誤差、RMS 控制力、位置 ISE

import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import os


# ==========================================
# 1. AdvancedMotorWrapper
#    （含感測器雜訊、平滑度懲罰、控制工程指標累積）
# ==========================================
class AdvancedMotorWrapper(gym.Wrapper):
    def __init__(self, env,
                 delay_steps=1, alpha=0.6,
                 force_penalty=0.005, jerk_penalty=0.02,
                 position_penalty=0.5, velocity_penalty=0.1,
                 obs_noise_std=0.01, obs_noise_enable=True,
                 snap_penalty=0.02, smoothness_penalty=0.05):
        super().__init__(env)
        # --- 物理硬體參數 ---
        self.delay_steps = delay_steps
        self.alpha = alpha

        # --- LQR 獎勵權重參數 ---
        self.force_penalty = force_penalty
        self.jerk_penalty = jerk_penalty
        self.position_penalty = position_penalty
        self.velocity_penalty = velocity_penalty

        # --- 感測器雜訊參數 ---
        self.obs_noise_std = obs_noise_std
        self.obs_noise_enable = obs_noise_enable

        # --- 平滑度懲罰參數 ---
        self.snap_penalty = snap_penalty
        self.smoothness_penalty = smoothness_penalty

        # --- 狀態記憶體 ---
        self.action_queue = deque(maxlen=delay_steps + 1)
        self.last_raw_action = np.zeros(self.action_space.shape)
        self.last_last_raw_action = np.zeros(self.action_space.shape)
        self.last_real_action = np.zeros(self.action_space.shape)
        self.action_history = deque(maxlen=5)

        # 擴增觀測空間：把馬達真實出力加入 AI 的大腦輸入中
        orig_obs_space = env.observation_space
        new_shape = (orig_obs_space.shape[0] + self.action_space.shape[0],)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=new_shape, dtype=np.float32
        )

        self._reset_metrics()

    def _reset_metrics(self):
        self.metric_angle_history = []
        self.metric_pos_history = []
        self.metric_action_history = []
        self.metric_step_count = 0
        self.metric_settling_time = None

    def _compute_episode_metrics(self):
        angles = np.array(self.metric_angle_history)
        positions = np.array(self.metric_pos_history)
        actions = np.array(self.metric_action_history)
        n = len(angles)

        ise  = float(np.sum(angles ** 2))
        iae  = float(np.sum(np.abs(angles)))
        itae = float(np.sum(np.arange(1, n + 1) * np.abs(angles)))

        metrics = {
            "ISE":                 ise,
            "IAE":                 iae,
            "ITAE":                itae,
            # 修正 3：加入每步平均版本，排除 episode 長度膨脹影響
            "ISE_per_step":        ise  / n,
            "IAE_per_step":        iae  / n,
            "ITAE_per_step":       itae / n,
            "Overshoot_max_angle": float(np.max(np.abs(angles))),
            "Settling_Time_steps": float(self.metric_settling_time if self.metric_settling_time else n),
            "Steady_State_Error":  float(np.mean(np.abs(angles[int(n * 0.8):]))) if n >= 5 else 0.0,
            "RMS_Control_Effort":  float(np.sqrt(np.mean(actions ** 2))),
            "Position_ISE":         float(np.sum(positions ** 2)),
            "Position_ISE_per_step": float(np.sum(positions ** 2) / n),
        }
        return metrics

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 加大初始擾動，讓 Settling Time 出現有意義的收斂數據
        unwrapped = self.env.unwrapped
        qpos = unwrapped.data.qpos.copy()
        qvel = unwrapped.data.qvel.copy()
        qpos[1] += np.random.uniform(-0.2, 0.2)   # 極角擾動 ±0.2 rad
        qvel[1] += np.random.uniform(-0.1, 0.1)   # 極速擾動 ±0.1 rad/s
        unwrapped.set_state(qpos, qvel)
        obs = unwrapped._get_obs()

        for _ in range(self.delay_steps + 1):
            self.action_queue.append(np.zeros(self.action_space.shape))
        self.last_raw_action = np.zeros(self.action_space.shape)
        self.last_last_raw_action = np.zeros(self.action_space.shape)
        self.last_real_action = np.zeros(self.action_space.shape)
        self.action_history.clear()

        self._reset_metrics()

        if self.obs_noise_enable:
            noisy_obs = obs + np.random.normal(0, self.obs_noise_std, size=obs.shape)
        else:
            noisy_obs = obs

        augmented_obs = np.concatenate((noisy_obs, self.last_real_action))
        return augmented_obs, info

    def step(self, action):
        # ── 步驟 1：低通濾波 + 延遲處理 → delayed_action ──
        smoothed_action = self.alpha * action + (1 - self.alpha) * self.last_real_action
        self.last_real_action = smoothed_action
        self.action_queue.append(smoothed_action)
        delayed_action = self.action_queue.popleft()

        # ── 步驟 2：與物理引擎互動，取得乾淨 obs ──
        obs, _, terminated, truncated, info = self.env.step(delayed_action)

        # ── 步驟 3：用「乾淨 obs」計算 LQR 獎勵（不可用雜訊版）──
        cart_pos   = obs[0]
        pole_angle = obs[1]
        cart_vel   = obs[2]
        pole_vel   = obs[3]

        q_angle    = 10.0 * (pole_angle ** 2)
        q_pos      = 5.0  * (cart_pos   ** 2)  # 修正 3：進一步加強位置懲罰，抑制 Position ISE 後段暴增
        q_pole_vel = 0.5  * (pole_vel   ** 2)
        q_cart_vel = 0.1  * (cart_vel   ** 2)

        # ── 步驟 4：計算所有懲罰項 ──
        r_action = self.force_penalty * np.sum(np.square(delayed_action))
        r_jerk   = self.jerk_penalty  * np.sum(np.square(action - self.last_raw_action))

        # 🌟 Snap（二階差分懲罰，描述動作加速度的變化）
        r_snap = self.snap_penalty * np.sum(
            np.square(action - 2 * self.last_raw_action + self.last_last_raw_action)
        )

        # 🌟 Moving Average Deviation（移動平均偏差懲罰，在 history 累積夠才啟動）
        if len(self.action_history) >= 3:
            action_mean = np.mean(list(self.action_history), axis=0)
            r_smooth = self.smoothness_penalty * np.sum(np.square(action - action_mean))
        else:
            r_smooth = 0.0

        # ── 步驟 5：計算穩態死區超級獎勵 ──
        modified_reward = 1.0 - (q_angle + q_pos + q_pole_vel + q_cart_vel
                                 + r_action + r_jerk + r_snap + r_smooth)

        if abs(pole_angle) < 0.02 and abs(pole_vel) < 0.05:
            modified_reward += 2.0
            if abs(cart_pos) < 0.1 and abs(cart_vel) < 0.1:
                modified_reward += 5.0

        # ── 步驟 6：Clip 獎勵，防止梯度爆炸 ──
        modified_reward = float(np.clip(modified_reward, -5.0, 10.0))

        # ── 步驟 7：累積控制工程指標（使用乾淨 obs）──
        self.metric_step_count += 1
        self.metric_angle_history.append(pole_angle)
        self.metric_pos_history.append(cart_pos)
        self.metric_action_history.append(float(delayed_action[0]))

        if self.metric_settling_time is None and abs(pole_angle) < 0.01:  # 修正 2：門檻從 0.02 收緊至 0.01，避免指標失效
            self.metric_settling_time = self.metric_step_count

        if terminated or truncated:
            info["control_metrics"] = self._compute_episode_metrics()

        # ── 步驟 8：更新 last_last_raw_action（順序不能錯）──
        self.last_last_raw_action = self.last_raw_action.copy()

        # ── 步驟 9：更新 last_raw_action ──
        self.last_raw_action = action.copy()

        # ── 步驟 10：更新 action_history ──
        self.action_history.append(action.copy())

        # ── 步驟 11：對乾淨 obs 加入感測器雜訊 ──
        if self.obs_noise_enable:
            noisy_obs = obs + np.random.normal(0, self.obs_noise_std, size=obs.shape)
        else:
            noisy_obs = obs

        # ── 步驟 12：拼接 noisy_obs + last_real_action 回傳給 Agent ──
        augmented_obs = np.concatenate((noisy_obs, self.last_real_action))

        return augmented_obs, modified_reward, terminated, truncated, info


# ==========================================
# 2. ControlMetricsCallback
#    （控制工程指標寫入 TensorBoard）
# ==========================================
class ControlMetricsCallback(BaseCallback):
    """
    從 env info 提取控制工程指標，每個 episode 結束時寫入 TensorBoard。
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "control_metrics" in info:
                metrics = info["control_metrics"]
                self.episode_count += 1
                for key, value in metrics.items():
                    self.logger.record(f"control/{key}", value)
                self.logger.dump(self.num_timesteps)  # 強制 flush，確保每集寫入
        return True

    def _on_training_end(self) -> None:
        """
        🌟 訓練結束時，強制計算並寫入最後一集的控制指標。
        即使最後一集尚未因 terminated/truncated 結束，
        也把已累積的數據寫入 TensorBoard，避免數據遺失。
        """
        try:
            env = self.training_env.envs[0]
            wrapper = env
            while wrapper is not None:
                if hasattr(wrapper, '_compute_episode_metrics'):
                    break
                wrapper = getattr(wrapper, 'env', None)

            if wrapper is not None and wrapper.metric_step_count > 0:
                metrics = wrapper._compute_episode_metrics()
                self.episode_count += 1
                for key, value in metrics.items():
                    self.logger.record(f"control/{key}", value)
                self.logger.record("control/_final_episode", 1.0)
                self.logger.dump(self.num_timesteps)
                print(f"📊 已回收最後一集數據（{wrapper.metric_step_count} 步）寫入 TensorBoard")
        except Exception as e:
            print(f"⚠️ 回收最後一集數據時發生錯誤：{e}")


# ==========================================
# 3. 主訓練程式
# ==========================================
if __name__ == "__main__":
    # 3.1 🌍 建立環境並解除步數限制
    base_env = gym.make('InvertedPendulum-v4', max_episode_steps=2000)  # 縮短單集步數，讓 TensorBoard 後段有足夠紀錄點
    env = AdvancedMotorWrapper(base_env)

    # 3.2 🎲 設定探索雜訊（sigma=0.2，補償 TD3 延遲 Actor 更新）
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    # 🌟 統一 TensorBoard 儲存路徑
    log_dir = "tensorboard_logs/td3/"
    os.makedirs(log_dir, exist_ok=True)

    print("🚀 啟動 TD3 進階控制訓練計畫（感測器雜訊 + 平滑度懲罰 + 控制工程指標）...")

    # 3.3 🤖 建立 TD3 模型
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log=log_dir,
        policy_delay=2,           # Actor 每 2 步才更新一次（TD3 核心）
        target_policy_noise=0.2,  # Target Policy Smoothing 雜訊
        target_noise_clip=0.5,    # 雜訊裁切範圍
        learning_rate=1e-3,
        batch_size=256,
        buffer_size=200000,
    )

    # 3.4 🏋️ 開始訓練（含控制工程指標 Callback）
    callback = ControlMetricsCallback()
    model.learn(
        total_timesteps=150000,
        tb_log_name="0513_Advanced_control",
        callback=callback
    )

    # 3.5 💾 儲存模型
    model.save("output/models/05113")
    print("✅ 訓練完成！模型已儲存為 'output/models/05113.zip'")
