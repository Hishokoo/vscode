# 05114trytrain_mujoco_td3.py
# 相較於 05114 舊版的改進：
# - 🌟 載入 05113.zip 作為起點（已在 delay=1 訓練 150K 步），不從隨機初始化開始
# - 🌟 觀測空間改回 5 維（與 05113 相容），移除 delay_obs
# - 🌟 全程 delay=(1,2) 各 50% 混合訓練，不分階段
# - 🌟 擾動參數：probability=0.3、magnitude=(0.1,0.3)、duration=(2,5)
# - 🌟 訓練步數：150K、輸出覆寫 05114 命名

import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import os


# ==========================================
# 1. AdvancedMotorWrapper
#    （含感測器雜訊、平滑度懲罰、Domain Randomization、控制工程指標累積）
# ==========================================
class AdvancedMotorWrapper(gym.Wrapper):
    def __init__(self, env,
                 delay_steps=1, alpha=0.6,
                 force_penalty=0.005, jerk_penalty=0.02,
                 position_penalty=0.5, velocity_penalty=0.1,
                 obs_noise_std=0.01, obs_noise_enable=True,
                 snap_penalty=0.02, smoothness_penalty=0.05,
                 delay_randomize=False, delay_random_range=(1, 2),
                 disturbance_enable=False, disturbance_probability=1.0,
                 disturbance_start_range=(50, 300),
                 disturbance_magnitude_range=(0.3, 0.8),
                 disturbance_duration_range=(3, 8)):
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

        # --- Domain Randomization 參數 ---
        self.delay_randomize = delay_randomize
        self.delay_random_range = delay_random_range
        self.disturbance_enable = disturbance_enable
        self.disturbance_probability = disturbance_probability
        self.disturbance_start_range = disturbance_start_range
        self.disturbance_magnitude_range = disturbance_magnitude_range
        self.disturbance_duration_range = disturbance_duration_range

        # 🌟 用於跨步追蹤擾動剩餘步數
        self._disturbance_steps_remaining = 0
        self._disturbance_active_magnitude = 0.0
        self._disturbance_active_direction = 1.0
        self.disturbance_plan = None

        # --- 狀態記憶體 ---
        self.action_queue = deque(maxlen=delay_steps + 1)
        self.last_raw_action = np.zeros(self.action_space.shape)
        self.last_last_raw_action = np.zeros(self.action_space.shape)
        self.last_real_action = np.zeros(self.action_space.shape)
        self.action_history = deque(maxlen=5)

        # 擴增觀測空間：馬達真實出力（與 05113 相容，5 維）
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
        angles    = np.array(self.metric_angle_history)
        positions = np.array(self.metric_pos_history)
        actions   = np.array(self.metric_action_history)
        n = len(angles)

        ise  = float(np.sum(angles ** 2))
        iae  = float(np.sum(np.abs(angles)))
        itae = float(np.sum(np.arange(1, n + 1) * np.abs(angles)))

        metrics = {
            "Episode_Length":        float(n),           # 🌟 新增：每集即時步數
            "ISE":                   ise,
            "IAE":                   iae,
            "ITAE":                  itae,
            "ISE_per_step":          ise  / n,
            "IAE_per_step":          iae  / n,
            "ITAE_per_step":         itae / n,
            "Overshoot_max_angle":   float(np.max(np.abs(angles))),
            "Settling_Time_steps":   float(self.metric_settling_time if self.metric_settling_time else n),
            "Steady_State_Error":    float(np.mean(np.abs(angles[int(n * 0.8):]))) if n >= 5 else 0.0,
            "RMS_Control_Effort":    float(np.sqrt(np.mean(actions ** 2))),
            "Position_ISE":          float(np.sum(positions ** 2)),
            "Position_ISE_per_step": float(np.sum(positions ** 2) / n),
        }
        return metrics

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 加大初始擾動，讓 Settling Time 出現有意義的收斂數據
        unwrapped = self.env.unwrapped
        qpos = unwrapped.data.qpos.copy()
        qvel = unwrapped.data.qvel.copy()
        qpos[1] += np.random.uniform(-0.2, 0.2)
        qvel[1] += np.random.uniform(-0.1, 0.1)
        unwrapped.set_state(qpos, qvel)
        obs = unwrapped._get_obs()

        # 🌟 Domain Randomization：每集隨機抽取延遲步數
        if self.delay_randomize:
            self.delay_steps = np.random.randint(
                self.delay_random_range[0],
                self.delay_random_range[1] + 1
            )
            # 重建 action_queue 以符合新的延遲長度
            self.action_queue = deque(maxlen=self.delay_steps + 1)

        for _ in range(self.delay_steps + 1):
            self.action_queue.append(np.zeros(self.action_space.shape))
        self.last_raw_action = np.zeros(self.action_space.shape)
        self.last_last_raw_action = np.zeros(self.action_space.shape)
        self.last_real_action = np.zeros(self.action_space.shape)
        self.action_history.clear()

        # 🌟 每集依機率決定是否生成外力擾動計畫
        if self.disturbance_enable and np.random.random() < self.disturbance_probability:
            self.disturbance_plan = {
                "start_step": np.random.randint(
                    self.disturbance_start_range[0],
                    self.disturbance_start_range[1] + 1
                ),
                "duration": np.random.randint(
                    self.disturbance_duration_range[0],
                    self.disturbance_duration_range[1] + 1
                ),
                "magnitude": float(np.random.uniform(
                    self.disturbance_magnitude_range[0],
                    self.disturbance_magnitude_range[1]
                )),
                "direction": float(np.random.choice([-1, 1])),
            }
        else:
            self.disturbance_plan = None

        # 重置跨步擾動狀態
        self._disturbance_steps_remaining = 0
        self._disturbance_active_magnitude = 0.0
        self._disturbance_active_direction = 1.0

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

        # ── 步驟 2：🌟 外力擾動（跨步分散施力，每步施加一次角速度衝量）──
        # 檢查是否啟動新的擾動
        if (self.disturbance_plan is not None
                and self._disturbance_steps_remaining == 0
                and self.metric_step_count >= self.disturbance_plan["start_step"]):
            self._disturbance_steps_remaining = self.disturbance_plan["duration"]
            self._disturbance_active_magnitude = self.disturbance_plan["magnitude"]
            self._disturbance_active_direction = self.disturbance_plan["direction"]
            self.disturbance_plan = None  # 清除計畫，避免重複觸發

        # 若擾動進行中，每步施加一次角速度衝量
        if self._disturbance_steps_remaining > 0:
            qpos = self.env.unwrapped.data.qpos.copy()
            qvel = self.env.unwrapped.data.qvel.copy()
            qvel[1] += self._disturbance_active_magnitude * self._disturbance_active_direction
            self.env.unwrapped.set_state(qpos, qvel)
            self._disturbance_steps_remaining -= 1

        # ── 步驟 3：與物理引擎互動，取得乾淨 obs ──
        obs, _, terminated, truncated, info = self.env.step(delayed_action)

        # ── 步驟 4：用「乾淨 obs」計算 LQR 獎勵（不可用雜訊版）──
        cart_pos   = obs[0]
        pole_angle = obs[1]
        cart_vel   = obs[2]
        pole_vel   = obs[3]

        q_angle    = 10.0 * (pole_angle ** 2)
        q_pos      = 5.0  * (cart_pos   ** 2)
        q_pole_vel = 0.5  * (pole_vel   ** 2)
        q_cart_vel = 0.1  * (cart_vel   ** 2)

        # ── 步驟 5：計算所有懲罰項 ──
        r_action = self.force_penalty * np.sum(np.square(delayed_action))
        r_jerk   = self.jerk_penalty  * np.sum(np.square(action - self.last_raw_action))

        # 🌟 Snap（二階差分懲罰，描述動作加速度的變化）
        r_snap = self.snap_penalty * np.sum(
            np.square(action - 2 * self.last_raw_action + self.last_last_raw_action)
        )

        # 🌟 Moving Average Deviation（移動平均偏差懲罰）
        if len(self.action_history) >= 3:
            action_mean = np.mean(list(self.action_history), axis=0)
            r_smooth = self.smoothness_penalty * np.sum(np.square(action - action_mean))
        else:
            r_smooth = 0.0

        # ── 步驟 6：計算穩態死區超級獎勵 ──
        modified_reward = 1.0 - (q_angle + q_pos + q_pole_vel + q_cart_vel
                                 + r_action + r_jerk + r_snap + r_smooth)

        if abs(pole_angle) < 0.02 and abs(pole_vel) < 0.05:
            modified_reward += 2.0
            if abs(cart_pos) < 0.1 and abs(cart_vel) < 0.1:
                modified_reward += 5.0

        # ── 步驟 7：Clip 獎勵，防止梯度爆炸 ──
        modified_reward = float(np.clip(modified_reward, -5.0, 10.0))

        # ── 步驟 8：累積控制工程指標（使用乾淨 obs）──
        self.metric_step_count += 1
        self.metric_angle_history.append(pole_angle)
        self.metric_pos_history.append(cart_pos)
        self.metric_action_history.append(float(delayed_action[0]))

        if self.metric_settling_time is None and abs(pole_angle) < 0.01:
            self.metric_settling_time = self.metric_step_count

        if terminated or truncated:
            info["control_metrics"] = self._compute_episode_metrics()

        # ── 步驟 9：更新 last_last_raw_action（順序不能錯）──
        self.last_last_raw_action = self.last_raw_action.copy()

        # ── 步驟 10：更新 last_raw_action ──
        self.last_raw_action = action.copy()

        # ── 步驟 11：更新 action_history ──
        self.action_history.append(action.copy())

        # ── 步驟 12：對乾淨 obs 加入感測器雜訊 ──
        if self.obs_noise_enable:
            noisy_obs = obs + np.random.normal(0, self.obs_noise_std, size=obs.shape)
        else:
            noisy_obs = obs

        # ── 步驟 13：拼接 noisy_obs + last_real_action 回傳給 Agent ──
        augmented_obs = np.concatenate((noisy_obs, self.last_real_action))
        return augmented_obs, modified_reward, terminated, truncated, info


# ==========================================
# 2. ControlMetricsCallback
#    （控制工程指標寫入 TensorBoard）
# ==========================================
class ControlMetricsCallback(BaseCallback):
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
                self.logger.dump(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
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
                # ✅ 不再寫入 _final_episode 標記
                self.logger.dump(self.num_timesteps)
                print(f"📊 已回收最後一集數據（{wrapper.metric_step_count} 步）寫入 TensorBoard")
        except Exception as e:
            print(f"⚠️ 回收最後一集數據時發生錯誤：{e}")


# ==========================================
# 3. 主訓練程式
# ==========================================
if __name__ == "__main__":
    # 3.1 🌍 建立環境並解除步數限制
    base_env = gym.make('InvertedPendulum-v4', max_episode_steps=2000)
    env = AdvancedMotorWrapper(
        base_env,
        delay_randomize=True,
        delay_random_range=(1, 2),           # 🌟 修正一：縮小至 1~2（原 1~4 Critic 無法收斂）
        disturbance_enable=True,
        disturbance_probability=0.3,                  # 30% 機率觸發，保留探索空間
        disturbance_start_range=(50, 300),
        disturbance_magnitude_range=(0.1, 0.3),       # 縮小衝量：最高 1.5 rad/s（原最高 6.4）
        disturbance_duration_range=(2, 5),            # 縮短持續步數（原 3~8）
    )

    # 3.2 🎲 設定探索雜訊
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    # 🌟 統一 TensorBoard 儲存路徑
    log_dir = "tensorboard_logs/td3/"
    os.makedirs(log_dir, exist_ok=True)

    print("🚀 啟動 TD3 Domain Randomization 訓練計畫（載入 05113 → delay=(1,2) 混合 + 分散施力外擾）...")

    # 3.3 🤖 載入 05113 模型作為起點（obs 5 維相容）
    model = TD3.load(
        "output/models/05113",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
    )
    model.action_noise = action_noise

    # 3.4 🏋️ 開始訓練（含控制工程指標 Callback）
    callback = ControlMetricsCallback()
    model.learn(
        total_timesteps=150000,
        tb_log_name="05114_Domain_Randomization",
        callback=callback,
        reset_num_timesteps=True,
    )

    # 3.5 💾 儲存模型
    model.save("output/models/05114")
    print("✅ 訓練完成！模型已儲存為 'output/models/05114.zip'")
