# 06111trytrain_mujoco_td3.py
# 課程學習 Stage 1：從 05113.zip 載入，純 delay=2 訓練 150K 步
# 目的：強迫模型學會應對 delay=2，產生足夠的 delay=2 存活樣本
# 輸出：rl_project/machinezip/06111.zip，作為 0611 optimize loop 的起點

import os
import gymnasium as gym
import numpy as np
from collections import deque
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


# ==========================================
# 1. AdvancedMotorWrapper
# ==========================================
class AdvancedMotorWrapper(gym.Wrapper):
    def __init__(self, env,
                 delay_steps=1, alpha=0.6,
                 force_penalty=0.005, jerk_penalty=0.02,
                 position_penalty=0.5, velocity_penalty=0.1,
                 obs_noise_std=0.01, obs_noise_enable=True,
                 snap_penalty=0.02, smoothness_penalty=0.05,
                 delay_randomize=False, delay_random_range=(1, 2),
                 delay2_prob=0.5,
                 disturbance_enable=False, disturbance_probability=1.0,
                 disturbance_start_range=(50, 300),
                 disturbance_magnitude_range=(0.3, 0.8),
                 disturbance_duration_range=(3, 8)):
        super().__init__(env)
        self.delay_steps = delay_steps
        self.alpha = alpha
        self.force_penalty = force_penalty
        self.jerk_penalty = jerk_penalty
        self.position_penalty = position_penalty
        self.velocity_penalty = velocity_penalty
        self.obs_noise_std = obs_noise_std
        self.obs_noise_enable = obs_noise_enable
        self.snap_penalty = snap_penalty
        self.smoothness_penalty = smoothness_penalty
        self.delay_randomize = delay_randomize
        self.delay_random_range = delay_random_range
        self.delay2_prob = delay2_prob
        self.disturbance_enable = disturbance_enable
        self.disturbance_probability = disturbance_probability
        self.disturbance_start_range = disturbance_start_range
        self.disturbance_magnitude_range = disturbance_magnitude_range
        self.disturbance_duration_range = disturbance_duration_range

        self._disturbance_steps_remaining = 0
        self._disturbance_active_magnitude = 0.0
        self._disturbance_active_direction = 1.0
        self.disturbance_plan = None

        self.action_queue = deque(maxlen=delay_steps + 1)
        self.last_raw_action = np.zeros(self.action_space.shape)
        self.last_last_raw_action = np.zeros(self.action_space.shape)
        self.last_real_action = np.zeros(self.action_space.shape)
        self.action_history = deque(maxlen=5)

        orig_obs_space = env.observation_space
        new_shape = (orig_obs_space.shape[0] + self.action_space.shape[0],)  # 4+1=5
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
        return {
            "Episode_Length":        float(n),
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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        unwrapped = self.env.unwrapped
        qpos = unwrapped.data.qpos.copy()
        qvel = unwrapped.data.qvel.copy()
        qpos[1] += np.random.uniform(-0.2, 0.2)
        qvel[1] += np.random.uniform(-0.1, 0.1)
        unwrapped.set_state(qpos, qvel)
        obs = unwrapped._get_obs()

        if self.delay_randomize:
            self.delay_steps = int(np.random.choice(
                [1, 2], p=[1.0 - self.delay2_prob, self.delay2_prob]
            ))
            self.action_queue = deque(maxlen=self.delay_steps + 1)

        for _ in range(self.delay_steps + 1):
            self.action_queue.append(np.zeros(self.action_space.shape))
        self.last_raw_action = np.zeros(self.action_space.shape)
        self.last_last_raw_action = np.zeros(self.action_space.shape)
        self.last_real_action = np.zeros(self.action_space.shape)
        self.action_history.clear()

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
        smoothed_action = self.alpha * action + (1 - self.alpha) * self.last_real_action
        self.last_real_action = smoothed_action
        self.action_queue.append(smoothed_action)
        delayed_action = self.action_queue.popleft()

        if (self.disturbance_plan is not None
                and self._disturbance_steps_remaining == 0
                and self.metric_step_count >= self.disturbance_plan["start_step"]):
            self._disturbance_steps_remaining = self.disturbance_plan["duration"]
            self._disturbance_active_magnitude = self.disturbance_plan["magnitude"]
            self._disturbance_active_direction = self.disturbance_plan["direction"]
            self.disturbance_plan = None

        if self._disturbance_steps_remaining > 0:
            qpos = self.env.unwrapped.data.qpos.copy()
            qvel = self.env.unwrapped.data.qvel.copy()
            qvel[1] += self._disturbance_active_magnitude * self._disturbance_active_direction
            self.env.unwrapped.set_state(qpos, qvel)
            self._disturbance_steps_remaining -= 1

        obs, _, terminated, truncated, info = self.env.step(delayed_action)

        cart_pos   = obs[0]
        pole_angle = obs[1]
        cart_vel   = obs[2]
        pole_vel   = obs[3]

        q_angle    = 10.0 * (pole_angle ** 2)
        q_pos      = 5.0  * (cart_pos   ** 2)
        q_pole_vel = 0.5  * (pole_vel   ** 2)
        q_cart_vel = 0.1  * (cart_vel   ** 2)

        r_action = self.force_penalty * np.sum(np.square(delayed_action))
        r_jerk   = self.jerk_penalty  * np.sum(np.square(action - self.last_raw_action))
        r_snap   = self.snap_penalty  * np.sum(
            np.square(action - 2 * self.last_raw_action + self.last_last_raw_action)
        )

        if len(self.action_history) >= 3:
            action_mean = np.mean(list(self.action_history), axis=0)
            r_smooth = self.smoothness_penalty * np.sum(np.square(action - action_mean))
        else:
            r_smooth = 0.0

        modified_reward = 1.0 - (q_angle + q_pos + q_pole_vel + q_cart_vel
                                 + r_action + r_jerk + r_snap + r_smooth)

        if abs(pole_angle) < 0.02 and abs(pole_vel) < 0.05:
            modified_reward += 2.0
            if abs(cart_pos) < 0.1 and abs(cart_vel) < 0.1:
                modified_reward += 5.0

        modified_reward = float(np.clip(modified_reward, -5.0, 10.0))

        self.metric_step_count += 1
        self.metric_angle_history.append(pole_angle)
        self.metric_pos_history.append(cart_pos)
        self.metric_action_history.append(float(delayed_action[0]))

        if self.metric_settling_time is None and abs(pole_angle) < 0.01:
            self.metric_settling_time = self.metric_step_count

        if terminated or truncated:
            info["control_metrics"] = self._compute_episode_metrics()

        self.last_last_raw_action = self.last_raw_action.copy()
        self.last_raw_action = action.copy()
        self.action_history.append(action.copy())

        if self.obs_noise_enable:
            noisy_obs = obs + np.random.normal(0, self.obs_noise_std, size=obs.shape)
        else:
            noisy_obs = obs

        augmented_obs = np.concatenate((noisy_obs, self.last_real_action))
        return augmented_obs, modified_reward, terminated, truncated, info


# ==========================================
# 2. ControlMetricsCallback
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
                self.logger.dump(self.num_timesteps)
                print(f"📊 已回收最後一集數據（{wrapper.metric_step_count} 步）寫入 TensorBoard")
        except Exception as e:
            print(f"⚠️ 回收最後一集數據時發生錯誤：{e}")


# ==========================================
# 3. 主訓練程式
# ==========================================
if __name__ == "__main__":
    TB_BASE = "tensorboard_logs/curriculum"
    OUTPUT_MODELS_DIR = "output/models"
    os.makedirs(TB_BASE, exist_ok=True)
    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)

    # 課程學習 Stage 1：固定 delay=2，不隨機化
    base_env = gym.make('InvertedPendulum-v4', max_episode_steps=2000)
    env = AdvancedMotorWrapper(
        base_env,
        delay_steps=2,
        delay_randomize=False,
        disturbance_enable=True,
        disturbance_probability=0.3,
        disturbance_start_range=(50, 300),
        disturbance_magnitude_range=(0.1, 0.3),
        disturbance_duration_range=(2, 5),
    )

    n_actions = env.action_space.shape[-1]
    print("🚀 課程學習 Stage 1：載入 05113.zip，純 delay=2 訓練 150K 步...")

    model = TD3.load(
        "output/models/05113",
        env=env,
        verbose=1,
        tensorboard_log=TB_BASE,
    )
    model.action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions)
    )

    callback = ControlMetricsCallback()
    model.learn(
        total_timesteps=150000,
        tb_log_name="06111",
        callback=callback,
        reset_num_timesteps=True,
    )

    model.save(f"{OUTPUT_MODELS_DIR}/06111")
    print(f"✅ Stage 1 完成，模型已儲存：{OUTPUT_MODELS_DIR}/06111.zip")
    print("   → 接著執行 0611auto_optimize_loop.py 開始 Stage 2（delay=(1,2) 混訓）")
