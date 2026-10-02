# 0611auto_optimize_loop.py
# 半自動優化迴圈：從 05113.zip 出發，自動訓練最多 10 輪（版本 06111~06120）
# 每輪結束後解析 tfevents，根據規則 D→A→C→B→E 調整參數，直到停止條件或跑滿 10 輪

import os
import struct
import math
import gymnasium as gym
import numpy as np
from collections import deque, defaultdict
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
# 3. tfevents 解析
# ==========================================
def read_varint(data, pos):
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80): break
        shift += 7
    return result, pos

def read_field(data, pos):
    tag, pos = read_varint(data, pos)
    fn, wt = tag >> 3, tag & 0x07
    if wt == 0:
        val, pos = read_varint(data, pos); return fn, wt, val, pos
    elif wt == 1:
        val = struct.unpack_from('<d', data, pos)[0]; return fn, wt, val, pos+8
    elif wt == 2:
        l, pos = read_varint(data, pos); val = data[pos:pos+l]; return fn, wt, val, pos+l
    elif wt == 5:
        val = struct.unpack_from('<f', data, pos)[0]; return fn, wt, val, pos+4
    return fn, wt, None, pos

def parse_summary_value(data):
    pos, tag_str, simple_value = 0, None, None
    while pos < len(data):
        fn, wt, val, pos = read_field(data, pos)
        if fn == 1 and wt == 2: tag_str = val.decode('utf-8', errors='replace')
        elif fn == 2 and wt == 5: simple_value = val
    return tag_str, simple_value

def parse_summary(data):
    values, pos = [], 0
    while pos < len(data):
        fn, wt, val, pos = read_field(data, pos)
        if fn == 1 and wt == 2:
            t, v = parse_summary_value(val)
            if t and v is not None: values.append((t, v))
    return values

def parse_event(data):
    pos, wall_time, step, summary = 0, None, None, None
    while pos < len(data):
        fn, wt, val, pos = read_field(data, pos)
        if fn == 1 and wt == 1: wall_time = val
        elif fn == 2 and wt == 0: step = val
        elif fn == 5 and wt == 2: summary = val
    return wall_time, step, summary

def read_tb(filepath):
    try:
        records = []
        with open(filepath, 'rb') as f:
            while True:
                lb = f.read(8)
                if len(lb) < 8: break
                l = struct.unpack('<Q', lb)[0]
                f.read(4); data = f.read(l); f.read(4)
                records.append(data)
        by_tag = defaultdict(list)
        for rec in records:
            wt, step, summary = parse_event(rec)
            if summary is not None and step is not None:
                for tag, val in parse_summary(summary):
                    by_tag[tag].append((step, val))
        for k in by_tag:
            by_tag[k] = sorted(by_tag[k], key=lambda x: x[0])
        return by_tag
    except Exception as e:
        print(f"⚠️ tfevents 解析失敗：{e}")
        return {}

def find_tfevents(log_dir):
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if 'tfevents' in f:
                return os.path.join(root, f)
    return None

def analyze_run(version_name, tb_log_base):
    import os, math
    # SB3 names the log dir as "{tb_log_name}_1", "_2", etc. Find the latest one.
    candidates = [
        d for d in os.listdir(tb_log_base)
        if d.startswith(f"{version_name}_") and os.path.isdir(os.path.join(tb_log_base, d))
    ] if os.path.isdir(tb_log_base) else []
    if not candidates:
        print(f"⚠️ 找不到 {tb_log_base}/{version_name}_* 的目錄，跳過分析")
        return None
    log_dir = os.path.join(tb_log_base, sorted(candidates)[-1])
    tf_path = find_tfevents(log_dir)
    if tf_path is None:
        print(f"⚠️ 找不到 {log_dir} 的 tfevents，跳過分析")
        return None

    d = read_tb(tf_path)
    ep_len = d.get('control/Episode_Length', [])
    critic  = d.get('train/critic_loss', [])
    ise     = d.get('control/ISE_per_step', [])
    sse     = d.get('control/Steady_State_Error', [])

    if not ep_len:
        print("⚠️ 無 Episode_Length 資料，跳過分析")
        return None

    total = len(ep_len)
    vals  = [v for s, v in ep_len]
    short_ratio = sum(1 for v in vals if v < 50) / total
    dist_ratio  = sum(1 for v in vals if 50 <= v < 300) / total
    long_ratio  = sum(1 for v in vals if v >= 300) / total
    avg_len     = sum(vals) / total

    critic_vals = [v for s, v in critic if not math.isinf(v)]
    q = len(critic_vals) // 4
    if q >= 2:
        early_c = sum(critic_vals[:q]) / q
        late_c  = sum(critic_vals[-q:]) / q
        critic_trend = (late_c - early_c) / (abs(early_c) + 1e-9)
    else:
        critic_trend = 0.0

    ise_vals = [v for s, v in ise if not math.isinf(v)]
    sse_vals = [v for s, v in sse if not math.isinf(v)]
    ise_final = sum(ise_vals[-10:]) / min(len(ise_vals), 10) if ise_vals else 0
    sse_final = sum(sse_vals[-10:]) / min(len(sse_vals), 10) if sse_vals else 0

    return {
        "total": total,
        "avg_len": avg_len,
        "short_ratio": short_ratio,
        "dist_ratio": dist_ratio,
        "long_ratio": long_ratio,
        "critic_trend": critic_trend,
        "ise_final": ise_final,
        "sse_final": sse_final,
    }


# ==========================================
# 4. 主迴圈
# ==========================================
if __name__ == "__main__":
    TB_BASE = "tensorboard_logs/curriculum"
    OUTPUT_MODELS_DIR = "output/models"

    versions = [
        "06112", "06113", "06114", "06115",
        "06116", "06117", "06118", "06119", "06120"
    ]

    params = {
        "delay2_prob":                0.5,
        "disturbance_probability":    0.3,
        "disturbance_magnitude":      (0.1, 0.3),
        "disturbance_duration":       (2, 5),
    }

    prev_zip = "output/models/06111"

    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
    os.makedirs(TB_BASE, exist_ok=True)

    for version in versions:
        print(f"\n{'='*40}")
        print(f"▶ 開始訓練輪次：{version}，起點：{prev_zip}.zip")
        print(f"  參數：{params}")

        base_env = gym.make('InvertedPendulum-v4', max_episode_steps=2000)
        env = AdvancedMotorWrapper(
            base_env,
            delay_randomize=True,
            delay_random_range=(1, 2),
            delay2_prob=params["delay2_prob"],
            disturbance_enable=True,
            disturbance_start_range=(50, 300),
            disturbance_probability=params["disturbance_probability"],
            disturbance_magnitude_range=params["disturbance_magnitude"],
            disturbance_duration_range=params["disturbance_duration"],
        )

        n_actions = env.action_space.shape[-1]
        model = TD3.load(
            prev_zip,
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
            tb_log_name=version,
            callback=callback,
            reset_num_timesteps=True,
        )
        model.save(f"{OUTPUT_MODELS_DIR}/{version}")
        print(f"💾 模型已儲存：{OUTPUT_MODELS_DIR}/{version}.zip")

        result = analyze_run(version, TB_BASE)
        if result is None:
            print(f"⚠️ {version} 分析失敗，強制停止")
            break

        r = result
        trend_sign = "上升" if r["critic_trend"] > 0 else "下降"
        print(f"""
{'='*40}
輪次：{version}
訓練步數：150,000
起點模型：{prev_zip}.zip
輸出模型：{OUTPUT_MODELS_DIR}/{version}.zip
----------------------------------------
📊 本輪分析：
  總集次：{r['total']}
  平均集長：{r['avg_len']:.1f} 步
  short_ratio (< 50步)：{r['short_ratio']*100:.1f}%  ← 停止條件門檻 15%
  dist_ratio (50-300步)：{r['dist_ratio']*100:.1f}%
  long_ratio (≥ 300步)：{r['long_ratio']*100:.1f}%
  critic_loss 趨勢：{r['critic_trend']*100:+.1f}%（{trend_sign}）
  ISE_per_step 終值：{r['ise_final']:.4f}
  Steady_State_Error 終值：{r['sse_final']:.4f}
----------------------------------------""")

        if r["short_ratio"] < 0.15:
            print(f"🛑 停止條件達成（short_ratio={r['short_ratio']*100:.1f}% < 15%），優化完成。")
            print("="*40)
            break

        old_params = {k: v for k, v in params.items()}

        if r["critic_trend"] > 0.5:                                      # 規則 D
            rule = "D"
            params["delay2_prob"]           = max(0.2, params["delay2_prob"] - 0.05)
            params["disturbance_probability"]= max(0.2, params["disturbance_probability"] - 0.05)
        elif r["short_ratio"] > 0.4:                                      # 規則 A
            rule = "A"
            params["delay2_prob"] = max(0.2, params["delay2_prob"] - 0.05)
        elif r["dist_ratio"] > 0.25:                                      # 規則 C
            rule = "C"
            lo, hi = params["disturbance_magnitude"]
            params["disturbance_magnitude"] = (lo, max(0.2, hi - 0.05))
            dlo, dhi = params["disturbance_duration"]
            params["disturbance_duration"]  = (dlo, max(3, dhi - 1))
        elif r["short_ratio"] <= 0.4 and r["dist_ratio"] < 0.1:          # 規則 B
            rule = "B"
            params["disturbance_probability"] = min(0.5, params["disturbance_probability"] + 0.05)
        else:                                                              # 規則 E
            rule = "E"

        def fmt_change(key, old, new):
            if old == new:
                return f"{new}（不變）"
            return f"{old} → {new}"

        print(f"""🔧 套用規則 {rule}：
  delay2_prob：{fmt_change('delay2_prob', old_params['delay2_prob'], params['delay2_prob'])}
  disturbance_probability：{fmt_change('disturbance_probability', old_params['disturbance_probability'], params['disturbance_probability'])}
  disturbance_magnitude：{fmt_change('disturbance_magnitude', old_params['disturbance_magnitude'], params['disturbance_magnitude'])}
  disturbance_duration：{fmt_change('disturbance_duration', old_params['disturbance_duration'], params['disturbance_duration'])}
----------------------------------------
⏭ 下一輪：{versions[versions.index(version)+1] if version != versions[-1] else '（最後一輪）'}，起點：{OUTPUT_MODELS_DIR}/{version}.zip
{'='*40}""")

        prev_zip = f"{OUTPUT_MODELS_DIR}/{version}"

    print("\n✅ 全部輪次完成。")
