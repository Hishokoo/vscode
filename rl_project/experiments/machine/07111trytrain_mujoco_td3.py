# 07111trytrain_mujoco_td3.py
# Sim-to-Real 領域隨機化重校（以 05114 為基底，改動僅限領域隨機化，獎勵函數不動）
#
# 相較於 05114 的改進：
# - 🌟 過於苛刻的部分放寬：delay_steps (1,2)→(0,1)、alpha 0.6 固定→隨機 0.85–0.98
# - 🌟 觀測雜訊改 per-channel 重新標定：位置/角度 std 從 0.01 降到 (0.001, 0.001)，
#   速度 std 維持 0.02（差分估測會放大雜訊，不能一起降）
# - 🌟 新增原本完全沒模擬的實體效應：乾摩擦（滑軌/轉軸）、感測器角度零點偏移、
#   作動器增益誤差、指令抖動（模擬封包遲到/掉幀時驅動器沿用上一個值）
# - 🌟 刪除兩個死參數：position_penalty / velocity_penalty 賦值後從未被讀取，
#   真正生效的是 step() 裡硬寫的 q_pos=5.0 與 q_cart_vel=0.1
# - 🌟 獎勵函數與 LQR 權重完全不動，只改領域隨機化，確保與 05114 的壓力測試比較只有單一變因
#
# 依 prompts/07111_code_review.md 套用的 A 組修正：
# - 🔧 初始角度擾動 ±0.2 → ±0.15：原設定與終止條件 |θ|>0.2 重疊，實測 1.21% 的 episode
#   開局即死，且與 delay_steps 耦合（delay=1 首步必送 0 指令），會污染 ep_len_mean 判讀
# - 🔧 新增 Settling_Time_sustained（最後一次離開 ±0.02 誤差帶之後的索引），舊的
#   Settling_Time_steps（首次進入 ±0.01）保留不動以維持與 05114 可比性
# - 🔧 新增防呆與診斷輸出（n==0 保護、啟動時列印實際生效超參數、initial_angle_abs 指標）
# - 🔒 新增位置終止條件 rail_limit=0.9 m：原生環境只在 |角度|>0.2 終止，位置不在終止條件裡，
#   撞上 ±1.0 m 硬限位只是被 MuJoCo 擋住、episode 照常進行。模擬裡撞牆沒有後果，
#   實體上卻是撞限位開關。設 0.9 留 0.1 m 安全邊界，正常運作不觸發（實測 05114
#   位置中位數僅 0.115 m、最大 0.547 m），純粹作為上機安全保險絲。
#
# 🔒 獨立性：本腳本「從隨機初始化訓練」，執行時不讀取任何 .zip，
#    只要有原始碼 + gymnasium/SB3 就能完整重現，不依賴 05114 或任何既有模型檔。
#    超參數與 05111（TD3 家族唯一從零訓練的版本）一致：lr=1e-3、sigma=0.2、
#    batch=256、buffer=200K、policy_delay=2；步數 150K → 300K（環境多了四層隨機化）。
#    註：原設計是 TD3.load("output/models/05114") 續訓，連帶需要
#    learning_starts=10000 / lr=3e-4 / sigma=0.1 三項「續訓保護」來防災難性遺忘；
#    改為從零之後沒有策略需要保護，這三項已一併移除。
#    代價：與 05114 的比較從「只差領域隨機化」變成「領域隨機化 + 訓練起點都不同」，
#    解讀壓力測試結果時必須把這點納入考量。
#
# 詳細背景與參數推導見 prompts/07111_sim2real_dr.md
# 程式碼檢查與各項證據見 prompts/07111_code_review.md

import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import os


# ==========================================
# 1. AdvancedMotorWrapper
#    （感測器雜訊/零點偏移、平滑度懲罰、Domain Randomization、控制工程指標累積）
# ==========================================
class AdvancedMotorWrapper(gym.Wrapper):
    def __init__(self, env,
                 delay_steps=1, alpha=0.6,
                 force_penalty=0.005, jerk_penalty=0.02,
                 obs_noise_std=(0.001, 0.001, 0.02, 0.02), obs_noise_enable=True,
                 snap_penalty=0.02, smoothness_penalty=0.05,
                 delay_randomize=False, delay_random_range=(0, 1),
                 disturbance_enable=False, disturbance_probability=1.0,
                 disturbance_start_range=(50, 300),
                 disturbance_magnitude_range=(0.3, 0.8),
                 disturbance_duration_range=(3, 8),
                 # ===== 🌟 07111 新增：sim-to-real 領域隨機化 =====
                 alpha_randomize=False, alpha_range=(0.85, 0.98),
                 friction_randomize=False,
                 friction_slide_range=(0.0, 4.0),
                 friction_hinge_range=(0.0, 0.01),
                 angle_bias_enable=False, angle_bias_range=(-0.01, 0.01),
                 gain_error_enable=False, gain_error_range=(0.85, 1.15),
                 jitter_enable=False, jitter_prob=0.02,
                 rail_limit=0.9):
        super().__init__(env)
        # --- 物理硬體參數 ---
        self.delay_steps = delay_steps
        self.alpha = alpha

        # --- LQR 獎勵權重參數 ---
        self.force_penalty = force_penalty
        self.jerk_penalty = jerk_penalty

        # --- 感測器雜訊參數（允許純量或長度 4 的 per-channel 向量）---
        self.obs_noise_std = np.asarray(obs_noise_std, dtype=np.float64)
        self.obs_noise_enable = obs_noise_enable

        # --- 平滑度懲罰參數 ---
        self.snap_penalty = snap_penalty
        self.smoothness_penalty = smoothness_penalty

        # --- Domain Randomization 參數（延遲 / 外力擾動）---
        self.delay_randomize = delay_randomize
        self.delay_random_range = delay_random_range
        self.disturbance_enable = disturbance_enable
        self.disturbance_probability = disturbance_probability
        self.disturbance_start_range = disturbance_start_range
        self.disturbance_magnitude_range = disturbance_magnitude_range
        self.disturbance_duration_range = disturbance_duration_range

        # --- 🌟 Sim-to-Real 領域隨機化參數 ---
        self.alpha_randomize = alpha_randomize
        self.alpha_range = alpha_range
        self.friction_randomize = friction_randomize
        self.friction_slide_range = friction_slide_range
        self.friction_hinge_range = friction_hinge_range
        self.angle_bias_enable = angle_bias_enable
        self.angle_bias_range = angle_bias_range
        self.gain_error_enable = gain_error_enable
        self.gain_error_range = gain_error_range
        self.jitter_enable = jitter_enable
        self.jitter_prob = jitter_prob
        # 🔒 位置終止門檻（公尺）。滑軌硬限位為 ±1.0 m，設 0.9 留 0.1 m 安全邊界。
        #    設為 None 可關閉（例如要重現 05114 的原生行為時）。
        self.rail_limit = rail_limit
        self._rail_limit_hit = False

        # 🌟 必須在任何修改之前記下原始齒輪比，否則每集會以當前值累乘而爆掉
        self._base_gear = float(env.unwrapped.model.actuator_gear[0, 0])
        self._friction_slide = 0.0
        self._friction_hinge = 0.0
        self._angle_bias = 0.0
        self._gain_error = 1.0
        self._last_applied_action = np.zeros(self.action_space.shape)

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

        # 擴增觀測空間：第 5 維為「濾波後、進延遲佇列前」的指令（5 維）
        # TODO(07112): 這一維的語意含糊且跨 episode 漂移。它與真正作用在滑車上的力
        #   相差延遲、jitter、gain_error、摩擦四層，稱不上「馬達真實出力」。
        #   更實質的問題：delay=0 時它只是當步推力的 (1−α) 成分；delay=1 時它就是當步推力的全部。
        #   policy 是無記憶前饋網路，看不到 delay_steps，同一個數字在不同 episode 代表不同物理量。
        #   預定改法二選一：(a) 加 VecFrameStack(n_stack=4) 讓網路能從時間差推斷延遲；
        #   (b) 把 delay_steps 固定為 1（真實系統延遲通常穩定），把隨機化預算挪到別處。
        #   本輪不動：會改變觀測語意，讓與 05114 的壓力測試對照更難解讀。
        #   （註：改為從零訓練後，「維持 5 維以相容 05114 權重」這個約束已不再成立，
        #     所以這項在 07112 是可以自由改的，不需顧慮權重載入。）
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

        # 🔧 除零防護：目前呼叫路徑保證 n >= 1，但不倚賴外部檢查
        if n == 0:
            return {}

        ise  = float(np.sum(angles ** 2))
        iae  = float(np.sum(np.abs(angles)))
        itae = float(np.sum(np.arange(1, n + 1) * np.abs(angles)))

        # 🔧 Settling Time（控制工程定義）：最後一次離開誤差帶「之後」的索引。
        #    與舊的 Settling_Time_steps（首次進入 ±0.01）並存，後者保留以便與 05114 對照。
        #    全程都在帶內 → 0；直到最後一步仍在帶外 → 自然等於 n。
        SETTLE_BAND = 0.02
        outside = np.where(np.abs(angles) > SETTLE_BAND)[0]
        settling_sustained = float(outside[-1] + 1) if len(outside) > 0 else 0.0

        metrics = {
            "Episode_Length":        float(n),
            "ISE":                   ise,
            "IAE":                   iae,
            "ITAE":                  itae,
            "ISE_per_step":          ise  / n,
            "IAE_per_step":          iae  / n,
            "ITAE_per_step":         itae / n,
            "Overshoot_max_angle":   float(np.max(np.abs(angles))),
            # 舊定義：首次進入 ±0.01，保留不動以維持與 05114 的可比性
            "Settling_Time_steps":   float(self.metric_settling_time
                                           if self.metric_settling_time is not None else n),
            # 🔧 新定義：最後一次離開 ±0.02 誤差帶之後的索引（真正的 settling time）
            "Settling_Time_sustained": settling_sustained,
            "Steady_State_Error":    float(np.mean(np.abs(angles[int(n * 0.8):]))) if n >= 5 else 0.0,
            "RMS_Control_Effort":    float(np.sqrt(np.mean(actions ** 2))),
            "Position_ISE":          float(np.sum(positions ** 2)),
            "Position_ISE_per_step": float(np.sum(positions ** 2) / n),
            # 🌟 記錄本集實際抽到的領域隨機化參數，方便日後交叉比對哪種組合會失敗
            "dr_delay_steps":        float(self.delay_steps),
            "dr_alpha":              float(self.alpha),
            "dr_friction_slide":     float(self._friction_slide),
            "dr_angle_bias":         float(self._angle_bias),
            "dr_gain_error":         float(self._gain_error),
            # 🔧 記錄開局角度，事後可濾掉「初始化就接近邊界」的污染樣本
            "initial_angle_abs":     float(abs(self.metric_angle_history[0])),
            # 🔒 本集是否因撞到位置邊界而終止（1.0 = 是）。
            #    這個比例若偏高，代表策略在往邊界衝，是上機前必須先解決的問題。
            "rail_limit_hit":        float(self._rail_limit_hit),
            "max_cart_pos_abs":      float(np.max(np.abs(positions))),
        }
        return metrics

    def _observe(self, clean_obs):
        """
        把物理引擎的乾淨 obs 轉成 agent 實際看到的觀測。
        順序：零點偏移 → 感測器雜訊 → 串接馬達實際出力。
        獎勵與 control/ 指標一律使用 clean_obs，不經過這個函式。
        """
        measured = np.asarray(clean_obs, dtype=np.float64).copy()
        measured[1] += self._angle_bias  # bias 只污染角度通道
        if self.obs_noise_enable:
            measured = measured + np.random.normal(0.0, self.obs_noise_std, size=measured.shape)
        return np.concatenate((measured, self.last_real_action))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 加大初始擾動，讓 Settling Time 出現有意義的收斂數據
        # 🔧 ±0.2 → ±0.15：終止條件是 |θ|>0.2，環境自帶 ±0.01 reset 雜訊疊加後，
        #    原設定有實測 1.21% 的 episode 開局即死（與策略無關的污染樣本），
        #    且 delay=1 的 episode 首步必送 0 指令，在邊界帶死亡率更高，
        #    會讓 dr_delay_steps 帶上與策略無關的偏差。保留 0.05 rad 安全邊界。
        unwrapped = self.env.unwrapped
        qpos = unwrapped.data.qpos.copy()
        qvel = unwrapped.data.qvel.copy()
        qpos[1] += np.random.uniform(-0.15, 0.15)
        qvel[1] += np.random.uniform(-0.1, 0.1)
        unwrapped.set_state(qpos, qvel)
        obs = unwrapped._get_obs()

        # 🌟 1. 乾摩擦隨機化（絕對賦值，不會跨集累乘）
        if self.friction_randomize:
            self._friction_slide = np.random.uniform(*self.friction_slide_range)
            self._friction_hinge = np.random.uniform(*self.friction_hinge_range)
            unwrapped.model.dof_frictionloss[0] = self._friction_slide   # 滑軌，單位 N
            unwrapped.model.dof_frictionloss[1] = self._friction_hinge   # 轉軸，單位 N·m

        # 🌟 2. 作動器增益誤差（以 _base_gear 為基準，避免累乘）
        if self.gain_error_enable:
            self._gain_error = np.random.uniform(*self.gain_error_range)
            unwrapped.model.actuator_gear[0, 0] = self._base_gear * self._gain_error

        # 🌟 3. 感測器零點偏移
        self._angle_bias = (np.random.uniform(*self.angle_bias_range)
                             if self.angle_bias_enable else 0.0)

        # 🌟 4. 馬達一階響應隨機化
        if self.alpha_randomize:
            self.alpha = float(np.random.uniform(*self.alpha_range))

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
        self._last_applied_action = np.zeros(self.action_space.shape)
        self._rail_limit_hit = False
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

        augmented_obs = self._observe(obs)
        return augmented_obs, info

    def step(self, action):
        # TODO(07112): 作動鏈的物理順序反了。目前是「低通濾波 → 延遲佇列 → jitter → MuJoCo」，
        #   但低通濾波模擬的是馬達電感與轉子慣性，位於整條鏈的最下游，
        #   物理正確順序應為「延遲（通訊/運算）→ jitter（掉幀 hold）→ 低通濾波 → 出力」。
        #   具體不自洽：jitter 觸發時驅動器 hold 住舊值，但 last_real_action（濾波器狀態）
        #   已按新指令更新過，hold 的效果沒進到馬達動態。
        #   改法會連帶改變第 5 維定義，須與下方 TODO(07112) 的觀測語意一起處理。
        #   本輪不動：會破壞與 05114 的單一變因對照。
        # ── 步驟 1：低通濾波 + 延遲處理 → new_cmd ──
        smoothed_action = self.alpha * action + (1 - self.alpha) * self.last_real_action
        self.last_real_action = smoothed_action
        self.action_queue.append(smoothed_action)
        new_cmd = self.action_queue.popleft()

        # 🌟 指令抖動：模擬封包遲到 / 掉幀，驅動器沿用上一個值而非新指令
        if self.jitter_enable and np.random.random() < self.jitter_prob:
            applied_action = self._last_applied_action.copy()
        else:
            applied_action = new_cmd
        self._last_applied_action = applied_action.copy()

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
        obs, _, terminated, truncated, info = self.env.step(applied_action)

        # ── 步驟 4：用「乾淨 obs」計算 LQR 獎勵（不可用雜訊/bias 版）──
        cart_pos   = obs[0]
        pole_angle = obs[1]
        cart_vel   = obs[2]
        pole_vel   = obs[3]

        q_angle    = 10.0 * (pole_angle ** 2)
        q_pos      = 5.0  * (cart_pos   ** 2)
        q_pole_vel = 0.5  * (pole_vel   ** 2)
        q_cart_vel = 0.1  * (cart_vel   ** 2)

        # ── 步驟 5：計算所有懲罰項 ──
        # TODO(07112): r_action 用 applied_action，但 r_jerk / r_snap / r_smooth 用 action（raw）。
        #   jitter 觸發的那 10% 步數裡，agent 會因為一個它沒下、也無法預測的指令被罰，
        #   這筆懲罰是純噪音，只會推高 TD loss 的下限。
        #   預定改法：r_action 也改用 action，四項懲罰統一針對網路自身輸出
        #   （實體出力的代價已由 gain_error 與摩擦間接反映在狀態裡）。
        #   本輪不動：會改變獎勵函數。
        r_action = self.force_penalty * np.sum(np.square(applied_action))
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

        # TODO(07112): angle_bias 上限 ±0.01 與死區半寬 0.02 造成餘裕不對稱。
        #   +2 的獎勵一定拿得到（0.01 < 0.02），但 agent 不知道當集偏向哪一邊，
        #   唯一的理性應對是假設最壞情況、對齊到兩邊都安全的 ±0.01，
        #   等於學出比必要更保守的策略。預定改法：angle_bias_range 縮到 ±0.005，
        #   或死區放寬到 0.04，兩者擇一。本輪不動：會改變獎勵函數。
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
        self.metric_action_history.append(float(applied_action[0]))

        if self.metric_settling_time is None and abs(pole_angle) < 0.01:
            self.metric_settling_time = self.metric_step_count

        # 🔒 位置終止（上機安全保險絲）
        #   InvertedPendulum-v4 原生只在 |角度|>0.2 終止，位置完全不在終止條件裡。
        #   滑軌雖有 ±1.0 m 硬限位（jnt_limited=1），但撞上去只是被 MuJoCo 擋住、
        #   episode 照常進行 —— 在模擬裡「撞牆」沒有任何後果，實體上卻是撞限位開關或撞機構。
        #   設在 0.9 留 0.1 m 安全邊界：實測 05114 運行中位置中位數僅 0.115 m、最大 0.547 m，
        #   正常運作時不會觸發，只在策略真的往邊界衝時介入，教它「碰到邊界 = 失敗」。
        if self.rail_limit is not None and abs(cart_pos) > self.rail_limit:
            terminated = True
            self._rail_limit_hit = True

        if terminated or truncated:
            info["control_metrics"] = self._compute_episode_metrics()

        # ── 步驟 9：更新 last_last_raw_action（順序不能錯）──
        self.last_last_raw_action = self.last_raw_action.copy()

        # ── 步驟 10：更新 last_raw_action ──
        self.last_raw_action = action.copy()

        # ── 步驟 11：更新 action_history ──
        self.action_history.append(action.copy())

        # ── 步驟 12：組裝 agent 實際看到的觀測（零點偏移 + 雜訊 + 串接出力）──
        augmented_obs = self._observe(obs)
        return augmented_obs, modified_reward, terminated, truncated, info


# ==========================================
# 2. ControlMetricsCallback
#    （控制工程指標寫入 TensorBoard，含 dr_xxx 領域隨機化取樣值）
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
    # 3.1 🌍 建立環境並解除步數限制
    base_env = gym.make('InvertedPendulum-v4', max_episode_steps=2000)
    env = AdvancedMotorWrapper(
        base_env,
        # --- 作動路徑：大幅放寬（原本過於苛刻，見 prompts/07111_sim2real_dr.md）---
        # delay_random_range=(0,1) 為均勻 50/50。實體延遲推估：25 Hz 下一步=40 ms，
        # 而編碼器+MCU 的端到端延遲僅 2~10 ms（= 0.05~0.25 步），所以 delay=0 才是現實值，
        # delay=1 相當於現實的 4~20 倍。保留它當安全餘裕 —— 實測 delay=1 在 jitter=0 時
        # 仍是 100% 滿分，成本極低，而實體延遲有可能比預期高。
        delay_randomize=True,  delay_random_range=(0, 1),
        alpha_randomize=True,  alpha_range=(0.85, 0.98),

        # --- 感測：per-channel 重新標定 ---
        obs_noise_std=(0.001, 0.001, 0.02, 0.02),
        angle_bias_enable=True, angle_bias_range=(-0.01, 0.01),

        # --- 新增的實體效應（05114 完全沒有）---
        friction_randomize=True,
        friction_slide_range=(0.0, 4.0),
        friction_hinge_range=(0.0, 0.01),
        gain_error_enable=True, gain_error_range=(0.85, 1.15),
        # jitter_prob 0.10 → 0.02：原值是憑直覺設的，無量測根據。
        # 實際掉幀率：編碼器直連 MCU(有線) <0.1%、USB serial+Python 1~5%、WiFi/視覺 5~10%。
        # 0.10 等於把有線系統當成不穩定無線在訓練。實測 0.10 會讓 delay=1 的滿分率
        # 從 100% 掉到 17.5%，是目前唯一的效能瓶頸（其餘六項 DR 代價都在 0~5%）。
        jitter_enable=True,     jitter_prob=0.02,

        # --- 外力擾動：沿用 05114，不動 ---
        disturbance_enable=True,
        disturbance_probability=0.3,
        disturbance_start_range=(50, 300),
        disturbance_magnitude_range=(0.1, 0.3),
        disturbance_duration_range=(2, 5),
    )

    # 3.2 🎲 設定探索雜訊
    # sigma=0.2：與 05111（同樣從零訓練）一致。隨機初始化需要充分探索，
    # 不採用續訓版本的 0.1 —— 那個值是為了保護已收斂策略，這裡沒有策略要保護。
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    # 🌟 統一 TensorBoard 儲存路徑（腳本一律以 rl_project/ 為工作目錄執行）
    log_dir = "tensorboard_logs/machine/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("output/models", exist_ok=True)

    print("🚀 啟動 TD3 Sim-to-Real 領域隨機化重校訓練（從零初始化，不依賴任何既有模型）...")

    # 3.3 🤖 建立 TD3 模型（🔒 從隨機初始化開始，完全不依賴 05114.zip）
    #    超參數與 05111（TD3 家族唯一從零訓練的版本）完全一致，
    #    確保「從零訓練」這件事本身不引入新變因。
    #    刻意不用 TD3.load()：本腳本要能在只有原始碼、沒有任何 .zip 的環境重現。
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

    # 🔧 診斷輸出：確認實際生效的超參數
    print(f"   lr={model.learning_rate}  learning_starts={model.learning_starts}  "
          f"batch={model.batch_size}  train_freq={model.train_freq}  "
          f"sigma={model.action_noise._sigma if model.action_noise else None}")
    print(f"   buffer size at start = {model.replay_buffer.size()}  (從零開始，應為 0)")

    # 3.4 🏋️ 開始訓練（含控制工程指標 Callback）
    # ℹ️ 從零訓練的預期曲線：ep_len_mean 由低往高爬升，不會出現續訓才有的「斷崖」。
    #    因此原本「前 20K 步掉到 100 以下就中止」的災難性遺忘檢查已不適用。
    #    改為觀察：若 100K 步後 ep_len_mean 仍停在 50 以下，代表領域隨機化過強，
    #    優先調降 friction_slide_range 上限與 jitter_prob 再重跑。
    # 步數 150K → 300K：從零開始且環境比 05111 多了摩擦/bias/增益誤差/抖動四層隨機化。
    callback = ControlMetricsCallback()
    model.learn(
        total_timesteps=300000,
        tb_log_name="07111_Sim2Real_DR",
        callback=callback,
    )

    # 3.5 💾 儲存模型
    model.save("output/models/07111")
    print("✅ 訓練完成！模型已儲存為 'output/models/07111.zip'")
