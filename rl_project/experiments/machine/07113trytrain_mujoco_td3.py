# 07113trytrain_mujoco_td3.py
# IP570 直線倒立擺「純 RL 改善滑車位置穩定性」版
#
# 這一版和 07112 的關係：被控體、感測/通訊模型、DR 範圍、終止條件、觀測縮放與獎勵函數全部維持 07112，
#   只改「評估、探索、時間尺度、動作平滑性、critic 輸入」。規格見 prompts/07113_pure_rl_changes.md，
#   被控體參數推導見 prompts/07112_system_id.md 與 07112 檔頭。
#   07112 檔頭寫的「07113 改 residual RL」計畫作廢：部署時仍是純神經網路策略，不疊加 LQI。
#
# 為什麼不改獎勵：07112（300K、seed 0）在同一個獎勵下 LQI 的 return 是 RL 的 2 倍（5349 vs 2690），
#   名目 x 標準差 RL 7.61 cm、LQI 1.24 cm（6.1×）。獎勵本身已偏好「滑車不動」，是 TD3 沒最佳化到位。
#   成功判準：名目 osc_x_std_tail 的 RL/LQI 倍數 ≤ 2，且 angle_limit_hit 不高於 07112。
#
# ─────────────────────────────────────────────────────────────────────────────
# 🌟 1. 評估修正（其餘改動都靠它判斷）
#   --eval-freq 50000 / --eval-episodes 10：訓練中評估，種子 10000 起（挑 best 用）
#   --final-eval-episodes 50：訓練結束對 final 與 best 各跑名目 / 全開 DR，LQI 同組對照，
#       種子 20000 起（與挑 best 的種子錯開，避免 best 分數偏高）
#       → output/results/{run_name}_final_eval.json（全部參數 + 摘要 + 每集原始 metrics）
#   多 seed 彙整：experiments/machine/07113_aggregate.py；批次執行：07113_run_all.py
#
# 🌟 2. 探索雜訊：07112 的 σ=0.2 duty 比 LQI 平衡時的 RMS duty（0.13）還大，target 雜訊 0.2 會把 Q 抹平
#   --explore-sigma 0.08 → --explore-sigma-final 0.02（排程同 07112：前 60% 固定、60%→90% 線性下降）
#   --noise-type normal | ou（OU：theta=0.15, dt=1e-2，時間相關、對慢模態 x 較有探索效果）
#   --target-noise 0.05 / --target-noise-clip 0.1
#
# 🌟 3. 時間尺度 --ctrl-hz 50 | 25
#   每個動作 20 ms 對慢模態 x 的 Q 值差距太小，被 critic 誤差淹沒（Tallec et al. 2019）。
#   所有跟頻率有關的量集中在 Timing（由 main() 建立後明確傳入，不依賴 import 時綁定的預設值）：
#       50 Hz：10 子步、dt 0.02、1000 步、γ 0.995      25 Hz：20 子步、dt 0.04、500 步、γ 0.99
#       γ = 0.99^(dt/0.04)，維持約 4 秒視野；擾動起點 2–15 s、持續 40–100 ms 以秒定義後換算
#   TensorBoard 另記 time/sim_seconds，可同時比較「相同步數」與「相同模擬時間」。
#   ⚠️ 上機時 MCU/PC 的控制週期必須與訓練一致（看 {run_name}_config.json 的 timing）。
#
# 🌟 4. CAPS 動作平滑（Mysore et al. 2021）：--caps-lambda-t / --caps-lambda-s / --caps-sigma
#   actor_loss = −Q/mean|Q| + λ_T‖π(s_t)−π(s_t+1)‖ + λ_S‖π(s)−π(s+ε)‖，ε 只加在 6 維 "obs"
#   兩個 λ 皆 0 時直接用原版 TD3（行為與原版完全一致）。獎勵裡的 jerk/snap 懲罰保留不動。
#
# 🌟 5. Asymmetric critic（Pinto et al. 2018）：--asym-critic
#   actor 仍只看 6 維觀測（上機用），critic 另看 N_PRIV 維特權資訊（真實狀態、本集物理/感測參數、
#   本步延遲、外力）。觀測變成 Dict{"obs","priv"}；上機只需 actor：
#       duty = model.actor.mu(obs6)   或載入 output/models/{run_name}_actor.pt
#
# 🌟 --preset 07112：σ 0.2→0.05、target 0.2/0.5、50 Hz、無 CAPS、無 asym（明確指定的參數優先）
#   run 名稱：07113_{tag}_{ctrl_hz}hz_s{seed}；TensorBoard、模型、結果檔都用這個名稱
#
# 🔒 路徑：以本檔位置推出專案根目錄 rl_project/（parents[2]），output/、tensorboard_logs/ 不論從哪裡執行
#    都落在 rl_project/ 底下（07112 從 vscode/ 執行時會跑到 vscode/output/）。
#
# 🔒 先講清楚的物理事實（實測於本模型的 LQR，無感測雜訊）：
#   - 無摩擦 → 完全靜止。滑車庫倫摩擦 2.5 N → x 標準差 12 mm 的 0.5 Hz 極限環；
#     電位器摩擦 1e-3 N·m（規格上限）→ θ 標準差 0.011 rad。
#     所以「完全不動」在實機上本來就做不到，RL 的振盪指標要拿同環境、同頻率的 LQI 當地板來比。
#
# 🔒 獨立性：從隨機初始化訓練，不讀任何 .zip，只需原始碼 + mujoco + gymnasium + SB3。
#    不使用 VecNormalize：觀測的固定縮放寫在 ObservationBuilder 裡，上機不需要額外的統計檔。

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch as th
from torch.nn import functional as F
import gymnasium as gym
import mujoco
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3.policies import Actor, TD3Policy


# ==========================================
# 0. 路徑與時間設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # rl_project/
MODEL_DIR = PROJECT_ROOT / "output" / "models"
RESULT_DIR = PROJECT_ROOT / "output" / "results"
LOG_DIR = PROJECT_ROOT / "tensorboard_logs" / "machine"

PHYS_DT = 0.002          # MuJoCo timestep（implicitfast），不隨控制頻率變
EPISODE_SEC = 20.0       # 一集固定 20 秒
OSC_MIN_SEC = 2.0        # 振盪指標至少需要的集長
EVAL_SEED0 = 10_000      # 訓練中評估（挑 best）
FINAL_EVAL_SEED0 = 20_000  # 最終評估，與上面錯開


def _sec_to_steps(sec, dt):
    """四捨五入（.5 進位）並至少 1 步；加 1e-9 吸收 0.1/0.02=5.000000000000001 這類浮點誤差。"""
    return max(1, int(math.floor(sec / dt + 0.5 + 1e-9)))


@dataclass(frozen=True)
class Timing:
    """所有跟控制頻率有關的量。由 main() 依 --ctrl-hz 建立，明確傳給環境 / builder / LQI / callbacks。"""
    ctrl_hz: int
    n_substeps: int
    ctrl_dt: float
    max_episode_steps: int
    gamma: float
    dist_start_range: tuple       # 步
    dist_duration_range: tuple    # 步
    osc_min_steps: int

    @classmethod
    def from_hz(cls, hz):
        n_sub = int(round(1.0 / (hz * PHYS_DT)))
        dt = PHYS_DT * n_sub
        st = lambda sec: _sec_to_steps(sec, dt)
        return cls(ctrl_hz=int(hz), n_substeps=n_sub, ctrl_dt=dt,
                   max_episode_steps=st(EPISODE_SEC),
                   gamma=round(0.99 ** (dt / 0.04), 4),          # 50 Hz → 0.995，25 Hz → 0.99
                   dist_start_range=tuple(st(v) for v in DISTURBANCE["start_sec"]),
                   dist_duration_range=tuple(st(v) for v in DISTURBANCE["duration_sec"]),
                   osc_min_steps=st(OSC_MIN_SEC))


# ==========================================
# 1. IP570 物理參數（名目值 + DR 範圍）
#    [規] 規格表   [圖] 圖紙/照片量測   [推] 由規格推導   [估] 規格未給、依材質/類比估算
#    到貨後依 md 的量測清單逐項換成實測值，DR 範圍收斂到「實測值 ±10~20%」
# ==========================================
PLANT_NOMINAL = dict(
    pulley_r=0.012,      # m        [估] 同步輪節圓半徑。照片/圖紙估 ⌀20–28 mm。⚠️ 到貨第一個量
    J_motor=1.3e-6,      # kg·m²    [估] 520 級鐵芯轉子 + 減速箱，折算到馬達軸
    gear_N=20.0,         # -        [規] 減速比 1:20
    Kt_out=0.17,         # N·m/A    [推] 輸出軸扭矩常數 = η·Ke_out（η≈0.85）；規格堵轉值推得 0.23，等於 η>1，偏樂觀
    Ke_out=0.197,        # V·s/rad  [推] (12 − I0·R)/57.5，無載轉速 549 rpm
    R_total=4.8,         # Ω        [推] 12 V / 2.8 A = 4.3 Ω，加驅動晶片導通電阻約 0.5 Ω
    V_supply=12.0,       # V        [規]
    m_cart=0.20,         # kg       [估] 滑塊鋁板 + 直線軸承 + 電位器 + 皮帶夾
    m_rod=0.062,         # kg       [估] 不鏽鋼 ⌀5 mm × 0.39 m（ρ=7930）
    m_hub=0.012,         # kg       [估] 擺桿夾頭，位於轉軸附近
    pole_L=0.39,         # m        [規] 擺桿長度（轉軸到頂端；437−390≈47 mm 為軸心高度，與圖紙 328.8−282.8=46 吻合）
    Fc_cart=2.5,         # N        [推] 減速箱無載電流 ~0.15 A 換算 2.1 N + 軸承/皮帶約 0.4 N
    b_cart=0.0,          # N·s/m    [估] 反電動勢以外的黏滯阻尼
    hinge_fric=3e-4,     # N·m      [規] WDD35D4 起動力矩 ≤ 1e-3 N·m，運轉時通常更小
    hinge_damp=5e-5,     # N·m·s    [估]
)

# (low, high, 分布)；"log" = 對數均勻
PLANT_DR = dict(
    pulley_r=(0.010, 0.014, "uniform"),
    J_motor=(0.8e-6, 2.2e-6, "log"),
    Kt_out=(0.14, 0.23, "uniform"),
    Ke_out=(0.187, 0.207, "uniform"),
    R_total=(4.3, 5.5, "uniform"),
    V_supply=(11.4, 12.3, "uniform"),
    m_cart=(0.15, 0.30, "uniform"),
    m_rod=(0.03, 0.10, "uniform"),       # 鋁 ⌀6 ~ 不鏽鋼 ⌀6，材質未確認所以放寬
    m_hub=(0.005, 0.02, "uniform"),
    Fc_cart=(1.0, 5.0, "uniform"),
    b_cart=(0.0, 5.0, "uniform"),
    hinge_fric=(0.0, 1.0e-3, "uniform"),
    hinge_damp=(0.0, 2.0e-4, "uniform"),
)


def derive_plant(p):
    """由物理參數推出 MuJoCo 欄位。所有跟 r 有關的量一起變（相關的 DR，不是各自獨立抽）。"""
    r = p["pulley_r"]
    K_F = p["V_supply"] * p["Kt_out"] / (p["R_total"] * r)          # N / duty
    B = p["Kt_out"] * p["Ke_out"] / (p["R_total"] * r ** 2)         # N·s/m
    armature = p["J_motor"] * p["gear_N"] ** 2 / r ** 2             # kg（折算質量）
    L = p["pole_L"]
    m_p = p["m_rod"] + p["m_hub"]
    lc = (p["m_rod"] * L / 2.0) / m_p                                # 夾頭視為在轉軸上
    I_c = p["m_rod"] * L ** 2 / 12.0 + p["m_rod"] * (L / 2.0 - lc) ** 2 + p["m_hub"] * lc ** 2
    I_pivot = I_c + m_p * lc ** 2
    m_per_count = 2.0 * math.pi * r / ENCODER_CPR
    return dict(K_F=K_F, B=B, armature=armature, m_pole=m_p, lc=lc, I_pole_c=I_c,
                I_pole_pivot=I_pivot, v_free=K_F / B, m_per_count=m_per_count,
                tau_mech=(p["m_cart"] + m_p + armature) / B)


# ==========================================
# 2. 感測 / 通訊規格（韌體端必須照同樣的方式取樣）
# ==========================================
ENCODER_CPR = 260 * 4            # [規] 260 線（輸出軸）× 四倍頻
ADC_MAX = 4095                   # STM32 12-bit
POT_ELEC_DEG = 345.0             # [規] WDD35D4 理論電氣轉角 345°±2°
ADC_COUNTS_PER_RAD = ADC_MAX / math.radians(POT_ELEC_DEG)   # ≈ 680 count/rad
ADC_ZERO_NOMINAL = 2048          # 直立時的 ADC 讀值（實機：開機校正後的值）
ADC_OVERSAMPLE = 8               # 每拍 ADC 取樣次數（韌體端平均後回報）

SENSOR_DR = dict(
    angle_bias=(-0.01, 0.01),    # rad，零點校正殘差（含 0.1% 線性度在直立附近的局部偏差）
    angle_scale=(0.99, 1.01),    # 電氣轉角 ±2° → ±0.6%，取 ±1%
    adc_noise=(0.5, 3.0),        # 單次 ADC 取樣雜訊 σ（count）
    enc_cal=(0.99, 1.01),        # m/count 校正誤差
)
SENSOR_NOMINAL = dict(angle_bias=0.0, angle_scale=1.0, adc_noise=1.0, enc_cal=1.0)

COMM_DR = dict(latency_ms=(2.0, 10.0), drop_prob=0.02)
COMM_NOMINAL = dict(latency_ms=6.0, drop_prob=0.0)


def _ms_to_substeps(ms):
    return int(round(ms / (PHYS_DT * 1000)))


COMM_DR_SUBSTEPS = tuple(_ms_to_substeps(v) for v in COMM_DR["latency_ms"])   # (1, 5)

# 外力擾動：模擬手指敲擺桿頂端。以「名目擺桿上造成的角速度增量」定義衝量大小，
# 實際效果隨當集抽到的擺桿質量變化（同一下敲擊，輕的擺會被敲得比較偏，這是物理上對的）。
# 起點與持續時間以秒定義，由 Timing 換算成步（50 Hz：100–750 步、2–5 步；25 Hz：50–375 步、1–3 步）
DISTURBANCE = dict(probability=0.3, start_sec=(2.0, 15.0), duration_sec=(0.04, 0.10),
                   dtheta_dot_range=(0.3, 1.0))

INIT_RANGE = dict(x=0.03, theta=0.10, theta_dot=0.2)

# 終止條件
THETA_LIMIT = 0.2       # rad；LQI 實測 θ0=0.15 用掉 0.10 m 行程，θ0=0.2 會撞 0.15 m 軟限
RAIL_LIMIT = 0.15       # m；硬限位 0.1925，留 4 cm 給煞停

# 觀測固定縮放（取代 VecNormalize，上機時同一份常數）
OBS_SCALE = dict(x=0.15, theta=0.2, x_dot=0.5, theta_dot=2.0, integral=0.3)


# ==========================================
# 3. 獎勵常數
# ==========================================
REWARD = dict(
    q_theta=10.0, q_x=20.0, q_theta_dot=0.5, q_x_dot=0.1,     # 二次項（LQI 設計用同一組）
    r_u=0.05, r_jerk=0.2, r_snap=0.1,                         # 動作懲罰（作用在網路輸出 u）
    bonus_upright=2.0, bonus_center=5.0,                      # 平滑位能井
    sig_theta=0.02, sig_theta_dot=0.1, sig_x=0.03, sig_x_dot=0.1,
    well="cauchy",       # "cauchy" 或 "gauss"，見 _well()
    clip=(-5.0, 10.0),
)


def _well(z2):
    """位能井形狀。z2 = Σ(s_i/σ_i)²。兩者在中心的曲率相同（1 − z²），差在尾巴：
       gauss  = exp(−z²)       離中心 2σ 以外幾乎沒有梯度
       cauchy = 1 / (1 + z²)   長尾，整段行程都有往中心的拉力"""
    if REWARD["well"] == "cauchy":
        return 1.0 / (1.0 + z2)
    return math.exp(-z2)
LQI_INTEGRAL_WEIGHT = 5.0   # 只用於 LQI 基準設計，不在獎勵裡


# ==========================================
# 4. MuJoCo 模型（XML 內嵌，數值由 apply_plant 覆寫）
# ==========================================
IP570_XML = f"""
<mujoco model="ip570_linear_pendulum">
  <compiler angle="radian"/>
  <option timestep="{PHYS_DT}" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 -1 1" dir="0 1 -1"/>
    <geom name="rail" type="capsule" fromto="-0.2125 0 -0.02 0.2125 0 -0.02" size="0.004"
          contype="0" conaffinity="0" rgba="0.7 0.7 0.7 1"/>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-0.1925 0.1925"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>
      <geom type="box" size="0.03 0.025 0.02" contype="0" conaffinity="0" rgba="0.8 0.8 0.85 1"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" limited="true" range="-1.57 1.57"/>
        <inertial pos="0 0 0.17" mass="0.07" diaginertia="1e-3 1e-3 1e-7"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.39" size="0.0025" contype="0" conaffinity="0"
              rgba="0.75 0.75 0.78 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="motor" joint="slider" ctrllimited="true" ctrlrange="-1 1"
             gainprm="1" biastype="affine" biasprm="0 0 0"/>
  </actuator>
</mujoco>
"""


def apply_plant(model, data, p):
    """把一組物理參數寫進 MuJoCo 模型（每集絕對賦值，不會跨集累乘）。"""
    q = derive_plant(p)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cart")
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pole")
    model.body_mass[cid] = p["m_cart"]
    model.body_mass[pid] = q["m_pole"]
    model.body_ipos[pid] = [0.0, 0.0, q["lc"]]
    model.body_inertia[pid] = [q["I_pole_c"], q["I_pole_c"], 1e-7]
    model.dof_armature[0] = q["armature"]
    model.dof_frictionloss[0] = p["Fc_cart"]
    model.dof_damping[0] = p["b_cart"]
    model.dof_frictionloss[1] = p["hinge_fric"]
    model.dof_damping[1] = p["hinge_damp"]
    model.actuator_gainprm[0, 0] = q["K_F"]
    model.actuator_biasprm[0, :3] = [0.0, 0.0, -q["B"]]
    mujoco.mj_setConst(model, data)   # 重算依賴質量的衍生常數（約束的有效質量等）
    return q


# ==========================================
# 5. ObservationBuilder —— 模擬與實機「同一份程式碼」
#    只依賴 numpy。輸入是原始感測值（編碼器 count、ADC 讀值、MCU 回報的 duty），
#    輸出是策略看到的 6 維觀測。實機部署時以 importlib 載入本檔，直接用這個 class。
#    dt 必須明確傳入（= Timing.ctrl_dt），速度低通 α 與積分都依它計算。
# ==========================================
class ObservationBuilder:
    OBS_DIM = 6   # [x, θ, ẋ, θ̇, u_applied, ∫x]

    def __init__(self, dt, m_per_count=None, adc_zero=ADC_ZERO_NOMINAL,
                 counts_per_rad=ADC_COUNTS_PER_RAD, vel_cutoff_hz=10.0,
                 integral_clip=OBS_SCALE["integral"], scale=None):
        if m_per_count is None:
            m_per_count = 2.0 * math.pi * PLANT_NOMINAL["pulley_r"] / ENCODER_CPR
        self.dt = dt
        self.m_per_count = m_per_count
        self.adc_zero = adc_zero
        self.counts_per_rad = counts_per_rad
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * vel_cutoff_hz * dt)   # 一階低通
        self.integral_clip = integral_clip
        self.scale = dict(OBS_SCALE) if scale is None else dict(scale)
        self.reset(0, adc_zero)

    def _convert(self, enc_count, adc_value):
        x = enc_count * self.m_per_count
        theta = (adc_value - self.adc_zero) / self.counts_per_rad
        return x, theta

    def reset(self, enc_count, adc_value):
        """控制啟用的那一刻呼叫（實機：手扶正、按下啟動時）。"""
        self.x, self.theta = self._convert(enc_count, adc_value)
        self.x_dot = 0.0
        self.theta_dot = 0.0
        self.integral = 0.0
        self.u_applied = 0.0
        return self._pack()

    def update(self, enc_count, adc_value, u_applied):
        x, theta = self._convert(enc_count, adc_value)
        raw_xd = (x - self.x) / self.dt
        raw_td = (theta - self.theta) / self.dt
        self.x_dot += self.alpha * (raw_xd - self.x_dot)
        self.theta_dot += self.alpha * (raw_td - self.theta_dot)
        self.x, self.theta = x, theta
        self.integral = float(np.clip(self.integral + x * self.dt,
                                      -self.integral_clip, self.integral_clip))
        self.u_applied = float(u_applied)
        return self._pack()

    def physical(self):
        """物理單位的狀態（給 LQI 與記錄用）。"""
        return np.array([self.x, self.theta, self.x_dot, self.theta_dot,
                         self.u_applied, self.integral], dtype=np.float64)

    def _pack(self):
        s = self.scale
        return np.array([self.x / s["x"], self.theta / s["theta"],
                         self.x_dot / s["x_dot"], self.theta_dot / s["theta_dot"],
                         self.u_applied, self.integral / s["integral"]], dtype=np.float32)

    def to_config(self):
        return dict(dt=self.dt, m_per_count=self.m_per_count, adc_zero=self.adc_zero,
                    counts_per_rad=self.counts_per_rad, alpha=self.alpha,
                    integral_clip=self.integral_clip, scale=self.scale)


# ==========================================
# 6. IP570 環境
# ==========================================
# 特權觀測（只給 asymmetric critic，不上機）
PRIV_LAYOUT = (["x_true", "theta_true", "x_dot_true", "theta_dot_true"]
               + ["log_K_F", "log_B", "log_armature", "log_m_pole", "log_lc"]
               + ["Fc_cart", "b_cart", "hinge_fric", "hinge_damp"]
               + list(SENSOR_DR)
               + ["latency_substeps", "dist_force"])
N_PRIV = len(PRIV_LAYOUT)   # 19
PRIV_LOG_KEYS = ("K_F", "B", "armature", "m_pole", "lc")      # log(值 / 名目值)
PRIV_PLANT_KEYS = ("Fc_cart", "b_cart", "hinge_fric", "hinge_damp")


def _unit(v, rng):
    """把 [lo, hi] 線性對應到 [−1, 1]。"""
    lo, hi = rng[0], rng[1]
    return (v - 0.5 * (lo + hi)) / (0.5 * (hi - lo))


class IP570Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, timing, plant_dr=True, sensor_dr=True, comm_dr=True, disturbance=True,
                 privileged_obs=False):
        super().__init__()
        self.timing = timing
        self.plant_dr = plant_dr
        self.sensor_dr = sensor_dr
        self.comm_dr = comm_dr
        self.disturbance_enable = disturbance
        self.privileged_obs = privileged_obs
        self.max_episode_steps = timing.max_episode_steps

        self.model = mujoco.MjModel.from_xml_string(IP570_XML)
        self.data = mujoco.MjData(self.model)
        self._pole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pole")
        self._nominal_derived = derive_plant(PLANT_NOMINAL)
        # 名目最大外力：最大 Δθ̇ 的衝量在最短持續時間內打完
        self._dist_force_max = (DISTURBANCE["dtheta_dot_range"][1] * self._nominal_derived["I_pole_pivot"]
                                / PLANT_NOMINAL["pole_L"]
                                / (timing.dist_duration_range[0] * timing.ctrl_dt))

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        obs_box = gym.spaces.Box(-np.inf, np.inf, shape=(ObservationBuilder.OBS_DIM,), dtype=np.float32)
        if privileged_obs:
            self.observation_space = gym.spaces.Dict({
                "obs": obs_box,
                "priv": gym.spaces.Box(-np.inf, np.inf, shape=(N_PRIV,), dtype=np.float32),
            })
        else:
            self.observation_space = obs_box
        self.builder = ObservationBuilder(dt=timing.ctrl_dt)
        self.p = dict(PLANT_NOMINAL)
        self.q = derive_plant(self.p)
        self._reset_metrics()

    # ---------- 參數抽樣 ----------
    def _sample_plant(self):
        p = dict(PLANT_NOMINAL)
        if self.plant_dr:
            for k, (lo, hi, dist) in PLANT_DR.items():
                if dist == "log":
                    p[k] = float(math.exp(self.np_random.uniform(math.log(lo), math.log(hi))))
                else:
                    p[k] = float(self.np_random.uniform(lo, hi))
        return p

    def _sample_sensor(self):
        if not self.sensor_dr:
            return dict(SENSOR_NOMINAL)
        return {k: float(self.np_random.uniform(*v)) for k, v in SENSOR_DR.items()}

    # ---------- 原始感測值合成（模擬 MCU 端）----------
    def _read_raw_sensors(self):
        x_true, th_true = float(self.data.qpos[0]), float(self.data.qpos[1])
        enc = int(np.round(x_true / self._m_per_count_true))
        center = self.builder.adc_zero + (th_true * self.sens["angle_scale"]
                                          + self.sens["angle_bias"]) * ADC_COUNTS_PER_RAD
        samples = np.round(center + self.np_random.normal(0.0, self.sens["adc_noise"],
                                                          size=ADC_OVERSAMPLE))
        adc = float(np.mean(np.clip(samples, 0, ADC_MAX)))
        return enc, adc

    # ---------- 特權觀測 ----------
    def _priv(self, lat, force):
        s, qn = OBS_SCALE, self._nominal_derived
        v = [self.data.qpos[0] / s["x"], self.data.qpos[1] / s["theta"],
             self.data.qvel[0] / s["x_dot"], self.data.qvel[1] / s["theta_dot"]]
        v += [math.log(self.q[k] / qn[k]) for k in PRIV_LOG_KEYS]
        v += [_unit(self.p[k], PLANT_DR[k]) for k in PRIV_PLANT_KEYS]
        v += [_unit(self.sens[k], SENSOR_DR[k]) for k in SENSOR_DR]
        v += [_unit(lat, COMM_DR_SUBSTEPS), force / self._dist_force_max]
        return np.asarray(v, dtype=np.float32)

    def _wrap_obs(self, obs, lat, force):
        if not self.privileged_obs:
            return obs
        return {"obs": obs, "priv": self._priv(lat, force)}

    # ---------- reset ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.p = self._sample_plant()
        self.q = apply_plant(self.model, self.data, self.p)
        self.sens = self._sample_sensor()

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = self.np_random.uniform(-INIT_RANGE["x"], INIT_RANGE["x"])
        self.data.qpos[1] = self.np_random.uniform(-INIT_RANGE["theta"], INIT_RANGE["theta"])
        self.data.qvel[1] = self.np_random.uniform(-INIT_RANGE["theta_dot"], INIT_RANGE["theta_dot"])
        mujoco.mj_forward(self.model, self.data)

        # builder 用「校正後的」m/count；真實值與之差 enc_cal
        self.builder.m_per_count = self.q["m_per_count"]
        self._m_per_count_true = self.q["m_per_count"] * self.sens["enc_cal"]

        if self.comm_dr:
            self._lat_range = COMM_DR_SUBSTEPS
            self._drop_prob = COMM_DR["drop_prob"]
        else:
            n = _ms_to_substeps(COMM_NOMINAL["latency_ms"])
            self._lat_range = (n, n)
            self._drop_prob = COMM_NOMINAL["drop_prob"]

        self.u_applied = 0.0          # MCU 目前輸出的 duty
        self.u_prev = 0.0             # 網路上一拍輸出（jerk/snap 用）
        self.u_prev2 = 0.0
        self.step_count = 0
        self._latency_log = []

        self.dist_plan = None
        self._dist_remaining = 0
        self._dist_force = 0.0
        if self.disturbance_enable and self.np_random.random() < DISTURBANCE["probability"]:
            tm = self.timing
            dur = int(self.np_random.integers(tm.dist_duration_range[0],
                                              tm.dist_duration_range[1] + 1))
            dthd = float(self.np_random.uniform(*DISTURBANCE["dtheta_dot_range"]))
            impulse = dthd * self._nominal_derived["I_pole_pivot"] / PLANT_NOMINAL["pole_L"]  # N·s at tip
            self.dist_plan = dict(
                start=int(self.np_random.integers(tm.dist_start_range[0],
                                                  tm.dist_start_range[1] + 1)),
                duration=dur,
                force=float(self.np_random.choice([-1.0, 1.0])) * impulse / (dur * tm.ctrl_dt),
                dtheta_dot=dthd)

        self._reset_metrics()
        enc, adc = self._read_raw_sensors()
        obs = self.builder.reset(enc, adc)
        return self._wrap_obs(obs, 0.5 * sum(self._lat_range), 0.0), {}

    # ---------- step ----------
    def step(self, action):
        u = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))

        # 通訊：掉幀時 MCU 沿用舊值；新指令在第 L 個子步才到
        if self._drop_prob > 0 and self.np_random.random() < self._drop_prob:
            u_new = self.u_applied
        else:
            u_new = u
        lat = int(self.np_random.integers(self._lat_range[0], self._lat_range[1] + 1))
        self._latency_log.append(lat)

        # 外力擾動排程
        if (self.dist_plan is not None and self._dist_remaining == 0
                and self.step_count >= self.dist_plan["start"]):
            self._dist_remaining = self.dist_plan["duration"]
            self._dist_force = self.dist_plan["force"]
            self.dist_plan = None
        if self._dist_remaining > 0:
            F = self._dist_force
            arm = self.p["pole_L"] - self.q["lc"]     # 頂端施力 = 質心力 + 對質心力矩
            self.data.xfrc_applied[self._pole_id] = [F, 0.0, 0.0, 0.0, F * arm, 0.0]
            self._dist_remaining -= 1
        else:
            F = 0.0
            self.data.xfrc_applied[self._pole_id] = 0.0

        for k in range(self.timing.n_substeps):
            self.data.ctrl[0] = self.u_applied if k < lat else u_new
            mujoco.mj_step(self.model, self.data)
        self.u_applied = u_new

        # 乾淨狀態（只給獎勵、終止與指標用）
        x, th = float(self.data.qpos[0]), float(self.data.qpos[1])
        xd, thd = float(self.data.qvel[0]), float(self.data.qvel[1])

        # 策略看到的觀測（原始讀值 → builder）
        enc, adc = self._read_raw_sensors()
        obs = self._wrap_obs(self.builder.update(enc, adc, self.u_applied), lat, F)

        # ── 獎勵 ──
        R = REWARD
        q_cost = (R["q_theta"] * th ** 2 + R["q_x"] * x ** 2
                  + R["q_theta_dot"] * thd ** 2 + R["q_x_dot"] * xd ** 2)
        a_cost = (R["r_u"] * u ** 2
                  + R["r_jerk"] * (u - self.u_prev) ** 2
                  + R["r_snap"] * (u - 2.0 * self.u_prev + self.u_prev2) ** 2)
        k_th = _well((th / R["sig_theta"]) ** 2 + (thd / R["sig_theta_dot"]) ** 2)
        k_x = _well((x / R["sig_x"]) ** 2 + (xd / R["sig_x_dot"]) ** 2)
        reward = 1.0 - q_cost - a_cost + R["bonus_upright"] * k_th + R["bonus_center"] * k_th * k_x
        reward = float(np.clip(reward, *R["clip"]))

        self.u_prev2, self.u_prev = self.u_prev, u
        self.step_count += 1

        # ── 指標 ──
        self.m_theta.append(th)
        self.m_x.append(x)
        self.m_u.append(self.u_applied)
        self.m_bonus.append(k_th * k_x)
        if self.m_settle_first is None and abs(th) < 0.01:
            self.m_settle_first = self.step_count

        angle_hit = abs(th) > THETA_LIMIT
        rail_hit = abs(x) > RAIL_LIMIT
        terminated = bool(angle_hit or rail_hit)
        truncated = bool(self.step_count >= self.max_episode_steps and not terminated)
        info = {}
        if terminated or truncated:
            self._angle_hit, self._rail_hit = angle_hit, rail_hit
            info["control_metrics"] = self._compute_episode_metrics()
        return obs, reward, terminated, truncated, info

    # ---------- 指標 ----------
    def _reset_metrics(self):
        self.m_theta, self.m_x, self.m_u, self.m_bonus = [], [], [], []
        self.m_settle_first = None
        self._angle_hit = False
        self._rail_hit = False

    def _compute_episode_metrics(self):
        th = np.asarray(self.m_theta)
        x = np.asarray(self.m_x)
        u = np.asarray(self.m_u)
        n = len(th)
        if n == 0:
            return {}
        ise = float(np.sum(th ** 2))
        iae = float(np.sum(np.abs(th)))
        itae = float(np.sum(np.arange(1, n + 1) * np.abs(th)))
        outside = np.where(np.abs(th) > 0.02)[0]
        m = {
            "Episode_Length": float(n),
            "ISE": ise, "IAE": iae, "ITAE": itae,
            "ISE_per_step": ise / n, "IAE_per_step": iae / n, "ITAE_per_step": itae / n,
            "Overshoot_max_angle": float(np.max(np.abs(th))),
            "Settling_Time_steps": float(self.m_settle_first if self.m_settle_first is not None else n),
            "Settling_Time_sustained": float(outside[-1] + 1) if len(outside) else 0.0,
            "Steady_State_Error": float(np.mean(np.abs(th[int(n * 0.8):]))) if n >= 5 else 0.0,
            "RMS_Control_Effort": float(np.sqrt(np.mean(u ** 2))),
            "Duty_Saturation_frac": float(np.mean(np.abs(u) > 0.98)),
            "Position_ISE": float(np.sum(x ** 2)),
            "Position_ISE_per_step": float(np.sum(x ** 2) / n),
            "max_cart_pos_abs": float(np.max(np.abs(x))),
            "rail_limit_hit": float(self._rail_hit),
            "angle_limit_hit": float(self._angle_hit),
            "initial_angle_abs": float(abs(th[0])),
            "bonus_center_mean": float(np.mean(self.m_bonus)),
        }
        # 振盪指標（後 50%）：區分摩擦極限環 / 偏移漂移 / 延遲振盪
        if n >= self.timing.osc_min_steps:
            h = n // 2
            tt, xx = th[h:], x[h:]
            zc = np.sum(np.diff(np.sign(tt - tt.mean())) != 0)
            m.update({
                "osc_theta_std_tail": float(np.std(tt)),
                "osc_x_std_tail": float(np.std(xx)),
                "osc_x_mean_abs_tail": float(abs(np.mean(xx))),
                "osc_theta_zc_hz_tail": float(zc / 2.0 / (len(tt) * self.timing.ctrl_dt)),
            })
        # 本集實際抽到的參數，事後交叉比對哪種組合會失敗
        for k in PLANT_DR:
            m[f"dr_{k}"] = float(self.p[k])
        for k in SENSOR_DR:
            m[f"dr_{k}"] = float(self.sens[k])
        m["dr_K_F"] = float(self.q["K_F"])
        m["dr_B"] = float(self.q["B"])
        m["dr_armature"] = float(self.q["armature"])
        m["dr_latency_ms_mean"] = float(np.mean(self._latency_log) * PHYS_DT * 1000) if self._latency_log else 0.0
        return m


# ==========================================
# 7. LQI 基準（同環境、同觀測、同二次代價）
#    在名目模型上數值線性化，解離散 Riccati。只用 numpy，不依賴 scipy。
#    狀態 z = [x, θ, ẋ, θ̇, u_applied, ∫x]，與 builder.physical() 一一對應。
# ==========================================
def _linearize_nominal(timing, latency_substeps=3, eps=1e-5):
    p = dict(PLANT_NOMINAL)
    p["Fc_cart"] = 0.0
    p["hinge_fric"] = 0.0
    model = mujoco.MjModel.from_xml_string(IP570_XML)
    data = mujoco.MjData(model)
    apply_plant(model, data, p)

    def f(s, u_prev, u):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = s[:2]
        data.qvel[:] = s[2:]
        for k in range(timing.n_substeps):
            data.ctrl[0] = u_prev if k < latency_substeps else u
            mujoco.mj_step(model, data)
        return np.r_[data.qpos, data.qvel].copy()

    s0 = np.zeros(4)
    A = np.zeros((4, 4))
    for i in range(4):
        e = np.zeros(4); e[i] = eps
        A[:, i] = (f(s0 + e, 0, 0) - f(s0 - e, 0, 0)) / (2 * eps)
    B0 = (f(s0, 0, eps) - f(s0, 0, -eps)) / (2 * eps)
    B1 = (f(s0, eps, 0) - f(s0, -eps, 0)) / (2 * eps)

    Az = np.zeros((6, 6)); Bz = np.zeros((6, 1))
    Az[:4, :4] = A; Az[:4, 4] = B1; Bz[:4, 0] = B0
    Bz[4, 0] = 1.0                              # u_applied(k+1) = u(k)
    Az[5, :] = timing.ctrl_dt * Az[0, :]; Az[5, 5] += 1.0   # I(k+1) = I(k) + dt·x(k+1)（與 builder 一致）
    Bz[5, 0] = timing.ctrl_dt * Bz[0, 0]
    return A, Az, Bz


def _dare(A, B, Q, R, N, iters=20000, tol=1e-10):
    """離散 Riccati。有 scipy 就用 scipy；沒有就做值迭代（每步對稱化、以 K 收斂判停，避免數值漂移）。"""
    try:
        from scipy.linalg import solve_discrete_are
        P = solve_discrete_are(A, B, Q, R, s=N)
        return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A + N.T)
    except ImportError:
        pass
    P = Q.copy()
    K_old = None
    for _ in range(iters):
        G = R + B.T @ P @ B
        H = B.T @ P @ A + N.T
        K = np.linalg.solve(G, H)
        P = Q + A.T @ P @ A - H.T @ K
        P = 0.5 * (P + P.T)
        if K_old is not None and np.max(np.abs(K - K_old)) < tol:
            break
        K_old = K
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A + N.T)


def design_lqi(timing):
    """在 timing 的控制週期上重新設計（25 Hz 下 LQI 也是 25 Hz 的，倍數才可比）。"""
    _, Az, Bz = _linearize_nominal(timing)
    R_ = REWARD
    Q = np.diag([R_["q_x"], R_["q_theta"], R_["q_x_dot"], R_["q_theta_dot"],
                 R_["r_jerk"], LQI_INTEGRAL_WEIGHT])
    R = np.array([[R_["r_u"] + R_["r_jerk"]]])
    N = np.zeros((6, 1)); N[4, 0] = -R_["r_jerk"]     # jerk = (u − u_prev)² 的交叉項
    return _dare(Az, Bz, Q, R, N).reshape(-1)


class LQIController:
    """以 builder 的物理單位觀測做回授：u = −K z。"""
    def __init__(self, K):
        self.K = np.asarray(K, dtype=np.float64)

    def __call__(self, builder):
        return np.array([np.clip(-self.K @ builder.physical(), -1.0, 1.0)], dtype=np.float32)


# ==========================================
# 8. 演算法擴充：CAPS 與 asymmetric critic
# ==========================================
def _perturb_obs(obs, sigma):
    """CAPS 空間平滑用：只對策略看得到的觀測加雜訊；dict 觀測的 "priv" 不動。"""
    if isinstance(obs, dict):
        out = dict(obs)
        out["obs"] = obs["obs"] + sigma * th.randn_like(obs["obs"])
        return out
    return obs + sigma * th.randn_like(obs)


class TD3CAPS(TD3):
    """
    TD3 + CAPS（Mysore et al., ICRA 2021）。train() 以 SB3 2.8.0 的 TD3.train() 為底，
    只改 actor 更新段；critic 更新、polyak、batch norm stats、logger 與原版一致。
    Q 項除以 batch 的 mean|Q|（detach），讓 λ 不受 Q 量級影響（做法借自 TD3+BC）。
    """
    def __init__(self, *args, lambda_t=0.0, lambda_s=0.0, caps_sigma=0.02, **kwargs):
        self.lambda_t, self.lambda_s, self.caps_sigma = lambda_t, lambda_s, caps_sigma
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        caps_t, caps_s, q_abs = [], [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                # ── CAPS actor loss ──
                obs = replay_data.observations
                a = self.actor(obs)
                q = self.critic.q1_forward(obs, a)
                q_scale = q.abs().mean().detach()
                q_term = -q.mean() / (q_scale + 1e-8)
                a_next = self.actor(replay_data.next_observations)
                l_t = ((a - a_next).norm(dim=1) * (1 - replay_data.dones.squeeze(1))).mean()
                a_near = self.actor(_perturb_obs(obs, self.caps_sigma))
                l_s = (a - a_near).norm(dim=1).mean()
                actor_loss = q_term + self.lambda_t * l_t + self.lambda_s * l_s
                actor_losses.append(actor_loss.item())
                caps_t.append(l_t.item())
                caps_s.append(l_s.item())
                q_abs.append(q_scale.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/caps_temporal", np.mean(caps_t))
            self.logger.record("train/caps_spatial", np.mean(caps_s))
            self.logger.record("train/q_abs_mean", np.mean(q_abs))
        self.logger.record("train/critic_loss", np.mean(critic_losses))


class ObsOnly(BaseFeaturesExtractor):
    """actor：只取 6 維 obs（恆等映射，所以上機可直接 actor.mu(obs6)）。"""
    def __init__(self, space):
        super().__init__(space, features_dim=space["obs"].shape[0])

    def forward(self, o):
        return o["obs"]


class ObsPriv(BaseFeaturesExtractor):
    """critic：obs + priv。"""
    def __init__(self, space):
        super().__init__(space, features_dim=space["obs"].shape[0] + space["priv"].shape[0])

    def forward(self, o):
        return th.cat([o["obs"], o["priv"]], dim=1)


class AsymTD3Policy(TD3Policy):
    """actor / actor_target 用 ObsOnly；critic / critic_target 用 features_extractor_class（= ObsPriv）。"""
    def __init__(self, *args, **kwargs):
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, ObsOnly(self.observation_space))
        return Actor(**kw).to(self.device)


# ==========================================
# 9. Callbacks / 評估
# ==========================================
def _log_sim_seconds(cb):
    cb.logger.record("time/sim_seconds", cb.num_timesteps * cb.timing.ctrl_dt)


class ControlMetricsCallback(BaseCallback):
    """訓練中每集的控制指標（含探索雜訊；判讀振盪請看 eval_* 而不是這組）。"""
    def __init__(self, timing, verbose=0):
        super().__init__(verbose)
        self.timing = timing
        self.episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "control_metrics" in info:
                self.episode_count += 1
                for key, value in info["control_metrics"].items():
                    self.logger.record(f"control/{key}", value)
                _log_sim_seconds(self)
                self.logger.dump(self.num_timesteps)
        return True


def make_eval_envs(timing, privileged_obs):
    """(名目模型, 全開 DR)"""
    return (IP570Env(timing, plant_dr=False, sensor_dr=False, comm_dr=False, disturbance=False,
                     privileged_obs=privileged_obs),
            IP570Env(timing, plant_dr=True, sensor_dr=True, comm_dr=True, disturbance=True,
                     privileged_obs=privileged_obs))


def run_episodes(env, act_fn, seeds):
    """固定種子跑數集，回傳每集 (return, metrics)。act_fn(obs, env) -> action"""
    out = []
    for sd in seeds:
        obs, _ = env.reset(seed=int(sd))
        ret, done = 0.0, False
        while not done:
            a = act_fn(obs, env)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            done = term or trunc
        out.append((ret, info.get("control_metrics", {})))
    return out


def summarize(results, max_episode_steps):
    keys = ["Episode_Length", "IAE_per_step", "Position_ISE_per_step", "RMS_Control_Effort",
            "Duty_Saturation_frac", "max_cart_pos_abs", "rail_limit_hit", "angle_limit_hit",
            "osc_theta_std_tail", "osc_x_std_tail", "osc_x_mean_abs_tail",
            "osc_theta_zc_hz_tail", "bonus_center_mean", "Settling_Time_sustained"]
    s = {"return_mean": float(np.mean([r for r, _ in results])),
         "full_length_frac": float(np.mean([m.get("Episode_Length", 0) >= max_episode_steps
                                            for _, m in results]))}
    for k in keys:
        vals = [m[k] for _, m in results if k in m]
        if vals:
            s[k] = float(np.mean(vals))
    return s


def _policy_act(model):
    return lambda obs, env: model.predict(obs, deterministic=True)[0]


def _lqi_act(K):
    lqi = LQIController(K)
    return lambda obs, env: lqi(env.builder)


class PeriodicEvalCallback(BaseCallback):
    """
    每 eval_freq 步：deterministic 策略在「名目模型」與「全開 DR」各跑 n_episodes 集。
    種子固定（EVAL_SEED0 起）→ 每次評估面對同一批初始條件與 DR 抽樣，曲線可直接跨時間比較。
    best 存檔點以這組種子的 DR return 挑選；最終評估改用 FINAL_EVAL_SEED0 起的另一組種子。
    LQI 在同一批種子上的表現每次都一起寫入（lqi_*），作為 TensorBoard 上的參考線。
    """
    def __init__(self, timing, privileged_obs=False, eval_freq=50_000, n_episodes=10,
                 save_path=None, lqi_gain=None, verbose=1):
        super().__init__(verbose)
        assert n_episodes <= FINAL_EVAL_SEED0 - EVAL_SEED0
        self.timing = timing
        self.eval_freq = eval_freq
        self.seeds = np.arange(EVAL_SEED0, EVAL_SEED0 + n_episodes)
        self.save_path = save_path
        self.env_nom, self.env_dr = make_eval_envs(timing, privileged_obs)
        self.lqi_gain = lqi_gain
        self.lqi_summary = {}
        self.best = -np.inf
        self.best_saved = False

    def _summ(self, env, act):
        return summarize(run_episodes(env, act, self.seeds), self.timing.max_episode_steps)

    def _on_training_start(self):
        if self.lqi_gain is not None:
            act = _lqi_act(self.lqi_gain)
            self.lqi_summary = {"lqi_nominal": self._summ(self.env_nom, act),
                                "lqi_dr": self._summ(self.env_dr, act)}
            if self.verbose:
                for tag, s in self.lqi_summary.items():
                    print(f"   [{tag}] return={s['return_mean']:.0f}  full_len={s['full_length_frac']:.2f}  "
                          f"θ_std={s.get('osc_theta_std_tail', float('nan')):.4f}  "
                          f"x_std={s.get('osc_x_std_tail', float('nan')):.4f}")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        act = _policy_act(self.model)
        res = {"eval_nominal": self._summ(self.env_nom, act),
               "eval_dr": self._summ(self.env_dr, act)}
        res.update(self.lqi_summary)
        for tag, s in res.items():
            for k, v in s.items():
                self.logger.record(f"{tag}/{k}", v)
        _log_sim_seconds(self)
        self.logger.dump(self.num_timesteps)
        score = res["eval_dr"]["return_mean"]
        if self.verbose:
            print(f"   [eval @ {self.num_timesteps}] nominal={res['eval_nominal']['return_mean']:.0f}  "
                  f"dr={score:.0f}  (LQI dr={self.lqi_summary.get('lqi_dr', {}).get('return_mean', float('nan')):.0f})")
        if self.save_path and score > self.best:
            self.best = score
            self.model.save(self.save_path)
            self.best_saved = True
        return True


class NoiseDecayCallback(BaseCallback):
    """探索雜訊 σ：前 start_frac 固定 sigma0，之後線性降到 sigma1（到 end_frac），再維持。
    Normal 與 OU 雜訊的 σ 都存在 _sigma，同一套寫法適用。"""
    def __init__(self, total_timesteps, sigma0, sigma1, start_frac=0.6, end_frac=0.9):
        super().__init__(0)
        self.T = total_timesteps
        self.s0, self.s1, self.a, self.b = sigma0, sigma1, start_frac, end_frac

    def _on_step(self) -> bool:
        f = self.num_timesteps / self.T
        w = min(max((f - self.a) / (self.b - self.a), 0.0), 1.0)
        sigma = self.s0 + (self.s1 - self.s0) * w
        if self.model.action_noise is not None:
            self.model.action_noise._sigma = sigma * np.ones_like(self.model.action_noise._sigma)
        if self.n_calls % 5000 == 0:
            self.logger.record("train/action_noise_sigma", sigma)
        return True


def lr_schedule(progress_remaining):
    """前 70% 為 1e-3，之後線性降到 3e-4。progress_remaining 由 1 降到 0。"""
    if progress_remaining > 0.3:
        return 1e-3
    return 3e-4 + (1e-3 - 3e-4) * (progress_remaining / 0.3)


def final_evaluation(policies, timing, privileged_obs, n_episodes):
    """policies: {名稱: act_fn}。名目 / 全開 DR 各跑 n_episodes 集，種子 FINAL_EVAL_SEED0 起。"""
    seeds = np.arange(FINAL_EVAL_SEED0, FINAL_EVAL_SEED0 + n_episodes)
    envs = dict(zip(("nominal", "dr"), make_eval_envs(timing, privileged_obs)))
    summary, episodes = {}, {}
    for name, act in policies.items():
        for env_tag, env in envs.items():
            res = run_episodes(env, act, seeds)
            key = f"{name}_{env_tag}"
            summary[key] = summarize(res, timing.max_episode_steps)
            episodes[key] = [dict(seed=int(sd), episode_return=float(r), **m) for sd, (r, m) in zip(seeds, res)]
    return seeds, summary, episodes


def export_actor(model, path):
    """只匯出 actor.mu（6 維 obs → duty）。上機不需要 SB3，用同樣層結構重建 nn.Sequential 即可。"""
    mu = model.actor.mu
    th.save(dict(state_dict=mu.state_dict(),
                 layers=[repr(m) for m in mu],
                 net_arch=list(model.policy.net_arch["pi"] if isinstance(model.policy.net_arch, dict)
                               else model.policy.net_arch),
                 obs_dim=ObservationBuilder.OBS_DIM, act_dim=1,
                 obs_layout=["x", "theta", "x_dot", "theta_dot", "u_applied", "integral_x"],
                 usage="duty = mu(torch.as_tensor(obs6[None], dtype=torch.float32))[0]，輸出已含 tanh"),
            path)


def check_actor_export(model, timing, privileged_obs, n_steps=50):
    """驗證 actor.mu(obs6) 與 model.predict(obs, deterministic=True) 相同（回傳最大差值）。"""
    env, _ = make_eval_envs(timing, privileged_obs)
    obs, _ = env.reset(seed=FINAL_EVAL_SEED0)
    max_diff = 0.0
    for _ in range(n_steps):
        a_pred = model.predict(obs, deterministic=True)[0]
        obs6 = obs["obs"] if isinstance(obs, dict) else obs
        with th.no_grad():
            a_mu = model.actor.mu(th.as_tensor(obs6[None], dtype=th.float32, device=model.device)).cpu().numpy()[0]
        max_diff = max(max_diff, float(np.max(np.abs(a_pred - a_mu))))
        obs, _, term, trunc, _ = env.step(a_pred)
        if term or trunc:
            break
    return max_diff


# ==========================================
# 10. 主程式
# ==========================================
def print_plant_summary():
    q = derive_plant(PLANT_NOMINAL)
    print("   IP570 名目模型："
          f"K_F={q['K_F']:.1f} N/duty  B={q['B']:.1f} N·s/m  armature={q['armature']:.2f} kg  "
          f"v_free={q['v_free']:.2f} m/s  τ_mech={q['tau_mech']*1000:.0f} ms")
    print(f"                 擺 {q['m_pole']*1000:.0f} g, lc={q['lc']*1000:.0f} mm, "
          f"不穩定極點≈{math.sqrt(q['m_pole']*9.81*q['lc']/q['I_pole_pivot']):.2f} rad/s  "
          f"m/count={q['m_per_count']*1e6:.1f} µm  θ 量化={1/ADC_COUNTS_PER_RAD*1000:.2f} mrad")


# 受 --preset 影響的參數。"none" = 07113 預設；"07112" = 重現 07112（評估設定仍用 07113 的新評估）
PRESETS = {
    "none": dict(ctrl_hz=50, explore_sigma=0.08, explore_sigma_final=0.02, noise_type="normal",
                 target_noise=0.05, target_noise_clip=0.1,
                 caps_lambda_t=0.0, caps_lambda_s=0.0, caps_sigma=0.02, asym_critic=False),
    "07112": dict(ctrl_hz=50, explore_sigma=0.2, explore_sigma_final=0.05, noise_type="normal",
                  target_noise=0.2, target_noise_clip=0.5,
                  caps_lambda_t=0.0, caps_lambda_s=0.0, caps_sigma=0.02, asym_critic=False),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="07113：IP570 純 RL（評估 / 探索 / 時間尺度 / CAPS / asym critic）")
    ap.add_argument("--preset", choices=list(PRESETS), default="none",
                    help="07112 = σ 0.2→0.05、target 0.2/0.5、50 Hz、無 CAPS、無 asym；明確指定的參數優先")
    ap.add_argument("--tag", default="default", help="設定標籤，用於 run 名稱與輸出檔名")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--eval-freq", type=int, default=50_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--final-eval-episodes", type=int, default=50)
    # 以下預設為 None → 由 preset 填入
    ap.add_argument("--ctrl-hz", type=int, choices=[50, 25], default=None)
    ap.add_argument("--explore-sigma", type=float, default=None, help="探索雜訊起始 σ（duty）")
    ap.add_argument("--explore-sigma-final", type=float, default=None, help="探索雜訊最終 σ（duty）")
    ap.add_argument("--noise-type", choices=["normal", "ou"], default=None)
    ap.add_argument("--target-noise", type=float, default=None, help="TD3 target policy smoothing σ")
    ap.add_argument("--target-noise-clip", type=float, default=None)
    ap.add_argument("--caps-lambda-t", type=float, default=None, help="CAPS 時間平滑權重，0 = 關閉")
    ap.add_argument("--caps-lambda-s", type=float, default=None, help="CAPS 空間平滑權重，0 = 關閉")
    ap.add_argument("--caps-sigma", type=float, default=None, help="CAPS 加在縮放後觀測上的雜訊 σ")
    ap.add_argument("--asym-critic", action="store_true", default=None, help="critic 另看特權資訊")
    ap.add_argument("--lqi-only", action="store_true", help="只跑 LQI 基準並列印，不訓練")
    args = ap.parse_args(argv)
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def main():
    args = parse_args()
    timing = Timing.from_hz(args.ctrl_hz)
    run_name = f"07113_{args.tag}_{timing.ctrl_hz}hz_s{args.seed}"
    use_caps = args.caps_lambda_t > 0 or args.caps_lambda_s > 0

    for d in (MODEL_DIR, RESULT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"🚀 07113：{run_name}（純 RL；preset={args.preset}）")
    print(f"   專案根目錄 {PROJECT_ROOT}")
    print_plant_summary()
    print(f"   時間設定：CTRL_DT={timing.ctrl_dt}  N_SUBSTEPS={timing.n_substeps}  "
          f"MAX_EPISODE_STEPS={timing.max_episode_steps}  gamma={timing.gamma}  "
          f"擾動起點={timing.dist_start_range} 步、持續={timing.dist_duration_range} 步")

    K = design_lqi(timing)
    print(f"   LQI 增益 K（{timing.ctrl_hz} Hz，線性化 {timing.n_substeps} 子步，"
          f"z=[x, θ, ẋ, θ̇, u_applied, ∫x]）= {np.round(K, 3).tolist()}")

    if args.lqi_only:
        cb = PeriodicEvalCallback(timing, n_episodes=args.eval_episodes, lqi_gain=K)
        print(f"   ObservationBuilder.dt={cb.env_nom.builder.dt}")
        cb._on_training_start()
        return

    env = IP570Env(timing, plant_dr=True, sensor_dr=True, comm_dr=True, disturbance=True,
                   privileged_obs=args.asym_critic)
    n_actions = env.action_space.shape[-1]
    # σ 是在 SB3 的正規化動作空間 [-1,1] 上加的；動作本身就是 duty，所以 0.08 = 8% duty
    if args.noise_type == "ou":
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                    sigma=args.explore_sigma * np.ones(n_actions),
                                                    theta=0.15, dt=1e-2)
    else:
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.explore_sigma * np.ones(n_actions))

    algo_kwargs = dict(
        action_noise=action_noise,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        policy_delay=2,
        target_policy_noise=args.target_noise,
        target_noise_clip=args.target_noise_clip,
        learning_rate=lr_schedule,
        batch_size=256,
        buffer_size=200_000,
        gamma=timing.gamma,        # 約 4 秒有效視野：0.99^(dt/0.04)
        seed=args.seed,
    )
    if args.asym_critic:
        policy = AsymTD3Policy
        algo_kwargs["policy_kwargs"] = dict(features_extractor_class=ObsPriv)
    else:
        policy = "MlpPolicy"
    if use_caps:
        algo_cls = TD3CAPS
        algo_kwargs.update(lambda_t=args.caps_lambda_t, lambda_s=args.caps_lambda_s, caps_sigma=args.caps_sigma)
    else:
        algo_cls = TD3         # CAPS 關閉 → 原版 TD3，行為與原版完全一致
    model = algo_cls(policy, env, **algo_kwargs)

    print(f"   algo={algo_cls.__name__}  policy={type(model.policy).__name__}  "
          f"replay={type(model.replay_buffer).__name__}  obs_space={env.observation_space}")
    print(f"   ObservationBuilder.dt={env.builder.dt}  noise={args.noise_type} σ {args.explore_sigma}→"
          f"{args.explore_sigma_final}  target {args.target_noise}/{args.target_noise_clip}  "
          f"CAPS λ_T={args.caps_lambda_t} λ_S={args.caps_lambda_s} σ={args.caps_sigma}")
    print(f"   lr(start)={lr_schedule(1.0)}  learning_starts={model.learning_starts}  "
          f"batch={model.batch_size}  net={model.policy.net_arch}")

    best_path = MODEL_DIR / f"{run_name}_best"
    eval_cb = PeriodicEvalCallback(timing, privileged_obs=args.asym_critic, eval_freq=args.eval_freq,
                                   n_episodes=args.eval_episodes, save_path=str(best_path), lqi_gain=K)
    callbacks = CallbackList([
        ControlMetricsCallback(timing),
        eval_cb,
        NoiseDecayCallback(total_timesteps=args.timesteps,
                           sigma0=args.explore_sigma, sigma1=args.explore_sigma_final),
    ])
    # ℹ️ 判讀：看振盪請用 eval_nominal/osc_* 對照 lqi_nominal/osc_*（同頻率的 LQI）；
    #    跨頻率比較請把 x 軸換成 time/sim_seconds。
    model.learn(total_timesteps=args.timesteps, tb_log_name=run_name, callback=callbacks)

    final_path = MODEL_DIR / run_name
    model.save(str(final_path))
    actor_path = MODEL_DIR / f"{run_name}_actor.pt"
    export_actor(model, actor_path)
    diff = check_actor_export(model, timing, args.asym_critic)
    print(f"   actor.pt 匯出檢查：|actor.mu(obs6) − predict| 最大差 = {diff:.2e}")

    # ── 最終評估：final / best / LQI，種子 FINAL_EVAL_SEED0 起 ──
    policies = {"rl_final": _policy_act(model)}
    if eval_cb.best_saved:
        policies["rl_best"] = _policy_act(algo_cls.load(str(best_path), device=model.device))
    policies["lqi"] = _lqi_act(K)
    print(f"   最終評估：{list(policies)} × (nominal, dr) × {args.final_eval_episodes} 集，"
          f"種子 {FINAL_EVAL_SEED0} 起")
    seeds, summary, episodes = final_evaluation(policies, timing, args.asym_critic, args.final_eval_episodes)
    for key, s_ in summary.items():
        print(f"   [{key:<16}] return={s_['return_mean']:7.0f}  full_len={s_['full_length_frac']:.2f}  "
              f"x_std={s_.get('osc_x_std_tail', float('nan')):.4f}  "
              f"θ_std={s_.get('osc_theta_std_tail', float('nan')):.4f}  "
              f"angle_hit={s_.get('angle_limit_hit', float('nan')):.2f}")
    result = dict(
        version="07113", run_name=run_name, setting=f"07113_{args.tag}_{timing.ctrl_hz}hz",
        tag=args.tag, seed=args.seed, args=vars(args), timing=asdict(timing),
        algo=algo_cls.__name__, lqi_gain=K.tolist(),
        final_eval_seeds=[int(seeds[0]), int(seeds[-1])],
        summary=summary, episodes=episodes,
    )
    result_path = RESULT_DIR / f"{run_name}_final_eval.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    config = dict(
        version="07113", run_name=run_name, args=vars(args),
        timing=asdict(timing), ctrl_dt=timing.ctrl_dt, phys_dt=PHYS_DT, n_substeps=timing.n_substeps,
        gamma=timing.gamma,
        obs_layout=["x", "theta", "x_dot", "theta_dot", "u_applied", "integral_x"],
        builder=ObservationBuilder(dt=timing.ctrl_dt,
                                   m_per_count=derive_plant(PLANT_NOMINAL)["m_per_count"]).to_config(),
        encoder_cpr=ENCODER_CPR, adc_max=ADC_MAX, pot_elec_deg=POT_ELEC_DEG,
        adc_oversample=ADC_OVERSAMPLE,
        theta_limit=THETA_LIMIT, rail_limit=RAIL_LIMIT,
        plant_nominal=PLANT_NOMINAL, plant_derived=derive_plant(PLANT_NOMINAL),
        plant_dr={k: list(v) for k, v in PLANT_DR.items()}, sensor_dr=SENSOR_DR, comm_dr=COMM_DR,
        disturbance=DISTURBANCE, reward=REWARD, lqi_gain=K.tolist(),
        asym_critic=bool(args.asym_critic), priv_layout=PRIV_LAYOUT if args.asym_critic else None,
        deployment=dict(
            control_period_s=timing.ctrl_dt,
            note=f"MCU/PC 控制週期必須為 {timing.ctrl_dt * 1000:.0f} ms（{timing.ctrl_hz} Hz），與訓練一致",
            actor_file=actor_path.name,
            how=("只需 actor：duty = actor.mu(obs6)，obs6 由 ObservationBuilder(dt=ctrl_dt) 產生；"
                 + ("critic 用到的特權資訊不上機。" if args.asym_critic else "")),
        ),
    )
    with open(MODEL_DIR / f"{run_name}_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"✅ 完成：{final_path.name}.zip、{best_path.name}.zip、{actor_path.name}、"
          f"{run_name}_config.json → {MODEL_DIR}")
    print(f"         {result_path.name} → {RESULT_DIR}")


if __name__ == "__main__":
    main()
