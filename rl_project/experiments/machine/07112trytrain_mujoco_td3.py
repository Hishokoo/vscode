# 07112trytrain_mujoco_td3.py
# IP570 直線倒立擺「上機前系統識別」版：自寫 MuJoCo 模型 + 電壓致動器 + 感測器合成 + 獎勵平滑化
#
# 這一版和 07111 的關係：不是在 InvertedPendulum-v4 上再加隨機化，而是整個換掉被控體。
#   07111 的環境（0.6 m 擺桿、約 5 kg 的擺、±1 m 滑軌、±300 N 力源）和 IP570 差距太大，
#   放在原生環境上的任何 DR 範圍都量不準。所以 07112 從規格表重建一個 IP570 模型，從零訓練。
#   與 05114 / 07111 的壓力測試比較基準就此斷開，07112 用同一環境內的 LQI 當新基準。
#
# 參數推導、每個數字的來源與到貨後量測清單見 prompts/07112_system_id.md
#
# ─────────────────────────────────────────────────────────────────────────────
# 🌟 被控體：依 IP570 規格重建（取代 InvertedPendulum-v4）
#   - 滑軌有效行程 385 mm → 硬限位 ±0.1925 m，軟體終止 ±0.15 m（07111 的 rail_limit=0.9 在實機上會直接撞底）
#   - 擺桿 0.39 m（規格），不鏽鋼 ⌀5 mm 估約 62 g，加夾頭 12 g（規格未給，見 md）
#   - 動作 = PWM duty ∈ [-1, 1]，不再是力。MuJoCo general 致動器直接表達直流馬達：
#       F = K_F·duty − B·ẋ，K_F = V·Kt/(R·r)，B = Kt·Ke/(R·r²)  （反電動勢阻尼由物理引擎產生）
#   - 馬達轉子慣量折算到滑車：armature = J·N²/r² ≈ 3.6 kg，是滑車本體（約 0.2 kg）的 18 倍。
#     ⚠️ 這修正了上一輪討論「機械時間常數只有 5–9 ms」的說法：那裡漏算了轉子慣量。
#        實際 m_eq/B ≈ 80 ms，比 20 ms 控制週期長，滑車是「慢速度源」而不是瞬間到速。
#   - 原本模擬馬達動態用的 α 低通濾波器整個拿掉（馬達動態現在由物理產生）
#   - integrator=implicitfast、timestep 2 ms、控制週期 20 ms（50 Hz，07111 是 25 Hz）
#
# 🌟 感測器：從 MuJoCo 狀態「合成原始讀值」，再丟進與實機共用的 ObservationBuilder
#   - 編碼器：260 線 × 4 倍頻 = 1040 count/rev（輸出軸）→ 量化成 count
#   - 角度：WDD35D4 型導電塑膠電位器（電氣轉角 345°±2°）+ 12-bit ADC → 1.47e-3 rad/階，
#           每拍取 8 次 ADC 平均（韌體端照做），含零點偏移與比例誤差
#   - 速度：由 builder 做「差分 + 一階低通」，不再對真實速度加高斯雜訊（實機的速度就是這樣算的）
#
# 🌟 通訊：取代 delay_steps ∈ {0, 1} 的整步延遲
#   - 子步延遲 2–10 ms（PC 推論 + USB serial 來回），新指令在週期內第 L 個子步才生效
#   - 掉幀 2%：MCU 沿用上一拍 duty
#   - 第 5 維 = MCU 回報的「當下實際套用的 duty」，實機可 100% 重現（解決 07111 TODO：第五維語意）
#
# 🌟 新增第 6 維：位置積分 ∫x dt（抗飽和夾限）
#   - 角度零點偏移 b 會讓無記憶策略的平衡點偏到 x ≈ −(K_θ/K_x)·b。本環境實測 b=0.01 rad：
#     無積分偏 2.0 cm，加積分後 0.09 cm。選積分而不選 frame stack：延遲現在只有子步級、且第 5 維已帶出在途指令，
#     frame stack 能補的主要是延遲，補不了偏移（偏移需要長時間累積的量）。
#
# 🌟 獎勵：位能井平滑化（解決「死區內沒有梯度」）
#   - 階梯 +2 / +5 改成連續位能井：2·Kθ + 5·Kθ·Kx，
#       Kθ = 1/(1 + (θ/0.02)² + (θ̇/0.1)²)，Kx = 1/(1 + (x/0.03)² + (ẋ/0.1)²)   （Cauchy 長尾井）
#     中心曲率與高斯相同，但尾巴長，整段行程都有往中心的拉力。100K 冒煙測試（單一種子、40 集 DR 評估）：
#       高斯 σx=0.03 → DR 跑滿 50%、撞軟限 50%；高斯 σx=0.06 → 52.5%；Cauchy → 77.5%、撞軟限 17.5%
#     高斯版保留為 REWARD["well"]="gauss" 可切換
#   - 二次項權重依新尺度重設（x 改用 20，因為行程只剩 ±0.15 m）。LQI 基準就是用同一組二次項設計的
#   - 四項動作懲罰全部改用網路自身輸出 u（解決 07111 TODO：jitter 懲罰歸屬）；
#     移除 moving-average 項（與 jerk+snap 重複）
#
# 🌟 訓練流程
#   - γ 0.99 → 0.995：控制頻率加倍後，0.99 只看得到約 2 秒。實測 γ=0.99 在 40K 步時角度已穩，
#     但滑車在約 4 秒後一路漂去撞軟限（評估 100% 是 rail_limit_hit），撞牆的後果被折扣到幾乎看不見。
#     0.99^(20/40) ≈ 0.995，恢復與 07111 相同的約 4 秒視野
#   - 探索雜訊 σ：前 60% 固定 0.2，60%→90% 線性降到 0.05，之後維持
#   - 學習率：前 70% 為 1e-3，之後線性降到 3e-4（後段低學習率微調）
#   - 每 25K 步做 deterministic 評估：名目模型 / 全開 DR 各 5 集（固定種子，跨次可比），
#     同時記錄 LQI 在同樣 5 集的表現當參考線
#   - 新增振盪指標（後 50% 的 θ/x 標準差、過零頻率、x 平均偏移），用來區分
#     「摩擦極限環 / 偏移漂移 / 延遲振盪」
#
# ⚠️ 冒煙測試現況（100K 步、排程壓縮成 100K、單一種子；正式是 300K）：
#     名目模型 100% 跑滿 20 秒，但滑車 x 標準差約 6 cm、θ 過零約 1.2 Hz；LQI 是 1.2 cm、0.45 Hz。
#     也就是「會平衡，但還在搖」。300K 跑完先比 eval_nominal/osc_x_std_tail 與 lqi_nominal/osc_x_std_tail，
#     差距仍在 2 倍以上，就考慮 07113 改成 residual RL（策略輸出疊加在 LQI 上），見 md 第 5 節。
#
# 🔒 先講清楚的物理事實（實測於本模型的 LQR，無感測雜訊）：
#   - 無摩擦 → 完全靜止。滑車庫倫摩擦 2.5 N → x 標準差 12 mm 的 0.5 Hz 極限環；
#     電位器摩擦 1e-3 N·m（規格上限）→ θ 標準差 0.011 rad。
#     所以「完全不動」在實機上本來就做不到，RL 的振盪指標要拿同環境的 LQI 當地板來比。
#   - 07111 的 friction_hinge_range=(0, 0.01) 放到 IP570 上是規格上限的 10 倍。在原生環境裡無害，
#     只因為原生擺桿約 5 kg、重力恢復力矩是 IP570 的 120 倍以上。摩擦要以擺的重力力矩為尺度。
#
# 🔒 獨立性：從隨機初始化訓練，不讀任何 .zip，只需原始碼 + mujoco + gymnasium + SB3。
#    不使用 VecNormalize：觀測的固定縮放寫在 ObservationBuilder 裡，上機不需要額外的統計檔。
#    訓練結束另存 output/models/07112_config.json（builder 常數、名目參數、LQI 增益），上機時用。
#
# 🔒 路徑：以本檔位置推出專案根目錄 rl_project/（parents[2]），output/、tensorboard_logs/ 不論從哪裡執行
#    都落在 rl_project/ 底下。

import argparse
import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import gymnasium as gym
import mujoco
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


# ==========================================
# 0. 路徑與時間常數
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # rl_project/
MODEL_DIR = PROJECT_ROOT / "output" / "models"
LOG_DIR = PROJECT_ROOT / "tensorboard_logs" / "machine"

PHYS_DT = 0.002          # MuJoCo timestep（implicitfast）
N_SUBSTEPS = 10          # 每個控制週期的物理子步數
CTRL_DT = PHYS_DT * N_SUBSTEPS   # 0.02 s = 50 Hz，模擬與實機必須一致
MAX_EPISODE_STEPS = 1000         # 20 s


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

# 外力擾動：模擬手指敲擺桿頂端。以「名目擺桿上造成的角速度增量」定義衝量大小，
# 實際效果隨當集抽到的擺桿質量變化（同一下敲擊，輕的擺會被敲得比較偏，這是物理上對的）。
DISTURBANCE = dict(probability=0.3, start_range=(100, 750), duration_range=(2, 5),
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
# ==========================================
class ObservationBuilder:
    OBS_DIM = 6   # [x, θ, ẋ, θ̇, u_applied, ∫x]

    def __init__(self, dt=CTRL_DT, m_per_count=None, adc_zero=ADC_ZERO_NOMINAL,
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
class IP570Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, plant_dr=True, sensor_dr=True, comm_dr=True, disturbance=True,
                 max_episode_steps=MAX_EPISODE_STEPS):
        super().__init__()
        self.plant_dr = plant_dr
        self.sensor_dr = sensor_dr
        self.comm_dr = comm_dr
        self.disturbance_enable = disturbance
        self.max_episode_steps = max_episode_steps

        self.model = mujoco.MjModel.from_xml_string(IP570_XML)
        self.data = mujoco.MjData(self.model)
        self._pole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pole")
        self._nominal_derived = derive_plant(PLANT_NOMINAL)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(ObservationBuilder.OBS_DIM,),
                                                dtype=np.float32)
        self.builder = ObservationBuilder()
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
            self._lat_range = tuple(int(round(v / (PHYS_DT * 1000))) for v in COMM_DR["latency_ms"])
            self._drop_prob = COMM_DR["drop_prob"]
        else:
            n = int(round(COMM_NOMINAL["latency_ms"] / (PHYS_DT * 1000)))
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
            dur = int(self.np_random.integers(DISTURBANCE["duration_range"][0],
                                              DISTURBANCE["duration_range"][1] + 1))
            dthd = float(self.np_random.uniform(*DISTURBANCE["dtheta_dot_range"]))
            impulse = dthd * self._nominal_derived["I_pole_pivot"] / PLANT_NOMINAL["pole_L"]  # N·s at tip
            self.dist_plan = dict(
                start=int(self.np_random.integers(DISTURBANCE["start_range"][0],
                                                  DISTURBANCE["start_range"][1] + 1)),
                duration=dur,
                force=float(self.np_random.choice([-1.0, 1.0])) * impulse / (dur * CTRL_DT),
                dtheta_dot=dthd)

        self._reset_metrics()
        enc, adc = self._read_raw_sensors()
        obs = self.builder.reset(enc, adc)
        return obs, {}

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
            self.data.xfrc_applied[self._pole_id] = 0.0

        for k in range(N_SUBSTEPS):
            self.data.ctrl[0] = self.u_applied if k < lat else u_new
            mujoco.mj_step(self.model, self.data)
        self.u_applied = u_new

        # 乾淨狀態（只給獎勵、終止與指標用）
        x, th = float(self.data.qpos[0]), float(self.data.qpos[1])
        xd, thd = float(self.data.qvel[0]), float(self.data.qvel[1])

        # 策略看到的觀測（原始讀值 → builder）
        enc, adc = self._read_raw_sensors()
        obs = self.builder.update(enc, adc, self.u_applied)

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
        if n >= 100:
            h = n // 2
            tt, xx = th[h:], x[h:]
            zc = np.sum(np.diff(np.sign(tt - tt.mean())) != 0)
            m.update({
                "osc_theta_std_tail": float(np.std(tt)),
                "osc_x_std_tail": float(np.std(xx)),
                "osc_x_mean_abs_tail": float(abs(np.mean(xx))),
                "osc_theta_zc_hz_tail": float(zc / 2.0 / (len(tt) * CTRL_DT)),
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
def _linearize_nominal(latency_substeps=3, eps=1e-5):
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
        for k in range(N_SUBSTEPS):
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
    Az[5, :] = CTRL_DT * Az[0, :]; Az[5, 5] += 1.0   # I(k+1) = I(k) + dt·x(k+1)（與 builder 一致）
    Bz[5, 0] = CTRL_DT * Bz[0, 0]
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


def design_lqi():
    _, Az, Bz = _linearize_nominal()
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
# 8. Callbacks
# ==========================================
class ControlMetricsCallback(BaseCallback):
    """訓練中每集的控制指標（含探索雜訊；判讀振盪請看 eval_* 而不是這組）。"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "control_metrics" in info:
                self.episode_count += 1
                for key, value in info["control_metrics"].items():
                    self.logger.record(f"control/{key}", value)
                self.logger.dump(self.num_timesteps)
        return True


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


def summarize(results):
    keys = ["Episode_Length", "IAE_per_step", "Position_ISE_per_step", "RMS_Control_Effort",
            "Duty_Saturation_frac", "max_cart_pos_abs", "rail_limit_hit", "angle_limit_hit",
            "osc_theta_std_tail", "osc_x_std_tail", "osc_x_mean_abs_tail",
            "osc_theta_zc_hz_tail", "bonus_center_mean", "Settling_Time_sustained"]
    s = {"return_mean": float(np.mean([r for r, _ in results])),
         "full_length_frac": float(np.mean([m.get("Episode_Length", 0) >= MAX_EPISODE_STEPS
                                            for _, m in results]))}
    for k in keys:
        vals = [m[k] for _, m in results if k in m]
        if vals:
            s[k] = float(np.mean(vals))
    return s


class PeriodicEvalCallback(BaseCallback):
    """
    每 eval_freq 步：deterministic 策略在「名目模型」與「全開 DR」各跑 n_episodes 集。
    種子固定 → 每次評估面對同一批初始條件與 DR 抽樣，曲線可直接跨時間比較。
    LQI 在同一批種子上的表現每次都一起寫入（lqi_*），作為 TensorBoard 上的參考線。
    """
    def __init__(self, eval_freq=25_000, n_episodes=5, save_path=None, lqi_gain=None, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.seeds = np.arange(10_000, 10_000 + n_episodes)
        self.save_path = save_path
        self.env_nom = IP570Env(plant_dr=False, sensor_dr=False, comm_dr=False, disturbance=False)
        self.env_dr = IP570Env(plant_dr=True, sensor_dr=True, comm_dr=True, disturbance=True)
        self.lqi = LQIController(lqi_gain) if lqi_gain is not None else None
        self.lqi_summary = {}
        self.best = -np.inf

    def _on_training_start(self):
        if self.lqi is not None:
            act = lambda obs, env: self.lqi(env.builder)
            self.lqi_summary = {
                "lqi_nominal": summarize(run_episodes(self.env_nom, act, self.seeds)),
                "lqi_dr": summarize(run_episodes(self.env_dr, act, self.seeds)),
            }
            if self.verbose:
                for tag, s in self.lqi_summary.items():
                    print(f"   [{tag}] return={s['return_mean']:.0f}  full_len={s['full_length_frac']:.2f}  "
                          f"θ_std={s.get('osc_theta_std_tail', float('nan')):.4f}  "
                          f"x_std={s.get('osc_x_std_tail', float('nan')):.4f}")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        act = lambda obs, env: self.model.predict(obs, deterministic=True)[0]
        res = {"eval_nominal": summarize(run_episodes(self.env_nom, act, self.seeds)),
               "eval_dr": summarize(run_episodes(self.env_dr, act, self.seeds))}
        res.update(self.lqi_summary)
        for tag, s in res.items():
            for k, v in s.items():
                self.logger.record(f"{tag}/{k}", v)
        self.logger.dump(self.num_timesteps)
        score = res["eval_dr"]["return_mean"]
        if self.verbose:
            print(f"   [eval @ {self.num_timesteps}] nominal={res['eval_nominal']['return_mean']:.0f}  "
                  f"dr={score:.0f}  (LQI dr={self.lqi_summary.get('lqi_dr', {}).get('return_mean', float('nan')):.0f})")
        if self.save_path and score > self.best:
            self.best = score
            self.model.save(self.save_path)
        return True


class NoiseDecayCallback(BaseCallback):
    """探索雜訊 σ：前 start_frac 固定 sigma0，之後線性降到 sigma1（到 end_frac），再維持。"""
    def __init__(self, total_timesteps, sigma0=0.2, sigma1=0.05, start_frac=0.6, end_frac=0.9):
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


# ==========================================
# 9. 主程式
# ==========================================
def print_plant_summary():
    q = derive_plant(PLANT_NOMINAL)
    print("   IP570 名目模型："
          f"K_F={q['K_F']:.1f} N/duty  B={q['B']:.1f} N·s/m  armature={q['armature']:.2f} kg  "
          f"v_free={q['v_free']:.2f} m/s  τ_mech={q['tau_mech']*1000:.0f} ms")
    print(f"                 擺 {q['m_pole']*1000:.0f} g, lc={q['lc']*1000:.0f} mm, "
          f"不穩定極點≈{math.sqrt(q['m_pole']*9.81*q['lc']/q['I_pole_pivot']):.2f} rad/s  "
          f"m/count={q['m_per_count']*1e6:.1f} µm  θ 量化={1/ADC_COUNTS_PER_RAD*1000:.2f} mrad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--eval-freq", type=int, default=25_000)
    ap.add_argument("--lqi-only", action="store_true", help="只跑 LQI 基準並列印，不訓練")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("🚀 07112：IP570 系統識別模型 + 電壓致動器 + 平滑獎勵，從零訓練")
    print_plant_summary()

    K = design_lqi()
    print(f"   LQI 增益 K（z=[x, θ, ẋ, θ̇, u_applied, ∫x]）= {np.round(K, 3).tolist()}")

    if args.lqi_only:
        cb = PeriodicEvalCallback(n_episodes=10, lqi_gain=K)
        cb._on_training_start()
        return

    env = IP570Env(plant_dr=True, sensor_dr=True, comm_dr=True, disturbance=True)
    n_actions = env.action_space.shape[-1]
    # σ 是在 SB3 的正規化動作空間 [-1,1] 上加的；動作本身就是 duty，所以 0.2 = 20% duty
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    model = TD3(
        "MlpPolicy", env,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        learning_rate=lr_schedule,
        batch_size=256,
        buffer_size=200_000,
        gamma=0.995,              # 50 Hz 下維持與 07111（25 Hz、γ=0.99）相同的約 4 秒有效視野：0.99^(20/40)
        seed=args.seed,
    )
    print(f"   lr(start)={lr_schedule(1.0)}  learning_starts={model.learning_starts}  "
          f"batch={model.batch_size}  net={model.policy.net_arch}  obs_dim={env.observation_space.shape}")

    callbacks = CallbackList([
        ControlMetricsCallback(),
        PeriodicEvalCallback(eval_freq=args.eval_freq, n_episodes=5,
                             save_path=str(MODEL_DIR / "07112_best"), lqi_gain=K),
        NoiseDecayCallback(total_timesteps=args.timesteps),
    ])
    # ℹ️ 判讀：ep_len_mean 應由低往高爬；看振盪請用 eval_nominal/osc_* 對照 lqi_nominal/osc_*。
    #    100K 步後 eval_dr/full_length_frac 仍為 0 → 先檢查 rail_limit_hit 與 angle_limit_hit 哪個在觸發。
    model.learn(total_timesteps=args.timesteps, tb_log_name="07112_IP570_SysID", callback=callbacks)

    model.save(str(MODEL_DIR / "07112"))
    config = dict(
        version="07112",
        ctrl_dt=CTRL_DT, phys_dt=PHYS_DT, n_substeps=N_SUBSTEPS,
        obs_layout=["x", "theta", "x_dot", "theta_dot", "u_applied", "integral_x"],
        builder=ObservationBuilder(m_per_count=derive_plant(PLANT_NOMINAL)["m_per_count"]).to_config(),
        encoder_cpr=ENCODER_CPR, adc_max=ADC_MAX, pot_elec_deg=POT_ELEC_DEG,
        adc_oversample=ADC_OVERSAMPLE,
        theta_limit=THETA_LIMIT, rail_limit=RAIL_LIMIT,
        plant_nominal=PLANT_NOMINAL, plant_derived=derive_plant(PLANT_NOMINAL),
        plant_dr={k: list(v) for k, v in PLANT_DR.items()}, sensor_dr=SENSOR_DR, comm_dr=COMM_DR,
        reward=REWARD, lqi_gain=K.tolist(),
    )
    with open(MODEL_DIR / "07112_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print("✅ 完成：output/models/07112.zip、07112_best.zip、07112_config.json")


if __name__ == "__main__":
    main()
