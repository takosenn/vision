# targetとAgentが合体

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
from matplotlib.widgets import Button
from matplotlib.lines import Line2D
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import array

matplotlib.rcParams["font.family"] = "MS Gothic"  # Windows標準の日本語フォントを指定

# ドローン Agent数 6台
num_agents = 6

# --- パラメータ設定（論文 Example1 Fig.3 準拠） ---
center = (0, 0)
radius = 20  # targetの軌道半径
frames = 10000
xlim = (-10, 10)  # x軸の限界
ylim = (-10, 10)  # y軸の限界
R = 2  # targetとAgentの理想の距離
d_i = 2 * np.pi / num_agents  # Agentiとその隣接Agenti+-の理想角度
frame_time = 0.02  # interval=50msの場合    アニメーション全体の速度を調整
fps = 1 / frame_time
Omega = 2 / fps  # Ω=2
# ランダムウォークのパラメータ
random_walk_sigma = 0.2  # 1フレームごとの速度変化の標準偏差
max_speed = 1  # targetの最大速度
Agent_handles: list = []  # 各Agentの緑の球(target)のハンドル
visionSensor_handles: list = []  # 各AgentのvisionSensorのハンドル


client = RemoteAPIClient()
sim = client.require("sim")

# 各Agentの緑の球(target)のハンドル
for i in range(num_agents):
    object_name = f"Quadcopter[{i+1}]"
    Agent_handle = sim.getObject(f"/{object_name}/target")
    Agent_handles.append(Agent_handle)
    print(f"取得: {object_name}")

# --- 追加: 各AgentのvisionSensorのハンドル取得 ---
for i in range(num_agents):
    vision_sensor_name = f"Quadcopter[{i+1}]"
    visionSensor_handle = sim.getObjectHandle(f"/{vision_sensor_name}/visionSensor")
    visionSensor_handles.append(visionSensor_handle)
    print(f"取得: {vision_sensor_name}")

# 中央のtargetのハンドル
target_handle = sim.getObject("/Quadcopter[0]/target")
print("取得: Quadcopter[0]")

# シミュレーション開始
if sim.getSimulationState() == sim.simulation_stopped:
    sim.startSimulation()
    print("Simulation started")
    time.sleep(1.0)

# --- 初期化 ---
fig, ax = plt.subplots()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Example1")

# 目標の軌道
(point,) = ax.plot([0], [radius], "ro", label="Target")

# --- エージェントの初期角度を第i象限に配置（i=1:第1象限, i=2:第2象限, ...） ---
agent_positions = np.zeros((num_agents, 2))
radius_limit = 6  # 配置半径（中心からの距離）

for i in range(num_agents):
    theta = 2 * np.pi * i / num_agents

    r = radius_limit  # ランダム性を排除し、一定の半径で配置
    agent_positions[i, 0] = center[0] + r * np.cos(theta)
    agent_positions[i, 1] = center[1] + r * np.sin(theta)

# 色分け用カラーマップ（tab10を利用）
agent_colors = plt.get_cmap("tab10").colors[:num_agents]
agent_dots = ax.scatter(
    agent_positions[:, 0], agent_positions[:, 1], c=agent_colors, label="Agents"
)
# --- エージェント番号と色の凡例を追加 ---
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=f"Agent {i+1}",
        markerfacecolor=agent_colors[i],
        markersize=10,
    )
    for i in range(num_agents)
]
ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

# 各エージェントが隣接エージェント（前後の番号）の座標を知る
neighbor_indices = [
    ((i - 1) % num_agents, (i + 1) % num_agents) for i in range(num_agents)
]
agent_neighbors = []
for i in range(num_agents):
    left_idx = (i - 1) % num_agents
    right_idx = (i + 1) % num_agents
    left_pos = agent_positions[left_idx]
    right_pos = agent_positions[right_idx]
    agent_neighbors.append((left_pos, right_pos))
# agent_neighbors[i] = (左隣の座標, 右隣の座標)

for i in range(num_agents):
    left_idx = (i - 1) % num_agents
    right_idx = (i + 1) % num_agents
    dist_left = np.linalg.norm(agent_positions[i] - agent_positions[left_idx])
    dist_right = np.linalg.norm(agent_positions[i] - agent_positions[right_idx])

# --- アニメーション関数 ---

# --- targetのランダムウォーク用初期化 ---
target_pos = np.array([0.0, 0.0])  # 初期位置（円運動の初期値と同じ）
target_velocity = np.zeros(2)  # 初期速度


def init():
    point.set_data([0], [radius])
    agent_dots.set_offsets(agent_positions)
    return point, agent_dots


# --- グラフ用データ保存リスト ---
ro_i_history: list = [[] for _ in range(num_agents)]
eta_i_history: list = [[] for _ in range(num_agents)]
omega_i_history: list = [[] for _ in range(num_agents)]
alpha_i_history: list = [[] for _ in range(num_agents)]
u_vec_history: list = [[] for _ in range(num_agents)]
relative_velocity_history: list = [[] for _ in range(num_agents)]
# e_i_1, e_i_2の時間積分値（各エージェントごと）
e_i_1_integral = [0.0 for _ in range(num_agents)]
e_i_2_integral = [0.0 for _ in range(num_agents)]


def get_distance_from_vision_sensor(visionSensor_handle):
    result, state, aux = sim.readVisionSensor(visionSensor_handle)
    if aux and len(aux) > 1 and len(aux[1]) > 0:
        depth_buffer = aux[1]
        ro_i = min(depth_buffer)
        return ro_i
    else:
        return None  # 取得失敗時


def animate(i):
    global target_pos, target_velocity
    # targetのランダムウォーク
    # 速度にランダムな変化を加える。一瞬で枠外に飛び出さないように
    target_velocity += np.random.normal(0, random_walk_sigma, size=2)
    # 最大速度制限
    speed = np.linalg.norm(target_velocity)
    if speed > max_speed:
        target_velocity = target_velocity / speed * max_speed
    # 位置を更新
    target_pos += target_velocity * frame_time
    x, y = target_pos
    point.set_data([x], [y])
    agent_dots.set_offsets(agent_positions)

    if not hasattr(animate, "prev_agent_pos"):
        animate.prev_agent_pos = agent_positions.copy()
    if not hasattr(animate, "prev_target_pos"):
        animate.prev_target_pos = np.array([x, y])

    for j in range(num_agents):
        # --- visionSensorから距離取得 ---
        result = sim.getVisionSensorDepth(visionSensor_handles[j],0,[0,0],[0,0])
        if isinstance(result, tuple) and len(result) == 2:
            depth_bytes, resolution = result    #resolutionは解像度[256,256]を表す
            # bytes → float32配列に変換
            floatingNumbers = sim.unpackFloatTable(depth_bytes , 0,  0,  0)
            if len(floatingNumbers) > 0 :
                ro_i = min(floatingNumbers)  # 画面内の最短距離[m]
                # もし中心ピクセルだけ使いたい場合
                # width, height = resolution
                # center_idx = (height // 2) * width + (width // 2)
                # ro_i = arr[center_idx]
                print(f"Agent{j+1} visionSensor 測定成功: 距離 = {ro_i:.3f} [m]")
            else:
                # データが空の場合
                ro_i = np.linalg.norm(agent_positions[j] - np.array([x, y]))
                print(
                    f"aAgent{j+1} visionSensor 測定失敗: 距離 = {ro_i:.3f} [m] (計算値)"
                )
        else:
            # 取得失敗時
            ro_i = np.linalg.norm(agent_positions[j] - np.array([x, y]))
            print(f"Agent{j+1} visionSensor 測定失敗: 距離 = {ro_i:.3f} [m] (計算値)")
        # ローカル座標系の定義: x軸=target方向, y軸=その直交方向

        vec = agent_positions[j] - np.array([x, y])  # vecは他で使うので計算
        e_r = vec / ro_i  # target方向の単位ベクトル（ローカルx軸）
        e_theta = np.array([-e_r[1], e_r[0]])  # ローカルy軸
        # ローカル座標系でtargetや隣接エージェントの情報を取得
        # targetの相対速度（ローカル）
        agent_velocity = (agent_positions[j] - animate.prev_agent_pos[j]) / frame_time
        relative_velocity = agent_velocity - target_velocity
        relative_velocity_local = np.array(
            [np.dot(relative_velocity, e_r), np.dot(relative_velocity, e_theta)]
        )
        # 隣接エージェントのローカル角度
        idx_plus = (j + 1) % num_agents
        idx_minus = (j - 1) % num_agents
        vec_plus = agent_positions[idx_plus] - agent_positions[j]
        vec_minus = agent_positions[idx_minus] - agent_positions[j]
        theta_plus_local = np.arctan2(np.dot(vec_plus, e_theta), np.dot(vec_plus, e_r))
        theta_minus_local = np.arctan2(
            np.dot(vec_minus, e_theta), np.dot(vec_minus, e_r)
        )
        theta_now_local = 0.0  # 自分自身から見たtarget方向は常に0
        # ローカル角速度
        if not hasattr(animate, "prev_theta_local"):
            animate.prev_theta_local = np.zeros(num_agents)
        omega_i_local = theta_now_local - animate.prev_theta_local[j]
        omega_i_local = (omega_i_local + np.pi) % (2 * np.pi) - np.pi
        animate.prev_theta_local[j] = theta_now_local
        # 隣接エージェントのローカル角速度
        if not hasattr(animate, "prev_theta_plus_local"):
            animate.prev_theta_plus_local = np.zeros(num_agents)
        if not hasattr(animate, "prev_theta_minus_local"):
            animate.prev_theta_minus_local = np.zeros(num_agents)
        omega_i_plus_local = theta_plus_local - animate.prev_theta_plus_local[j]
        omega_i_plus_local = (omega_i_plus_local + np.pi) % (2 * np.pi) - np.pi
        omega_i_minus_local = theta_minus_local - animate.prev_theta_minus_local[j]
        omega_i_minus_local = (omega_i_minus_local + np.pi) % (2 * np.pi) - np.pi
        animate.prev_theta_plus_local[j] = theta_plus_local
        animate.prev_theta_minus_local[j] = theta_minus_local
        # ローカル角距離
        alpha_i_local = abs(theta_plus_local - theta_now_local)
        alpha_i_minus_local = abs(theta_minus_local - theta_now_local)
        # --- 制御プロトコルu_iの計算（ローカル座標系） ---
        eta = relative_velocity_local[0]
        eta_norm = abs(eta)
        if i == 0:
            e_i_1 = 0
            e_i_2 = 0
        else:
            tau_i_1 = 0.5
            tau_i_2 = 0.5
            e_i_1 = tau_i_1 * abs(ro_i - R + eta_norm)
            e_i_2 = tau_i_2 * abs(ro_i * (omega_i_local + Omega - omega_i_local))
        e_i_1_integral[j] += e_i_1 * frame_time
        e_i_2_integral[j] += e_i_2 * frame_time
        fi = (d_i * alpha_i_local - d_i * alpha_i_minus_local) / (2 * d_i)
        zi = (
            d_i * (omega_i_plus_local - omega_i_local)
            - d_i * (omega_i_local - omega_i_minus_local)
        ) / (2 * d_i)
        u_r = (
            -ro_i * omega_i_local**2
            - eta_norm
            - e_i_1_integral[j] * np.sign(ro_i - R + eta_norm)
        )
        u_theta = (
            (omega_i_local + Omega + fi) * eta_norm
            + zi * ro_i
            + e_i_2_integral[j] * np.sign(fi + Omega - omega_i_local)
        )
        if ro_i > 1.5 * R or ro_i < 0.5 * R:
            u_r = u_r * 0.5
        else:
            u_r = u_r * 0.2
        if alpha_i_local < np.pi / 3.4 or alpha_i_local > np.pi / 2.6:
            u_theta = u_theta * 2
        else:
            u_theta = u_theta * 1
        # --- ローカル→グローバル変換 ---
        theta_global = np.arctan2(e_r[1], e_r[0])
        A = np.array(
            [
                [np.cos(theta_global), -np.sin(theta_global)],
                [np.sin(theta_global), np.cos(theta_global)],
            ]
        )
        u_vec_local = np.array([u_r, u_theta])
        u_vec = A @ u_vec_local
        # 位置を仮更新
        new_pos = agent_positions[j] + u_vec * frame_time
        # targetとの距離を計算
        dist_to_target = np.linalg.norm(new_pos - target_pos)
        if dist_to_target >= R:
            agent_positions[j] = new_pos
        else:
            # R未満なら、targetから距離Rの位置に補正
            direction = (new_pos - target_pos) / np.linalg.norm(new_pos - target_pos)
            agent_positions[j] = target_pos + direction * R
        # omega_i, omega_i_plus, omega_i_minusを[rad/sec]に変換
        omega_i_sec = omega_i_local * fps
        # u_r, u_theta, u_vecを[m/sec]に変換
        u_vec_sec = u_vec * fps
        ro_i_history[j].append(ro_i)
        eta_i_history[j].append(eta)
        omega_i_history[j].append(omega_i_sec)  # [rad/sec]で保存
        alpha_i_history[j].append(alpha_i_local)  # [rad]で保存
        u_vec_history[j].append(u_vec_sec.copy())

    animate.prev_agent_pos = agent_positions.copy()
    animate.prev_target_pos = np.array([x, y])

    # Coppeliasim側でAgentの緑の球(target)の位置同期
    for j in range(num_agents):
        Agents_pos_3d = [agent_positions[j][0], agent_positions[j][1], 2.0]
        sim.setObjectPosition(Agent_handles[j], -1, Agents_pos_3d)
    # Coppeliasim側でtargetの緑の球(target)の位置同期
    target_pos_3d = [target_pos[0], target_pos[1], 2.0]
    sim.setObjectPosition(target_handle, -1, target_pos_3d)
    return point, agent_dots


ani = FuncAnimation(
    fig, animate, frames=frames, init_func=init, blit=True, interval=frame_time * 1000
)


# --- 再生/停止ボタンのみ ---
class AnimationControl:
    def __init__(self, anim):
        self.anim = anim
        self.running = True

    def toggle(self, event):
        if self.running:
            self.anim.event_source.stop()
        else:
            self.anim.event_source.start()
        self.running = not self.running


button_ax = plt.axes((0.85, 0.05, 0.1, 0.075))
button = Button(button_ax, "再生/停止")
control = AnimationControl(ani)
button.on_clicked(control.toggle)

plt.show()

# シミュレーション停止
print("Stopping simulation")
sim.stopSimulation()
