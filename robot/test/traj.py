# IDLE → MoveJ() → MOVING → 空格键 → PAUSING → PAUSED
#           ↑                                     ↓
#           ←────────────── r 键 ←───────────────┘
#           ↓                                     ↑
#           └→ MOVING(重启) → 空格键 → PAUSING → PAUSED
#                               ↓
#             ←────────────── r 键 ←─────────────┘


import numpy as np
import time
import threading
from scipy.spatial.transform import Rotation
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
from pyroboplan.core.utils import extract_cartesian_pose
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.planning.cartesian_planner import CartesianPlanner, CartesianPlannerOptions
from pyroboplan.visualization.meshcat_utils import visualize_frames
import matplotlib.pyplot as plt
from pathlib import Path
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from pynput import keyboard  # 用于键盘监听

DELTA_T = 0.001

class Robot:
    def __init__(
        self, 
        urdf_path: str, 
        mesh_dir: str,
        vizualizer: bool = False,
        target_frame: str = "link_7",
    ):
        """
        初始化机器人，并添加状态管理和中断标志。
        """
        self.model, self.collision_model, self.visual_model = self.load_robot(urdf_path, mesh_dir)
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        self.q = np.zeros(self.model.nq)
        self.target_frame = target_frame
        self.q_vel_limits = np.array([1.5] * self.model.nq)
        self.q_limits = np.array([[-np.pi]*self.model.nq, [np.pi]*self.model.nq]).T
        self.q_acc_limits = np.array([10.0] * self.model.nq)
        
        # 新增：状态和中断标志
        self.state = "IDLE"  # 状态: "IDLE", "MOVING", "PAUSING", "PAUSED","RESTARTING"
        self.pause_requested = False
        self.restart_requested = False  # 新增：重启请求标志
        self._state_lock = threading.Lock()
        self.original_target = None  # 新增：保存原始目标位置
        self.is_restarting_motion = False

        self.current_velocity = np.zeros(self.model.nq)  # 记录当前指令速度，用于Ruckig
        


        # 新增：用于存储和对比的轨迹数据
        self._time_plan = []      # 原始规划的时间点
        self._q_plan = []         # 原始规划的关节位置
        self._qd_plan = []        # 原始规划的关节速度
        self._qdd_plan = []       # 原始规划的关节加速度
        self._time_offset=0
        self._time_actual = []    # 实际执行的时间点（累计，包含暂停重启）
        self._q_actual = []       # 实际执行的关节位置
        self._qd_actual = []      # 实际执行的关节速度
        self._qdd_actual = []     # 实际执行的关节加速度

        # 可视化初始化
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model, data=self.data)
        self.viz.initViewer(open=vizualizer)
        self.viz.loadViewerModel()
        if(vizualizer):
            self.viz.display(self.q)
            time.sleep(2)

        
        
    def load_robot(self, urdf_path: str, mesh_dir: str):
        """加载机器人URDF模型。"""
        model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(urdf_path, mesh_dir)
        return model, collision_model, visual_model
    
    def get_cartesian_pose(self, q: np.ndarray):
        """获取末端笛卡尔位姿。"""
        return extract_cartesian_pose(self.model, self.target_frame, q, data=self.data)
    
    def inverse_kinematics(self, target_position: np.ndarray, q_start: np.ndarray, max_retries: int = 5):
        """使用Differential IK求解逆运动学。"""
        ik = DifferentialIk(self.model, data=self.data, collision_model=self.collision_model, 
                            options=DifferentialIkOptions(max_retries=max_retries))
        q_sol = ik.solve(self.target_frame, target_position, q_start)
        return q_sol
    
    def cartesian_planning(self, q_start: np.ndarray, tforms: list, dt: float = 0.05, max_retries: int = 10):
        """笛卡尔路径规划。"""
        options = CartesianPlannerOptions(
            use_trapezoidal_scaling=True,
            max_linear_velocity=0.1,
            max_linear_acceleration=0.5,
            max_angular_velocity=1.0,
            max_angular_acceleration=1.0,
        )
        ik = DifferentialIk(self.model, data=self.data, collision_model=self.collision_model, 
                            options=DifferentialIkOptions(max_retries=max_retries))
        planner = CartesianPlanner(self.model, self.target_frame, tforms, ik, options=options)
        success, t_vec, q_vec = planner.generate(q_start, dt)
        tforms_to_show = planner.generated_tforms[::5]
        if not success:
            print("笛卡尔路径规划失败。")
            return None, None, None, False
        return tforms_to_show, t_vec, q_vec, success

    def display_trajectory(self, q_vec: np.ndarray, t_vec: np.ndarray):
        """可视化关节轨迹。"""
        plt.ion()
        plt.figure()
        plt.title("关节位置轨迹")

        for joint_idx in range(q_vec.shape[1]):
            plt.plot(t_vec, q_vec[:, joint_idx], label=f"关节 {joint_idx}")

        plt.xlabel("时间 [秒]")
        plt.ylabel("关节位置 [弧度]")
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_frames(self, tforms: list, q_start: np.ndarray):
        """可视化路径上的坐标系。"""
        self.viz.displayFrames(True, frame_ids=[self.model.getFrameId(self.target_frame)])
        self.viz.display(np.zeros(self.model.nq))
        self.viz.display(q_start)
        visualize_frames(self.viz, "cartesian_plan", tforms, line_length=0.05, line_width=1)
        time.sleep(4)
        
    def servoJ(self, q_target: np.ndarray, delta_t: float = DELTA_T):
        """
        关节位置伺服控制。
        返回: 0 成功, -1 失败
        """
        if not isinstance(q_target, np.ndarray):
            q_target = np.array(q_target, dtype=np.float64)
        if hasattr(self, "q_limits"):
            for i in range(len(q_target)):
                qmin, qmax = self.q_limits[i]
                if not (qmin <= q_target[i] <= qmax):
                    print(f"[servoJ] 目标位置超限: 关节 {i} = {np.degrees(q_target[i]):.2f} 度")
                    return -1

        q_offset = np.abs(q_target - self.q)
        for i in range(len(q_target)):
            v_max = self.q_vel_limits[i]
            if q_offset[i] > v_max * delta_t:
                print(f"[servoJ] 关节速度超限: 关节 {i} = {np.degrees(q_offset[i]/delta_t):.2f} 度/秒")
                return -1

        self.q = q_target.copy()

        if hasattr(self, "viz"):
            self.viz.display(self.q)
        time.sleep(delta_t)
        return 0

    def servol(self, pose_target: pinocchio.SE3, q_start: np.ndarray, duration: float = DELTA_T):
        """基于笛卡尔位姿的伺服控制。"""
        q_sol = self.inverse_kinematics(self.target_frame, pose_target.translation, q_start)
        if q_sol is not None:
            self.servoJ(q_sol, duration)
        else:
            print("[Robot] servol 失败: 无有效逆解。")

    def MoveJ(self, q_target: np.ndarray, v_max: float = 1.0, a_max: float = 2.0, dt: float = DELTA_T, traj_rviz: bool = False):
        """
        支持键盘中断的关节空间运动。
        按下空格键触发平滑暂停。
        """
        if self.state == "MOVING":
            print("[MoveJ] 机器人已在运动中，请等待完成或暂停。")
            return
            
        self.state = "MOVING"
        self._time_offset=0
        self.pause_requested = False
        self.restart_requested = False
        self.is_restarting_motion = False  # 标记这不是重启运动
        self.original_target = q_target.copy()  # 保存原始目标位置

        
        self._should_plot = traj_rviz
        q_start = self.q.copy()
        delta_q = q_target - q_start
        max_delta = np.max(np.abs(delta_q))
        print(f"[MoveJ] 开始运动: 最大关节位移 = {max_delta:.4f} rad")

        # 关节极限检查
        for i in range(len(q_target)):
            v_max_i = self.q_vel_limits[i]
            a_max_i = self.q_acc_limits[i]
            if np.abs(v_max) > v_max_i:
                print(f"[MoveJ] 失败: 关节 {i} 速度超限。")
                self.state = "IDLE"
                return
            if np.abs(a_max) > a_max_i:
                print(f"[MoveJ] 失败: 关节 {i} 加速度超限。")
                self.state = "IDLE"
                return
                
        if max_delta < 1e-5:
            print("[MoveJ] 已在目标位置。")
            self.state = "IDLE"
            return

        # 归一化处理
        v_max_norm = v_max / max_delta
        a_max_norm = a_max / max_delta

        # 轨迹阶段计算
        
        t_acc = v_max_norm / a_max_norm
        s_acc = 0.5 * a_max_norm * t_acc**2

        if s_acc * 2 > 1.0:  # 三角轨迹
            t_acc = np.sqrt(1.0 / a_max_norm)
            t_total = 2 * t_acc
            has_steady = False
        else:  # 梯形轨迹
            t_steady = (1.0 - 2 * s_acc) / v_max_norm
            t_total = 2 * t_acc + t_steady
            has_steady = True
        
        # 绘制原有轨迹
        self._time_plan.clear()
        self._q_plan.clear()
        self._qd_plan.clear()
        self._qdd_plan.clear()
        
        sim_time = 0.0
        while sim_time <= t_total:
            # 使用和主循环相同的公式计算 s, v, a
            if has_steady:
                if sim_time < t_acc:  # 加速
                    a = a_max_norm
                    v = a_max_norm * sim_time
                    s = 0.5 * a * sim_time**2
                elif sim_time < t_acc + t_steady:  # 匀速
                    a = 0.0
                    v = v_max_norm
                    s = s_acc + v_max_norm * (sim_time - t_acc)
                else:  # 减速
                    dec_time = sim_time - (t_acc + t_steady)
                    a = -a_max_norm
                    v = v_max_norm + a * dec_time
                    s = s_acc + v_max_norm * t_steady + v_max_norm * dec_time + 0.5 * a * dec_time**2
            else:  # 三角轨迹
                if sim_time < t_acc:  # 加速
                    a = a_max_norm
                    v = a * sim_time
                    s = 0.5 * a * sim_time**2
                else:  # 减速
                    dec_time = sim_time - t_acc
                    a = -a_max_norm
                    v = a_max_norm * t_acc + a * dec_time
                    s = 0.5 * a_max_norm * t_acc**2 + a_max_norm * t_acc * dec_time + 0.5 * a * dec_time**2
            q_plan = q_start + s * delta_q
            qd_plan = v * delta_q
            qdd_plan = a * delta_q
            
            self._time_plan.append(sim_time)
            self._q_plan.append(q_plan.copy())
            self._qd_plan.append(qd_plan.copy())
            self._qdd_plan.append(qdd_plan.copy())
            
            sim_time += dt
        # === 原始规划轨迹生成结束 ===
        
        # 清空实际执行轨迹记录（为新一次MoveJ做准备）
        self._time_actual.clear()
        self._q_actual.clear()
        self._qd_actual.clear()
        self._qdd_actual.clear()

        self.movej_q0 = q_start.copy()
        self.movej_q1 = q_target.copy()
        self.movej_delta = q_target - q_start
        self.movej_max_delta = max_delta

        self.movej_t_acc = t_acc
        self.movej_t_steady = t_steady if has_steady else 0
        self.movej_t_total = t_total
        self.movej_has_steady = has_steady
        self.movej_a = a_max_norm
        self.movej_v = v_max_norm
        
        

        # 主运动循环
        self.current_time = 0.0
        # time_traj, q_traj, qd_traj, qdd_traj = [], [], [], []

        while self.current_time <= t_total and self.state == "MOVING":
            # 检查暂停请求
            if self.pause_requested:
                print(f"[MoveJ] 暂停请求于 t={self.current_time:.3f}s，开始沿路径参数 s 平滑停止...")
                self.state = "PAUSING"

                delta_q = self.movej_delta.copy()
                q0 = self.movej_q0.copy()

                delta_norm_sq = np.dot(delta_q, delta_q)
                if delta_norm_sq < 1e-12:
                    print("[MoveJ] delta_q 过小，无法执行暂停。")
                    self.q_pause = self.q.copy()
                    self.pause_s = 0.0
                    self.state = "PAUSED"
                    self.pause_requested = False
                    return

                # ------------------------------------------------------------------
                # 1) 先把当前关节状态投影到路径参数 s
                # ------------------------------------------------------------------
                s_now = np.dot(self.q - q0, delta_q) / delta_norm_sq
                s_now = float(np.clip(s_now, 0.0, 1.0))

                # q_dot = s_dot * delta_q  ->  s_dot = <q_dot, delta_q> / ||delta_q||^2
                sdot_now = np.dot(self.current_velocity, delta_q) / delta_norm_sq
                sdot_now = float(max(0.0, sdot_now))

                # q_ddot = s_ddot * delta_q
                sddot_now = np.dot(self.current_acceleration, delta_q) / delta_norm_sq
                sddot_now = float(sddot_now)

                # ------------------------------------------------------------------
                # 2) 从关节约束换算出 s 的约束
                # ------------------------------------------------------------------


                sdot_max = v_max_norm
                sddot_max = a_max_norm

                # ------------------------------------------------------------------
                # 3) 用 1 维 Ruckig，但“位置层”表示 s_dot
                #    即：
                #       new_position      -> s_dot
                #       new_velocity      -> s_ddot
                #       new_acceleration  -> s_jerk
                # ------------------------------------------------------------------
                otg_s = Ruckig(1, dt)
                inp_s = InputParameter(1)
                out_s = OutputParameter(1)

                # 把 Ruckig 的 position 层当作 s_dot
                inp_s.current_position = [sdot_now]
                inp_s.current_velocity = [sddot_now]
                inp_s.current_acceleration = [0.0]

                # 目标“位置=0” 实际表示目标 s_dot = 0
                inp_s.target_position = [0.0]
                inp_s.target_velocity = [0.0]
                inp_s.target_acceleration = [0.0]

                # 这里的 max_velocity / max_acceleration 实际对应：
                #   |s_ddot| <= sddot_max
                #   |s_jerk| <= sjerk_max
                #
                # 因为 position 层被你映射成了 s_dot，所以 Ruckig 的速度约束对应的是 s_ddot 约束
                inp_s.max_velocity = [sddot_max]

                # jerk 约束你如果没有单独定义，可以先给大一点
                sjerk_max = 1000
                inp_s.max_acceleration = [sjerk_max]

                inp_s.min_position = [0.0]

                # ------------------------------------------------------------------
                # 4) 循环执行停止
                # ------------------------------------------------------------------
                res = Result.Working
                s_curr = s_now

                while res == Result.Working:
                    res = otg_s.update(inp_s, out_s)
                    if res == Result.Error:
                        print("[MoveJ] Ruckig 路径参数停止轨迹规划出错。")
                        break

                    # 由于我们把“位置层”映射成 s_dot，这里这样读：
                    sdot_new = float(out_s.new_position[0])
                    sddot_new = float(out_s.new_velocity[0])

                    # 用积分得到新的 s
                    s_next = s_curr + sdot_new * dt
                    s_next = float(np.clip(s_next, 0.0, 1.0))

                    # 映射回关节空间
                    q_desired = q0 + s_next * delta_q
                    qd_desired = sdot_new * delta_q
                    qdd_desired = sddot_new * delta_q

                    self.current_time += dt
                    result = self.servoJ(q_desired, dt)
                    if result != 0:
                        print("[MoveJ] 停止过程中 servoJ 失败。")
                        break

                    # 同步内部状态
                    self.current_velocity = qd_desired.copy()
                    self.current_acceleration = qdd_desired.copy()

                    if self._should_plot:
                        self._time_actual.append(self.current_time)
                        self._q_actual.append(np.array(q_desired, dtype=np.float64))
                        self._qd_actual.append(np.array(qd_desired, dtype=np.float64))
                        self._qdd_actual.append(np.array(qdd_desired, dtype=np.float64))

                    s_curr = s_next
                    out_s.pass_to_input(inp_s)

                    # 停止判据
                    if abs(sdot_new) < 1e-6 and abs(sddot_new) < 1e-5:
                        break

                self.q_pause = self.q.copy()
                self.pause_s = float(np.clip(s_curr, 0.0, 1.0))

                print(f"[MoveJ] 已暂停在位置: {np.degrees(self.q)} 度")
                print("pause s =", self.pause_s)

                self.state = "PAUSED"
                self.pause_requested = False
                return

            # 正常轨迹计算
            if has_steady:
                if self.current_time < t_acc:  # 加速
                    a = a_max_norm
                    v = a_max_norm * self.current_time
                    s = 0.5 * a * self.current_time**2
                elif self.current_time < t_acc + t_steady:  # 匀速
                    a = 0.0
                    v = v_max_norm
                    s = s_acc + v_max_norm * (self.current_time - t_acc)
                else:  # 减速
                    dec_time = self.current_time - (t_acc + t_steady)
                    a = -a_max_norm
                    v = v_max_norm + a * dec_time
                    s = s_acc + v_max_norm * t_steady + v_max_norm * dec_time + 0.5 * a * dec_time**2
            else:  # 三角轨迹
                if self.current_time < t_acc:  # 加速
                    a = a_max_norm
                    v = a * self.current_time
                    s = 0.5 * a * self.current_time**2
                else:  # 减速
                    dec_time = self.current_time - t_acc
                    a = -a_max_norm
                    v = a_max_norm * t_acc + a * dec_time
                    s = 0.5 * a_max_norm * t_acc**2 + a_max_norm * t_acc * dec_time + 0.5 * a * dec_time**2

            # 计算关节空间值
            q_current = q_start + s * delta_q
            qd_current = v * delta_q
            qdd_current = a * delta_q
            
            # 记录当前速度，供可能的暂停使用
            self.current_velocity = qd_current.copy()
            self.current_acceleration = qdd_current.copy()

            # 执行伺服
            result = self.servoJ(q_current, dt)
            if result != 0:
                print("[MoveJ] servoJ 失败，运动终止。")
                self.state = "IDLE"
                break
            if self._should_plot:  # 使用实例属性判断
                self._q_actual.append(q_current.copy())
                self._qd_actual.append(qd_current.copy())
                self._qdd_actual.append(qdd_current.copy())
                self._time_actual.append(self.current_time)

            self.current_time += dt

        if self.state == "MOVING":
            print("[MoveJ] 运动完成。")
            self.state = "IDLE"
            self.is_restarting_motion = False
        
    def resumeMoveJ(self, dt: float = DELTA_T, traj_rviz: bool = False):
        """
        从暂停点重新开始MoveJ运动。
        基于剩余的路径，重新生成一条从零速开始的T型（梯形/三角）速度轨迹。
        """
        if self.state != "PAUSED":
            print("[resumeMoveJ] 当前未暂停")
            return
        if self.original_target is None:
            print("[resumeMoveJ] 未找到原始目标")
            return

        print(f"[resumeMoveJ] 从 s={self.pause_s:.4f} 处重新规划运动...")

        self.is_restarting_motion = True
        self.state = "MOVING"
        self.pause_requested = False
        self._time_offset=self._time_actual[-1]

        # 获取原始路径参数
        q0 = self.movej_q0
        q1 = self.movej_q1
        delta_q = self.movej_delta
        s0 = self.pause_s
        v_max_norm = self.movej_v
        a_max_norm = self.movej_a

        # 计算剩余路径
        s_remaining = 1.0 - s0
        if s_remaining < 1e-6:
            print("[resumeMoveJ] 已非常接近终点，直接跳到目标。")
            self.servoJ(q1, dt)
            self.state = "IDLE"
            self.is_restarting_motion = False
            return

        # ===== 1. 为剩余路径重新规划T型速度曲线 =====
        # 判断是梯形还是三角轨迹
        # 能达到的最大速度 v_reach = sqrt(a * s_remaining)
        # 如果 v_reach >= v_max_norm，则为梯形轨迹，否则为三角轨迹
        s_acc_needed_to_reach_vmax = 0.5 * (v_max_norm ** 2) / a_max_norm
        has_steady_new = False
        t_acc_new = 0.0
        t_steady_new = 0.0
        t_total_new = 0.0

        if s_remaining > 2 * s_acc_needed_to_reach_vmax:
            # 梯形轨迹：有匀速段
            has_steady_new = True
            t_acc_new = v_max_norm / a_max_norm
            s_acc_new = 0.5 * a_max_norm * (t_acc_new ** 2)
            t_steady_new = (s_remaining - 2 * s_acc_new) / v_max_norm
            t_total_new = 2 * t_acc_new + t_steady_new
        else:
            # 三角轨迹：无法加速到最大速度
            has_steady_new = False
            t_acc_new = np.sqrt(s_remaining / a_max_norm)
            t_total_new = 2 * t_acc_new
            v_reach = a_max_norm * t_acc_new
            print(f"  [resumeMoveJ] 三角轨迹，可达速度 v_reach={v_reach:.3f} (norm)")

        print(f"  [resumeMoveJ] 重规划: s_rem={s_remaining:.3f}, 梯形?{has_steady_new}, t_acc={t_acc_new:.3f}s, t_steady={t_steady_new:.3f}s, t_total={t_total_new:.3f}s")
        # ===== 2. 主循环：基于新参数生成轨迹 =====
        t_new = 0.0
        while t_new <= t_total_new and self.state == "MOVING":
            # 检查暂停请求（保留原有暂停逻辑）
            if self.pause_requested:
                print(f"[MoveJ] 暂停请求于 t={self.current_time:.3f}s，开始沿路径参数 s 平滑停止...")
                self.state = "PAUSING"

                delta_q = self.movej_delta.copy()
                q0 = self.movej_q0.copy()

                delta_norm_sq = np.dot(delta_q, delta_q)
                if delta_norm_sq < 1e-12:
                    print("[MoveJ] delta_q 过小，无法执行暂停。")
                    self.q_pause = self.q.copy()
                    self.pause_s = 0.0
                    self.state = "PAUSED"
                    self.pause_requested = False
                    return

                # ------------------------------------------------------------------
                # 1) 先把当前关节状态投影到路径参数 s
                # ------------------------------------------------------------------
                s_now = np.dot(self.q - q0, delta_q) / delta_norm_sq
                s_now = float(np.clip(s_now, 0.0, 1.0))

                # q_dot = s_dot * delta_q  ->  s_dot = <q_dot, delta_q> / ||delta_q||^2
                sdot_now = np.dot(self.current_velocity, delta_q) / delta_norm_sq
                sdot_now = float(max(0.0, sdot_now))

                # q_ddot = s_ddot * delta_q
                sddot_now = np.dot(self.current_acceleration, delta_q) / delta_norm_sq
                sddot_now = float(sddot_now)

                # ------------------------------------------------------------------
                # 2) 从关节约束换算出 s 的约束
                # ------------------------------------------------------------------


                sdot_max = v_max_norm
                sddot_max = a_max_norm

                # ------------------------------------------------------------------
                # 3) 用 1 维 Ruckig，但“位置层”表示 s_dot
                #    即：
                #       new_position      -> s_dot
                #       new_velocity      -> s_ddot
                #       new_acceleration  -> s_jerk
                # ------------------------------------------------------------------
                otg_s = Ruckig(1, dt)
                inp_s = InputParameter(1)
                out_s = OutputParameter(1)

                # 把 Ruckig 的 position 层当作 s_dot
                inp_s.current_position = [sdot_now]
                inp_s.current_velocity = [sddot_now]
                inp_s.current_acceleration = [0.0]

                # 目标“位置=0” 实际表示目标 s_dot = 0
                inp_s.target_position = [0.0]
                inp_s.target_velocity = [0.0]
                inp_s.target_acceleration = [0.0]

                # 这里的 max_velocity / max_acceleration 实际对应：
                #   |s_ddot| <= sddot_max
                #   |s_jerk| <= sjerk_max
                #
                # 因为 position 层被你映射成了 s_dot，所以 Ruckig 的速度约束对应的是 s_ddot 约束
                inp_s.max_velocity = [sddot_max]

                # jerk 约束你如果没有单独定义，可以先给大一点
                sjerk_max = 1000
                inp_s.max_acceleration = [sjerk_max]

                inp_s.min_position = [0.0]

                # ------------------------------------------------------------------
                # 4) 循环执行停止
                # ------------------------------------------------------------------
                res = Result.Working
                s_curr = s_now

                while res == Result.Working:
                    res = otg_s.update(inp_s, out_s)
                    if res == Result.Error:
                        print("[MoveJ] Ruckig 路径参数停止轨迹规划出错。")
                        break

                    # 由于我们把“位置层”映射成 s_dot，这里这样读：
                    sdot_new = float(out_s.new_position[0])
                    sddot_new = float(out_s.new_velocity[0])

                    # 用积分得到新的 s
                    s_next = s_curr + sdot_new * dt
                    s_next = float(np.clip(s_next, 0.0, 1.0))

                    # 映射回关节空间
                    q_desired = q0 + s_next * delta_q
                    qd_desired = sdot_new * delta_q
                    qdd_desired = sddot_new * delta_q

                    self.current_time += dt
                    result = self.servoJ(q_desired, dt)
                    if result != 0:
                        print("[MoveJ] 停止过程中 servoJ 失败。")
                        break

                    # 同步内部状态
                    self.current_velocity = qd_desired.copy()
                    self.current_acceleration = qdd_desired.copy()

                    if self._should_plot:
                        self._time_actual.append(self.current_time)
                        self._q_actual.append(np.array(q_desired, dtype=np.float64))
                        self._qd_actual.append(np.array(qd_desired, dtype=np.float64))
                        self._qdd_actual.append(np.array(qdd_desired, dtype=np.float64))

                    s_curr = s_next
                    out_s.pass_to_input(inp_s)

                    # 停止判据
                    if abs(sdot_new) < 1e-6 and abs(sddot_new) < 1e-5:
                        break

                self.q_pause = self.q.copy()
                self.pause_s = float(np.clip(s_curr, 0.0, 1.0))

                print(f"[MoveJ] 已暂停在位置: {np.degrees(self.q)} 度")
                print("pause s =", self.pause_s)

                self.state = "PAUSED"
                self.pause_requested = False
                return

            # 基于新的时间 t_new 和新的阶段参数，计算 s, v, a
            if has_steady_new:
                if t_new < t_acc_new:
                    # 新轨迹的加速段
                    a = a_max_norm
                    v = a * t_new
                    ds = 0.5 * a * (t_new ** 2)  # 在新轨迹中的位移
                elif t_new < t_acc_new + t_steady_new:
                    # 新轨迹的匀速段
                    a = 0.0
                    v = v_max_norm
                    ds = 0.5 * a_max_norm * (t_acc_new ** 2) + v_max_norm * (t_new - t_acc_new)
                else:
                    # 新轨迹的减速段
                    dec_time = t_new - (t_acc_new + t_steady_new)
                    a = -a_max_norm
                    v = v_max_norm + a * dec_time
                    ds = (0.5 * a_max_norm * (t_acc_new ** 2) +
                          v_max_norm * t_steady_new +
                          v_max_norm * dec_time +
                          0.5 * a * (dec_time ** 2))
            else:
                # 新三角轨迹
                if t_new < t_acc_new:
                    a = a_max_norm
                    v = a * t_new
                    ds = 0.5 * a * (t_new ** 2)
                else:
                    dec_time = t_new - t_acc_new
                    a = -a_max_norm
                    v = a_max_norm * t_acc_new + a * dec_time
                    ds = (0.5 * a_max_norm * (t_acc_new ** 2) +
                          a_max_norm * t_acc_new * dec_time +
                          0.5 * a * (dec_time ** 2))

            # 总归一化路径参数 = 暂停点 + 新轨迹产生的位移
            s = s0 + ds
            if s > 1.0:
                s = 1.0

            # 计算关节空间指令
            q = q0 + s * delta_q
            qd = v * delta_q
            qdd = a * delta_q

            self.current_velocity = qd
            self.current_acceleration = qdd

            # 执行伺服
            res = self.servoJ(q, dt)
            if res != 0:
                print("[resumeMoveJ] servoJ失败")
                break

            # 记录轨迹（如果需要）
            if traj_rviz:
                # 注意：这里的时间记录是相对于原始轨迹起点的总时间
                # 您需要根据实际情况调整_time_offset或使用另一个变量
                self._time_actual.append(self._time_offset + t_new)
                self._q_actual.append(q.copy())
                self._qd_actual.append(qd.copy())
                self._qdd_actual.append(qdd.copy())

            t_new += dt

        if self.state == "MOVING":
            print(f"[resumeMoveJ] 重启运动完成，到达终点。")
            self.state = "IDLE"
        self.is_restarting_motion = False
    def MoveL(self, tforms: list, dt: float = DELTA_T, max_retries: int = 10):
        """笛卡尔直线运动（此版本暂不支持中断）。"""
        q_start = self.q.copy()
        tforms_to_show, t_vec, q_vec, success = self.cartesian_planning(
            q_start=q_start,
            tforms=tforms,
            dt=dt,
            max_retries=max_retries
        )
        if not success:
            print("[Robot] MoveL 失败：笛卡尔路径规划不成功")
            return
        for idx in range(1, q_vec.shape[1]):
            self.servoJ(q_vec[:, idx], dt)

    def plot_trajectory(self, time_series, q_traj, qd_traj, qdd_traj):
        """绘制轨迹曲线。"""
        q_traj = np.array(q_traj)
        qd_traj = np.array(qd_traj)
        qdd_traj = np.array(qdd_traj)
        num_joints = q_traj.shape[1]

        plt.figure(figsize=(12, 8))
        for j in range(num_joints):
            plt.subplot(3, 1, 1)
            plt.plot(time_series, q_traj[:, j], label=f'joint {j+1}')
            plt.ylabel("pose (rad)")
            plt.title("joint position")

            plt.subplot(3, 1, 2)
            plt.plot(time_series, qd_traj[:, j], label=f'joint {j+1}')
            plt.ylabel("vel (rad/s)")
            plt.title("joint velocity")

            plt.subplot(3, 1, 3)
            plt.plot(time_series, qdd_traj[:, j], label=f'joint {j+1}')
            plt.ylabel("acc (rad/s²)")
            plt.title("joint acceleration")
            plt.xlabel("t (s)")

        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()
    
    def plot_compare_trajectory(self):
        """
        Compare the original planned MoveJ trajectory with the actual executed trajectory
        (including pause / resume events)

        Verify:
            1. Whether the actual trajectory still lies on the original MoveJ geometric path
            2. Whether the actual velocity remains inside the trapezoidal (T-profile) velocity envelope
        """

        if len(self._time_plan) == 0 or len(self._time_actual) == 0:
            print("No trajectory data available for comparison")
            return

        q_plan = np.array(self._q_plan)
        qd_plan = np.array(self._qd_plan)
        time_plan = np.arange(q_plan.shape[0])

        q_actual = np.array(self._q_actual)
        qd_actual = np.array(self._qd_actual)
        time_actual = np.arange(q_actual.shape[0])

        num_joints = q_plan.shape[1]

        # ==============================
        # Geometric trajectory validation (MoveJ straight line)
        # ==============================

        q0 = q_plan[0]
        q1 = q_plan[-1]

        d = q1 - q0
        denom = np.dot(d, d)

        geom_dist = []

        for q in q_actual:

            v = q - q0
            alpha = np.dot(v, d) / denom

            proj = q0 + alpha * d

            dist = np.linalg.norm(q - proj)

            geom_dist.append(dist)

        geom_dist = np.array(geom_dist)

        print("\n====== MoveJ Geometric Trajectory Validation ======")
        print("Maximum deviation:", np.max(geom_dist))
        print("Average deviation:", np.mean(geom_dist))

        # ==============================
        # Trapezoidal velocity envelope
        # ==============================

        vel_limit = np.max(np.abs(qd_plan), axis=0)

        vel_violation = []

        for v in qd_actual:
            vel_violation.append(np.max(np.abs(v) - vel_limit))

        vel_violation = np.array(vel_violation)

        print("\n====== Trapezoidal Velocity Envelope Validation ======")
        print("Maximum violation:", np.max(vel_violation))
        print("Average violation:", np.mean(vel_violation))

        # ==============================
        # Plotting
        # ==============================

        fig, axs = plt.subplots(4, 1, figsize=(12, 12))

        fig.suptitle("MoveJ Trajectory Comparison (with Pause / Resume)", fontsize=14)

        # ======================
        # 1 Joint Position
        # ======================

        for j in range(num_joints):

            axs[0].plot(
                time_plan,
                q_plan[:, j],
                '--',
                label=f'Planned Joint{j+1}',
                alpha=0.7
            )

            axs[0].plot(
                time_actual,
                q_actual[:, j],
                '-',
                label=f'Actual Joint{j+1}',
                linewidth=1.5
            )

        axs[0].set_title("Joint Position")
        axs[0].set_ylabel("rad")
        axs[0].grid(True)

        # ======================
        # 2 Joint Velocity
        # ======================

        for j in range(num_joints):

            axs[1].plot(
                time_plan,
                qd_plan[:, j],
                '--',
                alpha=0.7
            )

            axs[1].plot(
                time_actual,
                qd_actual[:, j],
                '-',
                linewidth=1.5
            )

        axs[1].set_title("Joint Velocity")
        axs[1].set_ylabel("rad/s")
        axs[1].grid(True)

        # ======================
        # 3 Geometric Deviation
        # ======================

        axs[2].plot(
            time_actual,
            geom_dist,
            'r'
        )

        axs[2].set_title("Deviation from MoveJ Straight Line")
        axs[2].set_ylabel("rad")
        axs[2].grid(True)

        # ======================
        # 4 Velocity Envelope
        # ======================

        axs[3].plot(
            time_actual,
            vel_violation,
            'k'
        )

        axs[3].set_title("Velocity Violation vs T-profile Envelope")
        axs[3].set_ylabel("rad/s")
        axs[3].set_xlabel("time(s)")
        axs[3].grid(True)

        plt.tight_layout(rect=[0,0,1,0.96])

        plt.show()
        
        
class DianaRobot(Robot):
    def __init__(self, target_frame, visualizer: bool = True):
        pinocchio_model_dir = Path(__file__).parent.parent.parent / "assets" / "urdf"
        model_path = pinocchio_model_dir
        print(f"模型路径: {model_path}")
        
        urdf_model_path = (
            pinocchio_model_dir 
            / "diana7_description" 
            / "urdf" 
            / "diana_v2.urdf"
        ).resolve()
        urdf_path = urdf_model_path.as_posix()
        
        if not urdf_model_path.exists():
            raise FileNotFoundError(f"未找到URDF文件: {urdf_path}")
            
        mesh_dir = model_path.resolve().as_posix()
        super().__init__(urdf_path, mesh_dir, vizualizer=visualizer)

        # 设置Diana7的实际关节约束
        dblMinPos = np.array([-3.124139, -1.570796, -3.124139, 0.000000, -3.124139, -3.124139, -3.124139])
        dblMaxPos = np.array([3.124139, 1.570796, 3.124139, 3.054326, 3.124139, 3.124139, 3.124139])
        dblMaxVel = np.array([2.967060, 2.617994, 2.617994, 2.617994, 3.141593, 3.141593, 3.839724])
        dblMaxAcc = np.array([10.780899, 8.733977, 8.931373, 8.794889, 14.885564, 14.762169, 14.803359])

        self.model.lowerPositionLimit = dblMinPos
        self.model.upperPositionLimit = dblMaxPos
        self.q_vel_limits = dblMaxVel
        self.q_acc_limits = dblMaxAcc
        self.q_limits = np.array([dblMinPos, dblMaxPos]).T
        self.model.velocityLimit = dblMaxVel
        self.model.accelerationLimit = dblMaxAcc


# 键盘监听线程函数
def keyboard_listener(robot):
    """
    独立线程：监听键盘事件，按下空格键触发暂停，按下'r'键触发重启。
    """
    def on_press(key):
        try:
            if key == keyboard.Key.space:
                with robot._state_lock:
                    if robot.state == "MOVING" and not robot.pause_requested:
                        robot.pause_requested = True
                        print("\n[键盘监听] 空格键按下，暂停请求已发送。")
            elif hasattr(key, 'char') and key.char == 'r':  # 按下'r'键重启
                with robot._state_lock:
                    if robot.state == "PAUSED":
                        print("\n[键盘监听] 'r'键按下，重启运动。")
                        robot.restart_requested = True  # 仅设置标志，不调用函数
                        if robot.is_restarting_motion:
                            print("[键盘监听] 继续重启运动...")
                        
            elif hasattr(key, 'char') and key.char == 's':  # 新增：'s'键状态查询
                print(f"\n[状态查询] 当前状态: {robot.state}")
                print(f"          是否重启运动中: {robot.is_restarting_motion}")
                print(f"          当前位置: {np.degrees(robot.q)} 度")
                if robot.original_target is not None:
                    print(f"          原始目标: {np.degrees(robot.original_target)} 度")
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    # 初始化机器人
    target_frame = "link_7"
    robot = DianaRobot(target_frame, visualizer=True)
    
    # 启动键盘监听线程
    listener_thread = threading.Thread(target=keyboard_listener, args=(robot,), daemon=True)
    listener_thread.start()
    print("键盘监听已启动。")
    print("控制指令:")
    print("  空格键 - 暂停当前运动")
    print("  r 键   - 从暂停位置重启运动")
    print("  s 键   - 查询当前状态")
    print("\n您可以多次暂停和重启，直到到达最终目标位置。")
    time.sleep(1)
    
    # 定义起始关节角
    q_start = np.array([0.0, 0, 0, 0, 0, 0, 0])
    
    # 运动到起始点
    print("\n正在运动到起始点...")
    robot.MoveJ(q_start, v_max=1.8, a_max=8.0, dt=DELTA_T)
    print(f"已到达起始点: {np.degrees(q_start)} 度")
    time.sleep(1)
    
    # 定义目标点
    q_target = np.array([1.0, 0.8, 0.5, 2.0, 0.5, -0.8, 0.3])
    print(f"\n目标点: {np.degrees(q_target)} 度")
    
    # 执行可中断的MoveJ运动
    print("\n开始执行MoveJ运动。")
    print("演示：")
    print("1. 运动过程中按空格键暂停")
    print("2. 按'r'键重启")
    print("3. 在重启运动中可再次按空格键暂停")
    print("4. 再次按'r'键继续")
    print("5. 重复直到到达目标")
    robot.MoveJ(q_target, v_max=0.8, a_max=2.0, dt=DELTA_T, traj_rviz=True)
    print("\n进入主循环，等待指令...")
    try:
        while True:
            # 检查重启请求
            restart_needed = False
            with robot._state_lock:
                if robot.restart_requested and robot.state == "PAUSED":
                    restart_needed = True
                    robot.restart_requested = False  # 重置标志
            if robot.state == "IDLE" and not robot.restart_requested:
                print("\n[主线程] 所有运动完成，程序将退出。")
                break  # 退出while循环
            # 如果检测到重启请求，则在主线程中执行重启
            if restart_needed:
                print("\n[主线程] 执行重启运动。")
                # 注意：此调用在主线程中，不会阻塞键盘监听
                robot.resumeMoveJ(dt=DELTA_T, traj_rviz=True)
            
            time.sleep(0.01)  # 短暂睡眠，避免CPU占用过高
    except KeyboardInterrupt:
        print("\n程序退出。")
    # 检查最终状态
    print(f"\n最终状态: {robot.state}")
    print(f"最终关节位置: {np.degrees(robot.q)} 度")
    if robot.original_target is not None:
        error = np.linalg.norm(robot.q - robot.original_target)
        print(f"与目标位置误差: {error:.6f} rad")
    print("\n正在生成轨迹对比图...")
    robot.plot_compare_trajectory()
    # 保持程序运行
    print("\n程序运行结束。按Ctrl+C退出。")
    