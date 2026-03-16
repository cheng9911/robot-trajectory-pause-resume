import numpy as np
import time
from scipy.spatial.transform import Rotation
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
from pyroboplan.core.utils import extract_cartesian_pose
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.planning.cartesian_planner import CartesianPlanner, CartesianPlannerOptions
from pyroboplan.visualization.meshcat_utils import visualize_frames
import matplotlib.pyplot as plt
from pathlib import Path
import mujoco
import mujoco.viewer
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

# 检查 sys.path
print("sys.path:", sys.path)

from utils import mj
DELTA_T = 0.001
class Robot:
    def __init__(
        self, 
        urdf_path: str, 

       
        mesh_dir: str  ,
        vizualizer: bool = False,
        target_frame: str = "link_7",

       
    ):
        """
        Initialize the robot with the given URDF path.
        """
        self.model, self.collision_model, self.visual_model = self.load_robot(urdf_path, mesh_dir)
        geom_ids = [i for i in range(len(self.collision_model.geometryObjects))]
        for geom in self.collision_model.geometryObjects:
            print(geom.name, geom.meshPath if hasattr(geom, 'meshPath') else 'no meshPath')
        for i in range(len(geom_ids)):
            for j in range(i+2, len(geom_ids)):
                # 添加碰撞对 (避免同一个link重复碰撞)
                pair = pinocchio.CollisionPair(geom_ids[i], geom_ids[j])
                self.collision_model.addCollisionPair(pair)
        for i, pair in enumerate(self.collision_model.collisionPairs):
            geom1 = self.collision_model.geometryObjects[pair.first].name
            geom2 = self.collision_model.geometryObjects[pair.second].name
            print(f"Pair {i}: {geom1} <--> {geom2}")
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        # 初始化当前关节角度为零（或其他方式）
        self.q = np.zeros(self.model.nq)
        self.target_frame = target_frame
        self.q_vel_limits = np.array([1.5] * self.model.nq)  # 最大速度 (rad/s)
        self.q_limits = np.array([[-5*np.pi]*self.model.nq, [5*np.pi]*self.model.nq]).T  # 最小最大位置限制（可选）
        self.q_acc_limits = np.array([10.0] * self.model.nq)  # 最大加速度 (rad/s²)
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model, data=self.data)
        self.viz.initViewer(open=vizualizer)
        self.viz.loadViewerModel()
        self.viz.displayVisuals(True)  # 隐藏
        self.viz.displayCollisions(True)   # 显示
        
        if(vizualizer):
            self.viz.display(self.q)
            time.sleep(2)
        
        
    def load_robot(self, urdf_path: str,mesh_dir ):
        """
        Load the robot model from a URDF file.
        """
        model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(urdf_path,mesh_dir)
        return model, collision_model, visual_model
    
    def get_cartesian_pose(self, q: np.ndarray):
        """
        Get the Cartesian pose of the robot's end-effector for the given joint configuration.
        """
        return extract_cartesian_pose(self.model, self.target_frame, q, data=self.data)
    
    def get_joint_position(self):
        """
        Target joint Command ,not Actual joint Command;
        """
        return  self.q
    
    def inverse_kinematics(self,  target_position: pinocchio.SE3, q_start: np.ndarray, max_retries: int = 5):
        """
        Solve the inverse kinematics problem using Differential IK.
        """
        ik = DifferentialIk(self.model, data=self.data, collision_model=self.collision_model,
                            options=DifferentialIkOptions(max_retries=max_retries))
        q_sol = ik.solve(self.target_frame, target_position, q_start)
        return q_sol

    def cartesian_planning(self, q_start: np.ndarray, tforms: list, dt: float = 0.05,max_retries: int = 10):
        """
        Perform Cartesian motion planning.
        """
        self.options = CartesianPlannerOptions(use_trapezoidal_scaling=True, max_linear_velocity=0.1,
                                               max_linear_acceleration=0.5, max_angular_velocity=1.0,
                                               max_angular_acceleration=1.0, )
        options = self.options
        ik = DifferentialIk(self.model, data=self.data, collision_model=self.collision_model, 
                            options=DifferentialIkOptions(max_retries=max_retries))
        planner = CartesianPlanner(self.model, self.target_frame, tforms, ik, options=options)
        success, t_vec, q_vec = planner.generate(q_start, dt)
        tforms_to_show = planner.generated_tforms[::5]
        if not success:
            # 抛出异常失败
            print("Failed to generate Cartesian path.")
            return None, None, None, False
        return tforms_to_show,t_vec, q_vec,success
    
#     def check_and_highlight_collisions(self, collision_threshold: float = 0.01) -> bool:
#         # 1. 计算正向运动学
#         pinocchio.forwardKinematics(self.model, self.data, self.q)
#         pinocchio.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data, self.q)

#         # 2. 计算所有碰撞对的碰撞信息（距离和碰撞状态）
#         pinocchio.computeCollisions(
#             self.collision_model,
#             self.collision_data,
#             False
#         )

#         collision_indices = []

#         # 3. 遍历所有碰撞对的结果
#         for i, cr in enumerate(self.collision_data.collisionResults):
#             dist = cr.min_distance if hasattr(cr, 'min_distance') else None
#             is_collision = cr.isCollision() if hasattr(cr, 'isCollision') else False

#             # 获取碰撞对
#             pair = self.collision_model.collisionPairs[i]
#             geom1 = self.collision_model.geometryObjects[pair.first]
#             geom2 = self.collision_model.geometryObjects[pair.second]

#             if dist is not None:
#                 dist_str = f"{dist:.4f} m"
#             else:
#                 dist_str = "N/A"

#             # 打印碰撞信息
#             # print(f"[Collision Check] Pair {i}: {geom1.name} <--> {geom2.name}, "
#             #     f"distance: {dist_str}, collision: {is_collision}")

#             if is_collision or (dist is not None and dist < collision_threshold):
#                 print(f"[Collision Detected] 碰撞对 {i}: {geom1.name} <--> {geom2.name}, "
#                     f"距离 {dist_str} < 阈值 {collision_threshold}")
#                 collision_indices.append(i)

#                 # 4. 重置所有碰撞几何体颜色为默认灰色
#                 # 1. 重置颜色，保证 alpha=1，避免透明
#                 for geom in self.visual_model.geometryObjects:
#                     node_name = self.viz.getViewerNodeName(geom, pinocchio.GeometryType.VISUAL)
#                     if node_name is None:
#                         print(f"Warning: node_name is None for geom {geom.name}")
#                         continue
#                     if not isinstance(node_name, str):
#                         node_name = str(node_name)
#                     if node_name in self.viz.viewer:
#                         self.viz.viewer[node_name].set_property("color", [0.7, 0.7, 0.7, 1])

# # 2. 高亮碰撞对应视觉模型，确保 alpha=1
#                 for i in collision_indices:
#                     pair = self.collision_model.collisionPairs[i]
#                     geom1_collision = self.collision_model.geometryObjects[pair.first]
#                     geom2_collision = self.collision_model.geometryObjects[pair.second]

#                     geom1_visual = next((g for g in self.visual_model.geometryObjects if g.name == geom1_collision.name), None)
#                     geom2_visual = next((g for g in self.visual_model.geometryObjects if g.name == geom2_collision.name), None)

#                     if geom1_visual:
#                         node_name1 = self.viz.getViewerNodeName(geom1_visual, pinocchio.GeometryType.VISUAL)
#                         if node_name1 in self.viz.viewer:
#                             self.viz.viewer[node_name1].set_property("color", [1, 0, 0, 1])

#                     if geom2_visual:
#                         node_name2 = self.viz.getViewerNodeName(geom2_visual, pinocchio.GeometryType.VISUAL)
#                         if node_name2 in self.viz.viewer:
#                             self.viz.viewer[node_name2].set_property("color", [1, 0, 0, 1])

#                         if collision_indices:
#                             print(f"[Collision] 检测到 {len(collision_indices)} 个碰撞对，已高亮显示视觉模型")
#                             return True

#             print("[Collision] 无碰撞")
#             return False

   
    def check_and_highlight_collisions(self, collision_threshold: float = 0.01) -> bool:
        # 1. 计算正向运动学
        pinocchio.forwardKinematics(self.model, self.data, self.q)
        pinocchio.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data, self.q)

        # 2. 计算所有碰撞对的碰撞信息（距离和碰撞状态）
        pinocchio.computeCollisions(
            self.collision_model,
            self.collision_data,
            False
        )

        collision_indices = []

        # 3. 遍历所有碰撞对的结果
        for i, cr in enumerate(self.collision_data.collisionResults):
            dist = cr.min_distance if hasattr(cr, 'min_distance') else None
            is_collision = cr.isCollision() if hasattr(cr, 'isCollision') else False

            # 获取碰撞对
            pair = self.collision_model.collisionPairs[i]
            geom1 = self.collision_model.geometryObjects[pair.first]
            geom2 = self.collision_model.geometryObjects[pair.second]

            if dist is not None:
                dist_str = f"{dist:.4f} m"
            else:
                dist_str = "N/A"

            # 打印碰撞信息
            # print(f"[Collision Check] Pair {i}: {geom1.name} <--> {geom2.name}, "
            #     f"distance: {dist_str}, collision: {is_collision}")

            if is_collision or (dist is not None and dist < collision_threshold):
                print(f"[Collision Detected] 碰撞对 {i}: {geom1.name} <--> {geom2.name}, "
                    f"距离 {dist_str} < 阈值 {collision_threshold}")
                collision_indices.append(i)

        # 4. 重置所有碰撞几何体颜色为默认灰色
        for geom in self.collision_model.geometryObjects:
            node_name = self.viz.getViewerNodeName(geom, pinocchio.GeometryType.COLLISION)
            self.viz.viewer[node_name].set_property("color", [0.7, 0.7, 0.7])

        # 5. 高亮碰撞几何体为红色
        for i in collision_indices:
            pair = self.collision_model.collisionPairs[i]
            geom1 = self.collision_model.geometryObjects[pair.first]
            geom2 = self.collision_model.geometryObjects[pair.second]

            node_name1 = self.viz.getViewerNodeName(geom1, pinocchio.GeometryType.COLLISION)
            node_name2 = self.viz.getViewerNodeName(geom2, pinocchio.GeometryType.COLLISION)

            self.viz.viewer[node_name1].set_property("color", [1, 0, 0])
            self.viz.viewer[node_name2].set_property("color", [1, 0, 0])

        if collision_indices:
            print(f"[Collision] 检测到 {len(collision_indices)} 个碰撞对，已高亮显示")
            return True

        print("[Collision] 无碰撞")
        return False
    def display_trajectory(self, q_vec: np.ndarray, t_vec: np.ndarray):
        """
        Visualize the joint trajectories in a Matplotlib plot.
        """
        plt.ion()
        plt.figure()
        plt.title("Robot Joint Position Trajectories")

        for joint_idx in range(q_vec.shape[1]):
            plt.plot(t_vec, q_vec[:, joint_idx], label=f"Joint {joint_idx}")

        plt.xlabel("Time [s]")
        plt.ylabel("Joint Position [rad]")
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_frames(self,  tforms: list,q_start: np.ndarray):
        """
        Visualize the frames along the Cartesian path.
        """
        self.viz.displayFrames(True, frame_ids=[self.model.getFrameId(self.target_frame)])
        self.viz.display(np.zeros(self.model.nq))  # Display initial configuration
        self.viz.display(q_start)
        visualize_frames(self.viz, "cartesian_plan", tforms, line_length=0.05, line_width=1)
        time.sleep(4)
    def servoJ(self, q_target: np.ndarray, delta_t: float = DELTA_T):
        """
        Python版本的servoJ接口，限制速度并更新关节位置状态self.q

        参数:
            q_target: 目标关节角 (np.ndarray)
            delta_t: 控制周期（单位：秒），默认1ms
        返回:
            0 成功，-1 失败
        """
        collision_threshold=0.001  # 碰撞检测阈值（单位：米）
        # **位置限制检查（可选，如果你定义了self.q_limits）**
        if hasattr(self, "q_limits"):  # self.q_limits: (n_joints, 2)
            for i in range(len(q_target)):
                qmin, qmax = self.q_limits[i]
                if not (qmin <= q_target[i] <= qmax):
                    print(f"[servoJ] 目标位置超限: Joint {i} = {np.degrees(q_target[i]):.2f} deg")
                    return -1

        # **速度限制检查**
        q_offset = np.abs(q_target - self.q)
        for i in range(len(q_target)):
            v_max = self.q_vel_limits[i]  # rad/s
            if q_offset[i] > v_max * delta_t:
                print(q_offset, q_target, self.q)
                print(f"[servoJ] 关节速度超限: Joint {i} = {np.degrees(q_offset[i]/delta_t):.2f} deg/s")
                return -1
        # **临时更新状态用于碰撞检测**
        q_backup = self.q.copy()
        self.q = q_target.copy()

        # **碰撞检测**
        if self.check_and_highlight_collisions(collision_threshold):
            print("[servoJ] 碰撞检测失败，中止执行")
            self.q = q_backup
            exit
            return -1


        # **位置伺服更新**
        self.q = q_target.copy()

        # **可选：触发可视化或仿真接口**
        if hasattr(self, "viz"):
            self.viz.display(self.q)
        time.sleep(delta_t)

        return 0
    def servol(self, pose_target: pinocchio.SE3, q_start: np.ndarray, duration: float = DELTA_T):
        """
        输入目标位姿（SE3），求解逆解并下发。
        """
        q_sol = self.inverse_kinematics(self.target_frame, pose_target.translation, q_start)
        if q_sol is not None:
            self.servoJ(q_sol, duration)
        else:
            print("[Robot] servol failed: no valid IK solution.")
    
    def MoveJ(self, q_target: np.ndarray, v_max: float = 1.0, a_max: float = 2.0, dt: float = DELTA_T,traj_rviz: bool = False):
            """
            使用T型速度规划执行关节空间插值运动，调用servoj逐步下发。
            并记录位置、速度、加速度轨迹用于可视化。
            """
            q_start = self.q.copy()
            delta_q = q_target - q_start
            max_delta = np.max(np.abs(delta_q))
            print(f"max_delta: {max_delta}, delta_q: {delta_q}")

            for i in range(len(q_target)):
                v_max_i = self.q_vel_limits[i]
                a_max_i = self.q_acc_limits[i]
                if np.abs(v_max) > v_max_i:
                    print(f"[Robot] MoveJ failed: velocity limit exceeded for joint {i}.")
                    return
                if np.abs(a_max) > a_max_i:
                    print(f"[Robot] MoveJ failed: acceleration limit exceeded for joint {i}.")
                    return

            if max_delta < 1e-9:
                print("[Robot] Already at target position.")
                return
            v_max_norm = v_max / max_delta  # 归一化速度
            a_max_norm = a_max / max_delta  # 归一化加速度

            t_steady=0.0
            t_total=0.0
            t_acc=0.0
            # 轨迹规划阶段计算 --------------------------------------------------
            # 计算临界速度（判断能否达到设定速度）
            v_critical = np.sqrt(a_max_norm * 1.0)  # 最大可能达到的速度
            t_acc = v_max_norm / a_max_norm        # 加速阶段时间
            s_acc = 0.5 * a_max_norm * t_acc**2    # 加速阶段位移
            # t_steady=0.0
            # t_total=0.0
            # t_acc=0.0
            # 判断轨迹类型（梯形/三角形）
            if s_acc * 2 >  1.0:  # 无法达到设定速度，使用三角形轨迹
                t_acc = np.sqrt(1.0 / a_max_norm)
                t_total = 2 * t_acc
                has_steady = False
            else:                # 使用梯形轨迹
                t_steady = (1.0 - 2*s_acc) / v_max_norm  # 匀速阶段时间
                t_total = 2*t_acc + t_steady
                has_steady = True
            # print("tacc:",t_acc, "tsteady:", t_steady, "ttotal:", t_total)
            # print("0.5 * a * current_time**2",0.5 * a_max_norm * t_acc**2)

            # 轨迹生成 ----------------------------------------------------------
            time_traj = []
            s_traj, v_traj, a_traj = [], [], []
            q_traj, qd_traj, qdd_traj = [], [], []

            current_time = 0.0
            while current_time <= t_total:
                # 计算当前阶段参数
                if has_steady:
                    if current_time < t_acc:  # 加速阶段
                        a = a_max_norm
                        v = a_max_norm * current_time
                        s = 0.5 * a * current_time**2
                    elif current_time < t_acc + t_steady:  # 匀速阶段
                        a = 0.0
                        v = v_max_norm
                        s = s_acc + v_max_norm*(current_time - t_acc)
                    else:  # 减速阶段
                        dec_time = current_time - (t_acc + t_steady)
                        a = -a_max_norm
                        v = v_max_norm +a *dec_time
                        s = s_acc + v_max_norm*t_steady + v_max_norm*dec_time + 0.5*a*dec_time**2
                        # print("dec_time",dec_time)
                        # print("s",s)
                        # print("v_max_norm*dec_time + 0.5*a*dec_time**2",v*dec_time + 0.5*a*dec_time**2)
                        # print("tacc:",t_acc, "tsteady:", t_steady, "ttotal:", t_total)
                else:  # 三角形轨迹
                    
                    if current_time < t_acc:  # 加速阶段
                        a = a_max_norm
                        v = a * current_time
                        s = 0.5 * a * current_time**2
                    else:  # 减速阶段
                        dec_time = current_time - t_acc
                        a = -a_max_norm
                        v = a_max_norm*t_acc + a*dec_time
                        s = 0.5*a_max_norm*t_acc**2 + a_max_norm*t_acc*dec_time + 0.5*a*dec_time**2
                        
                # 边界保护
                # s = np.clip(s, 0.0, 1.0)
                # v = v if s < 1.0 else 0.0
                # a = a if s < 1.0 else 0.0

                # 转换到实际关节空间
                q_current = q_start + s * delta_q
                qd_current = v * delta_q
                qdd_current = a * delta_q
                

                time_traj.append(current_time)
                s_traj.append(s)
                v_traj.append(v)
                a_traj.append(a)
                q_traj.append(q_current.copy())
                qd_traj.append(qd_current.copy())
                qdd_traj.append(qdd_current.copy())
                result=self.servoJ(q_current, dt)
                if result != 0:
                    print("[MoveJ] servoJ失败，中止MoveJ")
                    break    
                current_time += dt
                


            
            # for i in range(len(q_traj)):
            #     result=self.servoJ(q_traj[i], dt)
            #     if result != 0:
            #         print("[MoveJ] servoJ失败，中止MoveJ")
            #         break    
            # print("q_current",q_current,"q_target",s)
            # 可视化
            if traj_rviz:
                self.plot_trajectory(time_traj, q_traj, qd_traj, qdd_traj)
            
            
    def MoveL(self, tforms: list, dt: float = DELTA_T, max_retries: int = 10):
        """
        基于笛卡尔路径规划执行 moveL（线性轨迹运动），逐点调用 servol。
        
        参数：
            tforms: list of pinocchio.SE3
                目标末端位姿序列（通常为线性插值生成）
            dt: float
                每个点的时间间隔，单位：秒
            max_retries: int
                IK 求解最大尝试次数
        """
        # 当前起始关节角
        q_start = self.q.copy()
        
        # 调用笛卡尔路径规划器
        tforms_to_show, t_vec, q_vec, success = self.cartesian_planning(
            q_start=q_start,
            tforms=tforms,
            dt=dt,
            max_retries=max_retries
        )

        if not success:
            print("[Robot] moveL 失败：笛卡尔路径规划不成功")
            return

        # 按照轨迹中的位姿逐点追踪执行
        for idx in range(1, q_vec.shape[1]):
            self.servoJ(q_vec[:, idx], dt)
        # for i, pose in enumerate(tforms_to_show):
        #     self.servol(pose_target=pose, q_start=self.q, duration=dt)
    def plot_trajectory(self, time_series, q_traj, qd_traj, qdd_traj):
        q_traj = np.array(q_traj)
        qd_traj = np.array(qd_traj)
        qdd_traj = np.array(qdd_traj)

        num_joints = q_traj.shape[1]

        plt.figure(figsize=(12, 8))

        for j in range(num_joints):
            plt.subplot(3, 1, 1)
            plt.plot(time_series, q_traj[:, j], label=f'Joint {j+1}')
            plt.ylabel("Position (rad)")
            plt.title("Joint Position")

            plt.subplot(3, 1, 2)
            plt.plot(time_series, qd_traj[:, j], label=f'Joint {j+1}')
            plt.ylabel("Velocity (rad/s)")
            plt.title("Joint Velocity")

            plt.subplot(3, 1, 3)
            plt.plot(time_series, qdd_traj[:, j], label=f'Joint {j+1}')
            plt.ylabel("Acceleration (rad/s²)")
            plt.title("Joint Acceleration")
            plt.xlabel("Time (s)")

        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()
class DianaRobot(Robot):
    def __init__(self,target_frame, visualizer: bool = True):
        pinocchio_model_dir = Path(__file__).parent.parent.parent/ "assets"/"urdf"
        print("pinocchio_model_dir:", pinocchio_model_dir.as_posix())
        model_path = pinocchio_model_dir 
        print(model_path)
        # 构建URDF绝对路径并转换为字符串
        urdf_model_path = (
            pinocchio_model_dir 
            
            / "diana7_description" 
            / "urdf" 
            / "diana_v2.urdf"
        ).resolve()  # 解析符号链接和相对路径
        urdf_path = urdf_model_path.as_posix()  # 转换为POSIX路径字符串
        if not urdf_model_path.exists():
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")
        # 2. 模型加载修正
        # 获取mesh资源目录（转换为字符串）
        mesh_dir =model_path.resolve().as_posix()

        # 初始化父类（加载模型、初始化Meshcat）
        super().__init__(urdf_path, mesh_dir, vizualizer=visualizer,target_frame=target_frame)

        # 设置关节约束（从机器人控制器或官方SDK读取）
        dblMinPos = np.array([-3.124139, -1.570796*10, -3.124139, 0.000000, -3.124139, -3.124139, -3.124139])
        dblMaxPos = np.array([3.124139, 1.570796*10, 3.124139, 3.054326, 3.124139, 3.124139, 3.124139])

        # 最大关节速度 (rad/s)
        dblMaxVel = np.array([2.967060, 2.617994, 2.617994, 2.617994, 3.141593, 3.141593, 3.839724])

        # 最大关节加速度 (rad/s²)
        dblMaxAcc = np.array([10.780899, 8.733977, 8.931373, 8.794889, 14.885564, 14.762169, 14.803359])
        # 1. 位置限位
        self.model.lowerPositionLimit = dblMinPos  # 设置关节下限
        self.model.upperPositionLimit = dblMaxPos  # 设置关节上限
        self.q_vel_limits  = dblMaxVel
        self.q_acc_limits = dblMaxAcc
        # 1. 位置限位
        self.q_limits=np.array([dblMinPos, dblMaxPos]).T  # 最小最大位置限制（可选）

        # 2. 速度限位 (需手动扩展模型属性)
        self.model.velocityLimit = dblMaxVel       # 设置关节速度限位
        self.model.accelerationLimit = dblMaxAcc
        print("=== Collision pairs ===")
        
class DianaMujocoEnv(Robot):
    def __init__(self,target_frame, visualizer: bool = True):
        pinocchio_model_dir = Path(__file__).parent.parent.parent/ "assets"/"urdf"
        print("pinocchio_model_dir:", pinocchio_model_dir.as_posix())
        model_path = pinocchio_model_dir  
        print(model_path)
        # 构建URDF绝对路径并转换为字符串
        urdf_model_path = (
            pinocchio_model_dir 
            
            / "diana7_description" 
            / "urdf" 
            / "diana_v2.urdf"
        ).resolve()  # 解析符号链接和相对路径
        urdf_path = urdf_model_path.as_posix()  # 转换为POSIX路径字符串
        if not urdf_model_path.exists():
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")
        # 2. 模型加载修正
        # 获取mesh资源目录（转换为字符串）
        mesh_dir =model_path.resolve().as_posix()

        # 初始化父类（加载模型、初始化Meshcat）
        super().__init__(urdf_path, mesh_dir, vizualizer=visualizer,target_frame=target_frame)

        # 设置关节约束（从机器人控制器或官方SDK读取）
        dblMinPos = np.array([-3.124139, -1.570796, -3.124139, 0.000000, -3.124139, -3.124139, -3.124139])
        dblMaxPos = np.array([3.124139, 1.570796, 3.124139, 3.054326, 3.124139, 3.124139, 3.124139])

        # 最大关节速度 (rad/s)
        dblMaxVel = np.array([2.967060, 2.617994, 2.617994, 2.617994, 3.141593, 3.141593, 3.839724])

        # 最大关节加速度 (rad/s²)
        dblMaxAcc = np.array([10.780899, 8.733977, 8.931373, 8.794889, 14.885564, 14.762169, 14.803359])
        # 1. 位置限位
        self.model.lowerPositionLimit = dblMinPos  # 设置关节下限
        self.model.upperPositionLimit = dblMaxPos  # 设置关节上限
        self.q_vel_limits  = dblMaxVel
        self.q_acc_limits = dblMaxAcc
        # 1. 位置限位
        self.q_limits=np.array([dblMinPos, dblMaxPos]).T  # 最小最大位置限制（可选）

        # 2. 速度限位 (需手动扩展模型属性)
        self.model.velocityLimit = dblMaxVel       # 设置关节速度限位
        self.model.accelerationLimit = dblMaxAcc

        # MuJoCo环境初始化
        self.sim_hz = 500
        self.control_hz = 25
        self.latest_action = None
        self.render_cache = None
        
        # 加载MuJoCo模型
        mujoco_model_dir = Path(__file__).parent.parent.parent / "assets"/"mujoco"
        mujoco_model_path = (mujoco_model_dir / "diana7" / "scene.xml").resolve()
        self.mj_model = mujoco.MjModel.from_xml_path(mujoco_model_path.as_posix())
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # 关节名称列表（需与MuJoCo模型中的关节顺序一致）
        self.diana_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4",
                                 "joint_5", "joint_6","joint_7"]
        [mj.set_joint_q(self.mj_model, self.mj_data, jn, self.q[i]) for i, jn in enumerate(self.diana_joint_names)]
        
        # 初始化关节位置
        
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # 查看器句柄
       
        self.height = 256
        self.width = 256
        self.fovy = np.pi / 4
        self.camera_matrix = np.eye(3)
        self.camera_matrix_inv = np.eye(3)
        self.num_points = 4096

        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_renderer_depth = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_renderer_depth.update_scene(self.mj_data, 0)
        self.mj_renderer_depth.enable_depth_rendering()
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.camera_matrix = np.array([
            [self.height / (2.0 * np.tan(self.fovy / 2.0)), 0.0, self.width / 2.0],
            [0.0, self.height / (2.0 * np.tan(self.fovy / 2.0)), self.height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.step_num = 0
        # self.observation = self._get_obs()


    def _get_obs(self):
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_renderer_depth.update_scene(self.mj_data, 0)
        img = self.mj_renderer.render()
        depth = self.mj_renderer_depth.render()

        point_cloud = np.zeros((self.height * self.width, 6))
        for h in range(self.height):
            for w in range(self.width):
                point_cloud[h * self.width + w, :3] = self.camera_matrix_inv @ np.array([w * 1.0, h * 1.0, 1.0]) * \
                                                      depth[h, w]
                point_cloud[h * self.width + w, 3:] = img[h, w, :]
        sampled_points = self.uniform_sampling(point_cloud)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(sampled_points[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(sampled_points[:, 3:] / 255.0)
        # o3d.visualization.draw_geometries([pcd])

        for i in range(len(self.diana_joint_names)):
            self.robot_q[i] = mj.get_joint_q(self.mj_model, self.mj_data, self.diana_joint_names[i])
        self.robot_T = self.robot.fkine(self.robot_q)
        agent_pos = self.robot.fkine(self.robot_q).t
        obs = {
            'agent_pos': agent_pos,
            'point_cloud': sampled_points
        }
        self.render_cache = img
        return obs
    def render(self, mode):
        if self.render_cache is None:
            self._get_obs()
        return self.render_cache
    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_renderer_depth is not None:
            self.mj_renderer_depth.close()
    

    def servoJ(self, q_target: np.ndarray, delta_t: float = DELTA_T):
        """
        Python版本的servoJ接口，限制速度并更新关节位置状态 self.q，同时调用 MuJoCo 仿真。

        参数:
            q_target: 目标关节角 (np.ndarray)
            delta_t: 控制周期（单位：秒），默认1ms
        返回:
            0 成功，-1 失败
        """
        # 检查关节位置限制（如果有定义）
        if hasattr(self, "q_limits"):  # self.q_limits: (n_joints, 2)
            for i in range(len(q_target)):
                qmin, qmax = self.q_limits[i]
                if not (qmin <= q_target[i] <= qmax):
                    print(f"[servoJ] 目标位置超限: Joint {i} = {np.degrees(q_target[i]):.2f} deg")
                    return -1

        # 检查关节速度限制
        q_offset = np.abs(q_target - self.q)
        for i in range(len(q_target)):
            v_max = self.q_vel_limits[i]  # rad/s
            if q_offset[i] > v_max * delta_t:
                print(q_offset, q_target, self.q)
                print(f"[servoJ] 关节速度超限: Joint {i} = {np.degrees(q_offset[i]/delta_t):.2f} deg/s")
                return -1

        # 更新控制命令
        self.q = q_target.copy()

        if hasattr(self, "mj_data") and hasattr(self, "mj_model"):
            # 更新模拟控制信号（假设前7个为目标关节）
            self.mj_data.ctrl[:len(self.q)] = self.q

            # 执行一次仿真步进
            mujoco.mj_step(self.mj_model, self.mj_data)

        # 可视化更新
        if hasattr(self, "viz"):
            self.viz.display(self.q)
        if hasattr(self, "mj_renderer") and hasattr(self, "mj_viewer"):
            self.mj_renderer.update_scene(self.mj_data, 0)
            self.mj_viewer.sync()
        time.sleep(delta_t)
        return 0

# 使用示例
if __name__ == "__main__":
    robot = DianaRobot(target_frame="link_7")
    
    # 示例1：直接控制关节
    q_start = np.array([0.0, 4.564, 0, 1.84, 0.089, -0.504,0])
    robot.MoveJ(q_start, v_max=1.8, a_max=8.0, dt=DELTA_T,traj_rviz= True)

    init = robot.get_cartesian_pose(q_start)

    rot = Rotation.from_euler("z", 60, degrees=True).as_matrix()
    rot_neg = Rotation.from_euler("z", -60, degrees=True).as_matrix()
    tforms = [
        init,
        init * pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.2])),
        init * pinocchio.SE3(rot, np.array([0.0, 0.25, 0.2])),
        init * pinocchio.SE3(rot_neg, np.array([0.0, -0.25, 0.2])),
        init * pinocchio.SE3(np.eye(3), np.array([0.2, 0.0, 0.0])),
        init,
    ]
    target_pose=init * pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.2]))
    # robot.MoveL(tforms, dt=DELTA_T)
   
    # 示例2：使用MuJoCo查看器
    # env = DianaMujocoEnv(target_frame="link_7")
    
    # with mujoco.viewer.launch_passive(env.mj_model, env.mj_data) as env.viewer:
    #     for _ in range(10000):
    #         mujoco.mj_step(env.mj_model, env.mj_data)
    #         env.viewer.sync()
    #         time.sleep(0.01)