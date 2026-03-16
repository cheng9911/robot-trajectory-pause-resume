import mujoco
import mujoco.viewer
import numpy as np

# 加载模型和仿真
model = mujoco.MjModel.from_xml_path("waist_model_simple.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 控制第0个 actuator（motor）
        data.ctrl[0] = np.sin(data.time)  # 随时间变化的控制信号

        mujoco.mj_step(model, data)
        viewer.sync()
