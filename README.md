# RoboSimXCtrl
 针对机器人在仿真环境中的运动控制器，且实现了运动学（Kinematics）,碰撞检测，轨迹启停相关功能。

## 环境配置

Installing pin:
```
conda install pinocchio -c conda-forge
```
Installing meshcat:
```
pip install meshcat
```
Installing gepetto-viewer:
```

conda install gepetto-viewer-corba -c conda-forge
```
Installing panda3d_viewer:
```
pip install panda3d_viewer
```

Installing pyroboplan:
```
pip3 install pyroboplan
pyroboplan需要3.10以上的版本，3.10以下的版本会报错，解决方案
git clone https://github.com/sea-bass/pyroboplan.git
修改pyproject.toml中的依赖版本，修改如下
dependencies = [
    "drake ",
    "pin == 3.4.0",
    "matplotlib ",
    "meshcat == 0.3.2",
    "scipy ",
    "toppra == 0.6.3",
    "plyfile ",
  ]
去除对python的版本限制
pip install -e .
```
Installing math:
```
pip3 install roboticstoolbox-python
pip install modern-robotics
pip install spatialmath-python
cd arm
pip install -e .

pip install pandas
```

## 使用参考
```python
test_mujoco.py
traj.py
traj_resume.py
```
