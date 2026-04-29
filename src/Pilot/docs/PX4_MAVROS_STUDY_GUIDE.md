# PX4 + MAVROS 深度学习指南（Pilot 包）

## 1. 你现在有两套链路

1. 自定义 Gazebo 插件链路（你之前的）
- 控制话题：`/pilot/cmd_vel`、`/pilot/cmd_attitude`
- 控制器位置：`quadrotor_controller_plugin.cpp`
- 适合：学动力学、混控基础

2. PX4 标准链路（这次新增）
- 控制话题：`/mavros/setpoint_*`
- 飞控：PX4 SITL
- 桥接：MAVROS
- 适合：学真实工程链路（与实机更接近）

---

## 2. 新增文件说明

1. `scripts/px4_offboard_teacher.py`
- 教学版 OFFBOARD 节点
- 实现：预发送 setpoint -> 切 OFFBOARD -> 解锁 -> 持续悬停

2. `launch/pilot_px4_sitl_mavros.launch`
- 一键启动：PX4 SITL + MAVROS + Offboard
- 依赖你本机有 `PX4-Autopilot` 仓库

3. `launch/pilot_px4_existing_mavros.launch`
- 只启动 Offboard 节点
- 适合你已手动启动 PX4/MAVROS 时使用

---

## 3. 推荐学习顺序

1. 先理解状态机
- 看 `/mavros/state` 中 `connected / mode / armed`

2. 再理解 OFFBOARD 进入条件
- 为什么要预发送 setpoint
- 为什么 setpoint 不能断流

3. 再尝试改参数
- 改 `target_z` 看起飞高度变化
- 改 `rate_hz` 观察 OFFBOARD 稳定性

4. 最后再扩展轨迹控制
- 从“定点悬停”到“8字轨迹”

---

## 4. 常用命令

```bash
# 仅启动 Offboard（前提：PX4 + MAVROS 已就绪）
roslaunch Pilot pilot_px4_existing_mavros.launch target_z:=2.5
```

```bash
# 一键启动（需要 PX4 仓库路径正确）
roslaunch Pilot pilot_px4_sitl_mavros.launch px4_dir:=$HOME/PX4-Autopilot target_z:=2.0
```

```bash
# 看状态
rostopic echo /mavros/state
```

```bash
# 看本地位置
rostopic echo /mavros/local_position/pose
```

---

## 5. 常见问题

1. `OFFBOARD` 切不进去
- 通常是 setpoint 预发送不足或发布频率太低

2. 解锁失败
- 先看 `mode` 是否已是 `OFFBOARD`
- 检查仿真时间与连接状态

3. 节点报 `mavros_msgs` 找不到
- 本机还没安装 MAVROS（需要先安装 `ros-noetic-mavros ros-noetic-mavros-msgs`）

