# urdf02_gazebo

基于 ROS Noetic + Gazebo 的小车 URDF/Xacro 仿真示例包，包含：
- 小车底盘与轮子建模
- 雷达、相机、深度相机（Kinect/OpenNI）插件
- 差速驱动插件（`cmd_vel` 控制）
- 自定义仿真世界 `box_house.world`

## 1. 环境要求

- Ubuntu 20.04
- ROS Noetic
- Gazebo（Noetic 默认 Gazebo 11）

## 2. 编译

在工作空间根目录执行：

```bash
catkin_make
source devel/setup.bash
```

建议把 `source ~/demo05_ws/devel/setup.bash` 加到 `~/.bashrc`。

## 3. 目录说明

- `launch/`
  - `demo02_car.launch`：在 Gazebo 空场景中生成小车
  - `demo03_env.launch`：在 `box_house.world` 场景中生成小车
  - `demo04_sensor.launch`：RViz + 状态发布（传感器调试）
- `urdf/`
  - `car.urdf.xacro`：总入口
  - `demo05_car_base.urdf.xacro`：底盘与轮子
  - `demo06_car_camera.urdf.xacro`：机身相机 link
  - `demo07_car_laser.urdf.xacro`：雷达与支架 link
  - `gazebo/*.xacro`：Gazebo 插件（运动/雷达/相机/深度相机）
- `worlds/box_house.world`：自定义仿真环境

## 4. 快速启动

### 4.1 在环境中启动小车

```bash
roslaunch urdf02_gazebo demo03_env.launch
```

可选参数：

```bash
roslaunch urdf02_gazebo demo03_env.launch start_gazebo:=false model_name:=mycar2 x:=0 y:=0 z:=0.2
```

- `start_gazebo:=false`：Gazebo 已经开着时只做模型生成
- `model_name`：避免与已存在模型重名

### 4.2 发送速度控制

```bash
rostopic pub /cmd_vel geometry_msgs/Twist \
"{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}" -r 10
```

## 5. 传感器话题（默认配置）

- 2D 雷达：`/scan`
- 普通相机：`/camera/image_raw`、`/camera/camera_info`
- 深度相机（挂在 `support`）：
  - `.../rgb/image_raw`
  - `.../depth/image_raw`
  - `.../depth/points`

> 注：深度相机插件 `libgazebo_ros_openni_kinect.so` 的最终话题前缀会受 `cameraName` 等参数影响。

## 6. 常见问题

### 6.1 `SpawnModel: Failure - entity already exists`

Gazebo 里已有同名模型。解决方式：

```bash
rosservice call /gazebo/delete_model "model_name: 'mycar'"
```

或换模型名：`model_name:=mycar2`。

### 6.2 `robot_description parameter is required and not set`

说明启动文件没先加载 `robot_description`。

正确写法示例：

```xml
<param name="robot_description" command="$(find xacro)/xacro $(find urdf02_gazebo)/urdf/car.urdf.xacro" />
```

### 6.3 xacro 解析失败（code 2）

重点检查：
- include 的 xacro 文件是否为空
- XML 是否完整闭合
- 关节/link 名字是否和插件参数一致

## 7. 开发建议

- 修改 URDF/Xacro 后先自检：

```bash
xacro src/urdf02_gazebo/urdf/car.urdf.xacro > /tmp/car.urdf
check_urdf /tmp/car.urdf
```

- Gazebo 与 RViz 同时调试时，建议固定 `frame` 与 `topic` 命名，避免后续导航节点对接困难。
