#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
px4_offboard_teacher.py
=======================

这个脚本是“教学版 PX4 Offboard 控制示例”（ROS1 + MAVROS）。

目标：
1) 帮你理解 PX4 Offboard 的最小闭环流程
2) 用最少接口实现：连接 -> 预发送 setpoint -> 切 OFFBOARD -> 解锁 -> 持续控制
3) 注释尽量写全，适合初学者逐行阅读

---------------------------------------------------------------------
【你将学到的核心事实】
---------------------------------------------------------------------
A. 为什么要“先预发送 setpoint 再切 OFFBOARD”？
   - PX4 为了安全，要求在进入 OFFBOARD 前，外部控制源必须已经稳定输出 setpoint。
   - 如果你没先发，切模式通常会被拒绝。

B. 为什么要“持续发布 setpoint”？
   - OFFBOARD 不是“一次命令永久生效”。
   - 如果 setpoint 中断（常见阈值约 0.5s），PX4 会触发 failsafe 退出 OFFBOARD。

C. 最常用的话题和服务：
   - 订阅：/mavros/state
   - 发布：/mavros/setpoint_position/local
   - 服务：/mavros/set_mode, /mavros/cmd/arming

D. 坐标系提醒（非常重要）：
   - MAVROS 的 local_position 默认是 ENU（x前东，y北，z上），与 ROS 常规一致。
   - 飞控内部常见 NED，但 MAVROS 已帮你做常用转换。

---------------------------------------------------------------------
【运行前提】
---------------------------------------------------------------------
1) PX4 SITL 已启动，且 MAVROS 已连接飞控
2) 你能看到以下命令有输出：
   rostopic echo /mavros/state
3) 你知道当前测试环境是仿真（SITL），不是实机

---------------------------------------------------------------------
【使用方法】
---------------------------------------------------------------------
rosrun Pilot px4_offboard_teacher.py

可选参数（示例）：
rosrun Pilot px4_offboard_teacher.py _target_x:=0 _target_y:=0 _target_z:=2.0 _rate_hz:=20

说明：
- target_x/y/z: 期望悬停点（local ENU）
- rate_hz: setpoint 发布频率（建议 >= 20Hz）
"""

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


class Px4OffboardTeacher:
    """教学版 Offboard 控制器。

    结构非常简单：
    1) 订阅飞控状态
    2) 周期发布位置 setpoint
    3) 通过服务切模式与解锁
    """

    def __init__(self):
        # ------------------------- 参数读取区 -------------------------
        # 这些参数都支持 rosparam / 启动参数覆盖。
        self.target_x = rospy.get_param("~target_x", 0.0)
        self.target_y = rospy.get_param("~target_y", 0.0)
        self.target_z = rospy.get_param("~target_z", 2.0)
        self.rate_hz = rospy.get_param("~rate_hz", 20.0)

        # 当前飞控状态缓存（由回调更新）
        self.current_state = State()

        # ------------------------- ROS 通信区 -------------------------
        # 订阅飞控状态：连接状态、当前模式、是否已解锁等
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_cb, queue_size=10)

        # 发布本地位置 setpoint（OFFBOARD 常用接口之一）
        self.local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=20)

        # 等待服务可用：切模式 & 解锁
        rospy.loginfo("[offboard_teacher] waiting for /mavros/set_mode ...")
        rospy.wait_for_service("/mavros/set_mode")
        rospy.loginfo("[offboard_teacher] waiting for /mavros/cmd/arming ...")
        rospy.wait_for_service("/mavros/cmd/arming")

        self.set_mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.arming_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)

        # 统一构造目标位姿（本例仅控制位置，不控制姿态）
        self.pose = PoseStamped()
        self.pose.pose.position.x = self.target_x
        self.pose.pose.position.y = self.target_y
        self.pose.pose.position.z = self.target_z

        # 时间节流：避免过于频繁地请求服务
        self.last_mode_req = rospy.Time(0)
        self.last_arm_req = rospy.Time(0)

    def state_cb(self, msg: State):
        """飞控状态回调。"""
        self.current_state = msg

    def wait_for_connection(self):
        """等待 MAVROS 与飞控建立连接。"""
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown() and not self.current_state.connected:
            rospy.logwarn_throttle(2.0, "[offboard_teacher] waiting FCU connection...")
            rate.sleep()
        rospy.loginfo("[offboard_teacher] FCU connected.")

    def pre_stream_setpoints(self, count: int = 120):
        """预发送 setpoint，满足 PX4 进入 OFFBOARD 的前置条件。

        默认 120 次，在 20Hz 下约 6 秒。
        """
        rospy.loginfo("[offboard_teacher] pre-streaming setpoints...")
        rate = rospy.Rate(self.rate_hz)
        for _ in range(count):
            if rospy.is_shutdown():
                return
            self.pose.header.stamp = rospy.Time.now()
            self.local_pos_pub.publish(self.pose)
            rate.sleep()
        rospy.loginfo("[offboard_teacher] pre-stream done.")

    def run(self):
        """主循环：持续发 setpoint，并尝试切 OFFBOARD + 解锁。"""
        self.wait_for_connection()
        self.pre_stream_setpoints()

        rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("[offboard_teacher] entering control loop...")

        while not rospy.is_shutdown():
            now = rospy.Time.now()

            # 1) 若不在 OFFBOARD，周期性尝试切模式（每 2 秒最多一次）
            if self.current_state.mode != "OFFBOARD" and (now - self.last_mode_req) > rospy.Duration(2.0):
                try:
                    mode_resp = self.set_mode_srv(base_mode=0, custom_mode="OFFBOARD")
                    if mode_resp.mode_sent:
                        rospy.loginfo("[offboard_teacher] OFFBOARD mode request sent.")
                    else:
                        rospy.logwarn("[offboard_teacher] OFFBOARD request rejected.")
                except rospy.ServiceException as exc:
                    rospy.logerr("[offboard_teacher] set_mode call failed: %s", exc)
                self.last_mode_req = now

            # 2) 若未解锁且已在 OFFBOARD，周期性尝试解锁
            elif (not self.current_state.armed) and (now - self.last_arm_req) > rospy.Duration(2.0):
                try:
                    arm_resp = self.arming_srv(True)
                    if arm_resp.success:
                        rospy.loginfo("[offboard_teacher] arm request sent/success.")
                    else:
                        rospy.logwarn("[offboard_teacher] arm request failed.")
                except rospy.ServiceException as exc:
                    rospy.logerr("[offboard_teacher] arming call failed: %s", exc)
                self.last_arm_req = now

            # 3) 无论任何状态，都要持续发布 setpoint（OFFBOARD 保活关键）
            self.pose.header.stamp = rospy.Time.now()
            self.local_pos_pub.publish(self.pose)

            # 状态监控日志（节流显示）
            rospy.loginfo_throttle(
                2.0,
                "[offboard_teacher] connected=%s mode=%s armed=%s target=(%.2f, %.2f, %.2f)",
                self.current_state.connected,
                self.current_state.mode,
                self.current_state.armed,
                self.target_x,
                self.target_y,
                self.target_z,
            )
            rate.sleep()


def main():
    rospy.init_node("px4_offboard_teacher", anonymous=False)
    node = Px4OffboardTeacher()
    node.run()


if __name__ == "__main__":
    main()
