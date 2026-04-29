#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
物块颜色和大小识别节点

这个节点适合初学阶段使用，核心思路是：
1. 从 RGB-D 相机读取彩色图像。
2. 把图像从 BGR 转成 HSV 色彩空间。
3. 用 HSV 阈值把红、绿、蓝、黄等颜色区域分割出来。
4. 对分割结果做简单去噪。
5. 查找轮廓，得到物块的位置和像素面积。
6. 如果能收到深度图和相机内参，就估算物块的真实宽高。

常用查看方式：
  rosrun nav_demo block_color_detector.py
  rqt_image_view /block_detector/debug_image
  rostopic echo /block_detector/detections
"""

from __future__ import print_function

import json

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


class BlockColorDetector(object):
    """识别彩色物块的 ROS 节点。"""

    def __init__(self):
        # CvBridge 用来在 ROS 的 sensor_msgs/Image 和 OpenCV 图像之间转换。
        self.bridge = CvBridge()

        # 从参数服务器读取话题名。
        # 你的模型里已经有 RGB-D 相机，默认使用 Kinect 的 RGB 和 Depth 话题。
        # 如果你想用普通 RGB 相机，可以启动时改成：
        #   roslaunch nav_demo block_color_detector.launch rgb_topic:=/camera/image_raw
        self.rgb_topic = rospy.get_param("~rgb_topic", "/kinect/rgb/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/kinect/depth/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/kinect/rgb/camera_info")

        # 面积太小的轮廓大多是噪声，不作为物块。
        self.min_area = rospy.get_param("~min_area", 600)

        # 物块大小分类阈值，单位是像素面积。
        # 注意：这是图像里的大小，不是真实世界大小。
        # 物块离相机越近，像素面积越大；离相机越远，像素面积越小。
        self.small_area = rospy.get_param("~small_area", 2500)
        self.large_area = rospy.get_param("~large_area", 9000)

        # 保存最近一帧深度图。RGB 回调来得更频繁时，直接使用最近的深度图。
        self.latest_depth = None

        # 相机内参。fx/fy 是焦距，单位是像素，用来把“像素宽高”换算成“米”。
        self.fx = None
        self.fy = None

        # 发布调试图像：把识别到的物块框出来，便于你在 rqt_image_view 里观察。
        self.debug_pub = rospy.Publisher("~debug_image", Image, queue_size=1)

        # 发布识别结果。为了初学阶段简单直观，这里用 JSON 字符串。
        # 后续如果要给其他节点长期使用，可以再改成自定义 msg。
        self.detection_pub = rospy.Publisher("~detections", String, queue_size=10)

        # 订阅 RGB 图、深度图和相机内参。
        self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)
        self.info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)

        # HSV 阈值表。
        # OpenCV 里的 HSV 取值范围不是常见的 0~360：
        #   H: 0~179
        #   S: 0~255
        #   V: 0~255
        #
        # 红色比较特殊，因为红色在 H 轴两端，所以分成 red1 和 red2 两段。
        # 如果你的物块颜色识别不准，优先调这里的 lower/upper。
        self.color_ranges = {
            "red": [
                ((0, 100, 80), (10, 255, 255)),
                ((160, 100, 80), (179, 255, 255)),
            ],
            "green": [
                ((35, 80, 60), (85, 255, 255)),
            ],
            "blue": [
                ((90, 80, 60), (130, 255, 255)),
            ],
            "yellow": [
                ((20, 100, 80), (35, 255, 255)),
            ],
        }

        rospy.loginfo("block_color_detector started")
        rospy.loginfo("RGB topic: %s", self.rgb_topic)
        rospy.loginfo("Depth topic: %s", self.depth_topic)
        rospy.loginfo("CameraInfo topic: %s", self.camera_info_topic)

    def camera_info_callback(self, msg):
        """读取相机内参。

        CameraInfo.K 是一个 3x3 相机内参矩阵，按行展开：
          [fx, 0,  cx,
           0,  fy, cy,
           0,  0,  1]

        fx/fy 越大，说明同样远处的物体在图像里成像越大。
        """
        self.fx = msg.K[0]
        self.fy = msg.K[4]

    def depth_callback(self, msg):
        """保存最近一帧深度图。

        Gazebo/OpenNI 深度图常见编码：
        - 32FC1：每个像素是 float，单位通常是米。
        - 16UC1：每个像素是 uint16，单位通常是毫米。
        """
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            rospy.logwarn("Failed to convert depth image: %s", exc)
            return

        # 统一转换成 float32，方便后面计算。
        depth_image = np.asarray(depth_image, dtype=np.float32)

        # 如果深度值很大，通常是 16UC1 的毫米单位，转成米。
        # 这个判断比较保守：正常室内深度很少大于 100 米。
        if np.nanmax(depth_image) > 100.0:
            depth_image = depth_image / 1000.0

        self.latest_depth = depth_image

    def rgb_callback(self, msg):
        """处理每一帧 RGB 图像。"""
        try:
            # ROS 里的 RGB 图像转成 OpenCV 常用的 BGR 格式。
            bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("Failed to convert RGB image: %s", exc)
            return

        # OpenCV 做颜色阈值时，HSV 比 BGR/RGB 更稳定。
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        detections = []
        debug_image = bgr_image.copy()

        # 分别检测每一种颜色。
        for color_name, ranges in self.color_ranges.items():
            mask = self.build_color_mask(hsv_image, ranges)

            # 找轮廓。轮廓可以理解为 mask 里一块连通的白色区域边界。
            contours = self.find_contours(mask)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w / 2.0
                center_y = y + h / 2.0

                depth_m = self.estimate_depth(center_x, center_y)
                real_width_m, real_height_m = self.estimate_real_size(w, h, depth_m)

                detection = {
                    "color": color_name,
                    "center_x": center_x,
                    "center_y": center_y,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "pixel_area": area,
                    "size_level": self.classify_size(area),
                    "depth_m": depth_m,
                    "real_width_m": real_width_m,
                    "real_height_m": real_height_m,
                }
                detections.append(detection)

                self.draw_detection(debug_image, detection)

        # 发布 JSON 结果。ensure_ascii=False 方便以后写中文字段时不被转义。
        self.detection_pub.publish(String(data=json.dumps(detections, ensure_ascii=False)))

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as exc:
            rospy.logwarn("Failed to publish debug image: %s", exc)

    def build_color_mask(self, hsv_image, ranges):
        """根据一组 HSV 范围生成二值 mask。

        mask 是一张黑白图：
        - 白色 255：表示这个像素属于目标颜色。
        - 黑色 0：表示这个像素不是目标颜色。
        """
        full_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv_image, lower_np, upper_np)
            full_mask = cv2.bitwise_or(full_mask, mask)

        # 形态学开运算：先腐蚀再膨胀，去掉小白点噪声。
        kernel = np.ones((5, 5), dtype=np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

        # 形态学闭运算：先膨胀再腐蚀，填补物块区域里的小黑洞。
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

        return full_mask

    def find_contours(self, mask):
        """兼容不同 OpenCV 版本的 findContours 返回值。"""
        result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(result) == 2:
            contours, _ = result
        else:
            _, contours, _ = result
        return contours

    def classify_size(self, area):
        """根据像素面积把物块粗略分为 small / medium / large。"""
        if area < self.small_area:
            return "small"
        if area > self.large_area:
            return "large"
        return "medium"

    def estimate_depth(self, center_x, center_y):
        """估算物块中心附近的深度，单位：米。

        只取中心一个像素容易受到噪声影响，所以这里取中心附近 5x5 小窗口，
        再用中位数作为深度值。中位数比平均值更不容易被异常点影响。
        """
        if self.latest_depth is None:
            return None

        depth = self.latest_depth
        height, width = depth.shape[:2]

        cx = int(round(center_x))
        cy = int(round(center_y))

        if cx < 0 or cx >= width or cy < 0 or cy >= height:
            return None

        x1 = max(0, cx - 2)
        x2 = min(width, cx + 3)
        y1 = max(0, cy - 2)
        y2 = min(height, cy + 3)

        window = depth[y1:y2, x1:x2]

        # 深度图里 0、inf、nan 都不是可靠深度，需要过滤掉。
        valid = window[np.isfinite(window)]
        valid = valid[valid > 0.0]

        if valid.size == 0:
            return None

        return float(np.median(valid))

    def estimate_real_size(self, pixel_w, pixel_h, depth_m):
        """用针孔相机模型估算真实宽高。

        简化公式：
          real_width  = pixel_width  * depth / fx
          real_height = pixel_height * depth / fy

        这个估计有几个前提：
        - 物块表面大致正对相机。
        - 深度值比较准确。
        - 相机内参 fx/fy 已经从 CameraInfo 收到。
        """
        if depth_m is None or self.fx is None or self.fy is None:
            return None, None

        real_width_m = float(pixel_w) * depth_m / float(self.fx)
        real_height_m = float(pixel_h) * depth_m / float(self.fy)
        return real_width_m, real_height_m

    def draw_detection(self, image, detection):
        """在调试图像上画识别框和文字。"""
        color_bgr = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
        }.get(detection["color"], (255, 255, 255))

        x = int(detection["bbox_x"])
        y = int(detection["bbox_y"])
        w = int(detection["bbox_w"])
        h = int(detection["bbox_h"])

        cv2.rectangle(image, (x, y), (x + w, y + h), color_bgr, 2)

        label = "%s %s area=%d" % (
            detection["color"],
            detection["size_level"],
            int(detection["pixel_area"]),
        )

        if detection["depth_m"] is not None:
            label += " z=%.2fm" % detection["depth_m"]

        cv2.putText(
            image,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_bgr,
            2,
            cv2.LINE_AA,
        )


def main():
    rospy.init_node("block_color_detector")
    BlockColorDetector()
    rospy.spin()


if __name__ == "__main__":
    main()
