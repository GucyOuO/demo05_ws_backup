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
6. 如果能收到深度图和相机内参，就把物块轮廓里的深度点反投影成 3D 点云。
7. 根据 3D 点云估算长方体的长/宽/高，或圆柱体的半径/高度。

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

        # 只在图像的一部分区域里识别物块，这个区域叫 ROI（Region Of Interest）。
        # 你的相机如果拍到小车自身，通常会出现在画面底部，所以默认忽略最底下 18%。
        # 这些值是比例，不是像素：
        #   0.0 表示最左/最上，1.0 表示最右/最下。
        self.roi_x_min_ratio = rospy.get_param("~roi_x_min_ratio", 0.05)
        self.roi_x_max_ratio = rospy.get_param("~roi_x_max_ratio", 0.95)
        self.roi_y_min_ratio = rospy.get_param("~roi_y_min_ratio", 0.00)
        self.roi_y_max_ratio = rospy.get_param("~roi_y_max_ratio", 0.82)

        # 长宽比过滤。
        # 真正的方块在图像里一般不会特别扁；黄色车体边缘常常是一条很长的横条。
        # aspect_ratio = 宽 / 高。比如 300x30 的长条，长宽比就是 10，会被过滤。
        self.min_aspect_ratio = rospy.get_param("~min_aspect_ratio", 0.35)
        self.max_aspect_ratio = rospy.get_param("~max_aspect_ratio", 2.80)

        # 如果某个彩色区域占了画面大半，通常不是远处物块，而是拍到了小车自身或墙面。
        self.max_bbox_width_ratio = rospy.get_param("~max_bbox_width_ratio", 0.70)
        self.max_bbox_height_ratio = rospy.get_param("~max_bbox_height_ratio", 0.70)

        # 物块大小分类阈值，单位是像素面积。
        # 注意：这是图像里的大小，不是真实世界大小。
        # 物块离相机越近，像素面积越大；离相机越远，像素面积越小。
        self.small_area = rospy.get_param("~small_area", 2500)
        self.large_area = rospy.get_param("~large_area", 9000)

        # 是否启用“当前测试世界的颜色-形状提示”。
        # 在 box_house.world 里，我把几个物体设置成了固定颜色：
        #   red / green / blue      -> 立方体或长方体
        #   yellow / purple / cyan  -> 圆柱体
        #
        # 为什么还需要这个提示？
        #   单个相机从侧面看圆柱时，圆柱的外轮廓可能像一个矩形，
        #   仅靠 2D 外轮廓不可能 100% 区分圆柱和长方体。
        #   所以程序会优先使用深度弧面特征；如果深度不可用，
        #   再使用这个颜色提示作为当前仿真实验的备用判断。
        self.use_color_shape_hints = rospy.get_param("~use_color_shape_hints", True)

        # 保存最近一帧深度图。RGB 回调来得更频繁时，直接使用最近的深度图。
        self.latest_depth = None
        self.warned_bad_depth_encoding = False

        # 相机内参。
        # fx/fy 是焦距，单位是像素，用来把“像素宽高”换算成“米”。
        # cx/cy 是图像中心点，也叫主点，用来把像素坐标反投影到 3D 坐标。
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # 发布调试图像：把识别到的物块框出来，便于你在 rqt_image_view 里观察。
        self.debug_pub = rospy.Publisher("~debug_image", Image, queue_size=1)

        # 发布识别结果。为了初学阶段简单直观，这里用 JSON 字符串。
        # 后续如果要给其他节点长期使用，可以再改成自定义 msg。
        self.detection_pub = rospy.Publisher("~detections", String, queue_size=10)

        # 发布人类更容易阅读的文字摘要。
        # /detections 适合程序读取，/summary 适合你直接 rostopic echo 看。
        self.summary_pub = rospy.Publisher("~summary", String, queue_size=10)

        # 订阅 RGB 图、深度图和相机内参。
        self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)
        self.info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)

        # HSV 阈值表。
        #
        # 为什么不用 RGB/BGR 直接判断颜色？
        #   RGB/BGR 三个通道会同时受到光照明暗影响。比如同一个红色物块，
        #   在亮处和暗处的 R/G/B 数值可能差很多。
        #   HSV 把颜色大致拆成：
        #     H: Hue，色相，也就是“是什么颜色”
        #     S: Saturation，饱和度，也就是“颜色纯不纯”
        #     V: Value，亮度，也就是“亮不亮”
        #   所以用 HSV 更适合做初学阶段的颜色识别。
        #
        # OpenCV 里的 HSV 取值范围不是常见的 H=0~360：
        #   H: 0~179
        #   S: 0~255
        #   V: 0~255
        #
        # 每个颜色都是一个或多个 HSV 范围：
        #   ((H_min, S_min, V_min), (H_max, S_max, V_max))
        #
        # 红色比较特殊，因为红色在 H 轴两端，所以分成两段：
        #   一段是 H 接近 0 的红，另一段是 H 接近 179 的红。
        #
        # 调参经验：
        #   如果某个颜色识别不准，优先调对应颜色的 H 范围。
        #   如果灰白背景也被识别进来，可以适当调高 S_min。
        #   如果暗处识别不出来，可以适当调低 V_min。
        self.color_ranges = {
            "red": [
                ((0, 100, 80), (10, 255, 255)),
                ((160, 100, 80), (179, 255, 255)),
            ],
            # 橙色位于红色和黄色之间。常见橙色物体的 H 大约在 10~20。
            "orange": [
                ((10, 100, 80), (20, 255, 255)),
            ],
            # 黄色 H 通常在 20~35。Gazebo/Yellow 一般会落在这个范围。
            "yellow": [
                ((20, 100, 80), (35, 255, 255)),
            ],
            # 绿色范围比较宽，草绿、亮绿、深绿都可能落在这里。
            "green": [
                ((35, 80, 60), (85, 255, 255)),
            ],
            # 青色/蓝绿色位于绿色和蓝色之间，Gazebo/Turquoise 通常在这里。
            "cyan": [
                ((80, 80, 60), (95, 255, 255)),
            ],
            # 蓝色 H 通常在 90~130。
            "blue": [
                ((90, 80, 60), (130, 255, 255)),
            ],
            # 紫色位于蓝色和红色之间。
            "purple": [
                ((130, 80, 60), (160, 255, 255)),
            ],
            # 粉色/洋红色和红色、紫色比较接近，所以 H 范围略窄。
            "pink": [
                ((145, 60, 100), (170, 255, 255)),
            ],
            # 白色没有明显色相，特点是饱和度 S 很低、亮度 V 很高。
            # 注意：白墙、地面反光也可能被识别成 white，所以需要配合面积和形状过滤。
            "white": [
                ((0, 0, 200), (179, 45, 255)),
            ],
            # 黑色也没有明显色相，特点是亮度 V 很低。
            # 注意：阴影、黑色轮胎、雷达外壳也可能被识别成 black。
            "black": [
                ((0, 0, 0), (179, 255, 50)),
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
        self.cx = msg.K[2]
        self.cy = msg.K[5]

    def depth_callback(self, msg):
        """保存最近一帧深度图。

        Gazebo/OpenNI 深度图常见编码：
        - 32FC1：每个像素是 float，单位通常是米。
        - 16UC1：每个像素是 uint16，单位通常是毫米。

        如果这里收到 rgb8/bgr8，说明这个话题其实是彩色图，不是真正的深度图。
        这种情况下不能拿它估算距离，否则会把颜色值误当成深度值。
        """
        if msg.encoding not in ("32FC1", "16UC1", "mono16"):
            self.latest_depth = None
            if not self.warned_bad_depth_encoding:
                rospy.logwarn(
                    "Depth topic %s encoding is %s, not a single-channel depth image. "
                    "Color detection will continue, but depth/real size will be None.",
                    self.depth_topic,
                    msg.encoding,
                )
                self.warned_bad_depth_encoding = True
            return

        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            rospy.logwarn("Failed to convert depth image: %s", exc)
            return

        # 统一转换成 float32，方便后面计算。
        depth_image = np.asarray(depth_image, dtype=np.float32)
        if len(depth_image.shape) != 2:
            self.latest_depth = None
            if not self.warned_bad_depth_encoding:
                rospy.logwarn(
                    "Depth topic %s is not single-channel. Shape is %s. "
                    "Color detection will continue, but depth/real size will be None.",
                    self.depth_topic,
                    depth_image.shape,
                )
                self.warned_bad_depth_encoding = True
            return

        # 如果深度值很大，通常是 16UC1 的毫米单位，转成米。
        # 这个判断比较保守：正常室内深度很少大于 100 米。
        finite_depth = depth_image[np.isfinite(depth_image)]
        if finite_depth.size > 0 and np.nanmax(finite_depth) > 100.0:
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
        roi = self.get_roi(debug_image.shape[1], debug_image.shape[0])
        self.draw_roi(debug_image, roi)

        # 分别检测每一种颜色。
        for color_name, ranges in self.color_ranges.items():
            mask = self.build_color_mask(hsv_image, ranges)
            mask = self.apply_roi_mask(mask, roi)

            # 找轮廓。轮廓可以理解为 mask 里一块连通的白色区域边界。
            contours = self.find_contours(mask)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                if not self.is_reasonable_block_shape(w, h, debug_image.shape[1], debug_image.shape[0]):
                    continue

                center_x = x + w / 2.0
                center_y = y + h / 2.0

                shape_info = self.estimate_shape_and_dimensions(contour, color_name)
                depth_m = shape_info.get("depth_m")
                if depth_m is None:
                    depth_m = self.estimate_depth(center_x, center_y)
                    shape_info["depth_m"] = depth_m
                real_width_m, real_height_m = self.estimate_real_size(w, h, depth_m)

                detection = {
                    "id": len(detections) + 1,
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
                detection.update(shape_info)
                detections.append(detection)

                self.draw_detection(debug_image, detection)

        # 发布 JSON 结果。ensure_ascii=False 方便以后写中文字段时不被转义。
        self.detection_pub.publish(String(data=json.dumps(detections, ensure_ascii=False)))
        self.summary_pub.publish(String(data=self.format_summary(detections)))

        # 在右侧增加一块信息面板，让 rqt_image_view 里能清楚看到每个物体的信息。
        debug_image = self.draw_info_panel(debug_image, detections)

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

    def get_roi(self, image_width, image_height):
        """把 ROI 比例转换成像素坐标。

        返回值是 (x1, y1, x2, y2)，表示只识别这个矩形区域里面的物体。
        """
        x1 = int(image_width * self.roi_x_min_ratio)
        x2 = int(image_width * self.roi_x_max_ratio)
        y1 = int(image_height * self.roi_y_min_ratio)
        y2 = int(image_height * self.roi_y_max_ratio)

        # 做边界保护，防止参数写错导致坐标超出图像。
        x1 = max(0, min(image_width - 1, x1))
        x2 = max(x1 + 1, min(image_width, x2))
        y1 = max(0, min(image_height - 1, y1))
        y2 = max(y1 + 1, min(image_height, y2))
        return x1, y1, x2, y2

    def apply_roi_mask(self, mask, roi):
        """把 ROI 外面的区域全部涂黑，让算法只看 ROI 里面。"""
        x1, y1, x2, y2 = roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return roi_mask

    def is_reasonable_block_shape(self, bbox_w, bbox_h, image_width, image_height):
        """过滤明显不像物块的轮廓。

        主要过滤两类误识别：
        1. 很长很扁的横条或竖条，例如小车底盘边缘。
        2. 占画面比例过大的色块，例如相机拍到自身模型或近处墙面。
        """
        if bbox_h <= 0:
            return False

        aspect_ratio = float(bbox_w) / float(bbox_h)
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False

        if float(bbox_w) / float(image_width) > self.max_bbox_width_ratio:
            return False

        if float(bbox_h) / float(image_height) > self.max_bbox_height_ratio:
            return False

        return True

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

    def get_contour_features(self, contour):
        """计算 2D 轮廓特征。

        这些特征只来自彩色图像，不需要深度图：
        - circularity：圆度。圆越标准，值越接近 1。
        - rectangularity：矩形度。实心矩形越标准，值越接近 1。
        - vertex_count：简化轮廓后的顶点数量，正面长方体常接近 4。
        - aspect_ratio：外接矩形宽高比。

        这些指标只能作为辅助，因为圆柱从侧面看也可能像矩形。
        真正区分圆柱/长方体，后面还会结合深度图。
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area <= 0.0 or perimeter <= 0.0:
            return {
                "area": float(area),
                "perimeter": float(perimeter),
                "circularity": 0.0,
                "rectangularity": 0.0,
                "vertex_count": 0,
                "aspect_ratio": 0.0,
            }

        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = float(w * h)
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        vertex_count = len(approx)
        rectangularity = float(area) / bbox_area if bbox_area > 0.0 else 0.0

        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "circularity": float(circularity),
            "rectangularity": float(rectangularity),
            "vertex_count": int(vertex_count),
            "aspect_ratio": float(aspect_ratio),
        }

    def classify_shape_from_contour(self, contour):
        """根据 2D 轮廓粗略判断形状。

        这里只给出第一印象，最终结果会被深度特征进一步修正：
        - 四边形 + 高矩形度：更像 cuboid。
        - 高圆度 + 低矩形度：更像 cylinder。
        """
        features = self.get_contour_features(contour)
        circularity = features["circularity"]
        rectangularity = features["rectangularity"]
        vertex_count = features["vertex_count"]
        aspect_ratio = features["aspect_ratio"]

        if 4 <= vertex_count <= 6 and rectangularity > 0.80:
            return "cuboid", float(min(1.0, rectangularity))

        if circularity > 0.82 and rectangularity < 0.86 and 0.65 <= aspect_ratio <= 1.35:
            return "cylinder", float(circularity)

        return "unknown", float(circularity)

    def shape_hint_from_color(self, color_name):
        """根据当前 box_house.world 的颜色设置给出备用形状提示。

        这个提示只在深度特征不够可靠时使用。
        如果以后你把颜色和形状重新组合，比如黄色长方体，
        可以把 use_color_shape_hints 参数改成 false，或者修改这里的映射。
        """
        if not self.use_color_shape_hints:
            return None

        if color_name in ("red", "green", "blue"):
            return "cuboid"

        if color_name in ("yellow", "purple", "cyan"):
            return "cylinder"

        return None

    def estimate_depth_curvature(self, points_x, points_z):
        """估算物体表面在左右方向上的深度弯曲程度。

        圆柱体的正面/侧面通常是弧面：
        - 中间离相机更近，Z 较小。
        - 左右边缘离相机更远，Z 较大。

        长方体正面通常是平面：
        - 左右方向的深度变化很小。

        返回值单位是米：
        - 越接近 0，越像平面。
        - 越大，越像圆柱弧面。
        """
        if points_x.size < 30:
            return None

        x_min = np.percentile(points_x, 5)
        x_max = np.percentile(points_x, 95)
        if x_max <= x_min:
            return None

        # 把物体左右方向分成 9 段，每段取深度中位数。
        # 用中位数可以减少深度噪声点的影响。
        bins = np.linspace(x_min, x_max, 10)
        medians = []
        centers = []
        for idx in range(len(bins) - 1):
            left = bins[idx]
            right = bins[idx + 1]
            mask = (points_x >= left) & (points_x < right)
            if np.count_nonzero(mask) < 5:
                continue
            centers.append((left + right) / 2.0)
            medians.append(float(np.median(points_z[mask])))

        if len(medians) < 5:
            return None

        medians = np.array(medians, dtype=np.float32)

        # 两侧深度均值 - 中间深度。
        # 圆柱弧面一般是正数：边缘更远，中间更近。
        edge_depth = (float(np.median(medians[:2])) + float(np.median(medians[-2:]))) / 2.0
        center_depth = float(np.median(medians[len(medians) // 2 - 1:len(medians) // 2 + 2]))
        return max(0.0, edge_depth - center_depth)

    def choose_shape(self, color_name, contour_shape, contour_score, contour_features,
                     object_width_m, object_depth_m, depth_curvature_m):
        """综合颜色提示、2D 轮廓、深度弧面来选择最终形状。

        判断优先级：
        1. 深度弧面明显：优先认为是 cylinder。
        2. 深度近似平面 + 矩形度高：优先认为是 cuboid。
        3. 当前测试世界的颜色提示。
        4. 退回 2D 轮廓判断。
        """
        hint = self.shape_hint_from_color(color_name)
        rectangularity = contour_features["rectangularity"]
        circularity = contour_features["circularity"]
        vertex_count = contour_features["vertex_count"]

        depth_width_ratio = None
        curvature_ratio = None
        if object_width_m and object_width_m > 0.001:
            depth_width_ratio = object_depth_m / object_width_m
            if depth_curvature_m is not None:
                curvature_ratio = depth_curvature_m / object_width_m

        # 圆柱弧面：左右边缘比中心明显更远。
        if curvature_ratio is not None and curvature_ratio > 0.035 and depth_width_ratio > 0.06:
            return "cylinder", min(0.99, 0.70 + curvature_ratio * 3.0), "depth_curvature"

        # 长方体平面：轮廓像矩形，深度变化比较小。
        if 4 <= vertex_count <= 6 and rectangularity > 0.80:
            if depth_width_ratio is None or depth_width_ratio < 0.10:
                return "cuboid", min(0.98, rectangularity), "flat_depth_rectangle"

        # 当前仿真世界颜色固定，作为深度不明显时的备用。
        if hint is not None:
            return hint, 0.90, "color_hint"

        # 圆形投影明显时，认为是圆柱。
        if circularity > 0.82 and rectangularity < 0.86:
            return "cylinder", circularity, "contour_circle"

        return contour_shape, contour_score, "contour"

    def estimate_shape_and_dimensions(self, contour, color_name):
        """估算物体形状和三维尺寸。

        这是本节点里最关键的“真实尺寸”步骤。

        基本思路：
        1. 先用颜色分割得到某个物块的轮廓 contour。
        2. 在这个轮廓里面找对应的深度像素。
        3. 用相机内参把每个深度像素反投影成 3D 点：
             X = (u - cx) * Z / fx
             Y = (v - cy) * Z / fy
             Z = depth
           其中：
             u/v 是图像里的像素坐标。
             X 是相机坐标系左右方向，单位米。
             Y 是相机坐标系上下方向，单位米。
             Z 是相机前方深度方向，单位米。
        4. 对这些 3D 点分别求 X/Y/Z 的范围，就得到可见点云的宽、高、深。

        重要限制：
        - RGB-D 相机只能看到朝向相机的一面或几面，看不到被遮挡的背面。
        - 长方体如果正面对着相机，真实厚度可能看不到，depth 方向会偏小。
        - 圆柱体半径这里主要用图像左右宽度估算，适合正面看到圆柱外轮廓时使用。
        """
        contour_features = self.get_contour_features(contour)
        shape, shape_score = self.classify_shape_from_contour(contour)
        shape_method = "contour"
        hint_shape = self.shape_hint_from_color(color_name)
        if self.latest_depth is None and hint_shape is not None:
            shape = hint_shape
            shape_score = 0.90
            shape_method = "color_hint"

        empty_result = {
            "shape": shape,
            "shape_score": shape_score,
            "shape_method": shape_method,
            "shape_confidence": shape_score,
            "contour_circularity": contour_features["circularity"],
            "contour_rectangularity": contour_features["rectangularity"],
            "contour_vertex_count": contour_features["vertex_count"],
            "depth_curvature_m": None,
            "depth_width_ratio": None,
            "depth_m": None,
            "object_width_m": None,
            "object_height_m": None,
            "object_depth_m": None,
            "cuboid_length_m": None,
            "cuboid_width_m": None,
            "cuboid_height_m": None,
            "cylinder_radius_m": None,
            "cylinder_height_m": None,
            "dimension_note": "no_valid_depth",
        }

        if self.latest_depth is None:
            return empty_result

        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            empty_result["dimension_note"] = "no_camera_info"
            return empty_result

        depth = self.latest_depth
        depth_h, depth_w = depth.shape[:2]

        # 如果 RGB 图和深度图分辨率不同，轮廓像素坐标就不能直接拿来索引深度图。
        # 初学阶段先直接返回 None，避免算出看似有值但其实错位的尺寸。
        x, y, w, h = cv2.boundingRect(contour)
        if x + w > depth_w or y + h > depth_h:
            empty_result["dimension_note"] = "rgb_depth_size_mismatch"
            return empty_result

        # 为这个轮廓单独创建一张黑白 mask，只保留轮廓内部区域。
        # mask[y, x] = 255 表示这个像素属于当前物块。
        contour_mask = np.zeros((depth_h, depth_w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

        # 只取物块轮廓内部的深度点，避免背景点参与尺寸计算。
        ys, xs = np.where(contour_mask > 0)
        if xs.size == 0:
            return empty_result

        zs = depth[ys, xs]

        # 过滤掉无效深度：
        #   nan / inf 表示没有有效测量。
        #   <=0 表示无效或距离为 0。
        valid = np.isfinite(zs) & (zs > 0.0)
        xs = xs[valid]
        ys = ys[valid]
        zs = zs[valid]

        if zs.size < 20:
            return empty_result

        median_depth_m = float(np.median(zs))

        # 深度图边缘可能有少量飞点。这里用 5%~95% 分位数裁掉极端值，
        # 比直接用 min/max 更稳。
        z_low = np.percentile(zs, 5)
        z_high = np.percentile(zs, 95)
        inlier = (zs >= z_low) & (zs <= z_high)
        xs = xs[inlier]
        ys = ys[inlier]
        zs = zs[inlier]

        if zs.size < 20:
            return empty_result

        # 像素 + 深度 -> 相机坐标系 3D 点。
        # X: 左右方向；Y: 上下方向；Z: 前后方向。
        points_x = (xs.astype(np.float32) - float(self.cx)) * zs / float(self.fx)
        points_y = (ys.astype(np.float32) - float(self.cy)) * zs / float(self.fy)
        points_z = zs

        object_width_m = float(np.percentile(points_x, 95) - np.percentile(points_x, 5))
        object_height_m = float(np.percentile(points_y, 95) - np.percentile(points_y, 5))
        object_depth_m = float(np.percentile(points_z, 95) - np.percentile(points_z, 5))

        # 统一取正数，避免坐标方向导致高度为负。
        object_width_m = abs(object_width_m)
        object_height_m = abs(object_height_m)
        object_depth_m = abs(object_depth_m)

        depth_curvature_m = self.estimate_depth_curvature(points_x, points_z)
        depth_width_ratio = None
        if object_width_m > 0.001:
            depth_width_ratio = object_depth_m / object_width_m

        shape, shape_score, shape_method = self.choose_shape(
            color_name,
            shape,
            shape_score,
            contour_features,
            object_width_m,
            object_depth_m,
            depth_curvature_m,
        )

        result = {
            "shape": shape,
            "shape_score": shape_score,
            "shape_method": shape_method,
            "shape_confidence": shape_score,
            "contour_circularity": contour_features["circularity"],
            "contour_rectangularity": contour_features["rectangularity"],
            "contour_vertex_count": contour_features["vertex_count"],
            "depth_curvature_m": depth_curvature_m,
            "depth_width_ratio": depth_width_ratio,
            # 使用整个物体轮廓内部的深度中位数。
            # 这比只取中心 5x5 更稳定，尤其是物体较远、深度图较稀疏时。
            "depth_m": median_depth_m,
            # 通用尺寸：
            #   width  是相机画面左右方向的可见宽度。
            #   height 是相机画面上下方向的可见高度。
            #   depth  是相机前后方向的可见深度范围。
            "object_width_m": object_width_m,
            "object_height_m": object_height_m,
            "object_depth_m": object_depth_m,
            # 长方体尺寸：
            #   length 对应左右方向可见长度。
            #   width  对应前后方向可见厚度。
            #   height 对应上下方向高度。
            "cuboid_length_m": None,
            "cuboid_width_m": None,
            "cuboid_height_m": None,
            # 圆柱体尺寸：
            #   radius 使用左右可见直径的一半估计。
            #   height 使用上下可见高度估计。
            "cylinder_radius_m": None,
            "cylinder_height_m": None,
            "dimension_note": "visible_depth_points_only",
        }

        if shape == "cuboid":
            result["cuboid_length_m"] = object_width_m
            result["cuboid_width_m"] = object_depth_m
            result["cuboid_height_m"] = object_height_m
        elif shape == "cylinder":
            result["cylinder_radius_m"] = object_width_m / 2.0
            result["cylinder_height_m"] = object_height_m

        return result

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
            "orange": (0, 140, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "purple": (255, 0, 255),
            "pink": (203, 192, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }.get(detection["color"], (255, 255, 255))
        text_bgr = color_bgr
        if detection["color"] in ("black", "blue", "purple"):
            # 深色文字在画面上不容易看清，所以深色物体使用白色文字。
            text_bgr = (255, 255, 255)

        x = int(detection["bbox_x"])
        y = int(detection["bbox_y"])
        w = int(detection["bbox_w"])
        h = int(detection["bbox_h"])

        cv2.rectangle(image, (x, y), (x + w, y + h), color_bgr, 2)

        label = "#%d %s %s" % (
            detection["id"],
            detection["color"],
            detection.get("shape", "unknown"),
        )

        if detection["depth_m"] is not None:
            label += " z=%.2fm" % detection["depth_m"]

        cv2.putText(
            image,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_bgr,
            2,
            cv2.LINE_AA,
        )

    def draw_roi(self, image, roi):
        """在调试图像上画出当前算法实际会看的区域。"""
        x1, y1, x2, y2 = roi
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    def fmt_m(self, value):
        """把米制数字格式化成短字符串。None 显示为 N/A。"""
        if value is None:
            return "N/A"
        return "%.2fm" % value

    def format_summary(self, detections):
        """生成适合 rostopic echo /block_color_detector/summary 的文字摘要。"""
        if not detections:
            return "No colored objects detected."

        lines = ["Detected %d object(s):" % len(detections)]
        for det in detections:
            base = "#%d color=%s shape=%s z=%s method=%s confidence=%.2f" % (
                det["id"],
                det["color"],
                det.get("shape", "unknown"),
                self.fmt_m(det.get("depth_m")),
                det.get("shape_method", "unknown"),
                float(det.get("shape_confidence", 0.0)),
            )
            lines.append(base)

            if det.get("shape") == "cuboid":
                lines.append(
                    "  cuboid: length=%s width=%s height=%s" % (
                        self.fmt_m(det.get("cuboid_length_m")),
                        self.fmt_m(det.get("cuboid_width_m")),
                        self.fmt_m(det.get("cuboid_height_m")),
                    )
                )
            elif det.get("shape") == "cylinder":
                lines.append(
                    "  cylinder: radius=%s height=%s" % (
                        self.fmt_m(det.get("cylinder_radius_m")),
                        self.fmt_m(det.get("cylinder_height_m")),
                    )
                )
            else:
                lines.append(
                    "  visible: width=%s height=%s depth=%s" % (
                        self.fmt_m(det.get("object_width_m")),
                        self.fmt_m(det.get("object_height_m")),
                        self.fmt_m(det.get("object_depth_m")),
                    )
                )

        return "\n".join(lines)

    def put_panel_line(self, panel, text, y, color=(230, 230, 230), scale=0.45, thickness=1):
        """在信息面板上写一行英文文本。

        OpenCV 默认字体不支持中文，所以 rqt 图像面板里使用英文缩写；
        代码注释和 ROS 话题字段保持详细说明。
        """
        cv2.putText(
            panel,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def draw_info_panel(self, image, detections):
        """在调试图右侧增加结构化信息面板。

        rqt_image_view 只能显示图像本身，看 JSON 结果不方便。
        所以这里把关键信息画到图像右边：
        - 编号、颜色、形状、距离 z。
        - 长方体的长宽高。
        - 圆柱体的半径和高度。
        - shape_method 表示形状判断依据。
        """
        height = image.shape[0]
        panel_width = 430
        panel = np.full((height, panel_width, 3), 34, dtype=np.uint8)

        self.put_panel_line(panel, "Block detector", 26, color=(255, 255, 255), scale=0.65, thickness=2)
        self.put_panel_line(panel, "count: %d" % len(detections), 52, color=(200, 220, 255), scale=0.50)

        if not detections:
            self.put_panel_line(panel, "No colored objects detected", 90, color=(180, 180, 180), scale=0.50)
            return np.hstack((image, panel))

        y = 86
        for det in detections[:6]:
            shape = det.get("shape", "unknown")
            header = "#%d  %s  %s  z=%s" % (
                det["id"],
                det["color"],
                shape,
                self.fmt_m(det.get("depth_m")),
            )
            self.put_panel_line(panel, header, y, color=(255, 255, 255), scale=0.50, thickness=2)
            y += 21

            if shape == "cuboid":
                dims = "L=%s  W=%s  H=%s" % (
                    self.fmt_m(det.get("cuboid_length_m")),
                    self.fmt_m(det.get("cuboid_width_m")),
                    self.fmt_m(det.get("cuboid_height_m")),
                )
            elif shape == "cylinder":
                dims = "R=%s  H=%s" % (
                    self.fmt_m(det.get("cylinder_radius_m")),
                    self.fmt_m(det.get("cylinder_height_m")),
                )
            else:
                dims = "visible W=%s H=%s D=%s" % (
                    self.fmt_m(det.get("object_width_m")),
                    self.fmt_m(det.get("object_height_m")),
                    self.fmt_m(det.get("object_depth_m")),
                )
            self.put_panel_line(panel, dims, y, color=(210, 245, 210), scale=0.45)
            y += 19

            method = "method=%s  conf=%.2f" % (
                det.get("shape_method", "unknown"),
                float(det.get("shape_confidence", 0.0)),
            )
            self.put_panel_line(panel, method, y, color=(190, 190, 190), scale=0.42)
            y += 26

            if y > height - 30:
                self.put_panel_line(panel, "... more objects in /detections", y, color=(180, 180, 180), scale=0.42)
                break

        return np.hstack((image, panel))


def main():
    rospy.init_node("block_color_detector")
    BlockColorDetector()
    rospy.spin()


if __name__ == "__main__":
    main()
