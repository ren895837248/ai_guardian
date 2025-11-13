"""
头部动作检测模块
检测点头和摇头动作
使用面部关键点（facial landmarks）跟踪头部运动
"""

import cv2
import dlib
import numpy as np
import logging
from collections import deque
from typing import Optional, Tuple, Literal

logger = logging.getLogger(__name__)


class HeadGestureDetector:
    """头部动作检测器，检测点头和摇头动作"""

    def __init__(self,
                 predictor_path: str = "shape_predictor_68_face_landmarks.dat",
                 history_size: int = 15,
                 nod_threshold: float = 15.0,
                 shake_threshold: float = 15.0):
        """
        初始化头部动作检测器

        Args:
            predictor_path: dlib面部关键点预测器模型路径
            history_size: 保存的历史位置数量
            nod_threshold: 点头检测的阈值（像素）
            shake_threshold: 摇头检测的阈值（像素）
        """
        self.predictor_path = predictor_path
        self.history_size = history_size
        self.nod_threshold = nod_threshold
        self.shake_threshold = shake_threshold

        # 初始化dlib检测器和预测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor: Optional[dlib.shape_predictor] = None

        # 加载面部关键点预测器
        self._load_predictor()

        # 存储鼻尖位置历史
        self.nose_positions_x = deque(maxlen=history_size)
        self.nose_positions_y = deque(maxlen=history_size)

        # 检测状态
        self.gesture_detected = False

    def _load_predictor(self) -> bool:
        """
        加载dlib面部关键点预测器

        Returns:
            bool: 是否成功加载
        """
        try:
            import os
            if not os.path.exists(self.predictor_path):
                logger.error(f"面部关键点模型文件不存在: {self.predictor_path}")
                logger.error("请下载模型文件: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                return False

            self.predictor = dlib.shape_predictor(self.predictor_path)
            logger.info("成功加载面部关键点预测器")
            return True
        except Exception as e:
            logger.error(f"加载面部关键点预测器时发生错误: {e}")
            return False

    def reset(self):
        """重置检测状态"""
        self.nose_positions_x.clear()
        self.nose_positions_y.clear()
        self.gesture_detected = False

    def detect_gesture(self, frame: np.ndarray, face_location: Optional[Tuple[int, int, int, int]] = None,
                      priority_face_name: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        检测头部动作（点头或摇头）

        Args:
            frame: 输入图像
            face_location: 优先人脸位置 (top, right, bottom, left)，如果提供则使用该位置
            priority_face_name: 优先跟踪的人脸名称，用于日志记录

        Returns:
            Tuple[bool, Optional[str]]: (是否检测到动作, 动作类型 'nod'/'shake'/None)
        """
        if self.predictor is None:
            logger.error("面部关键点预测器未加载")
            return False, None

        try:
            # 转换为灰度图以提高处理速度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 使用指定的人脸位置或自动检测
            if face_location is None:
                faces = self.detector(gray, 0)
                if len(faces) == 0:
                    # logger.debug("未检测到人脸")
                    return False, None

                # 如果有多个人脸但没有指定位置，选择最大的人脸
                if len(faces) > 1:
                    face = max(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()))
                    logger.debug(f"检测到{len(faces)}个人脸，选择最大的人脸进行头部动作检测")
                else:
                    face = faces[0]
            else:
                # 使用提供的优先人脸位置
                top, right, bottom, left = face_location
                face = dlib.rectangle(left, top, right, bottom)
                if priority_face_name:
                    logger.debug(f"使用优先人脸位置进行头部动作检测: {priority_face_name}")

            # 获取面部关键点
            landmarks = self.predictor(gray, face)

            # 获取鼻尖位置（关键点30）
            nose_tip = landmarks.part(30)
            nose_x = nose_tip.x
            nose_y = nose_tip.y

            # 记录位置
            self.nose_positions_x.append(nose_x)
            self.nose_positions_y.append(nose_y)

            # 需要足够的历史数据才能检测
            if len(self.nose_positions_x) < self.history_size:
                return False, None

            # 计算位置变化
            x_positions = list(self.nose_positions_x)
            y_positions = list(self.nose_positions_y)

            # 检测摇头（水平运动）
            x_range = max(x_positions) - min(x_positions)
            x_std = np.std(x_positions)

            # 检测点头（垂直运动）
            y_range = max(y_positions) - min(y_positions)
            y_std = np.std(y_positions)

            # 判断是否为摇头
            if x_range > self.shake_threshold and x_std > 5.0 and y_std < x_std * 0.6:
                gesture_info = f"摇头动作 (x_range: {x_range:.2f}, x_std: {x_std:.2f})"
                if priority_face_name:
                    logger.info(f"{priority_face_name} 检测到{gesture_info}")
                else:
                    logger.info(f"检测到{gesture_info}")
                self.reset()
                return True, 'shake'

            # 判断是否为点头
            if y_range > self.nod_threshold and y_std > 5.0 and x_std < y_std * 0.6:
                gesture_info = f"点头动作 (y_range: {y_range:.2f}, y_std: {y_std:.2f})"
                if priority_face_name:
                    logger.info(f"{priority_face_name} 检测到{gesture_info}")
                else:
                    logger.info(f"检测到{gesture_info}")
                self.reset()
                return True, 'nod'

            return False, None

        except Exception as e:
            logger.error(f"检测头部动作时发生错误: {e}")
            return False, None

    def detect_nod(self, frame: np.ndarray, face_location: Optional[Tuple[int, int, int, int]] = None,
                   priority_face_name: Optional[str] = None) -> bool:
        """
        专门检测点头动作

        Args:
            frame: 输入图像
            face_location: 人脸位置
            priority_face_name: 优先跟踪的人脸名称

        Returns:
            bool: 是否检测到点头
        """
        detected, gesture_type = self.detect_gesture(frame, face_location, priority_face_name)
        return detected and gesture_type == 'nod'

    def detect_shake(self, frame: np.ndarray, face_location: Optional[Tuple[int, int, int, int]] = None,
                     priority_face_name: Optional[str] = None) -> bool:
        """
        专门检测摇头动作

        Args:
            frame: 输入图像
            face_location: 人脸位置
            priority_face_name: 优先跟踪的人脸名称

        Returns:
            bool: 是否检测到摇头
        """
        detected, gesture_type = self.detect_gesture(frame, face_location, priority_face_name)
        return detected and gesture_type == 'shake'

    def draw_landmarks(self, frame: np.ndarray, face_location: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        在图像上绘制面部关键点（用于调试）

        Args:
            frame: 输入图像
            face_location: 人脸位置

        Returns:
            np.ndarray: 标注后的图像
        """
        if self.predictor is None:
            return frame

        output_frame = frame.copy()

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if face_location is None:
                faces = self.detector(gray, 0)
                if len(faces) == 0:
                    return output_frame
                face = faces[0]
            else:
                top, right, bottom, left = face_location
                face = dlib.rectangle(left, top, right, bottom)

            landmarks = self.predictor(gray, face)

            # 绘制鼻尖
            nose_tip = landmarks.part(30)
            cv2.circle(output_frame, (nose_tip.x, nose_tip.y), 3, (0, 0, 255), -1)

            # 绘制轨迹
            if len(self.nose_positions_x) > 1:
                points = [(int(x), int(y)) for x, y in zip(self.nose_positions_x, self.nose_positions_y)]
                for i in range(len(points) - 1):
                    cv2.line(output_frame, points[i], points[i + 1], (255, 0, 0), 2)

        except Exception as e:
            logger.error(f"绘制关键点时发生错误: {e}")

        return output_frame
