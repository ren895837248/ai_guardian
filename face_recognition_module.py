"""
人脸识别模块
使用face_recognition库进行人脸检测和识别
优化系统负载：使用较低的图像分辨率和跳帧处理
"""

import face_recognition
import cv2
import numpy as np
import pickle
import os
import logging
import time
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """人脸识别类，支持人脸注册和识别，优化系统性能，支持多人脸智能选择"""

    def __init__(self,
                 person_folder: str = "person",
                 scale_factor: float = 0.5,
                 recognition_threshold: float = 0.6):
        """
        初始化人脸识别器

        Args:
            person_folder: 存放已注册人脸图片的文件夹路径
            scale_factor: 图像缩放因子，降低分辨率以提升性能（0-1之间）
            recognition_threshold: 人脸识别阈值，越小越严格
        """
        self.person_folder = person_folder
        self.scale_factor = scale_factor
        self.recognition_threshold = recognition_threshold

        # 存储已知人脸的编码和名字
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []

        # 多人脸跟踪状态
        self.tracked_faces: Dict[str, Dict] = {}  # 存储每个人脸的跟踪信息
        self.current_priority_face: Optional[str] = None  # 当前优先跟踪的人脸

        # 加载已注册的人脸数据
        self.load_known_faces_from_folder()

    def load_known_faces_from_folder(self) -> bool:
        """
        从person文件夹加载所有jpg图片作为已注册人脸

        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(self.person_folder):
            logger.warning(f"person文件夹不存在: {self.person_folder}")
            logger.info(f"请创建 {self.person_folder} 文件夹并放入人脸图片（jpg格式）")
            return False

        try:
            # 获取所有jpg图片
            image_files = [f for f in os.listdir(self.person_folder)
                          if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]

            if len(image_files) == 0:
                logger.warning(f"{self.person_folder} 文件夹中没有jpg图片")
                return False

            logger.info(f"开始加载 {len(image_files)} 个人脸图片...")

            for image_file in image_files:
                image_path = os.path.join(self.person_folder, image_file)
                # 使用文件名（去掉扩展名）作为人名
                person_name = os.path.splitext(image_file)[0]

                try:
                    # 加载图片
                    image = face_recognition.load_image_file(image_path)

                    # 检测人脸并生成编码
                    face_encodings = face_recognition.face_encodings(image)

                    if len(face_encodings) == 0:
                        logger.warning(f"未在 {image_file} 中检测到人脸，跳过")
                        continue

                    if len(face_encodings) > 1:
                        logger.warning(f"{image_file} 中检测到多个人脸，使用第一个")

                    # 保存第一个人脸编码
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(person_name)
                    logger.info(f"成功加载: {person_name} ({image_file})")

                except Exception as e:
                    logger.error(f"加载 {image_file} 时发生错误: {e}")
                    continue

            logger.info(f"总共成功加载 {len(self.known_face_names)} 个已注册人脸")
            return len(self.known_face_names) > 0

        except Exception as e:
            logger.error(f"加载人脸数据时发生错误: {e}")
            return False


    def recognize_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        识别图像中的人脸

        Args:
            frame: 输入图像

        Returns:
            List[Dict]: 识别结果列表，每个字典包含 'name', 'location', 'confidence'
        """
        if len(self.known_face_encodings) == 0:
            logger.warning("没有已注册的人脸数据")
            return []

        try:
            # 缩小图像以提高处理速度
            small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # 检测人脸位置
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")  # 使用HOG模型，速度更快

            if len(face_locations) == 0:
                return []

            # 生成人脸编码
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            results = []

            for face_encoding, face_location in zip(face_encodings, face_locations):
                # 计算与已知人脸的距离
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                if len(face_distances) == 0:
                    continue

                best_match_index = np.argmin(face_distances)
                min_distance = face_distances[best_match_index]

                # 如果距离小于阈值，认为是匹配的
                if min_distance < self.recognition_threshold:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - min_distance

                    # 将位置还原到原始图像大小
                    top, right, bottom, left = face_location
                    top = int(top / self.scale_factor)
                    right = int(right / self.scale_factor)
                    bottom = int(bottom / self.scale_factor)
                    left = int(left / self.scale_factor)

                    results.append({
                        'name': name,
                        'location': (top, right, bottom, left),
                        'confidence': confidence
                    })

            return results

        except Exception as e:
            logger.error(f"识别人脸时发生错误: {e}")
            return []

    def _calculate_face_center_distance(self, face_location: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> float:
        """
        计算人脸中心到画面中心的距离

        Args:
            face_location: 人脸位置 (top, right, bottom, left)
            frame_shape: 图像尺寸 (height, width)

        Returns:
            float: 归一化距离（0-1）
        """
        top, right, bottom, left = face_location
        face_center_x = (left + right) / 2
        face_center_y = (top + bottom) / 2

        frame_height, frame_width = frame_shape[:2]
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2

        distance = np.sqrt((face_center_x - frame_center_x)**2 + (face_center_y - frame_center_y)**2)
        max_distance = np.sqrt(frame_center_x**2 + frame_center_y**2)

        return distance / max_distance

    def _update_face_tracking(self, name: str, confidence: float, location: Tuple[int, int, int, int]):
        """
        更新人脸跟踪信息

        Args:
            name: 人名
            confidence: 识别置信度
            location: 人脸位置
        """
        current_time = time.time()

        if name not in self.tracked_faces:
            self.tracked_faces[name] = {
                'first_seen': current_time,
                'last_seen': current_time,
                'consecutive_frames': 1,
                'best_confidence': confidence,
                'latest_location': location,
                'stable_duration': 0
            }
        else:
            face_info = self.tracked_faces[name]
            face_info['last_seen'] = current_time
            face_info['consecutive_frames'] += 1
            face_info['best_confidence'] = max(face_info['best_confidence'], confidence)
            face_info['latest_location'] = location
            face_info['stable_duration'] = current_time - face_info['first_seen']

    def _clean_expired_faces(self, max_age: float = 5.0):
        """
        清理长时间未出现的人脸跟踪信息

        Args:
            max_age: 最大保留时间（秒）
        """
        current_time = time.time()
        expired_faces = []

        for name, face_info in self.tracked_faces.items():
            if current_time - face_info['last_seen'] > max_age:
                expired_faces.append(name)

        for name in expired_faces:
            del self.tracked_faces[name]
            if self.current_priority_face == name:
                self.current_priority_face = None

    def select_priority_face(self, recognized_faces: List[Dict], frame_shape: Tuple[int, int]) -> Optional[Dict]:
        """
        从多个识别的人脸中选择优先跟踪的人脸

        Args:
            recognized_faces: 识别到的人脸列表
            frame_shape: 图像尺寸

        Returns:
            Optional[Dict]: 选中的优先人脸信息，如果没有合适的返回None
        """
        if not recognized_faces:
            return None

        if len(recognized_faces) == 1:
            face = recognized_faces[0]
            self._update_face_tracking(face['name'], face['confidence'], face['location'])
            self.current_priority_face = face['name']
            return face

        # 更新所有识别到的人脸跟踪信息
        recognized_names = set()
        for face in recognized_faces:
            self._update_face_tracking(face['name'], face['confidence'], face['location'])
            recognized_names.add(face['name'])

        # 清理过期的人脸跟踪信息
        self._clean_expired_faces()

        # 多人脸智能选择算法
        best_face = None
        best_score = -1

        for face in recognized_faces:
            name = face['name']
            confidence = face['confidence']
            location = face['location']

            # 获取跟踪信息
            track_info = self.tracked_faces.get(name, {})

            # 计算选择分数
            score = 0

            # 1. 已注册人脸优先 (权重: 40%)
            if name in self.known_face_names:
                score += 40

            # 2. 稳定性评分 (权重: 25%) - 连续出现时间越长越好
            stable_duration = track_info.get('stable_duration', 0)
            stability_score = min(stable_duration / 3.0, 1.0) * 25  # 3秒达到满分
            score += stability_score

            # 3. 置信度评分 (权重: 20%)
            confidence_score = confidence * 20
            score += confidence_score

            # 4. 距离中心评分 (权重: 10%) - 距离越近越好
            center_distance = self._calculate_face_center_distance(location, frame_shape)
            distance_score = (1 - center_distance) * 10
            score += distance_score

            # 5. 当前优先人脸加分 (权重: 5%) - 保持连续性
            if name == self.current_priority_face:
                score += 5

            logger.debug(f"人脸选择评分 - {name}: 总分={score:.2f} "
                        f"(稳定性={stability_score:.1f}, 置信度={confidence_score:.1f}, "
                        f"距离={distance_score:.1f})")

            if score > best_score:
                best_score = score
                best_face = face

        # 更新当前优先人脸
        if best_face:
            self.current_priority_face = best_face['name']
            logger.info(f"选择优先跟踪人脸: {best_face['name']} (评分: {best_score:.2f})")

        return best_face

    def get_priority_face_info(self) -> Optional[Dict]:
        """
        获取当前优先人脸的跟踪信息

        Returns:
            Optional[Dict]: 优先人脸的详细信息
        """
        if not self.current_priority_face:
            return None

        face_info = self.tracked_faces.get(self.current_priority_face)
        if not face_info:
            return None

        return {
            'name': self.current_priority_face,
            'location': face_info['latest_location'],
            'stable_duration': face_info['stable_duration'],
            'confidence': face_info['best_confidence'],
            'consecutive_frames': face_info['consecutive_frames']
        }

    def reset_tracking(self):
        """重置所有跟踪状态"""
        self.tracked_faces.clear()
        self.current_priority_face = None
