"""
主程序 - 集成ONVIF摄像头、人脸识别和头部动作检测
实现完整的人脸识别+头部动作检测流程
"""

import cv2
import time
import logging
from typing import Optional
from onvif_camera import ONVIFCamera
from face_recognition_module import FaceRecognizer
from head_gesture_detector import HeadGestureDetector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceGestureSystem:
    """人脸识别+头部动作检测系统，支持多人脸智能选择"""

    def __init__(self,
                 rtsp_url: str,
                 recognition_duration: float = 3.0,
                 frame_skip: int = 2):
        """
        初始化系统

        Args:
            rtsp_url: RTSP流地址
            recognition_duration: 需要持续识别到人脸的时间（秒）
            frame_skip: 跳帧数，用于降低系统负载（每处理1帧跳过N帧）
        """
        self.rtsp_url = rtsp_url
        self.recognition_duration = recognition_duration
        self.frame_skip = frame_skip

        # 初始化各模块
        self.camera = ONVIFCamera(rtsp_url)
        self.face_recognizer = FaceRecognizer()
        self.gesture_detector = HeadGestureDetector()

        # 状态变量
        self.current_priority_face: Optional[str] = None
        self.current_priority_location: Optional[Tuple[int, int, int, int]] = None
        self.recognition_start_time: Optional[float] = None
        self.in_gesture_detection_mode = False
        self.frame_counter = 0


    def run(self) -> None:
        """
        运行主循环
        """
        logger.info("启动人脸识别+头部动作检测系统")

        if not self.camera.connect():
            logger.error("无法连接到摄像头，程序退出")
            return

        try:
            while True:
                ret, frame = self.camera.read_frame()

                if not ret or frame is None:
                    logger.warning("无法读取帧")
                    time.sleep(0.1)
                    continue

                # 跳帧处理，降低系统负载
                self.frame_counter += 1
                if self.frame_counter % (self.frame_skip + 1) != 0:
                    continue

                # 模式1: 人脸识别模式
                if not self.in_gesture_detection_mode:
                    # 识别所有人脸
                    all_faces = self.face_recognizer.recognize_faces(frame)

                    if all_faces:
                        # 使用智能算法选择优先跟踪的人脸
                        priority_face = self.face_recognizer.select_priority_face(all_faces, frame.shape)

                        if priority_face:
                            current_person = priority_face['name']
                            current_location = priority_face['location']

                            if self.current_priority_face == current_person:
                                # 同一个优先人脸，计算持续时间
                                elapsed_time = time.time() - self.recognition_start_time

                                if elapsed_time >= self.recognition_duration:
                                    # 达到识别时间阈值，进入头部动作检测模式
                                    logger.info(f"优先人脸 {current_person} 已稳定识别 {self.recognition_duration} 秒，进入动作检测模式")
                                    self.in_gesture_detection_mode = True
                                    self.current_priority_location = current_location
                                    self.gesture_detector.reset()
                                else:
                                    # 更新位置并记录进度
                                    self.current_priority_location = current_location
                                    logger.debug(f"跟踪中: {current_person} ({elapsed_time:.1f}s / {self.recognition_duration}s)")
                            else:
                                # 识别到新的优先人脸
                                self.current_priority_face = current_person
                                self.current_priority_location = current_location
                                self.recognition_start_time = time.time()
                                logger.info(f"切换到优先人脸: {current_person}")

                                # 显示多人脸信息
                                if len(all_faces) > 1:
                                    other_faces = [f['name'] for f in all_faces if f['name'] != current_person]
                                    logger.info(f"当前场景检测到 {len(all_faces)} 个人脸: {current_person}(优先), {', '.join(other_faces)}")

                    else:
                        # 未识别到任何人脸，重置
                        if self.current_priority_face:
                            logger.info(f"失去跟踪目标: {self.current_priority_face}")
                        self.current_priority_face = None
                        self.current_priority_location = None
                        self.recognition_start_time = None

                # 模式2: 头部动作检测模式
                else:
                    # 使用优先人脸位置检测头部动作
                    detected, gesture_type = self.gesture_detector.detect_gesture(
                        frame,
                        self.current_priority_location,
                        self.current_priority_face
                    )

                    if detected:
                        if gesture_type == 'nod':
                            logger.info(f"🎯 {self.current_priority_face} 点头动作检测成功！返回True")
                        elif gesture_type == 'shake':
                            logger.info(f"🎯 {self.current_priority_face} 摇头动作检测成功！返回True")

                        # 返回人脸识别模式
                        self.in_gesture_detection_mode = False
                        self.current_priority_face = None
                        self.current_priority_location = None
                        self.recognition_start_time = None
                        # 重置人脸跟踪状态，重新开始
                        self.face_recognizer.reset_tracking()
                        logger.info("动作检测完成，返回人脸识别模式")

        except KeyboardInterrupt:
            logger.info("收到中断信号，正在退出...")

        except Exception as e:
            logger.error(f"运行过程中发生错误: {e}")

        finally:
            self.camera.disconnect()
            logger.info("系统已关闭")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='人脸识别+头部动作检测系统')
    parser.add_argument('--rtsp', type=str, required=True,
                       help='RTSP流地址，例如: rtsp://admin:password@192.168.1.100:554/stream1')
    parser.add_argument('--duration', type=float, default=3.0,
                       help='需要持续识别的时间（秒），默认3秒')
    parser.add_argument('--frame-skip', type=int, default=2,
                       help='跳帧数，用于降低系统负载，默认2（每处理1帧跳过2帧）')

    args = parser.parse_args()

    # 创建系统实例
    system = FaceGestureSystem(
        rtsp_url=args.rtsp,
        recognition_duration=args.duration,
        frame_skip=args.frame_skip
    )

    # 运行系统
    system.run()


if __name__ == '__main__':
    main()
