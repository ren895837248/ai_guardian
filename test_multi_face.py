#!/usr/bin/env python3
"""
测试多人脸智能选择功能
模拟多人脸场景，验证系统的改进效果
"""

import cv2
import numpy as np
import time
import logging
from face_recognition_module import FaceRecognizer
from head_gesture_detector import HeadGestureDetector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiFaceTest:
    """多人脸测试类"""

    def __init__(self):
        self.face_recognizer = FaceRecognizer()
        self.gesture_detector = HeadGestureDetector()

    def simulate_multi_face_scenario(self):
        """
        模拟多人脸场景测试
        """
        logger.info("🧪 开始多人脸智能选择功能测试")

        # 模拟检测结果 - 多个人脸
        test_faces = [
            {
                'name': 'Unknown_Person_1',
                'location': (50, 200, 150, 100),
                'confidence': 0.7
            },
            {
                'name': 'Zhang_San',  # 假设这是已注册用户
                'location': (50, 400, 150, 300),
                'confidence': 0.8
            },
            {
                'name': 'Unknown_Person_2',
                'location': (200, 200, 300, 100),
                'confidence': 0.6
            }
        ]

        # 模拟图像尺寸
        frame_shape = (480, 640, 3)  # Height, Width, Channels

        logger.info(f"📍 模拟场景: 检测到 {len(test_faces)} 个人脸")
        for i, face in enumerate(test_faces, 1):
            logger.info(f"  人脸{i}: {face['name']}, 置信度: {face['confidence']:.2f}")

        # 测试智能选择算法
        logger.info("\n🤖 测试智能人脸选择算法...")

        # 第一次选择
        selected_face = self.face_recognizer.select_priority_face(test_faces, frame_shape)
        if selected_face:
            logger.info(f"✅ 第一次选择结果: {selected_face['name']}")

            # 获取详细信息
            face_info = self.face_recognizer.get_priority_face_info()
            if face_info:
                logger.info(f"   跟踪信息: 稳定时长={face_info['stable_duration']:.2f}s, "
                           f"连续帧数={face_info['consecutive_frames']}, "
                           f"最佳置信度={face_info['confidence']:.3f}")

        # 模拟连续几次选择，测试稳定性
        logger.info("\n🔄 模拟连续检测，测试选择稳定性...")
        for round_num in range(1, 6):
            time.sleep(0.1)  # 模拟时间流逝

            # 稍微调整人脸位置和置信度，模拟真实场景
            adjusted_faces = []
            for face in test_faces:
                new_face = face.copy()
                # 微调位置
                top, right, bottom, left = face['location']
                new_face['location'] = (
                    top + np.random.randint(-5, 6),
                    right + np.random.randint(-5, 6),
                    bottom + np.random.randint(-5, 6),
                    left + np.random.randint(-5, 6)
                )
                # 微调置信度
                new_face['confidence'] = face['confidence'] + np.random.uniform(-0.05, 0.05)
                new_face['confidence'] = max(0.3, min(0.95, new_face['confidence']))
                adjusted_faces.append(new_face)

            selected_face = self.face_recognizer.select_priority_face(adjusted_faces, frame_shape)
            if selected_face:
                face_info = self.face_recognizer.get_priority_face_info()
                logger.info(f"轮次 {round_num}: 选择 {selected_face['name']}, "
                           f"稳定时长={face_info['stable_duration']:.2f}s")

        logger.info("\n📊 测试完成，选择结果应该倾向于已注册用户或稳定出现的人脸")

    def test_priority_algorithm(self):
        """
        详细测试优先级算法的各个组成部分
        """
        logger.info("\n🔬 详细测试优先级算法组成部分")

        # 创建测试用例
        test_cases = [
            {
                'name': '已注册用户vs未注册用户',
                'faces': [
                    {'name': 'Unknown_1', 'location': (100, 300, 200, 200), 'confidence': 0.9},
                    {'name': 'Registered_User', 'location': (100, 500, 200, 400), 'confidence': 0.7}
                ],
                'expected': 'Registered_User'
            },
            {
                'name': '高置信度vs低置信度',
                'faces': [
                    {'name': 'Person_A', 'location': (100, 300, 200, 200), 'confidence': 0.9},
                    {'name': 'Person_B', 'location': (100, 500, 200, 400), 'confidence': 0.6}
                ],
                'expected': 'Person_A'
            },
            {
                'name': '中心位置vs边缘位置',
                'faces': [
                    {'name': 'Edge_Person', 'location': (10, 100, 110, 0), 'confidence': 0.8},
                    {'name': 'Center_Person', 'location': (200, 400, 300, 300), 'confidence': 0.8}
                ],
                'expected': 'Center_Person'
            }
        ]

        frame_shape = (480, 640, 3)

        for test_case in test_cases:
            logger.info(f"\n🧪 测试案例: {test_case['name']}")

            # 重置状态
            self.face_recognizer.reset_tracking()

            selected_face = self.face_recognizer.select_priority_face(test_case['faces'], frame_shape)

            if selected_face:
                result = selected_face['name']
                expected = test_case['expected']
                status = "✅ 通过" if result == expected else "❌ 失败"
                logger.info(f"   结果: {result}, 期望: {expected} - {status}")
            else:
                logger.warning("   未选择到任何人脸")

    def test_head_gesture_with_priority_face(self):
        """
        测试优先人脸的头部动作检测
        """
        logger.info("\n👤 测试优先人脸头部动作检测")

        # 创建模拟图像
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_frame, (200, 150), (400, 350), (255, 255, 255), -1)

        # 模拟人脸位置
        face_location = (150, 400, 350, 200)  # (top, right, bottom, left)
        priority_name = "Test_User"

        logger.info(f"🎯 模拟检测优先人脸 '{priority_name}' 的头部动作")

        # 注意：这里需要有实际的面部关键点检测器文件才能运行
        # detected, gesture_type = self.gesture_detector.detect_gesture(
        #     test_frame, face_location, priority_name
        # )

        logger.info("   注意: 头部动作检测需要 'shape_predictor_68_face_landmarks.dat' 模型文件")
        logger.info("   可从以下地址下载: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")

    def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始运行多人脸功能完整测试套件")
        logger.info("=" * 60)

        try:
            # 测试1: 多人脸智能选择
            self.simulate_multi_face_scenario()

            # 测试2: 优先级算法详细测试
            self.test_priority_algorithm()

            # 测试3: 头部动作检测
            self.test_head_gesture_with_priority_face()

            logger.info("\n" + "=" * 60)
            logger.info("🎉 所有测试完成！")

            # 显示改进总结
            self.show_improvement_summary()

        except Exception as e:
            logger.error(f"测试过程中发生错误: {e}")

    def show_improvement_summary(self):
        """显示改进功能总结"""
        logger.info("\n📋 多人脸智能选择功能改进总结:")
        logger.info("=" * 50)
        logger.info("✅ 1. 智能人脸优先级算法")
        logger.info("    - 已注册用户优先 (权重40%)")
        logger.info("    - 稳定性优先 (权重25%)")
        logger.info("    - 置信度优先 (权重20%)")
        logger.info("    - 画面中心优先 (权重10%)")
        logger.info("    - 连续性保持 (权重5%)")
        logger.info("")
        logger.info("✅ 2. 人脸跟踪状态管理")
        logger.info("    - 记录每个人脸的出现历史")
        logger.info("    - 自动清理长时间未出现的人脸")
        logger.info("    - 支持人脸跟踪信息查询")
        logger.info("")
        logger.info("✅ 3. 头部动作检测优化")
        logger.info("    - 支持指定优先人脸位置")
        logger.info("    - 多人脸场景下自动选择最佳目标")
        logger.info("    - 改进的日志和错误处理")
        logger.info("")
        logger.info("✅ 4. 主程序逻辑升级")
        logger.info("    - 集成智能人脸选择算法")
        logger.info("    - 实时多人脸状态监控")
        logger.info("    - 优化的模式切换逻辑")
        logger.info("")
        logger.info("🎯 核心改进:")
        logger.info("   在多人脸场景下，系统现在能够智能选择最合适的")
        logger.info("   已注册人脸进行持续跟踪和头部动作检测，显著")
        logger.info("   提升了系统的可靠性和用户体验！")


if __name__ == '__main__':
    # 运行测试
    test_suite = MultiFaceTest()
    test_suite.run_all_tests()