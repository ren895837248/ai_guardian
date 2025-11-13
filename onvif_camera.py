"""
ONVIF摄像头连接模块
处理摄像头连接、异常重连等功能
"""

import cv2
import logging
import time
from typing import Optional, Tuple
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ONVIFCamera:
    """ONVIF摄像头连接类，支持异常处理和自动重连，支持分别指定连接参数"""

    def __init__(self,
                 ip: str,
                 port: int = 554,
                 username: str = "admin",
                 password: str = "admin",
                 stream_path: str = "stream1",
                 reconnect_interval: int = 5,
                 max_retries: int = 3):
        """
        初始化ONVIF摄像头连接

        Args:
            ip: 摄像头IP地址，例如: "192.168.1.100"
            port: RTSP端口，默认554
            username: 用户名，默认"admin"
            password: 密码，默认"admin"
            stream_path: 流路径，默认"stream1"（常见的还有"h264"、"live"等）
            reconnect_interval: 重连间隔时间（秒）
            max_retries: 最大重试次数
        """
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.stream_path = stream_path
        self.reconnect_interval = reconnect_interval
        self.max_retries = max_retries

        # 构建RTSP URL
        self.rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/{stream_path}"

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.retry_count = 0

    @classmethod
    def from_rtsp_url(cls, rtsp_url: str, reconnect_interval: int = 5, max_retries: int = 3):
        """
        从完整RTSP URL创建摄像头连接实例（兼容旧版本）

        Args:
            rtsp_url: 完整RTSP URL，例如: "rtsp://admin:password@192.168.1.100:554/stream1"
            reconnect_interval: 重连间隔时间（秒）
            max_retries: 最大重试次数

        Returns:
            ONVIFCamera: 摄像头连接实例
        """
        # 解析RTSP URL
        import re
        pattern = r'rtsp://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)'
        match = re.match(pattern, rtsp_url)

        if match:
            username, password, ip, port, stream_path = match.groups()
            return cls(ip, int(port), username, password, stream_path, reconnect_interval, max_retries)
        else:
            # 简化的格式解析 rtsp://ip:port/path
            simple_pattern = r'rtsp://([^:]+):(\d+)/(.+)'
            simple_match = re.match(simple_pattern, rtsp_url)
            if simple_match:
                ip, port, stream_path = simple_match.groups()
                return cls(ip, int(port), "admin", "admin", stream_path, reconnect_interval, max_retries)
            else:
                raise ValueError(f"无法解析RTSP URL: {rtsp_url}")

    def update_credentials(self, username: str, password: str):
        """
        更新用户凭据

        Args:
            username: 新用户名
            password: 新密码
        """
        self.username = username
        self.password = password
        self.rtsp_url = f"rtsp://{username}:{password}@{self.ip}:{self.port}/{self.stream_path}"

        # 如果当前已连接，需要重新连接以应用新凭据
        if self.is_connected:
            logger.info("检测到凭据更新，重新连接摄像头...")
            self.disconnect()
            self.connect()

    def get_connection_info(self) -> dict:
        """
        获取连接信息

        Returns:
            dict: 包含连接参数的字典
        """
        return {
            'ip': self.ip,
            'port': self.port,
            'username': self.username,
            'password': '***' if self.password else '',  # 隐藏密码
            'stream_path': self.stream_path,
            'rtsp_url': f"rtsp://{self.username}:***@{self.ip}:{self.port}/{self.stream_path}",
            'is_connected': self.is_connected
        }

    def connect(self) -> bool:
        """
        连接到摄像头

        Returns:
            bool: 连接是否成功
        """
        try:
            logger.info(f"正在连接到摄像头: {self.rtsp_url}")
            self.cap = cv2.VideoCapture(self.rtsp_url)

            # 设置缓冲区大小为1，减少延迟
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self.cap.isOpened():
                self.is_connected = True
                self.retry_count = 0
                logger.info("摄像头连接成功")
                return True
            else:
                logger.error("摄像头连接失败")
                self.is_connected = False
                return False

        except Exception as e:
            logger.error(f"连接摄像头时发生异常: {e}")
            self.is_connected = False
            return False

    def reconnect(self) -> bool:
        """
        尝试重新连接摄像头

        Returns:
            bool: 重连是否成功
        """
        if self.retry_count >= self.max_retries:
            logger.error(f"已达到最大重试次数 ({self.max_retries})")
            return False

        self.retry_count += 1
        logger.info(f"尝试重新连接 (第 {self.retry_count}/{self.max_retries} 次)")

        self.disconnect()
        time.sleep(self.reconnect_interval)
        return self.connect()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧图像

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (是否成功, 图像数据)
        """
        if not self.is_connected or self.cap is None:
            return False, None

        try:
            ret, frame = self.cap.read()

            if not ret:
                logger.warning("无法读取帧，尝试重新连接")
                if self.reconnect():
                    ret, frame = self.cap.read()
                    return ret, frame
                else:
                    return False, None

            return ret, frame

        except Exception as e:
            logger.error(f"读取帧时发生异常: {e}")
            return False, None

    def disconnect(self):
        """断开摄像头连接"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.is_connected = False
                logger.info("摄像头已断开连接")
        except Exception as e:
            logger.error(f"断开连接时发生异常: {e}")

    def __enter__(self):
        """支持上下文管理器"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动断开连接"""
        self.disconnect()
