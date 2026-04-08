import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, QSettings, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# 串口依赖：任务12新增
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except Exception:
    serial = None
    SERIAL_AVAILABLE = False


@dataclass
class Frame:
    timestamp: float
    index: int
    matrix: np.ndarray
    scenario: str


class MockDataSource:
    """
    模拟数据源：
    - 输出 16x16 压力矩阵，范围 0~1000
    - 每 20 帧更新一次“压力场景”
    """

    def __init__(self) -> None:
        self._running = False
        self._index = 0
        self._current_matrix = np.zeros((16, 16), dtype=np.float32)
        self._current_scenario = "idle"
        self._scenes = ["uniform", "center", "point", "line", "impact"]
        self._scene_ptr = 0
        self._rng = np.random.default_rng()

    def start(self) -> None:
        self._running = True
        self._index = 0
        self._scene_ptr = 0
        self._current_scenario = self._scenes[0]
        self._current_matrix = self._generate_scene_matrix(self._current_scenario)

    def stop(self) -> None:
        self._running = False

    def read_frame(self) -> Optional[Frame]:
        if not self._running:
            return None

        if self._index % 20 == 0:
            self._current_scenario = self._scenes[self._scene_ptr % len(self._scenes)]
            self._current_matrix = self._generate_scene_matrix(self._current_scenario)
            self._scene_ptr += 1

        frame = Frame(
            timestamp=time.time(),
            index=self._index,
            matrix=self._current_matrix.copy(),
            scenario=self._current_scenario,
        )
        self._index += 1

        # 数据源频率高于 UI 刷新频率
        time.sleep(0.01)  # ~100Hz
        return frame

    def _generate_scene_matrix(self, scene: str) -> np.ndarray:
        if scene == "uniform":
            matrix = self._scene_uniform()
        elif scene == "center":
            matrix = self._scene_center_high()
        elif scene == "point":
            matrix = self._scene_point_pressure()
        elif scene == "line":
            matrix = self._scene_line_pressure()
        else:
            matrix = self._scene_impact()

        rr, cc = np.indices((16, 16))
        checker = ((rr + cc) % 2).astype(np.float32)
        matrix += checker * self._rng.uniform(80, 180)

        return np.clip(matrix, 0.0, 1000.0).astype(np.float32)

    def _scene_uniform(self) -> np.ndarray:
        base = self._rng.uniform(300, 700)
        noise = self._rng.uniform(-220, 220, size=(16, 16))
        return (base + noise).astype(np.float32)

    def _scene_center_high(self) -> np.ndarray:
        rr, cc = np.indices((16, 16))
        center = np.array([7.5, 7.5], dtype=np.float32)
        dist = np.sqrt((rr - center[0]) ** 2 + (cc - center[1]) ** 2)
        radial = 900.0 - dist * 95.0
        noise = self._rng.uniform(-120, 120, size=(16, 16))
        return (radial + noise).astype(np.float32)

    def _scene_point_pressure(self) -> np.ndarray:
        rr, cc = np.indices((16, 16))
        matrix = self._rng.uniform(20, 120, size=(16, 16)).astype(np.float32)

        point_count = int(self._rng.integers(2, 5))
        for _ in range(point_count):
            r0 = float(self._rng.uniform(1, 14))
            c0 = float(self._rng.uniform(1, 14))
            amp = float(self._rng.uniform(700, 1000))
            sigma = float(self._rng.uniform(0.6, 1.5))
            gaussian = amp * np.exp(-(((rr - r0) ** 2 + (cc - c0) ** 2) / (2 * sigma**2)))
            matrix += gaussian.astype(np.float32)

        return matrix

    def _scene_line_pressure(self) -> np.ndarray:
        matrix = self._rng.uniform(30, 130, size=(16, 16)).astype(np.float32)

        row_id = int(self._rng.integers(0, 16))
        col_id = int(self._rng.integers(0, 16))
        matrix[row_id, :] += self._rng.uniform(500, 900, size=(16,))
        matrix[:, col_id] += self._rng.uniform(500, 900, size=(16,))

        return matrix

    def _scene_impact(self) -> np.ndarray:
        rr, cc = np.indices((16, 16))
        matrix = self._rng.uniform(0, 80, size=(16, 16)).astype(np.float32)

        r0 = float(self._rng.uniform(3, 12))
        c0 = float(self._rng.uniform(3, 12))
        amp = float(self._rng.uniform(900, 1000))
        sigma = float(self._rng.uniform(0.5, 1.0))
        impact = amp * np.exp(-(((rr - r0) ** 2 + (cc - c0) ** 2) / (2 * sigma**2)))
        matrix += impact.astype(np.float32)

        return matrix


class Worker(QObject):
    frame_received = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._running = False
        self._source = MockDataSource()

    @pyqtSlot()
    def run(self) -> None:
        self._running = True
        self._source.start()
        self.status_changed.emit("状态：已连接")

        while self._running:
            frame = self._source.read_frame()
            if frame is not None:
                self.frame_received.emit(frame)

        self._source.stop()
        self.status_changed.emit("状态：未连接")
        self.finished.emit()

    @pyqtSlot()
    def stop(self) -> None:
        self._running = False


class SerialByteSource:
    """
    串口字节源（只读 bytes，不做解析）
    功能：
    - 列出串口
    - 打开/关闭串口
    - 读取原始 bytes
    """

    def __init__(self) -> None:
        self._ser = None

    @staticmethod
    def list_ports() -> list[str]:
        if not SERIAL_AVAILABLE:
            return []
        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]

    def open(self, port: str, baudrate: int, timeout: float = 0.2) -> None:
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial 未安装，请先执行: pip install pyserial")
        self._ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            write_timeout=timeout,
        )

    def close(self) -> None:
        if self._ser is not None:
            try:
                if self._ser.is_open:
                    self._ser.close()
            finally:
                self._ser = None

    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def read_bytes(self, max_bytes: int = 1024) -> bytes:
        if not self.is_open():
            return b""
        # 不阻塞UI：worker线程内读取；优先读取缓冲区可用字节
        waiting = self._ser.in_waiting if self._ser is not None else 0
        size = min(max(1, waiting), max_bytes)
        return self._ser.read(size)


class SerialWorker(QObject):
    """
    串口后台读取线程（必须跑在 QThread 里）
    仅输出 bytes，不解析协议。
    """
    bytes_received = pyqtSignal(bytes)
    log_message = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, port: str, baudrate: int, timeout: float = 0.2) -> None:
        super().__init__()
        self._running = False
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._source = SerialByteSource()
        self._last_timeout_log_ts = 0.0

    @pyqtSlot()
    def run(self) -> None:
        self._running = True

        try:
            self._source.open(self._port, self._baudrate, timeout=self._timeout)
            self.status_changed.emit("串口：已打开")
            self.log_message.emit(f"串口已打开: {self._port} @ {self._baudrate}")
        except Exception as e:
            self.status_changed.emit("串口：打开失败")
            self.log_message.emit(f"[错误] 串口打开失败（可能端口占用/权限不足）: {e}")
            self.finished.emit()
            return

        try:
            while self._running:
                try:
                    data = self._source.read_bytes(max_bytes=2048)
                except Exception as e:
                    # 典型拔插异常会在这里出现
                    self.log_message.emit(f"[错误] 串口读取异常（可能设备拔出）: {e}")
                    self.status_changed.emit("串口：读取异常")
                    break

                if data:
                    self.bytes_received.emit(data)
                else:
                    # 超时：不刷屏，2秒最多一条
                    now = time.time()
                    if now - self._last_timeout_log_ts >= 2.0:
                        self._last_timeout_log_ts = now
                        self.log_message.emit("[提示] 串口读取超时：当前未收到新字节。")
        finally:
            try:
                self._source.close()
            except Exception as e:
                self.log_message.emit(f"[警告] 串口关闭时异常: {e}")

            self.status_changed.emit("串口：已关闭")
            self.log_message.emit("串口已关闭。")
            self.finished.emit()

    @pyqtSlot()
    def stop(self) -> None:
        self._running = False


class HeatmapGridWidget(QWidget):
    """
    16x16 自定义热力图网格：
    - 单元严格正方形
    - 间隙可配置
    - 自适应窗口尺寸
    """
    def __init__(self, rows: int = 16, cols: int = 16, gap_px: int = 2) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.gap_px = gap_px
        self.margin_px = 12
        self.vmin = 0.0
        self.vmax = 1000.0
        self._matrix = np.zeros((rows, cols), dtype=np.float32)
        self._lut = self._build_colormap_lut("jet")
        self.setMinimumSize(360, 360)

    def set_matrix(self, matrix: np.ndarray) -> None:
        if matrix.shape != (self.rows, self.cols):
            return
        self._matrix = matrix.astype(np.float32, copy=False)
        self.update()

    def set_gap(self, gap_px: int) -> None:
        self.gap_px = max(0, int(gap_px))
        self.update()

    def set_colormap(self, name: str) -> None:
        self._lut = self._build_colormap_lut(name)
        self.update()

    def _build_colormap_lut(self, name: str) -> np.ndarray:
        schemes = {
            "jet": [
                (0.00, (0, 0, 130)),
                (0.20, (0, 80, 255)),
                (0.45, (0, 220, 220)),
                (0.65, (80, 255, 80)),
                (0.82, (255, 230, 0)),
                (0.92, (255, 120, 0)),
                (1.00, (200, 0, 0)),
            ],
            "viridis": [
                (0.00, (68, 1, 84)),
                (0.25, (59, 82, 139)),
                (0.50, (33, 145, 140)),
                (0.75, (94, 201, 98)),
                (1.00, (253, 231, 37)),
            ],
            "plasma": [
                (0.00, (13, 8, 135)),
                (0.25, (126, 3, 167)),
                (0.50, (203, 71, 119)),
                (0.75, (248, 149, 64)),
                (1.00, (240, 249, 33)),
            ],
        }
        anchors = schemes.get(name, schemes["jet"])

        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        xp = np.array([p for p, _ in anchors], dtype=np.float32)
        rgb = np.array([c for _, c in anchors], dtype=np.float32)

        lut = np.zeros((256, 3), dtype=np.uint8)
        for k in range(3):
            lut[:, k] = np.interp(x, xp, rgb[:, k]).astype(np.uint8)
        return lut

    def _value_to_color(self, value: float) -> QColor:
        norm = (value - self.vmin) / max(1e-6, (self.vmax - self.vmin))
        norm = min(1.0, max(0.0, norm))
        idx = int(norm * 255)
        r, g, b = self._lut[idx]
        return QColor(int(r), int(g), int(b))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        painter.fillRect(self.rect(), QColor(18, 18, 22))

        w = self.width()
        h = self.height()
        gap = self.gap_px
        margin = self.margin_px

        avail_w = max(1, w - 2 * margin - (self.cols - 1) * gap)
        avail_h = max(1, h - 2 * margin - (self.rows - 1) * gap)

        # 核心：严格正方形
        cell = max(1, int(min(avail_w / self.cols, avail_h / self.rows)))

        grid_w = cell * self.cols + gap * (self.cols - 1)
        grid_h = cell * self.rows + gap * (self.rows - 1)
        x0 = (w - grid_w) // 2
        y0 = (h - grid_h) // 2

        painter.setPen(Qt.PenStyle.NoPen)
        for r in range(self.rows):
            for c in range(self.cols):
                val = float(self._matrix[r, c])
                painter.setBrush(self._value_to_color(val))
                x = x0 + c * (cell + gap)
                y = y0 + r * (cell + gap)
                painter.drawRect(x, y, cell, cell)

        border_pen = QPen(QColor(220, 220, 220, 140))
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(x0 - 1, y0 - 1, grid_w + 2, grid_h + 2)


class MainWindow(QMainWindow):
    UNIT_FACTORS = {
        # 设置仅影响显示层，不改变底层数据
        "raw": 1.0,
        "g": 1.0,
        "N": 0.00980665,
    }

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hand FSR Demo - Settings + Serial Bytes")
        self.resize(1250, 900)

        self._settings = QSettings("NJIT", "HandFSR")

        # Mock 线程
        self._thread: Optional[QThread] = None
        self._worker: Optional[Worker] = None

        # 串口线程
        self._serial_thread: Optional[QThread] = None
        self._serial_worker: Optional[SerialWorker] = None

        self._connected = False
        self._latest_frame: Optional[Frame] = None
        self._frame_count = 0
        self._ui_fps = 0.0
        self._last_ui_time: Optional[float] = None
        self._last_logged_data_update = -1

        # 串口字节统计
        self._serial_rx_total = 0
        self._serial_rx_last = 0
        self._serial_last_preview_ts = 0.0

        # 读取持久化设置
        self._unit = self._settings.value("display/unit", "raw", type=str)
        self._decimals = self._settings.value("display/decimals", 1, type=int)
        self._ui_interval_ms = self._settings.value("display/ui_interval_ms", 50, type=int)

        # 基础控件（Mock）
        self._btn_connect = QPushButton("连接(Mock)")
        self._btn_disconnect = QPushButton("断开(Mock)")
        self._btn_disconnect.setEnabled(False)

        self._status = QLabel("状态：未连接")
        self._stats = QLabel("sum: 0.0 | max: 0.0 | min: 0.0 | mean: 0.0 | UI_fps: 0.0 | 场景: -")

        self._combo_colormap = QComboBox()
        self._combo_colormap.addItems(["jet", "viridis", "plasma"])
        self._combo_colormap.setCurrentText("jet")

        self._spin_gap = QSpinBox()
        self._spin_gap.setRange(0, 8)
        self._spin_gap.setValue(2)
        self._spin_gap.setSingleStep(1)

        # 设置区域（任务8）
        self._combo_unit = QComboBox()
        self._combo_unit.addItems(["raw", "g", "N"])
        if self._unit not in ("raw", "g", "N"):
            self._unit = "raw"
        self._combo_unit.setCurrentText(self._unit)

        self._spin_decimals = QSpinBox()
        self._spin_decimals.setRange(0, 6)
        self._spin_decimals.setValue(max(0, min(6, int(self._decimals))))

        self._spin_ui_interval = QSpinBox()
        self._spin_ui_interval.setRange(20, 200)
        self._spin_ui_interval.setValue(max(20, min(200, int(self._ui_interval_ms))))
        self._spin_ui_interval.setSuffix(" ms")

        # 串口区域（任务12）
        self._btn_refresh_ports = QPushButton("刷新串口")
        self._combo_ports = QComboBox()
        self._combo_baud = QComboBox()
        self._combo_baud.addItems(["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"])
        self._combo_baud.setCurrentText("115200")

        self._btn_serial_open = QPushButton("打开串口")
        self._btn_serial_close = QPushButton("关闭串口")
        self._btn_serial_close.setEnabled(False)

        self._serial_status = QLabel("串口：未打开")
        self._serial_rx_label = QLabel("RX: 0 bytes")

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(170)

        self._heatmap = HeatmapGridWidget(rows=16, cols=16, gap_px=2)

        # 布局
        top = QHBoxLayout()
        top.addWidget(self._btn_connect)
        top.addWidget(self._btn_disconnect)
        top.addSpacing(12)
        top.addWidget(QLabel("色图:"))
        top.addWidget(self._combo_colormap)
        top.addSpacing(8)
        top.addWidget(QLabel("间隙(px):"))
        top.addWidget(self._spin_gap)
        top.addStretch()
        top.addWidget(self._status)

        settings_bar = QHBoxLayout()
        settings_bar.addWidget(QLabel("显示单位:"))
        settings_bar.addWidget(self._combo_unit)
        settings_bar.addSpacing(12)
        settings_bar.addWidget(QLabel("小数位:"))
        settings_bar.addWidget(self._spin_decimals)
        settings_bar.addSpacing(12)
        settings_bar.addWidget(QLabel("UI刷新周期:"))
        settings_bar.addWidget(self._spin_ui_interval)
        settings_bar.addStretch()

        serial_bar = QHBoxLayout()
        serial_bar.addWidget(self._btn_refresh_ports)
        serial_bar.addWidget(QLabel("端口:"))
        serial_bar.addWidget(self._combo_ports)
        serial_bar.addWidget(QLabel("波特率:"))
        serial_bar.addWidget(self._combo_baud)
        serial_bar.addWidget(self._btn_serial_open)
        serial_bar.addWidget(self._btn_serial_close)
        serial_bar.addSpacing(12)
        serial_bar.addWidget(self._serial_status)
        serial_bar.addSpacing(8)
        serial_bar.addWidget(self._serial_rx_label)
        serial_bar.addStretch()

        root = QVBoxLayout()
        root.addLayout(top)
        root.addLayout(settings_bar)
        root.addLayout(serial_bar)
        root.addWidget(self._stats)
        root.addWidget(self._heatmap, stretch=1)
        root.addWidget(self._log)

        center = QWidget()
        center.setLayout(root)
        self.setCentralWidget(center)

        # 信号：Mock
        self._btn_connect.clicked.connect(self.on_connect_clicked)
        self._btn_disconnect.clicked.connect(self.on_disconnect_clicked)
        self._combo_colormap.currentTextChanged.connect(self._heatmap.set_colormap)
        self._spin_gap.valueChanged.connect(self._heatmap.set_gap)

        # 信号：任务8设置
        self._combo_unit.currentTextChanged.connect(self.on_display_settings_changed)
        self._spin_decimals.valueChanged.connect(self.on_display_settings_changed)
        self._spin_ui_interval.valueChanged.connect(self.on_display_settings_changed)

        # 信号：任务12串口
        self._btn_refresh_ports.clicked.connect(self.on_refresh_ports_clicked)
        self._btn_serial_open.clicked.connect(self.on_serial_open_clicked)
        self._btn_serial_close.clicked.connect(self.on_serial_close_clicked)

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(self._spin_ui_interval.value())
        self._ui_timer.timeout.connect(self.refresh_ui)
        self._ui_timer.start()

        # 初始化
        self.apply_display_settings()
        self.on_refresh_ports_clicked()

        if not SERIAL_AVAILABLE:
            self.append_log("[警告] 未检测到 pyserial。请安装: pip install pyserial")
            self._serial_status.setText("串口：不可用（缺少 pyserial）")
            self._btn_serial_open.setEnabled(False)
            self._btn_refresh_ports.setEnabled(False)

    def append_log(self, text: str) -> None:
        self._log.appendPlainText(text)

    # ---------------- Mock 热力图逻辑（保持原有） ----------------
    @pyqtSlot()
    def on_connect_clicked(self) -> None:
        if self._thread is not None:
            return

        self._thread = QThread()
        self._worker = Worker()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.frame_received.connect(self.on_frame_received)
        self._worker.status_changed.connect(self.on_status_changed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self.on_thread_finished)

        self._thread.start()
        self._btn_connect.setEnabled(False)
        self._btn_disconnect.setEnabled(True)
        self.append_log("已连接：开始模拟压力数据。")

    @pyqtSlot()
    def on_disconnect_clicked(self) -> None:
        self.append_log("正在断开 Mock...")
        self.stop_worker()

    @pyqtSlot(object)
    def on_frame_received(self, frame: object) -> None:
        if isinstance(frame, Frame):
            self._latest_frame = frame
            self._frame_count += 1

            update_id = frame.index // 20
            if update_id != self._last_logged_data_update:
                self._last_logged_data_update = update_id
                self.append_log(f"数据更新: frame={frame.index}, scene={frame.scenario}")

    @pyqtSlot(str)
    def on_status_changed(self, text: str) -> None:
        self._connected = (text == "状态：已连接")
        self._status.setText(text)

    @pyqtSlot()
    def on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._connected = False
        self._btn_connect.setEnabled(True)
        self._btn_disconnect.setEnabled(False)
        self._status.setText("状态：未连接")
        self.append_log("Mock 已断开。")

    def stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.stop()

        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()

    # ---------------- 任务8：显示设置 ----------------
    @pyqtSlot()
    def on_display_settings_changed(self) -> None:
        self.apply_display_settings()
        self.save_display_settings()

    def apply_display_settings(self) -> None:
        self._unit = self._combo_unit.currentText()
        self._decimals = self._spin_decimals.value()
        self._ui_interval_ms = self._spin_ui_interval.value()

        # 只影响UI刷新，不影响底层数据源
        self._ui_timer.setInterval(self._ui_interval_ms)
        self.refresh_ui()

    def save_display_settings(self) -> None:
        self._settings.setValue("display/unit", self._unit)
        self._settings.setValue("display/decimals", self._decimals)
        self._settings.setValue("display/ui_interval_ms", self._ui_interval_ms)
        self._settings.sync()

    def format_value(self, value: float) -> str:
        factor = self.UNIT_FACTORS.get(self._unit, 1.0)
        converted = value * factor
        return f"{converted:.{self._decimals}f}"

    # ---------------- 任务12：串口字节读取 ----------------
    @pyqtSlot()
    def on_refresh_ports_clicked(self) -> None:
        self._combo_ports.clear()

        if not SERIAL_AVAILABLE:
            return

        ports = SerialByteSource.list_ports()
        if not ports:
            self._combo_ports.addItem("无可用串口")
            self._btn_serial_open.setEnabled(False)
            self.append_log("[提示] 未发现可用串口。")
            return

        self._combo_ports.addItems(ports)
        self._btn_serial_open.setEnabled(True)
        self.append_log(f"[串口] 已刷新，发现 {len(ports)} 个端口。")

    @pyqtSlot()
    def on_serial_open_clicked(self) -> None:
        if self._serial_thread is not None:
            return

        if not SERIAL_AVAILABLE:
            self.append_log("[错误] pyserial 不可用，无法打开串口。")
            return

        port = self._combo_ports.currentText().strip()
        if not port or port == "无可用串口":
            self.append_log("[错误] 请先选择有效串口。")
            return

        try:
            baudrate = int(self._combo_baud.currentText())
        except ValueError:
            self.append_log("[错误] 波特率非法。")
            return

        self._serial_thread = QThread()
        self._serial_worker = SerialWorker(port=port, baudrate=baudrate, timeout=0.2)
        self._serial_worker.moveToThread(self._serial_thread)

        self._serial_thread.started.connect(self._serial_worker.run)
        self._serial_worker.bytes_received.connect(self.on_serial_bytes_received)
        self._serial_worker.log_message.connect(self.on_serial_log_message)
        self._serial_worker.status_changed.connect(self.on_serial_status_changed)
        self._serial_worker.finished.connect(self._serial_thread.quit)
        self._serial_worker.finished.connect(self._serial_worker.deleteLater)
        self._serial_thread.finished.connect(self._serial_thread.deleteLater)
        self._serial_thread.finished.connect(self.on_serial_thread_finished)

        self._serial_rx_total = 0
        self._serial_rx_last = 0
        self._serial_last_preview_ts = 0.0
        self._serial_rx_label.setText("RX: 0 bytes")

        self._btn_serial_open.setEnabled(False)
        self._btn_serial_close.setEnabled(True)
        self._btn_refresh_ports.setEnabled(False)
        self._combo_ports.setEnabled(False)
        self._combo_baud.setEnabled(False)

        self._serial_thread.start()

    @pyqtSlot()
    def on_serial_close_clicked(self) -> None:
        self.append_log("[串口] 正在关闭...")
        self.stop_serial_worker()

    @pyqtSlot(bytes)
    def on_serial_bytes_received(self, data: bytes) -> None:
        # 上层拿到原始bytes（暂不解析）
        self._serial_rx_total += len(data)
        self._serial_rx_label.setText(f"RX: {self._serial_rx_total} bytes")

        # 控制日志频率：每秒最多打一条预览
        now = time.time()
        if now - self._serial_last_preview_ts >= 1.0:
            self._serial_last_preview_ts = now
            preview = data[:16].hex(" ")
            self.append_log(f"[串口RX] +{len(data)} bytes, preview: {preview}")

    @pyqtSlot(str)
    def on_serial_log_message(self, text: str) -> None:
        self.append_log(text)

    @pyqtSlot(str)
    def on_serial_status_changed(self, text: str) -> None:
        self._serial_status.setText(text)

    @pyqtSlot()
    def on_serial_thread_finished(self) -> None:
        self._serial_thread = None
        self._serial_worker = None

        self._btn_serial_open.setEnabled(SERIAL_AVAILABLE and self._combo_ports.currentText() != "无可用串口")
        self._btn_serial_close.setEnabled(False)
        self._btn_refresh_ports.setEnabled(SERIAL_AVAILABLE)
        self._combo_ports.setEnabled(True)
        self._combo_baud.setEnabled(True)

        self._serial_status.setText("串口：未打开")

    def stop_serial_worker(self) -> None:
        if self._serial_worker is not None:
            self._serial_worker.stop()

        if self._serial_thread is not None:
            self._serial_thread.quit()
            self._serial_thread.wait()

    # ---------------- UI 刷新 ----------------
    def refresh_ui(self) -> None:
        if not self._connected or self._latest_frame is None:
            return

        frame = self._latest_frame
        matrix = frame.matrix
        self._heatmap.set_matrix(matrix)

        total_raw = float(np.sum(matrix))
        vmax_raw = float(np.max(matrix))
        vmin_raw = float(np.min(matrix))
        mean_raw = float(np.mean(matrix))

        now = time.time()
        if self._last_ui_time is not None:
            dt = now - self._last_ui_time
            if dt > 0:
                self._ui_fps = 1.0 / dt
        self._last_ui_time = now

        self._stats.setText(
            f"sum: {self.format_value(total_raw)} {self._unit} | "
            f"max: {self.format_value(vmax_raw)} {self._unit} | "
            f"min: {self.format_value(vmin_raw)} {self._unit} | "
            f"mean: {self.format_value(mean_raw)} {self._unit} | "
            f"UI_fps: {self._ui_fps:.1f} | 场景: {frame.scenario}"
        )
        self._status.setText(f"状态：已连接 | 已收帧 {self._frame_count} | 最新帧 {frame.index}")

    def closeEvent(self, event) -> None:
        self.stop_worker()
        self.stop_serial_worker()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())