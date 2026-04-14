import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QSettings, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPen, QPalette, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

# 串口依赖（可选）
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
    - 每 20 帧更新一次"压力场景"
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
    """串口字节源（只读 bytes，不解析）"""

    def __init__(self) -> None:
        self._ser = None

    @staticmethod
    def list_ports() -> list[str]:
        if not SERIAL_AVAILABLE:
            return []
        return [p.device for p in serial.tools.list_ports.comports()]

    def open(self, port: str, baudrate: int, timeout: float = 0.2) -> None:
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial 未安装")
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
        waiting = self._ser.in_waiting if self._ser is not None else 0
        size = min(max(1, waiting), max_bytes)
        return self._ser.read(size)


class SerialWorker(QObject):
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
            self.status_changed.emit("已打开")
            self.log_message.emit(f"串口已打开: {self._port}@{self._baudrate}")
        except Exception as e:
            self.status_changed.emit("打开失败")
            self.log_message.emit(f"[错误] 串口打开失败: {e}")
            self.finished.emit()
            return

        try:
            while self._running:
                try:
                    data = self._source.read_bytes(max_bytes=2048)
                except Exception as e:
                    self.log_message.emit(f"[错误] 串口读取异常(可能拔插): {e}")
                    self.status_changed.emit("读取异常")
                    break

                if data:
                    self.bytes_received.emit(data)
                else:
                    now = time.time()
                    if now - self._last_timeout_log_ts >= 2.0:
                        self._last_timeout_log_ts = now
                        self.log_message.emit("[提示] 串口读取超时")
        finally:
            try:
                self._source.close()
            except Exception:
                pass
            self.status_changed.emit("已关闭")
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
    - 背景：系统默认色
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
        self._lut = self._build_jet_lut()
        self.setMinimumSize(360, 360)
        self.setAutoFillBackground(True)

    def set_matrix(self, matrix: np.ndarray) -> None:
        if matrix.shape != (self.rows, self.cols):
            return
        self._matrix = matrix.astype(np.float32, copy=False)
        self.update()

    def set_gap(self, gap_px: int) -> None:
        self.gap_px = max(0, int(gap_px))
        self.update()

    def _build_jet_lut(self) -> np.ndarray:
        """生成 jet 色彩查询表"""
        anchors = [
            (0.00, (0, 0, 130)),
            (0.20, (0, 80, 255)),
            (0.45, (0, 220, 220)),
            (0.65, (80, 255, 80)),
            (0.82, (255, 230, 0)),
            (0.92, (255, 120, 0)),
            (1.00, (200, 0, 0)),
        ]
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

        # 背景：系统默认色
        bg = self.palette().color(QPalette.ColorRole.Window)
        painter.fillRect(self.rect(), bg)

        w = self.width()
        h = self.height()
        gap = self.gap_px
        margin = self.margin_px

        avail_w = max(1, w - 2 * margin - (self.cols - 1) * gap)
        avail_h = max(1, h - 2 * margin - (self.rows - 1) * gap)

        # 核心：严格正方形单元
        cell = max(1, int(min(avail_w / self.cols, avail_h / self.rows)))

        grid_w = cell * self.cols + gap * (self.cols - 1)
        grid_h = cell * self.rows + gap * (self.rows - 1)
        x0 = (w - grid_w) // 2
        y0 = (h - grid_h) // 2

        # 绘制 16x16 单元
        painter.setPen(Qt.PenStyle.NoPen)
        for r in range(self.rows):
            for c in range(self.cols):
                val = float(self._matrix[r, c])
                painter.setBrush(self._value_to_color(val))
                x = x0 + c * (cell + gap)
                y = y0 + r * (cell + gap)
                painter.drawRect(x, y, cell, cell)

        # 边框
        border_pen = QPen(QColor(120, 120, 120, 120))
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(x0 - 1, y0 - 1, grid_w + 2, grid_h + 2)


class MainWindow(QMainWindow):
    """
    主窗口布局：
    - 顶部：紧凑控制栏
    - 中间：左右并列（左：热力图，右：数据分析）
    - 底部：单行状态栏
    """

    UNIT_FACTORS = {
        "raw": 1.0,
        "g": 1.0,
        "N": 0.00980665,
    }

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hand FSR Demo - Optimized Layout")
        self.resize(1400, 900)

        self._settings = QSettings("NJIT", "HandFSR")

        self._thread: Optional[QThread] = None
        self._worker: Optional[Worker] = None
        self._serial_thread: Optional[QThread] = None
        self._serial_worker: Optional[SerialWorker] = None

        self._connected = False
        self._latest_frame: Optional[Frame] = None
        self._frame_count = 0
        self._ui_fps = 0.0
        self._last_ui_time: Optional[float] = None

        self._serial_rx_total = 0
        self._serial_state_text = "未打开"

        # 会话统计
        self._session_max = float("-inf")
        self._session_min = float("inf")

        # 历史缓存：60秒窗口按20Hz约1200点，留余量
        self._t0 = time.time()
        self._time_hist = deque(maxlen=2400)
        self._total_hist = deque(maxlen=2400)
        self._current_hist = deque(maxlen=2400)

        # 读取持久化设置
        self._unit = self._settings.value("display/unit", "raw", type=str)
        self._decimals = self._settings.value("display/decimals", 1, type=int)
        self._ui_interval_ms = self._settings.value("display/ui_interval_ms", 50, type=int)

        # ========== 顶部控制栏（紧凑） ==========
        self._btn_connect = QPushButton("连接(Mock)")
        self._btn_disconnect = QPushButton("断开(Mock)")
        self._btn_disconnect.setEnabled(False)

        self._spin_gap = QSpinBox()
        self._spin_gap.setRange(0, 8)
        self._spin_gap.setValue(2)

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

        # 串口控件
        self._btn_refresh_ports = QPushButton("刷新串口")
        self._combo_ports = QComboBox()
        self._combo_baud = QComboBox()
        self._combo_baud.addItems(["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"])
        self._combo_baud.setCurrentText("115200")
        self._btn_serial_open = QPushButton("打开串口")
        self._btn_serial_close = QPushButton("关闭串口")
        self._btn_serial_close.setEnabled(False)

        # 控制栏布局
        controls = QHBoxLayout()
        controls.addWidget(self._btn_connect)
        controls.addWidget(self._btn_disconnect)
        controls.addSpacing(10)
        controls.addWidget(QLabel("间隙"))
        controls.addWidget(self._spin_gap)
        controls.addSpacing(10)
        controls.addWidget(QLabel("单位"))
        controls.addWidget(self._combo_unit)
        controls.addWidget(QLabel("小数"))
        controls.addWidget(self._spin_decimals)
        controls.addWidget(QLabel("刷新"))
        controls.addWidget(self._spin_ui_interval)
        controls.addSpacing(10)
        controls.addWidget(self._btn_refresh_ports)
        controls.addWidget(self._combo_ports)
        controls.addWidget(self._combo_baud)
        controls.addWidget(self._btn_serial_open)
        controls.addWidget(self._btn_serial_close)
        controls.addStretch()

        # ========== 左栏：热力图 ==========
        self._heatmap = HeatmapGridWidget(rows=16, cols=16, gap_px=2)
        
        left_title = QLabel("热力图（Jet 色彩）")
        left_title_font = QFont()
        left_title_font.setPointSize(11)
        left_title_font.setBold(True)
        left_title.setFont(left_title_font)

        left_panel = QVBoxLayout()
        left_panel.addWidget(left_title)
        left_panel.addWidget(self._heatmap, stretch=1)
        left_panel.setContentsMargins(8, 8, 8, 8)
        left_widget = QWidget()
        left_widget.setLayout(left_panel)

        # ========== 右栏：数据分析 ==========
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)

        data_font = QFont()
        data_font.setPointSize(10)

        value_font = QFont()
        value_font.setPointSize(12)
        value_font.setBold(True)

        right_title = QLabel("数据分析")
        right_title.setFont(title_font)

        # 数值显示区：标签 + 右对齐数值
        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(12)
        metrics_grid.setVerticalSpacing(8)

        lbl_total_name = QLabel("Sum_force")
        lbl_current_name = QLabel("Current_force")
        lbl_max_name = QLabel("Max")
        lbl_min_name = QLabel("Min")

        for lbl in [lbl_total_name, lbl_current_name, lbl_max_name, lbl_min_name]:
            lbl.setFont(data_font)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._val_total = QLabel("0.0")
        self._val_current = QLabel("0.0")
        self._val_max = QLabel("-")
        self._val_min = QLabel("-")

        for val in [self._val_total, self._val_current, self._val_max, self._val_min]:
            val.setFont(value_font)
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val.setStyleSheet("QLabel { background: white; border: 1px solid #BFBFBF; padding: 4px 8px; }")
            val.setMinimumWidth(140)

        metrics_grid.addWidget(lbl_total_name, 0, 0)
        metrics_grid.addWidget(self._val_total, 0, 1)
        metrics_grid.addWidget(lbl_current_name, 1, 0)
        metrics_grid.addWidget(self._val_current, 1, 1)
        metrics_grid.addWidget(lbl_max_name, 2, 0)
        metrics_grid.addWidget(self._val_max, 2, 1)
        metrics_grid.addWidget(lbl_min_name, 3, 0)
        metrics_grid.addWidget(self._val_min, 3, 1)

        # 时间窗口选择
        self._combo_time_window = QComboBox()
        self._combo_time_window.addItems(["10 s", "30 s", "60 s","All"])
        self._combo_time_window.setCurrentText("30 s")

        tw_bar = QHBoxLayout()
        tw_bar.addWidget(QLabel("时间窗"))
        tw_bar.addWidget(self._combo_time_window)
        tw_bar.addStretch()

        # 两条曲线图：白底黑字
        self._plot_total = pg.PlotWidget(title="总值-时间")
        self._setup_white_black_plot(self._plot_total)
        self._curve_total = self._plot_total.plot([], [], pen=pg.mkPen("k", width=2))

        self._plot_current = pg.PlotWidget(title="当前值-时间")
        self._setup_white_black_plot(self._plot_current)
        self._curve_current = self._plot_current.plot(
            [], [], pen=pg.mkPen("k", width=2, style=Qt.PenStyle.DashLine)
        )

        # 右栏布局
        right_panel = QVBoxLayout()
        right_panel.addWidget(right_title)
        right_panel.addLayout(metrics_grid)
        right_panel.addLayout(tw_bar)
        right_panel.addWidget(self._plot_total, stretch=1)
        right_panel.addWidget(self._plot_current, stretch=1)
        right_panel.setContentsMargins(8, 8, 8, 8)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)

        # ========== 主工作区：左右并列 ==========
        work_layout = QHBoxLayout()
        work_layout.addWidget(left_widget, stretch=3)
        work_layout.addWidget(right_widget, stretch=2)
        work_layout.setContentsMargins(0, 0, 0, 0)

        # ========== 底部单行状态栏 ==========
        self._statusbar = QStatusBar(self)
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("系统就绪")

        # ========== 主布局 ==========
        root = QVBoxLayout()
        root.addLayout(controls)
        root.addLayout(work_layout, stretch=1)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        # ========== 信号连接 ==========
        self._btn_connect.clicked.connect(self.on_connect_clicked)
        self._btn_disconnect.clicked.connect(self.on_disconnect_clicked)
        self._spin_gap.valueChanged.connect(self._heatmap.set_gap)

        self._combo_unit.currentTextChanged.connect(self.on_display_settings_changed)
        self._spin_decimals.valueChanged.connect(self.on_display_settings_changed)
        self._spin_ui_interval.valueChanged.connect(self.on_display_settings_changed)

        self._btn_refresh_ports.clicked.connect(self.on_refresh_ports_clicked)
        self._btn_serial_open.clicked.connect(self.on_serial_open_clicked)
        self._btn_serial_close.clicked.connect(self.on_serial_close_clicked)

        self._combo_time_window.currentTextChanged.connect(self.apply_plot_time_window)

        # UI 刷新定时器
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(self._spin_ui_interval.value())
        self._ui_timer.timeout.connect(self.refresh_ui)
        self._ui_timer.start()

        # 初始化
        self.apply_display_settings()
        self.on_refresh_ports_clicked()
        self.apply_plot_time_window()

        if not SERIAL_AVAILABLE:
            self._statusbar.showMessage("系统就绪 | 串口不可用(缺少 pyserial)")

    def _setup_white_black_plot(self, plot: pg.PlotWidget) -> None:
        """配置曲线图：白底、黑色坐标轴与文字"""
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.2)

        # 坐标轴黑色
        axis_left = plot.getAxis("left")
        axis_bottom = plot.getAxis("bottom")
        axis_left.setPen(pg.mkPen("k", width=1))
        axis_bottom.setPen(pg.mkPen("k", width=1))
        axis_left.setTextPen(pg.mkPen("k"))
        axis_bottom.setTextPen(pg.mkPen("k"))

        plot.setLabel("left", color="#000000")
        plot.setLabel("bottom", "Time(s)", color="#000000")

    def _get_time_window_seconds(self) -> Optional[float]:
        """根据选择的时间窗返回秒数；“截至当前”返回 None。"""
        text = self._combo_time_window.currentText().strip()
        if text.startswith("10"):
            return 10.0
        if text.startswith("60"):
            return 60.0
        if text.startswith("截至当前"):
            return None
        return 30.0

    def apply_plot_time_window(self) -> None:
        """应用时间窗口范围到两条曲线图。"""
        if len(self._time_hist) == 0:
            window = self._get_time_window_seconds()
            if window is None:
                window = 30.0
            self._plot_total.setXRange(0, window, padding=0.0)
            self._plot_current.setXRange(0, window, padding=0.0)
            return

        window = self._get_time_window_seconds()
        t_end = self._time_hist[-1]

        if window is None:
            t_start = 0.0
        else:
            t_start = max(0.0, t_end - window)

        self._plot_total.setXRange(t_start, t_end, padding=0.0)
        self._plot_current.setXRange(t_start, t_end, padding=0.0)


    # ========== Mock 逻辑 ==========
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

        # 新会话重置
        self._session_max = float("-inf")
        self._session_min = float("inf")
        self._time_hist.clear()
        self._total_hist.clear()
        self._current_hist.clear()
        self._t0 = time.time()

        self._thread.start()
        self._btn_connect.setEnabled(False)
        self._btn_disconnect.setEnabled(True)

    @pyqtSlot()
    def on_disconnect_clicked(self) -> None:
        self.stop_worker()

    @pyqtSlot(object)
    def on_frame_received(self, frame: object) -> None:
        if isinstance(frame, Frame):
            self._latest_frame = frame
            self._frame_count += 1

    @pyqtSlot(str)
    def on_status_changed(self, text: str) -> None:
        self._connected = (text == "状态：已连接")

    @pyqtSlot()
    def on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._connected = False
        self._btn_connect.setEnabled(True)
        self._btn_disconnect.setEnabled(False)

    def stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()

    # ========== 显示设置 ==========
    @pyqtSlot()
    def on_display_settings_changed(self) -> None:
        self.apply_display_settings()
        self.save_display_settings()

    def apply_display_settings(self) -> None:
        self._unit = self._combo_unit.currentText()
        self._decimals = self._spin_decimals.value()
        self._ui_interval_ms = self._spin_ui_interval.value()
        self._ui_timer.setInterval(self._ui_interval_ms)
        self.refresh_ui()

    def save_display_settings(self) -> None:
        self._settings.setValue("display/unit", self._unit)
        self._settings.setValue("display/decimals", self._decimals)
        self._settings.setValue("display/ui_interval_ms", self._ui_interval_ms)
        self._settings.sync()

    def format_value(self, value: float) -> str:
        """根据单位和小数位格式化数值"""
        factor = self.UNIT_FACTORS.get(self._unit, 1.0)
        converted = value * factor
        return f"{converted:.{self._decimals}f}"

    # ========== 串口字节读取 ==========
    @pyqtSlot()
    def on_refresh_ports_clicked(self) -> None:
        self._combo_ports.clear()
        if not SERIAL_AVAILABLE:
            return

        ports = SerialByteSource.list_ports()
        if not ports:
            self._combo_ports.addItem("无可用串口")
            self._btn_serial_open.setEnabled(False)
            return

        self._combo_ports.addItems(ports)
        self._btn_serial_open.setEnabled(True)

    @pyqtSlot()
    def on_serial_open_clicked(self) -> None:
        if self._serial_thread is not None or not SERIAL_AVAILABLE:
            return

        port = self._combo_ports.currentText().strip()
        if not port or port == "无可用串口":
            return

        try:
            baudrate = int(self._combo_baud.currentText())
        except ValueError:
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
        self._btn_serial_open.setEnabled(False)
        self._btn_serial_close.setEnabled(True)
        self._btn_refresh_ports.setEnabled(False)
        self._combo_ports.setEnabled(False)
        self._combo_baud.setEnabled(False)

        self._serial_thread.start()

    @pyqtSlot()
    def on_serial_close_clicked(self) -> None:
        self.stop_serial_worker()

    @pyqtSlot(bytes)
    def on_serial_bytes_received(self, data: bytes) -> None:
        self._serial_rx_total += len(data)

    @pyqtSlot(str)
    def on_serial_log_message(self, text: str) -> None:
        _ = text  # 当前界面不展开日志

    @pyqtSlot(str)
    def on_serial_status_changed(self, text: str) -> None:
        self._serial_state_text = text

    @pyqtSlot()
    def on_serial_thread_finished(self) -> None:
        self._serial_thread = None
        self._serial_worker = None

        self._btn_serial_open.setEnabled(SERIAL_AVAILABLE and self._combo_ports.currentText() != "无可用串口")
        self._btn_serial_close.setEnabled(False)
        self._btn_refresh_ports.setEnabled(SERIAL_AVAILABLE)
        self._combo_ports.setEnabled(True)
        self._combo_baud.setEnabled(True)

    def stop_serial_worker(self) -> None:
        if self._serial_worker is not None:
            self._serial_worker.stop()
        if self._serial_thread is not None:
            self._serial_thread.quit()
            self._serial_thread.wait()

    # ========== UI 刷新 ==========
    def refresh_ui(self) -> None:
        """
        主 UI 刷新循环（20Hz）：
        - 更新热力图
        - 更新统计数值
        - 更新曲线图
        - 更新状态栏
        """
        if not self._connected or self._latest_frame is None:
            self._statusbar.showMessage(
                f"设备: 未连接 | UI采样率: {self._ui_fps:.1f} Hz | 串口: {self._serial_state_text}"
            )
            return

        frame = self._latest_frame
        matrix = frame.matrix
        self._heatmap.set_matrix(matrix)

        # 计算统计量
        total_raw = float(np.sum(matrix))
        current_raw = float(np.max(matrix))  # 当前压力值定义为瞬时峰值

        self._session_max = max(self._session_max, current_raw)
        self._session_min = min(self._session_min, current_raw)

        # 累积历史数据
        t = time.time() - self._t0
        self._time_hist.append(t)
        self._total_hist.append(total_raw)
        self._current_hist.append(current_raw)

        # 更新曲线
        self._curve_total.setData(list(self._time_hist), list(self._total_hist))
        self._curve_current.setData(list(self._time_hist), list(self._current_hist))
        self.apply_plot_time_window()

        # 更新数值标签
        self._val_total.setText(f"{self.format_value(total_raw)} {self._unit}")
        self._val_current.setText(f"{self.format_value(current_raw)} {self._unit}")
        self._val_max.setText(f"{self.format_value(self._session_max)} {self._unit}")
        self._val_min.setText(f"{self.format_value(self._session_min)} {self._unit}")

        # 计算 UI 采样率
        now = time.time()
        if self._last_ui_time is not None:
            dt = now - self._last_ui_time
            if dt > 0:
                self._ui_fps = 1.0 / dt
        self._last_ui_time = now

        # 单行状态栏：关键信息
        self._statusbar.showMessage(
            f"设备: 已连接 | UI采样率: {self._ui_fps:.1f} Hz | 串口RX: {self._serial_rx_total} bytes | 串口: {self._serial_state_text}"
        )

    def closeEvent(self, event) -> None:
        self.stop_worker()
        self.stop_serial_worker()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())