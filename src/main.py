import codecs
import re
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QSettings, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPen, QPalette, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
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


@dataclass
class ParserStats:
    emitted_frames: int = 0
    accepted_rows: int = 0
    dropped_lines: int = 0
    resync_count: int = 0


_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


# ============ 工具函数：从日志读取传感器数据范围 ============
def load_sensor_range_from_logs(log_dir: str = "logs") -> tuple[float, float]:
    """
    【修复1】扫描日志文件夹，提取传感器数据的历史最小值和最大值
    用于初始化热力图的合理显示范围
    
    参数:
        log_dir: 日志文件夹路径
    
    返回:
        (min_value, max_value) 元组
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        # 若日志文件夹不存在，返回默认范围
        return 0.0, 300.0
    
    vmin = float("inf")
    vmax = float("-inf")
    
    # 扫描所有 .log 文件
    for log_file in log_path.glob("*.log"):
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # 日志格式: timestamp\trow_num float0 float1...
                    if "\t" not in line:
                        continue
                    parts = line.split("\t", 1)
                    if len(parts) < 2:
                        continue
                    
                    payload = parts[1]
                    # 提取所有浮点数
                    nums = []
                    for match in _FLOAT_RE.finditer(payload):
                        try:
                            nums.append(float(match.group()))
                        except ValueError:
                            pass
                    
                    # 跳过行号，只取16个值
                    if len(nums) >= 16:
                        # 若首个数是行号（1-16），则从索引1开始
                        if len(nums) == 17:
                            data_values = nums[1:17]
                        else:
                            data_values = nums[:16]
                        
                        for val in data_values:
                            vmin = min(vmin, val)
                            vmax = max(vmax, val)
        except Exception:
            pass
    
    # 若没有读到任何数据，返回默认范围
    if vmin == float("inf") or vmax == float("-inf"):
        return 0.0, 300.0
    
    # 返回 [min, max*1.1] 以留出余量
    return float(vmin), float(max(vmax * 1.1, 100.0))


class TextFrameParser:
    """
    文本流 -> 16x16 矩阵 的流式解析器
    - 支持半行/粘包
    - 支持 \n 和 \r\n
    - 行内稳健提取 float
    - 默认兼容"行号 + 16个浮点数"的格式
    """

    def __init__(
        self,
        rows: int = 16,
        cols: int = 16,
        encoding: str = "utf-8",
        errors: str = "ignore",
        max_consecutive_errors: int = 8,
        allow_leading_index: bool = True,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.encoding = encoding
        self.errors = errors
        self.max_consecutive_errors = max_consecutive_errors
        self.allow_leading_index = allow_leading_index

        self._decoder = codecs.getincrementaldecoder(self.encoding)(errors=self.errors)
        self._text_buffer = ""
        self._row_buffer: List[List[float]] = []
        self._consecutive_errors = 0
        self.stats = ParserStats()

    def reset(self) -> None:
        self._decoder = codecs.getincrementaldecoder(self.encoding)(errors=self.errors)
        self._text_buffer = ""
        self._row_buffer.clear()
        self._consecutive_errors = 0

    def feed(self, data: bytes) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        if not data:
            return frames

        decoded = self._decoder.decode(data, final=False)
        if decoded:
            self._text_buffer += decoded

        while "\n" in self._text_buffer:
            raw_line, self._text_buffer = self._text_buffer.split("\n", 1)
            line = raw_line.rstrip("\r")
            self._consume_line(line, frames)

        return frames

    def flush(self) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        tail = self._decoder.decode(b"", final=True)
        if tail:
            self._text_buffer += tail

        while "\n" in self._text_buffer:
            raw_line, self._text_buffer = self._text_buffer.split("\n", 1)
            line = raw_line.rstrip("\r")
            self._consume_line(line, frames)

        self._text_buffer = ""
        return frames

    def _consume_line(self, line: str, out_frames: List[np.ndarray]) -> None:
        if not line.strip():
            return

        row = self._parse_line_to_row(line)
        if row is None:
            self.stats.dropped_lines += 1
            self._consecutive_errors += 1

            if self._consecutive_errors >= self.max_consecutive_errors:
                self._row_buffer.clear()
                self._consecutive_errors = 0
                self.stats.resync_count += 1
            return

        self._consecutive_errors = 0
        self._row_buffer.append(row)
        self.stats.accepted_rows += 1

        if len(self._row_buffer) == self.rows:
            mat = np.asarray(self._row_buffer, dtype=np.float32)
            out_frames.append(mat)
            self._row_buffer.clear()
            self.stats.emitted_frames += 1

    def _parse_line_to_row(self, line: str) -> Optional[List[float]]:
        if "\t" in line:
            line = line.split("\t", 1)[1]

        line = line.strip().rstrip("|").strip()
        if not line:
            return None

        nums_str = _FLOAT_RE.findall(line)
        if not nums_str:
            return None

        try:
            nums = [float(x) for x in nums_str]
        except ValueError:
            return None

        if len(nums) == self.cols:
            return nums

        if self.allow_leading_index and len(nums) == self.cols + 1:
            first = nums[0]
            first_int = int(round(first))
            if abs(first - first_int) < 1e-6 and 1 <= first_int <= self.rows:
                return nums[1:]

        return None


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
    frame_received = pyqtSignal(object)
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
        self._parser = TextFrameParser(
            rows=16,
            cols=16,
            encoding="utf-8",
            errors="ignore",
            max_consecutive_errors=8,
            allow_leading_index=True,
        )
        self._frame_index = 0
        self._last_timeout_log_ts = 0.0
        self._read_cycle_count = 0
        self._last_parser_stats = None

    @pyqtSlot()
    def run(self) -> None:
        self._running = True

        try:
            self._source.open(self._port, self._baudrate, self._timeout)
            self.status_changed.emit("已打开")
            self.log_message.emit(f"串口已打开: {self._port}@{self._baudrate} | 编码: UTF-8")
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
                    matrices = self._parser.feed(data)
                    for matrix in matrices:
                        frame = Frame(
                            timestamp=time.time(),
                            index=self._frame_index,
                            matrix=matrix,
                            scenario="sensor",
                        )
                        self._frame_index += 1
                        self.frame_received.emit(frame)

                    self._read_cycle_count += 1
                    if self._read_cycle_count % 100 == 0:
                        stats = self._parser.stats
                        if self._last_parser_stats is None or (
                            stats.emitted_frames != self._last_parser_stats.emitted_frames
                        ):
                            self.log_message.emit(
                                f"[解析] frames={stats.emitted_frames} "
                                f"rows={stats.accepted_rows} "
                                f"dropped={stats.dropped_lines} "
                                f"resync={stats.resync_count}"
                            )
                            self._last_parser_stats = stats
                else:
                    now = time.time()
                    if now - self._last_timeout_log_ts >= 2.0:
                        self._last_timeout_log_ts = now
                        self.log_message.emit("[提示] 串口读取超时（可能无数据）")
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
        # 【修复1】初始化为合理范围，而不是 0-10
        self.vmin = 0.0
        self.vmax = 300.0
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

    def set_vrange(self, vmin: float, vmax: float) -> None:
        """【修复2】动态设置色彩范围 - 支持用户手动调整"""
        self.vmin = float(vmin)
        self.vmax = max(self.vmin + 0.5, float(vmax))
        self.update()

    def _build_jet_lut(self) -> np.ndarray:
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

        bg = self.palette().color(QPalette.ColorRole.Window)
        painter.fillRect(self.rect(), bg)

        w = self.width()
        h = self.height()
        gap = self.gap_px
        margin = self.margin_px

        avail_w = max(1, w - 2 * margin - (self.cols - 1) * gap)
        avail_h = max(1, h - 2 * margin - (self.rows - 1) * gap)

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

        border_pen = QPen(QColor(120, 120, 120, 120))
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(x0 - 1, y0 - 1, grid_w + 2, grid_h + 2)


class MainWindow(QMainWindow):
    """
    主窗口布局：
    - 顶部：串口与显示控制 + 【修复2】阈值调整控件
    - 中间：左右并列（左：热力图，右：数据分析）
    - 底部：单行状态栏
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hand FSR Demo - Sensor Mode")
        self.resize(1500, 900)

        self._settings = QSettings("NJIT", "HandFSR")

        self._serial_thread: Optional[QThread] = None
        self._serial_worker: Optional[SerialWorker] = None
        self._serial_is_running = False

        self._latest_frame: Optional[Frame] = None
        self._frame_count = 0
        self._data_fps = 0.0
        self._last_frame_arrival_time: Optional[float] = None
        self._ui_fps = 0.0
        self._last_ui_time: Optional[float] = None

        self._serial_rx_total = 0
        self._serial_state_text = "未打开"

        self._session_max = float("-inf")
        self._session_min = float("inf")

        self._t0 = time.time()
        self._time_hist = deque(maxlen=2400)
        self._total_hist = deque(maxlen=2400)
        self._current_hist = deque(maxlen=2400)

        self._decimals = self._settings.value("display/decimals", 1, type=int)
        self._ui_interval_ms = self._settings.value("display/ui_interval_ms", 50, type=int)

        self._data_source_text = "传感器"
        self._unit_text = "N"

        # 【修复1】从日志读取历史范围
        hist_vmin, hist_vmax = load_sensor_range_from_logs("logs")

        # ========== 顶部控制栏（扩展版）==========
        self._btn_refresh_ports = QPushButton("刷新串口")
        self._combo_ports = QComboBox()
        self._combo_baud = QComboBox()
        self._combo_baud.addItems(["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"])
        self._combo_baud.setCurrentText("9600")
        self._btn_serial_open = QPushButton("打开串口")
        self._btn_serial_close = QPushButton("关闭串口")
        self._btn_serial_close.setEnabled(False)

        self._spin_gap = QSpinBox()
        self._spin_gap.setRange(0, 8)
        self._spin_gap.setValue(2)

        self._spin_decimals = QSpinBox()
        self._spin_decimals.setRange(0, 6)
        self._spin_decimals.setValue(max(0, min(6, int(self._decimals))))

        self._spin_ui_interval = QSpinBox()
        self._spin_ui_interval.setRange(20, 100)
        self._spin_ui_interval.setValue(max(16, min(100, int(self._ui_interval_ms))))
        self._spin_ui_interval.setSuffix(" ms")

        # 【修复2】添加热力图阈值调整控件
        self._spin_vmin = QDoubleSpinBox()
        self._spin_vmin.setRange(-100.0, 1000.0)
        self._spin_vmin.setValue(hist_vmin)
        self._spin_vmin.setSingleStep(5.0)
        self._spin_vmin.setDecimals(1)

        self._spin_vmax = QDoubleSpinBox()
        self._spin_vmax.setRange(0.0, 1000.0)
        self._spin_vmax.setValue(hist_vmax)
        self._spin_vmax.setSingleStep(5.0)
        self._spin_vmax.setDecimals(1)

        controls = QHBoxLayout()
        controls.addWidget(self._btn_refresh_ports)
        controls.addWidget(QLabel("端口"))
        controls.addWidget(self._combo_ports)
        controls.addWidget(QLabel("波特率"))
        controls.addWidget(self._combo_baud)
        controls.addWidget(self._btn_serial_open)
        controls.addWidget(self._btn_serial_close)
        controls.addSpacing(12)
        controls.addWidget(QLabel("间隙"))
        controls.addWidget(self._spin_gap)
        controls.addWidget(QLabel("小数"))
        controls.addWidget(self._spin_decimals)
        controls.addWidget(QLabel("刷新"))
        controls.addWidget(self._spin_ui_interval)
        # 【修复2】添加热力图范围标签和控件
        controls.addWidget(QLabel("热力Min"))
        controls.addWidget(self._spin_vmin)
        controls.addWidget(QLabel("Max"))
        controls.addWidget(self._spin_vmax)
        controls.addWidget(QLabel("单位"))
        controls.addWidget(QLabel("N"))
        controls.addStretch()

        # ========== 左栏：热力图 ==========
        self._heatmap = HeatmapGridWidget(rows=16, cols=16, gap_px=2)
        # 【修复1】使用历史范围初始化热力图
        self._heatmap.set_vrange(hist_vmin, hist_vmax)

        left_title = QLabel("热力图（传感器）")
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

        right_title = QLabel("实时数据分析")
        right_title.setFont(title_font)

        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(12)
        metrics_grid.setVerticalSpacing(8)

        lbl_total_name = QLabel("总力 (N)")
        lbl_current_name = QLabel("当前峰值 (N)")
        lbl_max_name = QLabel("会话最大 (N)")
        lbl_min_name = QLabel("会话最小 (N)")

        for lbl in [lbl_total_name, lbl_current_name, lbl_max_name, lbl_min_name]:
            lbl.setFont(data_font)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._val_total = QLabel("0.0 N")
        self._val_current = QLabel("0.0 N")
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

        self._combo_time_window = QComboBox()
        self._combo_time_window.addItems(["10 s", "30 s", "60 s", "截至当前"])
        self._combo_time_window.setCurrentText("30 s")

        tw_bar = QHBoxLayout()
        tw_bar.addWidget(QLabel("时间窗"))
        tw_bar.addWidget(self._combo_time_window)
        tw_bar.addStretch()

        self._plot_total = pg.PlotWidget(title="总力 - 时间")
        self._setup_white_black_plot(self._plot_total)
        self._curve_total = self._plot_total.plot([], [], pen=pg.mkPen("k", width=2))

        self._plot_current = pg.PlotWidget(title="峰值 - 时间")
        self._setup_white_black_plot(self._plot_current)
        self._curve_current = self._plot_current.plot(
            [], [], pen=pg.mkPen("k", width=2, style=Qt.PenStyle.DashLine)
        )

        right_panel = QVBoxLayout()
        right_panel.addWidget(right_title)
        right_panel.addLayout(metrics_grid)
        right_panel.addLayout(tw_bar)
        right_panel.addWidget(self._plot_total, stretch=1)
        right_panel.addWidget(self._plot_current, stretch=1)
        right_panel.setContentsMargins(8, 8, 8, 8)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)

        work_layout = QHBoxLayout()
        work_layout.addWidget(left_widget, stretch=3)
        work_layout.addWidget(right_widget, stretch=2)
        work_layout.setContentsMargins(0, 0, 0, 0)

        # ========== 底部状态栏 ==========
        self._statusbar = QStatusBar(self)
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("系统就绪 | 数据源: 传感器")

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
        self._btn_refresh_ports.clicked.connect(self.on_refresh_ports_clicked)
        self._btn_serial_open.clicked.connect(self.on_serial_open_clicked)
        self._btn_serial_close.clicked.connect(self.on_serial_close_clicked)
        self._spin_gap.valueChanged.connect(self._heatmap.set_gap)
        self._spin_decimals.valueChanged.connect(self.on_display_settings_changed)
        self._spin_ui_interval.valueChanged.connect(self.on_display_settings_changed)
        self._combo_time_window.currentTextChanged.connect(self.apply_plot_time_window)
        # 【修复2】连接阈值控件信号
        self._spin_vmin.valueChanged.connect(self.on_vrange_changed)
        self._spin_vmax.valueChanged.connect(self.on_vrange_changed)

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
            self._statusbar.showMessage("系统就绪 | 数据源: 传感器 | 串口不可用(缺少 pyserial)")

    def _setup_white_black_plot(self, plot: pg.PlotWidget) -> None:
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.2)

        axis_left = plot.getAxis("left")
        axis_bottom = plot.getAxis("bottom")
        axis_left.setPen(pg.mkPen("k", width=1))
        axis_bottom.setPen(pg.mkPen("k", width=1))
        axis_left.setTextPen(pg.mkPen("k"))
        axis_bottom.setTextPen(pg.mkPen("k"))

        plot.setLabel("left", color="#000000")
        plot.setLabel("bottom", "Time(s)", color="#000000")

    def _get_time_window_seconds(self) -> Optional[float]:
        text = self._combo_time_window.currentText().strip()
        if text.startswith("10"):
            return 10.0
        if text.startswith("60"):
            return 60.0
        if text.startswith("截至当前"):
            return None
        return 30.0

    def apply_plot_time_window(self) -> None:
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

    @pyqtSlot()
    def on_refresh_ports_clicked(self) -> None:
        self._combo_ports.clear()
        if not SERIAL_AVAILABLE:
            self._combo_ports.addItem("pyserial 未装")
            self._btn_serial_open.setEnabled(False)
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
        if not port or port in ("pyserial 未装", "无可用串口"):
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
        self._serial_worker.frame_received.connect(self.on_frame_received)
        self._serial_worker.log_message.connect(self.on_serial_log_message)
        self._serial_worker.status_changed.connect(self.on_serial_status_changed)
        self._serial_worker.finished.connect(self._serial_thread.quit)
        self._serial_worker.finished.connect(self._serial_worker.deleteLater)
        self._serial_thread.finished.connect(self._serial_thread.deleteLater)
        self._serial_thread.finished.connect(self.on_serial_thread_finished)

        # 【修复3】重置所有与数据相关的状态
        self._serial_rx_total = 0
        self._frame_count = 0
        self._data_fps = 0.0
        self._last_frame_arrival_time = None
        self._session_max = float("-inf")
        self._session_min = float("inf")
        self._time_hist.clear()
        self._total_hist.clear()
        self._current_hist.clear()
        self._t0 = time.time()
        self._latest_frame = None  # 清除之前的帧
        self._serial_is_running = True

        self._btn_serial_open.setEnabled(False)
        self._btn_serial_close.setEnabled(True)
        self._btn_refresh_ports.setEnabled(False)
        self._combo_ports.setEnabled(False)
        self._combo_baud.setEnabled(False)

        self._statusbar.showMessage(f"系统就绪 | 数据源: {self._data_source_text} | 串口: 打开中...")

        self._serial_thread.start()

    @pyqtSlot()
    def on_serial_close_clicked(self) -> None:
        # 【修复3】立即标记为运行停止，阻止新数据写入
        self._serial_is_running = False
        self.stop_serial_worker()

    @pyqtSlot(bytes)
    def on_serial_bytes_received(self, data: bytes) -> None:
        self._serial_rx_total += len(data)

    @pyqtSlot(object)
    def on_frame_received(self, frame: object) -> None:
        # 【修复3】只有当 _serial_is_running 为 True 时才接受帧
        if isinstance(frame, Frame) and self._serial_is_running:
            self._latest_frame = frame
            self._frame_count += 1

            now = time.time()
            if self._last_frame_arrival_time is not None:
                dt = now - self._last_frame_arrival_time
                if dt > 0:
                    self._data_fps = 1.0 / dt
            self._last_frame_arrival_time = now

    @pyqtSlot(str)
    def on_serial_log_message(self, text: str) -> None:
        _ = text

    @pyqtSlot(str)
    def on_serial_status_changed(self, text: str) -> None:
        self._serial_state_text = text

    @pyqtSlot()
    def on_serial_thread_finished(self) -> None:
        self._serial_thread = None
        self._serial_worker = None
        # 【修复3】确保已设置为 False（由 on_serial_close_clicked 设置）
        self._serial_is_running = False
        self._latest_frame = None  # 清除最后一帧

        self._btn_serial_open.setEnabled(
            SERIAL_AVAILABLE and self._combo_ports.currentText() not in ("pyserial 未装", "无可用串口")
        )
        self._btn_serial_close.setEnabled(False)
        self._btn_refresh_ports.setEnabled(SERIAL_AVAILABLE)
        self._combo_ports.setEnabled(True)
        self._combo_baud.setEnabled(True)

    def stop_serial_worker(self) -> None:
        """【修复3】安全停止串口工作线程"""
        if self._serial_worker is not None:
            self._serial_worker.stop()
        if self._serial_thread is not None:
            self._serial_thread.quit()
            self._serial_thread.wait()

    @pyqtSlot()
    def on_display_settings_changed(self) -> None:
        self.apply_display_settings()
        self.save_display_settings()

    @pyqtSlot()
    def on_vrange_changed(self) -> None:
        """【修复2】热力图阈值改变时的处理"""
        vmin = self._spin_vmin.value()
        vmax = self._spin_vmax.value()
        # 确保 vmin < vmax
        if vmin >= vmax:
            vmax = vmin + 0.5
            self._spin_vmax.blockSignals(True)
            self._spin_vmax.setValue(vmax)
            self._spin_vmax.blockSignals(False)
        self._heatmap.set_vrange(vmin, vmax)

    def apply_display_settings(self) -> None:
        self._decimals = self._spin_decimals.value()
        self._ui_interval_ms = self._spin_ui_interval.value()
        self._ui_timer.setInterval(self._ui_interval_ms)
        self.refresh_ui()

    def save_display_settings(self) -> None:
        self._settings.setValue("display/decimals", self._decimals)
        self._settings.setValue("display/ui_interval_ms", self._ui_interval_ms)
        self._settings.sync()

    def format_value(self, value: float) -> str:
        return f"{value:.{self._decimals}f}"

    def refresh_ui(self) -> None:
        """【修复3】改进 UI 刷新逻辑，确保数据流完全停止"""
        if not self._serial_state_text:
            self._serial_state_text = "未打开"

        # 【修复3】若串口已关闭且无最新帧，只更新状态栏
        if not self._serial_is_running or self._latest_frame is None:
            self._statusbar.showMessage(
                f"数据源: {self._data_source_text} | 状态: {self._serial_state_text} | "
                f"数据FPS: {self._data_fps:.1f} | 已收帧: {self._frame_count} | 串口RX: {self._serial_rx_total} bytes"
            )
            return

        frame = self._latest_frame
        matrix = frame.matrix
        self._heatmap.set_matrix(matrix)

        total_raw = float(np.sum(matrix))
        current_raw = float(np.max(matrix))

        self._session_max = max(self._session_max, current_raw)
        self._session_min = min(self._session_min, current_raw)

        # 【修复3】仅当 _serial_is_running 为 True 时才累积数据到曲线
        if self._serial_is_running:
            t = time.time() - self._t0
            self._time_hist.append(t)
            self._total_hist.append(total_raw)
            self._current_hist.append(current_raw)

        self._curve_total.setData(list(self._time_hist), list(self._total_hist))
        self._curve_current.setData(list(self._time_hist), list(self._current_hist))
        self.apply_plot_time_window()

        self._val_total.setText(f"{self.format_value(total_raw)} {self._unit_text}")
        self._val_current.setText(f"{self.format_value(current_raw)} {self._unit_text}")
        self._val_max.setText(f"{self.format_value(self._session_max)} {self._unit_text}")
        self._val_min.setText(f"{self.format_value(self._session_min)} {self._unit_text}")

        now = time.time()
        if self._last_ui_time is not None:
            dt = now - self._last_ui_time
            if dt > 0:
                self._ui_fps = 1.0 / dt
        self._last_ui_time = now

        self._statusbar.showMessage(
            f"数据源: {self._data_source_text} | 状态: {self._serial_state_text} | "
            f"数据FPS: {self._data_fps:.1f} | UI FPS: {self._ui_fps:.1f} | "
            f"已收帧: {self._frame_count} | 串口RX: {self._serial_rx_total} bytes"
        )

    def closeEvent(self, event) -> None:
        self.stop_serial_worker()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())