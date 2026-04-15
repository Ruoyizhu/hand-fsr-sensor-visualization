import codecs
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QSettings, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPen, QFont
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


def _candidate_log_dirs(log_dir: str = "logs") -> List[Path]:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(log_dir),
        script_dir / log_dir,
        script_dir / "logs",
        script_dir.parent / "logs",
        script_dir.parent / "src" / "logs",
    ]

    result: List[Path] = []
    seen = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp)
        if key not in seen and rp.exists() and rp.is_dir():
            seen.add(key)
            result.append(rp)
    return result


def load_sensor_mapping_from_logs(log_dir: str = "logs") -> tuple[float, float]:
    """
    从历史日志估计原始量程，返回 (raw_min, raw_max)。
    使用分位数，降低极端值影响。
    """
    values: List[float] = []

    for root in _candidate_log_dirs(log_dir):
        for log_file in root.glob("*.log"):
            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "\t" not in line:
                            continue
                        payload = line.split("\t", 1)[1]
                        nums_str = _FLOAT_RE.findall(payload)
                        if len(nums_str) < 16:
                            continue

                        nums: List[float] = []
                        for s in nums_str:
                            try:
                                nums.append(float(s))
                            except ValueError:
                                pass

                        if len(nums) == 17:
                            row = nums[1:17]
                        elif len(nums) >= 16:
                            row = nums[:16]
                        else:
                            continue

                        # 硬件故障补偿：第3列 <- 第4列
                        row[2] = row[3]
                        values.extend(row)
            except Exception:
                pass

    if not values:
        return 0.0, 300.0

    arr = np.asarray(values, dtype=np.float32)
    p2 = float(np.percentile(arr, 2))
    p98 = float(np.percentile(arr, 98))

    raw_min = max(0.0, p2)
    raw_max = max(raw_min + 1.0, p98 * 1.1)
    return raw_min, raw_max


class TextFrameParser:
    """
    文本流 -> 16x16矩阵
    支持半行/粘包、行号+16值格式、坏行丢弃重同步。
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

        # 用 find 逐行提取，减少 split 大量临时对象
        start = 0
        while True:
            idx = self._text_buffer.find("\n", start)
            if idx < 0:
                break
            raw_line = self._text_buffer[:idx]
            self._text_buffer = self._text_buffer[idx + 1 :]
            line = raw_line.rstrip("\r")
            self._consume_line(line, frames)
            start = 0

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
    def __init__(self) -> None:
        self._ser = None

    @staticmethod
    def list_ports() -> list[str]:
        if not SERIAL_AVAILABLE:
            return []
        return [p.device for p in serial.tools.list_ports.comports()]

    def open(self, port: str, baudrate: int, timeout: float = 0.0) -> None:
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial 未安装")
        self._ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,  # 非阻塞读取
            write_timeout=timeout if timeout > 0 else 0.1,
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

    def read_latest_bytes(self, max_chunk: int = 32768, backlog_limit: int = 131072) -> tuple[bytes, bool]:
        """
        返回一个尽量新的数据块，必要时主动丢弃过大积压，避免显示延迟。
        """
        if not self.is_open():
            return b"", False

        dropped = False
        waiting = self._ser.in_waiting if self._ser is not None else 0

        if waiting > backlog_limit:
            try:
                self._ser.reset_input_buffer()
                return b"", True
            except Exception:
                pass

        if waiting <= 0:
            return b"", False

        size = min(waiting, max_chunk)
        data = self._ser.read(size)

        waiting_after = self._ser.in_waiting if self._ser is not None else 0
        if waiting_after > backlog_limit // 2:
            try:
                self._ser.reset_input_buffer()
                dropped = True
            except Exception:
                pass

        return data, dropped


class SerialWorker(QObject):
    bytes_stat = pyqtSignal(int)
    frame_received = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, port: str, baudrate: int, timeout: float = 0.0) -> None:
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

        # 实时性优化：投递频率控制在接近UI频率，减少Qt队列堆积导致的延迟
        self._emit_interval_s = 1.0 / 50.0
        self._last_emit_ts = 0.0
        self._latest_frame_cache: Optional[Frame] = None

        # 批量bytes统计，减少信号开销
        self._bytes_acc = 0
        self._last_bytes_emit_ts = 0.0
        self._bytes_emit_interval_s = 0.1

    @pyqtSlot()
    def run(self) -> None:
        self._running = True
        try:
            self._source.open(self._port, self._baudrate, self._timeout)
            self.status_changed.emit("已打开")
        except Exception:
            self.status_changed.emit("打开失败")
            self.finished.emit()
            return

        try:
            while self._running:
                try:
                    data, dropped = self._source.read_latest_bytes(
                        max_chunk=32768,
                        backlog_limit=131072,
                    )
                except Exception:
                    self.status_changed.emit("读取异常")
                    break

                now = time.perf_counter()

                if dropped:
                    self._parser.reset()
                    self._latest_frame_cache = None

                if data:
                    self._bytes_acc += len(data)

                    matrices = self._parser.feed(data)
                    if matrices:
                        # 丢旧保新：同一批只保留最新矩阵
                        last_mat = matrices[-1]
                        self._frame_index += len(matrices) - 1
                        self._latest_frame_cache = Frame(
                            timestamp=time.time(),
                            index=self._frame_index,
                            matrix=last_mat,
                            scenario="sensor",
                        )
                        self._frame_index += 1

                # 限频发帧：始终发最新帧，避免排队
                if self._latest_frame_cache is not None and (now - self._last_emit_ts) >= self._emit_interval_s:
                    self.frame_received.emit(self._latest_frame_cache)
                    self._latest_frame_cache = None
                    self._last_emit_ts = now

                # 批量发送bytes统计
                if self._bytes_acc > 0 and (now - self._last_bytes_emit_ts) >= self._bytes_emit_interval_s:
                    self.bytes_stat.emit(self._bytes_acc)
                    self._bytes_acc = 0
                    self._last_bytes_emit_ts = now

                # 无数据时短暂休眠，避免CPU满载
                if not data:
                    time.sleep(0.0005)

        finally:
            try:
                self._source.close()
            except Exception:
                pass
            if self._bytes_acc > 0:
                self.bytes_stat.emit(self._bytes_acc)
            self.status_changed.emit("已关闭")
            self.finished.emit()

    @pyqtSlot()
    def stop(self) -> None:
        self._running = False


class HeatmapGridWidget(QWidget):
    def __init__(self, rows: int = 16, cols: int = 16, gap_px: int = 2) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.gap_px = gap_px
        self.margin_px = 12
        self.vmin = 0.0
        self.vmax = 20.0
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
        self.vmin = float(vmin)
        self.vmax = max(self.vmin + 0.1, float(vmax))
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
        painter.fillRect(self.rect(), self.palette().window())

        w, h = self.width(), self.height()
        gap, margin = self.gap_px, self.margin_px
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
                v = float(self._matrix[r, c])
                painter.setBrush(self._value_to_color(v))
                x = x0 + c * (cell + gap)
                y = y0 + r * (cell + gap)
                painter.drawRect(x, y, cell, cell)

        pen = QPen(QColor(120, 120, 120, 120))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(x0 - 1, y0 - 1, grid_w + 2, grid_h + 2)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hand FSR Demo")
        self.resize(1500, 900)

        self._settings = QSettings("NJIT", "HandFSR")

        self._serial_thread: Optional[QThread] = None
        self._serial_worker: Optional[SerialWorker] = None
        self._stream_active = False

        # 最新帧邮箱：只保留最近一帧，避免旧帧排队
        self._pending_frame: Optional[Frame] = None
        self._has_pending_frame = False
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

        self._time_hist = deque(maxlen=2400)
        self._total_hist = deque(maxlen=2400)
        self._current_hist = deque(maxlen=2400)
        self._plot_t0_sensor_ts: Optional[float] = None
        self._last_plot_update_ts = 0.0
        self._plot_update_interval_s = 1.0 / 12.0  # 曲线12Hz，减轻负载
        self._last_plotted_count = 0

        self._decimals = self._settings.value("display/decimals", 1, type=int)
        self._ui_interval_ms = self._settings.value("display/ui_interval_ms", 20, type=int)
        self._unit_text = "N"

        self._raw_min, self._raw_max = load_sensor_mapping_from_logs("logs")
        self._force_max_n = 20.0

        self._zero_offset = np.zeros((16, 16), dtype=np.float32)
        self._calib_sum = np.zeros((16, 16), dtype=np.float64)
        self._calib_count = 0
        self._calib_active = False
        self._calib_need_frames = 10

        # 状态栏降频刷新，减少字符串拼接和UI更新开销
        self._last_status_update_ts = 0.0
        self._status_update_interval_s = 0.2

        self._btn_refresh_ports = QPushButton("刷新串口")
        self._combo_ports = QComboBox()
        self._combo_baud = QComboBox()
        self._combo_baud.addItems(["9600", "19200", "38400", "57600", "115200"])
        self._combo_baud.setCurrentText("9600")
        self._btn_open = QPushButton("打开串口")
        self._btn_close = QPushButton("关闭串口")
        self._btn_close.setEnabled(False)
        self._btn_zero = QPushButton("清零")

        self._spin_gap = QSpinBox()
        self._spin_gap.setRange(0, 8)
        self._spin_gap.setValue(2)

        self._spin_decimals = QSpinBox()
        self._spin_decimals.setRange(0, 6)
        self._spin_decimals.setValue(max(0, min(6, int(self._decimals))))

        self._spin_ui_interval = QSpinBox()
        self._spin_ui_interval.setRange(20, 100)
        self._spin_ui_interval.setValue(max(20, min(100, int(self._ui_interval_ms))))
        self._spin_ui_interval.setSuffix(" ms")

        saved_vmin = self._settings.value("display/vmin", 0.0, type=float)
        saved_vmax = self._settings.value("display/vmax", 20.0, type=float)

        self._spin_vmin = QDoubleSpinBox()
        self._spin_vmin.setRange(0.0, 100.0)
        self._spin_vmin.setDecimals(2)
        self._spin_vmin.setSingleStep(0.5)
        self._spin_vmin.setValue(saved_vmin)

        self._spin_vmax = QDoubleSpinBox()
        self._spin_vmax.setRange(0.1, 100.0)
        self._spin_vmax.setDecimals(2)
        self._spin_vmax.setSingleStep(0.5)
        self._spin_vmax.setValue(saved_vmax)

        self._heatmap = HeatmapGridWidget(rows=16, cols=16, gap_px=2)
        self._heatmap.set_vrange(self._spin_vmin.value(), self._spin_vmax.value())

        self._plot_total = pg.PlotWidget(title="总力 - 时间")
        self._plot_current = pg.PlotWidget(title="峰值 - 时间")
        self._setup_plot(self._plot_total)
        self._setup_plot(self._plot_current)
        self._curve_total = self._plot_total.plot([], [], pen=pg.mkPen("k", width=2))
        self._curve_current = self._plot_current.plot([], [], pen=pg.mkPen("k", width=2, style=Qt.PenStyle.DashLine))

        self._combo_time_window = QComboBox()
        self._combo_time_window.addItems(["10 s", "30 s", "60 s", "截至当前"])
        self._combo_time_window.setCurrentText("30 s")

        self._val_total = QLabel("0.0 N")
        self._val_current = QLabel("0.0 N")
        self._val_max = QLabel("-")
        self._val_min = QLabel("-")
        self._make_value_labels()

        controls = QHBoxLayout()
        controls.addWidget(self._btn_refresh_ports)
        controls.addWidget(QLabel("端口"))
        controls.addWidget(self._combo_ports)
        controls.addWidget(QLabel("波特率"))
        controls.addWidget(self._combo_baud)
        controls.addWidget(self._btn_open)
        controls.addWidget(self._btn_close)
        controls.addWidget(self._btn_zero)
        controls.addSpacing(12)
        controls.addWidget(QLabel("间隙"))
        controls.addWidget(self._spin_gap)
        controls.addWidget(QLabel("小数"))
        controls.addWidget(self._spin_decimals)
        controls.addWidget(QLabel("刷新"))
        controls.addWidget(self._spin_ui_interval)
        controls.addWidget(QLabel("热力Min"))
        controls.addWidget(self._spin_vmin)
        controls.addWidget(QLabel("Max"))
        controls.addWidget(self._spin_vmax)
        controls.addWidget(QLabel("单位 N"))
        controls.addStretch()

        left = QVBoxLayout()
        left_title = QLabel("热力图（传感器）")
        f = QFont()
        f.setBold(True)
        f.setPointSize(11)
        left_title.setFont(f)
        left.addWidget(left_title)
        left.addWidget(self._heatmap, stretch=1)
        left_w = QWidget()
        left_w.setLayout(left)

        metrics = QGridLayout()
        metrics.addWidget(QLabel("总力 (N)"), 0, 0)
        metrics.addWidget(self._val_total, 0, 1)
        metrics.addWidget(QLabel("当前峰值 (N)"), 1, 0)
        metrics.addWidget(self._val_current, 1, 1)
        metrics.addWidget(QLabel("会话最大 (N)"), 2, 0)
        metrics.addWidget(self._val_max, 2, 1)
        metrics.addWidget(QLabel("会话最小 (N)"), 3, 0)
        metrics.addWidget(self._val_min, 3, 1)

        tw = QHBoxLayout()
        tw.addWidget(QLabel("时间窗"))
        tw.addWidget(self._combo_time_window)
        tw.addStretch()

        right = QVBoxLayout()
        right_title = QLabel("实时数据分析")
        right_title.setFont(f)
        right.addWidget(right_title)
        right.addLayout(metrics)
        right.addLayout(tw)
        right.addWidget(self._plot_total, stretch=1)
        right.addWidget(self._plot_current, stretch=1)
        right_w = QWidget()
        right_w.setLayout(right)

        work = QHBoxLayout()
        work.addWidget(left_w, stretch=3)
        work.addWidget(right_w, stretch=2)

        root = QVBoxLayout()
        root.addLayout(controls)
        root.addLayout(work, stretch=1)
        root_w = QWidget()
        root_w.setLayout(root)
        self.setCentralWidget(root_w)

        self._statusbar = QStatusBar(self)
        self.setStatusBar(self._statusbar)

        self._btn_refresh_ports.clicked.connect(self.on_refresh_ports_clicked)
        self._btn_open.clicked.connect(self.on_open_clicked)
        self._btn_close.clicked.connect(self.on_close_clicked)
        self._btn_zero.clicked.connect(self.on_zero_clicked)
        self._spin_gap.valueChanged.connect(self._heatmap.set_gap)
        self._spin_decimals.valueChanged.connect(self.on_display_settings_changed)
        self._spin_ui_interval.valueChanged.connect(self.on_display_settings_changed)
        self._spin_vmin.valueChanged.connect(self.on_vrange_changed)
        self._spin_vmax.valueChanged.connect(self.on_vrange_changed)
        self._combo_time_window.currentTextChanged.connect(self.apply_plot_time_window)

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(self._spin_ui_interval.value())
        self._ui_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._ui_timer.timeout.connect(self.refresh_ui)
        self._ui_timer.start()

        self.apply_display_settings()
        self.on_refresh_ports_clicked()
        self.apply_plot_time_window()
        self._update_status("系统就绪", force=True)

    def _make_value_labels(self) -> None:
        for v in [self._val_total, self._val_current, self._val_max, self._val_min]:
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            v.setStyleSheet("QLabel { background: white; border: 1px solid #BFBFBF; padding: 4px 8px; }")
            v.setMinimumWidth(140)

    def _setup_plot(self, plot: pg.PlotWidget) -> None:
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setClipToView(True)
        plot.setDownsampling(auto=True, mode="peak")
        ax_l = plot.getAxis("left")
        ax_b = plot.getAxis("bottom")
        ax_l.setPen(pg.mkPen("k", width=1))
        ax_b.setPen(pg.mkPen("k", width=1))
        ax_l.setTextPen(pg.mkPen("k"))
        ax_b.setTextPen(pg.mkPen("k"))
        plot.setLabel("bottom", "Time(s)", color="#000000")

    def _update_status(self, prefix: str = "", force: bool = False) -> None:
        now = time.perf_counter()
        if (not force) and (now - self._last_status_update_ts < self._status_update_interval_s):
            return

        msg = (
            f"{prefix} | 数据源: 传感器 | 状态: {self._serial_state_text} | "
            f"已收帧: {self._frame_count} | 数据FPS: {self._data_fps:.1f} | "
            f"UI FPS: {self._ui_fps:.1f} | 串口RX: {self._serial_rx_total} bytes"
        )
        self._statusbar.showMessage(msg)
        self._last_status_update_ts = now

    def _get_time_window_seconds(self) -> Optional[float]:
        t = self._combo_time_window.currentText().strip()
        if t.startswith("10"):
            return 10.0
        if t.startswith("60"):
            return 60.0
        if t.startswith("截至当前"):
            return None
        return 30.0

    def apply_plot_time_window(self) -> None:
        if len(self._time_hist) == 0:
            w = self._get_time_window_seconds()
            if w is None:
                w = 30.0
            self._plot_total.setXRange(0, w, padding=0.0)
            self._plot_current.setXRange(0, w, padding=0.0)
            return

        t_end = self._time_hist[-1]
        w = self._get_time_window_seconds()
        t_start = 0.0 if w is None else max(0.0, t_end - w)
        self._plot_total.setXRange(t_start, t_end, padding=0.0)
        self._plot_current.setXRange(t_start, t_end, padding=0.0)

    def _build_correction(self, matrix: np.ndarray) -> np.ndarray:
        out = matrix.astype(np.float32, copy=True)
        if out.shape == (16, 16):
            out[:, 2] = out[:, 3]
        return out

    def _apply_zero_offset(self, matrix: np.ndarray) -> np.ndarray:
        out = matrix - self._zero_offset
        return np.maximum(out, 0.0).astype(np.float32)

    def _map_to_force_n(self, matrix: np.ndarray) -> np.ndarray:
        span = max(1e-6, self._raw_max - self._raw_min)
        norm = (matrix - self._raw_min) / span
        norm = np.clip(norm, 0.0, 1.0)
        mapped = norm * self._force_max_n
        return mapped.astype(np.float32)

    def _transform_for_heatmap_layout(self, matrix: np.ndarray) -> np.ndarray:
        """
        物理布局 -> 热力图布局坐标变换
        满足：
        实际左上 -> 图右下
        实际右上 -> 图右上
        实际左下 -> 图左下
        实际右下 -> 图左上

        实现方式：先上下+左右翻转，再转置（等价于沿副对角线镜像）
        """
        return np.flipud(np.fliplr(matrix)).T.astype(np.float32, copy=False)

    def _start_calibration(self) -> None:
        self._calib_sum = np.zeros((16, 16), dtype=np.float64)
        self._calib_count = 0
        self._calib_active = True
        self._zero_offset = np.zeros((16, 16), dtype=np.float32)
        self._update_status("正在自动标定零点", force=True)

    def _finish_calibration(self) -> None:
        if self._calib_count > 0:
            self._zero_offset = (self._calib_sum / float(self._calib_count)).astype(np.float32)
        else:
            self._zero_offset = np.zeros((16, 16), dtype=np.float32)
        self._calib_active = False
        self._update_status("标定完成", force=True)

    def _clear_runtime_buffers(self) -> None:
        self._pending_frame = None
        self._has_pending_frame = False
        self._latest_frame = None

        self._frame_count = 0
        self._data_fps = 0.0
        self._last_frame_arrival_time = None

        self._session_max = float("-inf")
        self._session_min = float("inf")

        self._time_hist.clear()
        self._total_hist.clear()
        self._current_hist.clear()
        self._plot_t0_sensor_ts = None
        self._last_plot_update_ts = 0.0
        self._last_plotted_count = 0

        self._curve_total.setData([], [])
        self._curve_current.setData([], [])

        self._heatmap.set_matrix(np.zeros((16, 16), dtype=np.float32))
        self._val_total.setText("0.0 N")
        self._val_current.setText("0.0 N")
        self._val_max.setText("-")
        self._val_min.setText("-")

    @pyqtSlot()
    def on_refresh_ports_clicked(self) -> None:
        self._combo_ports.clear()
        if not SERIAL_AVAILABLE:
            self._combo_ports.addItem("pyserial 未装")
            self._btn_open.setEnabled(False)
            return
        ports = SerialByteSource.list_ports()
        if not ports:
            self._combo_ports.addItem("无可用串口")
            self._btn_open.setEnabled(False)
            return
        self._combo_ports.addItems(ports)
        self._btn_open.setEnabled(True)

    @pyqtSlot()
    def on_open_clicked(self) -> None:
        if self._serial_thread is not None or not SERIAL_AVAILABLE:
            return

        port = self._combo_ports.currentText().strip()
        if not port or port in ("pyserial 未装", "无可用串口"):
            return

        try:
            baud = int(self._combo_baud.currentText())
        except ValueError:
            return

        self._serial_thread = QThread()
        self._serial_worker = SerialWorker(port=port, baudrate=baud, timeout=0.0)
        self._serial_worker.moveToThread(self._serial_thread)

        self._serial_thread.started.connect(self._serial_worker.run)
        self._serial_worker.bytes_stat.connect(self.on_serial_bytes_stat)
        self._serial_worker.frame_received.connect(self.on_frame_received)
        self._serial_worker.status_changed.connect(self.on_serial_status_changed)
        self._serial_worker.finished.connect(self._serial_thread.quit)
        self._serial_worker.finished.connect(self._serial_worker.deleteLater)
        self._serial_thread.finished.connect(self._serial_thread.deleteLater)
        self._serial_thread.finished.connect(self.on_serial_thread_finished)

        self._stream_active = True
        self._serial_rx_total = 0
        self._clear_runtime_buffers()
        self._start_calibration()

        self._btn_open.setEnabled(False)
        self._btn_close.setEnabled(True)
        self._btn_refresh_ports.setEnabled(False)
        self._combo_ports.setEnabled(False)
        self._combo_baud.setEnabled(False)

        self._serial_thread.start()
        self._update_status("打开串口中", force=True)

    @pyqtSlot()
    def on_close_clicked(self) -> None:
        self._stream_active = False
        self._calib_active = False
        self._pending_frame = None
        self._has_pending_frame = False
        self.stop_serial_worker()
        self._update_status("串口已关闭", force=True)

    @pyqtSlot()
    def on_zero_clicked(self) -> None:
        """
        清零逻辑：
        - 串口打开：重新标定 + 清空曲线
        - 串口关闭：清空显示与曲线
        """
        if self._stream_active:
            self._clear_runtime_buffers()
            self._start_calibration()
            self._update_status("手动清零：重新标定中", force=True)
        else:
            self._clear_runtime_buffers()
            self._zero_offset = np.zeros((16, 16), dtype=np.float32)
            self._update_status("手动清零：已重置", force=True)

    @pyqtSlot(int)
    def on_serial_bytes_stat(self, byte_count: int) -> None:
        self._serial_rx_total += int(byte_count)

    @pyqtSlot(object)
    def on_frame_received(self, frame: object) -> None:
        if not self._stream_active:
            return
        if not isinstance(frame, Frame):
            return

        now = time.time()
        if self._last_frame_arrival_time is not None:
            dt = now - self._last_frame_arrival_time
            if dt > 0:
                self._data_fps = 1.0 / dt
        self._last_frame_arrival_time = now

        # 只做轻量入队，重计算延后到UI定时器里处理最新帧
        self._pending_frame = frame
        self._has_pending_frame = True

    @pyqtSlot(str)
    def on_serial_status_changed(self, text: str) -> None:
        self._serial_state_text = text

    @pyqtSlot()
    def on_serial_thread_finished(self) -> None:
        self._serial_thread = None
        self._serial_worker = None
        self._btn_open.setEnabled(
            SERIAL_AVAILABLE and self._combo_ports.currentText() not in ("pyserial 未装", "无可用串口")
        )
        self._btn_close.setEnabled(False)
        self._btn_refresh_ports.setEnabled(SERIAL_AVAILABLE)
        self._combo_ports.setEnabled(True)
        self._combo_baud.setEnabled(True)

    def stop_serial_worker(self) -> None:
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
        vmin = self._spin_vmin.value()
        vmax = self._spin_vmax.value()
        if vmin >= vmax:
            vmax = vmin + 0.1
            self._spin_vmax.blockSignals(True)
            self._spin_vmax.setValue(vmax)
            self._spin_vmax.blockSignals(False)
        self._heatmap.set_vrange(vmin, vmax)
        self.save_display_settings()

    def apply_display_settings(self) -> None:
        self._decimals = self._spin_decimals.value()
        self._ui_interval_ms = self._spin_ui_interval.value()
        self._ui_timer.setInterval(self._ui_interval_ms)

    def save_display_settings(self) -> None:
        self._settings.setValue("display/decimals", self._decimals)
        self._settings.setValue("display/ui_interval_ms", self._ui_interval_ms)
        self._settings.setValue("display/vmin", self._spin_vmin.value())
        self._settings.setValue("display/vmax", self._spin_vmax.value())
        self._settings.sync()

    def _fmt(self, v: float) -> str:
        return f"{v:.{self._decimals}f}"

    def refresh_ui(self) -> None:
        # 高频刷新：只消费最新待处理帧
        if self._stream_active and self._has_pending_frame and self._pending_frame is not None:
            raw_frame = self._pending_frame
            self._pending_frame = None
            self._has_pending_frame = False

            corrected = self._build_correction(raw_frame.matrix)

            if self._calib_active:
                self._calib_sum += corrected.astype(np.float64)
                self._calib_count += 1
                if self._calib_count >= self._calib_need_frames:
                    self._finish_calibration()
            else:
                zero_fixed = self._apply_zero_offset(corrected)
                mapped = self._map_to_force_n(zero_fixed)

                # 关键：绘制前坐标变换，使热力图方向与物理传感器一致
                display_matrix = self._transform_for_heatmap_layout(mapped)
                self._heatmap.set_matrix(display_matrix)

                total_n = float(np.sum(mapped))
                peak_n = float(np.max(mapped))

                self._session_max = max(self._session_max, peak_n)
                self._session_min = min(self._session_min, peak_n)

                if self._plot_t0_sensor_ts is None:
                    self._plot_t0_sensor_ts = raw_frame.timestamp
                t = raw_frame.timestamp - self._plot_t0_sensor_ts

                self._time_hist.append(t)
                self._total_hist.append(total_n)
                self._current_hist.append(peak_n)

                self._frame_count += 1
                self._val_total.setText(f"{self._fmt(total_n)} {self._unit_text}")
                self._val_current.setText(f"{self._fmt(peak_n)} {self._unit_text}")
                self._val_max.setText(f"{self._fmt(self._session_max)} {self._unit_text}")
                self._val_min.setText(f"{self._fmt(self._session_min)} {self._unit_text}")

                self._latest_frame = Frame(
                    timestamp=raw_frame.timestamp,
                    index=raw_frame.index,
                    matrix=mapped,
                    scenario="sensor",
                )

        # 低频刷新：曲线（减少CPU占用），且只有新增数据才重绘
        now = time.perf_counter()
        hist_len = len(self._time_hist)
        if (
            hist_len > 0
            and hist_len != self._last_plotted_count
            and (now - self._last_plot_update_ts) >= self._plot_update_interval_s
        ):
            t_arr = np.fromiter(self._time_hist, dtype=np.float32)
            total_arr = np.fromiter(self._total_hist, dtype=np.float32)
            current_arr = np.fromiter(self._current_hist, dtype=np.float32)
            self._curve_total.setData(t_arr, total_arr)
            self._curve_current.setData(t_arr, current_arr)
            self.apply_plot_time_window()
            self._last_plot_update_ts = now
            self._last_plotted_count = hist_len

        # UI fps
        now_ui = time.time()
        if self._last_ui_time is not None:
            dt = now_ui - self._last_ui_time
            if dt > 0:
                self._ui_fps = 1.0 / dt
        self._last_ui_time = now_ui

        if self._calib_active:
            self._update_status(f"标定中({self._calib_count}/{self._calib_need_frames})")
        else:
            self._update_status()

    def closeEvent(self, event) -> None:
        self.on_close_clicked()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=False)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())