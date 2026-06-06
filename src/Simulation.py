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
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QGridLayout,
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
    使用分位数降低极端值影响。
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

        while True:
            idx = self._text_buffer.find("\n")
            if idx < 0:
                break
            raw_line = self._text_buffer[:idx]
            self._text_buffer = self._text_buffer[idx + 1 :]
            line = raw_line.rstrip("\r")
            self._consume_line(line, frames)

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
            timeout=timeout,  # 非阻塞
            write_timeout=0.1,
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
        读取尽量新的数据，必要时丢弃积压。
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

        # 发帧限频：与UI刷新频率接近，降低事件堆积
        self._emit_interval_s = 1.0 / 50.0
        self._last_emit_ts = 0.0
        self._latest_frame_cache: Optional[Frame] = None

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
                        last_mat = matrices[-1]
                        self._frame_index += len(matrices) - 1
                        self._latest_frame_cache = Frame(
                            timestamp=time.time(),
                            index=self._frame_index,
                            matrix=last_mat,
                            scenario="sensor",
                        )
                        self._frame_index += 1

                if self._latest_frame_cache is not None and (now - self._last_emit_ts) >= self._emit_interval_s:
                    self.frame_received.emit(self._latest_frame_cache)
                    self._latest_frame_cache = None
                    self._last_emit_ts = now

                if self._bytes_acc > 0 and (now - self._last_bytes_emit_ts) >= self._bytes_emit_interval_s:
                    self.bytes_stat.emit(self._bytes_acc)
                    self._bytes_acc = 0
                    self._last_bytes_emit_ts = now

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


class SimulatedWorker(QObject):
    """
    改进后的模拟左手手掌按压工作线程。
    - 手指朝向界面上方（指尖在矩阵行索引 0 附近）
    - 周期 15 s：缓慢按压、保持、缓慢释放、休息
    - 每个受力单元在阶段内持续平滑变化（时空正弦扰动）
    """
    bytes_stat = pyqtSignal(int)
    frame_received = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._running = False
        self._frame_index = 0
        self._emit_interval_s = 1.0 / 50.0
        self._last_emit_ts = 0.0
        self._bytes_acc = 0
        self._last_bytes_emit_ts = 0.0
        self._bytes_emit_interval_s = 0.1

        self._period_s = 8.0

        # 区域最大原始尺度（与原脚本尺度相近）
        self._palm_max = 20.0
        self._thumb_max = 9.0
        self._index_max = 13.0
        self._middle_max = 15.0
        self._ring_max = 13.0
        self._little_max = 9.0

        # 单元时空扰动参数
        self._cell_freq = 0.6  # Hz 局部低频抖动
        self._cell_amp = 0.08  # 相对幅度
        self._spatial_phase_scale = 0.12  # 空间相位差尺度

        # 预计算行列索引网格，用于时空扰动
        self._rows = 16
        self._cols = 16
        r_idx, c_idx = np.meshgrid(np.arange(self._rows), np.arange(self._cols), indexing='ij')
        self._r_idx = r_idx.astype(np.float32)
        self._c_idx = c_idx.astype(np.float32)

    def _ease_in_out(self, x: float) -> float:
        # 平滑的 ease-in-out（x in [0,1]）
        x = np.clip(x, 0.0, 1.0)
        return 0.5 * (1 - np.cos(np.pi * x))

    def _temporal_cell_variation(self, t: float) -> np.ndarray:
        """
        返回每个单元的乘法扰动矩阵，随时间连续变化。
        使用小幅正弦，带空间相位差，保证同一区域内的单元不一致。
        """
        # 基础相位随单元位置变化，保证相邻单元相位不同
        phase = (self._r_idx * self._cols + self._c_idx) * self._spatial_phase_scale
        omega = 2 * np.pi * self._cell_freq
        mat = 1.0 + self._cell_amp * np.sin(omega * t + phase)
        return mat.astype(np.float32)

    def _region_weight_circle(self, center_r: float, center_c: float, radius: float) -> np.ndarray:
        """
        返回一个基于距离的线性权重矩阵，中心到边界线性衰减到 0。
        """
        dr = self._r_idx - center_r
        dc = self._c_idx - center_c
        dist = np.sqrt(dr * dr + dc * dc)
        w = np.clip(1.0 - (dist / radius), 0.0, 1.0)
        return w.astype(np.float32)

    def _create_palm_pattern(self, intensity_base: float, t_global: float) -> np.ndarray:
        """
        使用多个带权重的圆形/椭圆形区域叠加表示掌心与五指，
        再乘以时空扰动矩阵，使每个单元在阶段内连续变化。
        五指指向界面上方（指尖在行索引靠近 0 的一端）。
        """
        matrix = np.zeros((self._rows, self._cols), dtype=np.float32)

        # 掌心：中心偏下
        palm_center_r, palm_center_c, palm_radius = 9.0, 7.5, 6.0
        palm_w = self._region_weight_circle(palm_center_r, palm_center_c, palm_radius)
        palm_val = self._palm_max * intensity_base
        matrix += palm_val * palm_w

        # 手指位置与半径（指尖在上，行较小）
        # 拇指（左侧靠下的拇指部分用较低列索引）
        thumb_center_r, thumb_center_c, thumb_radius = 5.0, 1.0, 3.8
        thumb_w = self._region_weight_circle(thumb_center_r, thumb_center_c, thumb_radius)
        matrix += self._thumb_max * intensity_base * thumb_w

        # 食指
        index_center_r, index_center_c, index_radius = 4.0, 5.0, 3.2
        index_w = self._region_weight_circle(index_center_r, index_center_c, index_radius)
        matrix += self._index_max * intensity_base * index_w

        # 中指（通常最长且最大）
        middle_center_r, middle_center_c, middle_radius = 3.5, 8.0, 3.4
        middle_w = self._region_weight_circle(middle_center_r, middle_center_c, middle_radius)
        matrix += self._middle_max * intensity_base * middle_w

        # 无名指
        ring_center_r, ring_center_c, ring_radius = 4.2, 11.0, 3.2
        ring_w = self._region_weight_circle(ring_center_r, ring_center_c, ring_radius)
        matrix += self._ring_max * intensity_base * ring_w

        # 小指（右侧）
        little_center_r, little_center_c, little_radius = 5.0, 14.0, 3.6
        little_w = self._region_weight_circle(little_center_r, little_center_c, little_radius)
        matrix += self._little_max * intensity_base * little_w

        # 指间留白，手指之间的间隙设为零阈值区域（通过掩模减弱）
        gap_mask = np.ones_like(matrix, dtype=np.float32)
        gap_mask[0:11, 2:4] = 0.0
        gap_mask[0:11, 6:7] = 0.0
        gap_mask[0:11, 9:10] = 0.0
        gap_mask[0:11, 12:13] = 0.0
        matrix *= gap_mask

        # 施加每单元的时空微扰，保证阶段内不恒定
        cell_var = self._temporal_cell_variation(t_global)
        matrix = matrix * cell_var

        # 保证非负
        matrix = np.maximum(matrix, 0.0).astype(np.float32)
        return matrix

    @pyqtSlot()
    def run(self) -> None:
        self._running = True
        self.status_changed.emit("已打开")
        start_ts = time.perf_counter()

        try:
            while self._running:
                now = time.perf_counter()
                elapsed = now - start_ts
                cycle_t = elapsed % self._period_s

                # 8s 周期分段：按压（0-6s），保持（6-6.5s），释放（6.5-8s）
                if cycle_t < 6.0:
                    # 缓慢上升 0 -> 1，使用 ease-in-out 保证平滑
                    frac = cycle_t / 6.0
                    intensity_base = self._ease_in_out(frac)
                elif cycle_t < 6.5:
                    # 保持段，轻微波动但整体接近 0.95~1.0
                    local_t = cycle_t - 6.0
                    oscill = 0.04 * np.sin(2 * np.pi * 0.2 * local_t)  # 低频慢波动
                    intensity_base = 0.96 + oscill
                elif cycle_t < 8.0:
                    # 缓慢下降 1 -> 0
                    frac = (cycle_t - 6.0) / 2.0
                    intensity_base = self._ease_in_out(1.0 - frac)
                else:
                    # 休息段，保持接近零并带小幅扰动
                    local_t = cycle_t - 8.0
                    intensity_base = 0.02 + 0.04 * 0.5 * (1 + np.sin(2 * np.pi * 0.5 * local_t))

                intensity_base = float(np.clip(intensity_base, 0.0, 1.0))

                # 生成矩阵（同时传递全局时间用于细节扰动）
                matrix = self._create_palm_pattern(intensity_base, elapsed)

                frame = Frame(timestamp=time.time(), index=self._frame_index, matrix=matrix, scenario="simulated")
                self._frame_index += 1

                # 以目标频率发送帧（50 Hz）
                if (now - self._last_emit_ts) >= self._emit_interval_s:
                    self.frame_received.emit(frame)
                    self._last_emit_ts = now

                if (now - self._last_bytes_emit_ts) >= self._bytes_emit_interval_s:
                    # 模拟字节流量，便于状态栏展示
                    self.bytes_stat.emit(256)
                    self._last_bytes_emit_ts = now

                time.sleep(0.01)
        finally:
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
        self.setMinimumSize(420, 420)
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
        self.resize(1520, 930)

        self._settings = QSettings("NJIT", "HandFSR")

        self._serial_thread: Optional[QThread] = None
        self._serial_worker: Optional[SimulatedWorker] = None
        self._stream_active = False

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
        self._plot_update_interval_s = 1.0 / 12.0
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

        self._plot_total = pg.PlotWidget()
        self._plot_current = pg.PlotWidget()
        self._setup_plot(self._plot_total, "总力曲线")
        self._setup_plot(self._plot_current, "峰值曲线")
        self._curve_total = self._plot_total.plot([], [], pen=pg.mkPen("#111111", width=2))
        self._curve_current = self._plot_current.plot([], [], pen=pg.mkPen("#111111", width=2, style=Qt.PenStyle.DashLine))

        self._combo_time_window = QComboBox()
        self._combo_time_window.addItems(["10 s", "30 s", "60 s", "截至当前"])
        self._combo_time_window.setCurrentText("30 s")

        # 2x2 数据卡片（标题居中、加粗加大；值居中）
        self._val_total = QLabel("0.0 N")
        self._val_current = QLabel("0.0 N")
        self._val_max = QLabel("-")
        self._val_min = QLabel("-")
        self._metrics_widget = self._build_metrics_grid()

        controls = QHBoxLayout()
        controls.setSpacing(10)
        controls.setContentsMargins(8, 8, 8, 8)
        controls.addWidget(self._btn_refresh_ports)
        controls.addWidget(QLabel("端口"))
        controls.addWidget(self._combo_ports)
        controls.addWidget(QLabel("波特率"))
        controls.addWidget(self._combo_baud)
        controls.addWidget(self._btn_open)
        controls.addWidget(self._btn_close)
        controls.addWidget(self._btn_zero)
        controls.addSpacing(8)
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

        # 控制行更明显
        controls_bar = QWidget()
        controls_bar.setLayout(controls)
        controls_bar.setMinimumHeight(70)
        controls_bar.setStyleSheet("QWidget { background: #f6f7f9; }")

        left = QVBoxLayout()
        left_title = QLabel("热力图")
        lf = QFont()
        lf.setBold(True)
        lf.setPointSize(12)
        left_title.setFont(lf)
        left_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(left_title)
        left.addWidget(self._heatmap, stretch=1)
        left.setSpacing(8)
        left_w = QWidget()
        left_w.setLayout(left)

        tw = QHBoxLayout()
        tw.addWidget(QLabel("时间窗"))
        tw.addWidget(self._combo_time_window)
        tw.addStretch()

        right = QVBoxLayout()
        right_title = QLabel("实时数据分析")
        right_title.setFont(lf)
        right_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(right_title)
        right.addWidget(self._metrics_widget)
        right.addLayout(tw)
        right.addWidget(self._plot_total, stretch=1)
        right.addWidget(self._plot_current, stretch=1)
        right.setSpacing(8)
        right_w = QWidget()
        right_w.setLayout(right)

        work = QHBoxLayout()
        # 热力图略大于数据模块
        work.addWidget(left_w, stretch=6)
        work.addWidget(right_w, stretch=5)
        work.setSpacing(12)
        work.setContentsMargins(6, 6, 6, 6)

        root = QVBoxLayout()
        root.addWidget(controls_bar)
        root.addLayout(work, stretch=1)
        root.setSpacing(8)
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

    def _build_metric_card(self, title: str, value_label: QLabel) -> QWidget:
        card = QWidget()
        card.setStyleSheet(
            "QWidget { background: white; border: 1px solid #BFBFBF; border-radius: 4px; }"
            "QLabel { border: none; }"
        )
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        title_label = QLabel(title)
        tf = QFont()
        tf.setBold(True)
        tf.setPointSize(11)
        title_label.setFont(tf)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        vf = QFont()
        vf.setBold(True)
        vf.setPointSize(13)
        value_label.setFont(vf)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay.addWidget(title_label)
        lay.addWidget(value_label)
        card.setLayout(lay)
        return card

    def _build_metrics_grid(self) -> QWidget:
        w = QWidget()
        g = QGridLayout()
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(8)

        g.addWidget(self._build_metric_card("总力 (N)", self._val_total), 0, 0)
        g.addWidget(self._build_metric_card("当前峰值 (N)", self._val_current), 0, 1)
        g.addWidget(self._build_metric_card("会话最大 (N)", self._val_max), 1, 0)
        g.addWidget(self._build_metric_card("会话最小 (N)", self._val_min), 1, 1)

        w.setLayout(g)
        return w

    def _setup_plot(self, plot: pg.PlotWidget, title_text: str) -> None:
        # 标题实心填充 + 居中
        plot.setTitle(f"<div style='color:#000000; padding:4px; text-align:center;'><b>{title_text}</b></div>")
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setClipToView(True)
        plot.setDownsampling(auto=True, mode="peak")
        plot.getAxis("left").setPen(pg.mkPen("k", width=1))
        plot.getAxis("bottom").setPen(pg.mkPen("k", width=1))
        plot.getAxis("left").setTextPen(pg.mkPen("k"))
        plot.getAxis("bottom").setTextPen(pg.mkPen("k"))
        # 纵轴单位
        plot.setLabel("left", "Force", units="N", color="#000000")
        plot.setLabel("bottom", "Time", units="s", color="#000000")

    def _update_status(self, prefix: str = "", force: bool = False) -> None:
        now = time.perf_counter()
        if (not force) and (now - self._last_status_update_ts < self._status_update_interval_s):
            return

        msg = (
            f"{prefix} | 数据源: 模拟信号 | 状态: {self._serial_state_text} | "
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
        # 固定显示 COM5 用于模拟
        self._combo_ports.addItem("COM5")
        self._btn_open.setEnabled(True)

    @pyqtSlot()
    def on_open_clicked(self) -> None:
        if self._serial_thread is not None:
            return

        # 使用模拟工作线程而不是真实的串口工作线程
        self._serial_thread = QThread()
        self._serial_worker = SimulatedWorker()
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
        self._update_status("打开模拟信号", force=True)

    @pyqtSlot()
    def on_close_clicked(self) -> None:
        self._stream_active = False
        self._calib_active = False
        self._pending_frame = None
        self._has_pending_frame = False
        self.stop_serial_worker()
        self._update_status("模拟信号已关闭", force=True)

    @pyqtSlot()
    def on_zero_clicked(self) -> None:
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

        self._pending_frame = frame
        self._has_pending_frame = True

    @pyqtSlot(str)
    def on_serial_status_changed(self, text: str) -> None:
        self._serial_state_text = text

    @pyqtSlot()
    def on_serial_thread_finished(self) -> None:
        self._serial_thread = None
        self._serial_worker = None
        self._btn_open.setEnabled(True)
        self._btn_close.setEnabled(False)
        self._btn_refresh_ports.setEnabled(True)
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

                # 绘制前做坐标变换
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