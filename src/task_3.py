import argparse
import codecs
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


@dataclass
class ParserStats:
    emitted_frames: int = 0
    accepted_rows: int = 0
    dropped_lines: int = 0
    resync_count: int = 0


class TextFrameParser:
    """
    文本流 -> 16x16矩阵 的流式解析器
    - feed(data: bytes) -> list[np.ndarray]
    - 支持半行/粘包
    - 支持 \\n 和 \\r\\n
    - 行内稳健提取浮点数
    - 连续错误过多自动重同步
    """

    def __init__(
        self,
        rows: int = 16,
        cols: int = 16,
        encoding: str = "ascii",
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

        # 增量按行处理，保留最后半行
        while "\n" in self._text_buffer:
            raw_line, self._text_buffer = self._text_buffer.split("\n", 1)
            line = raw_line.rstrip("\r")
            self._consume_line(line, frames)

        return frames

    def flush(self) -> List[np.ndarray]:
        """
        结束流时调用，冲刷解码器缓存。
        """
        frames: List[np.ndarray] = []
        tail = self._decoder.decode(b"", final=True)
        if tail:
            self._text_buffer += tail

        # flush 阶段仅处理完整行，残缺半行直接丢弃
        while "\n" in self._text_buffer:
            raw_line, self._text_buffer = self._text_buffer.split("\n", 1)
            line = raw_line.rstrip("\r")
            self._consume_line(line, frames)

        # 残余半行放弃
        self._text_buffer = ""
        return frames

    def _consume_line(self, line: str, out_frames: List[np.ndarray]) -> None:
        # 空行可忽略，不记错误
        if not line.strip():
            return

        row = self._parse_line_to_row(line)
        if row is None:
            self.stats.dropped_lines += 1
            self._consecutive_errors += 1

            # 连续错误过多，清空行缓存并重同步
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
        """
        解析单行：
        - 允许行尾有 '|'
        - 允许多空格
        - 兼容日志行前缀时间戳：若存在 tab，优先只取 tab 后文本
        - 兼容你的实际数据格式：行首可能有 1..16 这样的行号（共17个数字）
        """
        # 若来自采集日志（timestamp\tpayload），先去掉时间戳部分
        if "\t" in line:
            line = line.split("\t", 1)[1]

        # 行尾分隔符兼容
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

        # 标准情况：正好16个
        if len(nums) == self.cols:
            return nums

        # 实际串口常见情况：首列是行号 + 16个值 => 17个数字
        if self.allow_leading_index and len(nums) == self.cols + 1:
            return nums[1:]

        # 其他数量视为坏行
        return None


def run_self_test() -> int:
    print("[SELF-TEST] 开始自检...")

    parser = TextFrameParser(
        rows=16,
        cols=16,
        encoding="utf-8",
        errors="ignore",
        max_consecutive_errors=6,
        allow_leading_index=True,
    )

    # 构造两帧已知矩阵
    f1 = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    f2 = (np.arange(16 * 16, dtype=np.float32).reshape(16, 16) + 1000.0)

    def frame_to_text(mat: np.ndarray) -> str:
        lines = []
        for i in range(16):
            # 带行号（模拟你日志中的真实格式）
            vals = " ".join(f"{v:.6f}" for v in mat[i])
            lines.append(f"{i+1} {vals}")
        return "\n".join(lines) + "\n"

    noisy_stream = (
        "garbage line !!!\n"
        + frame_to_text(f1)
        + "### invalid ###\n"
        + "9 1 2 3\n"
        + frame_to_text(f2)
    )

    raw = noisy_stream.encode("utf-8")
    chunks = [raw[:37], raw[37:121], raw[121:500], raw[500:901], raw[901:]]

    out = []
    for c in chunks:
        out.extend(parser.feed(c))
    out.extend(parser.flush())

    ok = True
    if len(out) < 2:
        ok = False
        print(f"[SELF-TEST] 失败：期望至少2帧，实际 {len(out)} 帧")
    else:
        if not np.allclose(out[0], f1):
            ok = False
            print("[SELF-TEST] 失败：第1帧内容不匹配")
        if not np.allclose(out[1], f2):
            ok = False
            print("[SELF-TEST] 失败：第2帧内容不匹配")

    print(
        "[SELF-TEST] 统计: "
        f"frames={parser.stats.emitted_frames}, "
        f"accepted_rows={parser.stats.accepted_rows}, "
        f"dropped_lines={parser.stats.dropped_lines}, "
        f"resync={parser.stats.resync_count}"
    )

    if ok:
        print("[SELF-TEST] 通过")
        return 0
    print("[SELF-TEST] 未通过")
    return 1


def replay_capture_log(log_path: Path, encoding: str = "utf-8") -> int:
    """
    用你任务2生成的日志回放：
    每行格式: timestamp \\t raw_line
    """
    if not log_path.exists():
        print(f"[REPLAY] 文件不存在: {log_path}")
        return 2

    parser = TextFrameParser(
        rows=16,
        cols=16,
        encoding="utf-8",
        errors="ignore",
        max_consecutive_errors=8,
        allow_leading_index=True,
    )

    frames = []
    line_count = 0

    with open(log_path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            line_count += 1
            # 注意：这里直接把整行喂给解析器，它内部会自动去掉 tab 前时间戳
            payload = line.encode("utf-8", errors="ignore")
            frames.extend(parser.feed(payload))

    frames.extend(parser.flush())

    print(f"[REPLAY] 输入日志行数: {line_count}")
    print(f"[REPLAY] 输出帧数: {len(frames)}")
    print(
        "[REPLAY] 解析统计: "
        f"accepted_rows={parser.stats.accepted_rows}, "
        f"dropped_lines={parser.stats.dropped_lines}, "
        f"resync={parser.stats.resync_count}"
    )

    if frames:
        m0 = frames[0]
        print(
            "[REPLAY] 首帧摘要: "
            f"shape={m0.shape}, dtype={m0.dtype}, "
            f"min={float(np.min(m0)):.6f}, "
            f"max={float(np.max(m0)):.6f}, "
            f"mean={float(np.mean(m0)):.6f}"
        )
    else:
        print("[REPLAY] 未解析到任何帧，请检查编码/行格式/串口原始数据。")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="任务3：文本流 -> 16x16矩阵 流式解析器")
    ap.add_argument("--self-test", action="store_true", help="运行内置最小自检")
    ap.add_argument("--replay-log", type=str, default="", help="回放任务2生成的log文件")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    if args.replay_log:
        return replay_capture_log(Path(args.replay_log))

    print("请指定 --self-test 或 --replay-log <path>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())