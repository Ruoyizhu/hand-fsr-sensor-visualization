import argparse
import codecs
import datetime as dt
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

import serial


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="milliseconds")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="串口文本采集器（原始bytes接收 + 增量解码 + 按行暂存）"
    )
    parser.add_argument("--port", required=True, help="串口号，例如 COM5")
    parser.add_argument("--baud", type=int, default=9600, help="波特率，默认9600")
    parser.add_argument("--bytesize", type=int, default=8, choices=[5, 6, 7, 8], help="数据位")
    parser.add_argument(
        "--parity",
        default="N",
        choices=["N", "E", "O", "M", "S"],
        help="校验位：N/E/O/M/S，默认N",
    )
    parser.add_argument("--stopbits", type=float, default=1, choices=[1, 1.5, 2], help="停止位")
    parser.add_argument("--timeout", type=float, default=0.2, help="串口超时秒，默认0.2")
    parser.add_argument("--encoding", default="gbk", help="增量解码编码，默认gbk")
    parser.add_argument("--out-dir", default="logs", help="默认保存目录，默认 logs")
    parser.add_argument("--max-bytes-per-read", type=int, default=4096, help="单次最多读取字节数")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="最大采集秒数，0表示直到Ctrl+C",
    )
    parser.add_argument(
        "--no-confirm-save",
        action="store_true",
        help="不询问，结束后直接保存到 --out-dir",
    )
    return parser


def build_serial_kwargs(args) -> dict:
    parity_map = {
        "N": serial.PARITY_NONE,
        "E": serial.PARITY_EVEN,
        "O": serial.PARITY_ODD,
        "M": serial.PARITY_MARK,
        "S": serial.PARITY_SPACE,
    }
    stopbits_map = {
        1: serial.STOPBITS_ONE,
        1.5: serial.STOPBITS_ONE_POINT_FIVE,
        2: serial.STOPBITS_TWO,
    }
    return {
        "port": args.port,
        "baudrate": args.baud,
        "bytesize": args.bytesize,
        "parity": parity_map[args.parity],
        "stopbits": stopbits_map[args.stopbits],
        "timeout": args.timeout,
        "write_timeout": args.timeout,
    }


def ask_save_target(default_out_dir: Path) -> Path | None:
    while True:
        ans = input("是否保存本次采集到 logs/ 文件夹？[Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            default_out_dir.mkdir(parents=True, exist_ok=True)
            return default_out_dir
        if ans in ("n", "no"):
            custom = input("输入其他保存目录（直接回车表示丢弃）: ").strip()
            if custom == "":
                return None
            custom_dir = Path(custom).expanduser()
            custom_dir.mkdir(parents=True, exist_ok=True)
            return custom_dir
        print("输入无效，请输入 Y 或 N。")


def finalize_save(
    tmp_log: Path,
    tmp_meta: Path,
    final_dir: Path | None,
    final_stem: str,
) -> tuple[Path | None, Path | None]:
    if final_dir is None:
        # 丢弃临时文件
        if tmp_log.exists():
            tmp_log.unlink(missing_ok=True)
        if tmp_meta.exists():
            tmp_meta.unlink(missing_ok=True)
        return None, None

    final_log = final_dir / f"{final_stem}.log"
    final_meta = final_dir / f"{final_stem}.meta.json"

    shutil.move(str(tmp_log), str(final_log))
    shutil.move(str(tmp_meta), str(final_meta))
    return final_log, final_meta


def main() -> int:
    args = build_arg_parser().parse_args()

    start_wall = dt.datetime.now()
    stamp = start_wall.strftime("%Y%m%d_%H%M%S")
    final_stem = f"serial_capture_{stamp}"

    tmp_dir = Path(tempfile.gettempdir()) / "hand_fsr_capture_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_log_path = tmp_dir / f"{final_stem}.log.tmp"
    tmp_meta_path = tmp_dir / f"{final_stem}.meta.json.tmp"

    running = True

    def handle_stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_stop)

    ser = None
    log_fp = None

    total_bytes = 0
    total_lines = 0
    last_sec_bytes = 0
    last_sec_lines = 0
    last_print_t = time.time()
    start_t = time.time()

    decoder = codecs.getincrementaldecoder(args.encoding)(errors="ignore")
    text_buf = ""

    try:
        ser = serial.Serial(**build_serial_kwargs(args))
        # 行缓冲写入，降低异常退出丢数据概率
        log_fp = open(tmp_log_path, "w", encoding="utf-8", newline="\n", buffering=1)

        print(f"[INFO] 已打开串口: {args.port} @ {args.baud}")
        print(f"[INFO] 临时日志: {tmp_log_path}")
        print("[INFO] 按 Ctrl+C 结束采集")

        while running:
            if args.max_seconds > 0 and (time.time() - start_t) >= args.max_seconds:
                print("[INFO] 到达最大采集时长，准备退出。")
                break

            raw = ser.read(args.max_bytes_per_read)
            if raw:
                total_bytes += len(raw)
                last_sec_bytes += len(raw)

                decoded = decoder.decode(raw, final=False)
                if decoded:
                    text_buf += decoded
                    while "\n" in text_buf:
                        line, text_buf = text_buf.split("\n", 1)
                        line = line.rstrip("\r")
                        total_lines += 1
                        last_sec_lines += 1
                        log_fp.write(f"{now_iso()}\t{line}\n")

            now = time.time()
            if now - last_print_t >= 1.0:
                sec = now - last_print_t
                lps = last_sec_lines / sec
                bps = last_sec_bytes / sec
                fps_est = lps / 16.0
                total_frames_est = total_lines / 16.0
                print(
                    f"[STAT] lines_total={total_lines} "
                    f"bytes_total={total_bytes} "
                    f"lines/s={lps:.2f} "
                    f"bytes/s={bps:.1f} "
                    f"fps_est={fps_est:.2f} "
                    f"frames_est={total_frames_est:.2f}"
                )
                last_print_t = now
                last_sec_lines = 0
                last_sec_bytes = 0

        # 收尾：冲刷增量解码器
        tail = decoder.decode(b"", final=True)
        if tail:
            text_buf += tail

        if text_buf.strip():
            log_fp.write(f"{now_iso()}\t[PARTIAL]{text_buf.rstrip()}\n")

    except serial.SerialException as e:
        print(f"[ERROR] 串口异常: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[ERROR] 未知异常: {e}", file=sys.stderr)
        return 3
    finally:
        end_wall = dt.datetime.now()
        duration_s = max(0.0, (end_wall - start_wall).total_seconds())

        if log_fp is not None:
            log_fp.flush()
            log_fp.close()

        if ser is not None and ser.is_open:
            ser.close()

        # 先写临时meta，最终是否保存由用户决定
        meta = {
            "capture_start": start_wall.isoformat(timespec="seconds"),
            "capture_end": end_wall.isoformat(timespec="seconds"),
            "duration_s": round(duration_s, 3),
            "port": args.port,
            "baudrate": args.baud,
            "bytesize": args.bytesize,
            "parity": args.parity,
            "stopbits": args.stopbits,
            "timeout": args.timeout,
            "encoding": args.encoding,
            "total_bytes": total_bytes,
            "total_lines": total_lines,
            "estimated_frames_16lines": round(total_lines / 16.0, 3),
            "note": "Each line format: <iso_timestamp>\\t<raw_text_line>",
        }

        with open(tmp_meta_path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)

        target_dir = Path(args.out_dir)
        if args.no_confirm_save:
            target_dir.mkdir(parents=True, exist_ok=True)
            final_log, final_meta = finalize_save(tmp_log_path, tmp_meta_path, target_dir, final_stem)
        else:
            chosen_dir = ask_save_target(target_dir)
            final_log, final_meta = finalize_save(tmp_log_path, tmp_meta_path, chosen_dir, final_stem)

        if final_log is None:
            print("[INFO] 本次采集已丢弃（未保存到 logs/ 或其他目录）。")
        else:
            print(f"[INFO] 日志已保存: {final_log}")
            print(f"[INFO] 元信息已保存: {final_meta}")

        print("[INFO] 已安全退出。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())