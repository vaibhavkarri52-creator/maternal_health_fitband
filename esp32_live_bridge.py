#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import serial
import serial.tools.list_ports

HEADER = "timestamp_ms,ir_ppg,red_ppg,ax,ay,az"
THIS_DIR = Path(__file__).resolve().parent
KEYS = (
    "cp210", "ch340", "ch341", "ch9102", "ftdi", "ft232",
    "slab_usbtouart", "usbserial", "wchusbserial", "usbmodem",
    "ttyusb", "ttyacm", "uart", "silicon labs", "qinheng"
)


def find_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    matched = [x.device for x in ports if any(k in f"{x.device} {x.description} {x.manufacturer} {x.hwid}".lower() for k in KEYS)]
    if not matched:
        raise RuntimeError(f"No ESP32 USB serial port detected (found only: {[x.device for x in ports]}). Plug in ESP32 via USB data cable.")
    cu = [d for d in matched if "/cu." in d]
    return cu[0] if cu else matched[0]

def main():
    ap = argparse.ArgumentParser(description="Cross-Platform ESP32 Live Biosensor Bridge")
    ap.add_argument("--port", default=None, help="Serial port (COMx on Windows, /dev/cu.* on Mac)")
    ap.add_argument("--list-ports", action="store_true", help="List detected serial ports")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--sbp-offset", type=float, default=0.0)
    ap.add_argument("--dbp-offset", type=float, default=0.0)
    ap.add_argument("--stream", action="store_true", default=True, help="Stream evaluated epochs to web_app")
    ap.add_argument("--stream-url", default="http://localhost:8080/api/telemetry", help="Target webhook URL for live vitals")
    args = ap.parse_args()

    if args.list_ports:
        for p in serial.tools.list_ports.comports():
            print(f"  * {p.device:<28} | {p.description} [{p.hwid}]")
        return

    port = args.port or find_port()
    out_csv = (THIS_DIR / f"esp32_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv").resolve()

    print(f"[INFO] Connecting to {port} @ {args.baud} baud...")
    ser = serial.Serial(port, args.baud, timeout=2.0)
    time.sleep(2.0)
    ser.reset_input_buffer()

    target = args.duration * 50
    n = 0
    print(f"[REC] Recording {args.duration}s ({target} rows) -> {out_csv.name}")

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write(HEADER + "\n")
        try:
            while n < target:
                raw = ser.readline().decode("utf-8", errors="ignore").strip()
                parts = raw.split(",")
                if len(parts) != 6:
                    continue
                try:
                    [float(x) for x in parts]
                except ValueError:
                    continue
                f.write(raw + "\n")
                n += 1
                if n % 250 == 0:
                    print(f"  [{n // 50:>3}s / {args.duration}s] Packets: {n} | IR={parts[1]}")
        except KeyboardInterrupt:
            print("\n[INFO] Stopped early by user.")
        finally:
            ser.close()

    if n >= 1500:
        cmd = [
            sys.executable,
            str(THIS_DIR / "run_all_biosensors.py"),
            str(out_csv),
            "--raw-fs", "50",
            "--sbp-offset", str(args.sbp_offset),
            "--dbp-offset", str(args.dbp_offset),
        ]
        if args.stream:
            cmd.extend(["--stream", "--stream-url", args.stream_url])
        subprocess.run(cmd, check=True)
    else:
        print(f"[WARN] Only captured {n} rows (<1500 needed for a 30s epoch).")


if __name__ == "__main__":
    main()