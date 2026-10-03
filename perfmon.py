"""
Performance log: finds out WHY a run gets slower over time.

Every second it records the console loop rate, how many video frames were
shown, YOLO time, the OR panel's rate / draw time, process memory and (if
nvidia-smi is available) the GPU temperature, clock, load, power and the
reasons the driver is holding the clock down. Written to runs/perf_<tag>.csv.

At the end report() compares the first and the last minute and says in plain
words what got slower and the most likely reason (GPU heat / power limit,
memory growth, the video, YOLO or the panel).
"""
import csv
import os
import shutil
import subprocess
import threading
import time

_THROTTLE = {                      # nvidia-smi clocks_throttle_reasons bits
    0x2: "app clocks", 0x4: "power cap", 0x8: "HW slowdown",
    0x20: "SW thermal", 0x40: "HW thermal", 0x80: "HW power brake",
}


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:                                       # noqa: BLE001
        pass
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except Exception:                                       # noqa: BLE001
        return float("nan")


class _GpuSampler(threading.Thread):
    QUERY = ("temperature.gpu,clocks.sm,clocks.max.sm,utilization.gpu,"
             "power.draw,clocks_throttle_reasons.active")

    def __init__(self, every=2.0):
        super().__init__(daemon=True)
        self.exe = shutil.which("nvidia-smi")
        self.every = every
        self.latest = {}
        self._stop = threading.Event()

    def run(self):
        flags = 0x08000000 if os.name == "nt" else 0   # no console flash
        while self.exe and not self._stop.is_set():
            try:
                out = subprocess.run(
                    [self.exe, f"--query-gpu={self.QUERY}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=flags).stdout.strip().splitlines()[0]
                v = [x.strip() for x in out.split(",")]

                def num(s):
                    try:
                        return float(s)
                    except ValueError:
                        return float("nan")
                bits = int(v[5], 16) if v[5].startswith("0x") else 0
                self.latest = {
                    "gpu_temp_c": num(v[0]), "gpu_clock_mhz": num(v[1]),
                    "gpu_clock_max_mhz": num(v[2]), "gpu_util_pct": num(v[3]),
                    "gpu_power_w": num(v[4]),
                    "gpu_limit": "+".join(n for b, n in _THROTTLE.items()
                                          if bits & b)}
            except Exception:                               # noqa: BLE001
                self.exe = None
            self._stop.wait(self.every)

    def stop(self):
        self._stop.set()


class PerfMonitor:
    COLS = ["t", "loop_fps", "video_fps", "infer_ms", "panel_fps",
            "panel_ms", "rss_mb", "gpu_temp_c", "gpu_clock_mhz",
            "gpu_clock_max_mhz", "gpu_util_pct", "gpu_power_w", "gpu_limit"]

    def __init__(self, tag="run"):
        os.makedirs("runs", exist_ok=True)
        self.path = os.path.join("runs", f"perf_{tag}.csv")
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.COLS)
        self.gpu = _GpuSampler()
        self.gpu.start()
        self.t0 = time.perf_counter()
        self._win = time.perf_counter()
        self._loops = 0
        self._frame0 = None
        self._infer = []
        self._panel0 = 0
        self.rows = []

    def tick(self, frame_index, infer_ms, panel_frames=0, panel_ms=0.0):
        now = time.perf_counter()
        self._loops += 1
        self._infer.append(infer_ms)
        if self._frame0 is None:
            self._frame0, self._panel0 = frame_index, panel_frames
        dt = now - self._win
        if dt < 1.0:
            return
        g = self.gpu.latest
        row = {
            "t": round(now - self.t0, 1),
            "loop_fps": round(self._loops / dt, 1),
            "video_fps": round(max(0, frame_index - self._frame0) / dt, 1),
            "infer_ms": round(sorted(self._infer)[len(self._infer) // 2], 1),
            "panel_fps": round((panel_frames - self._panel0) / dt, 1),
            "panel_ms": round(panel_ms, 1),
            "rss_mb": round(_rss_mb(), 0),
        }
        row.update({k: g.get(k, "") for k in self.COLS[7:]})
        self._w.writerow([row[k] for k in self.COLS])
        self._f.flush()
        self.rows.append(row)
        self._win, self._loops, self._infer = now, 0, []
        self._frame0, self._panel0 = frame_index, panel_frames

    # ------------------------------------------------------------- report
    def report(self):
        self.gpu.stop()
        self._f.close()
        r = self.rows
        if len(r) < 40:
            print(f"[perf] log: {self.path} (run too short for a "
                  f"slowdown report; run for 2+ minutes)")
            return
        k = min(30, len(r) // 3)
        a, b = r[2:2 + k], r[-k:]            # skip the 2 s warm-up

        def avg(rows, key):
            v = [x[key] for x in rows if isinstance(x[key], (int, float))
                 and x[key] == x[key]]
            return sum(v) / len(v) if v else float("nan")

        print("\n---- performance: first vs last "
              f"{k} s  ({self.path}) ----")
        keys = [("loop_fps", "console loop", "fps"),
                ("video_fps", "video frames shown", "/s"),
                ("infer_ms", "YOLO time", "ms"),
                ("panel_fps", "OR panel", "fps"),
                ("panel_ms", "OR panel draw", "ms"),
                ("rss_mb", "memory", "MB"),
                ("gpu_temp_c", "GPU temperature", "C"),
                ("gpu_clock_mhz", "GPU clock", "MHz"),
                ("gpu_util_pct", "GPU load", "%"),
                ("gpu_power_w", "GPU power", "W")]
        res = {}
        for key, name, unit in keys:
            x, y = avg(a, key), avg(b, key)
            res[key] = (x, y)
            if x == x and y == y:
                print(f"  {name:20s} {x:8.1f} -> {y:8.1f} {unit}")
        limits = sorted({x["gpu_limit"] for x in b if x.get("gpu_limit")})
        if limits:
            print(f"  GPU clock held down by: {', '.join(limits)}")

        def worse(key, frac, higher_is_worse=True):
            x, y = res.get(key, (float("nan"),) * 2)
            if not (x == x and y == y) or x <= 0:
                return False
            return (y > x * (1 + frac)) if higher_is_worse else \
                (y < x * (1 - frac))

        why = []
        util = res.get("gpu_util_pct", (float("nan"),) * 2)
        inf = res.get("infer_ms", (float("nan"),) * 2)
        if util[1] == util[1] and util[1] < 3 and inf[1] == inf[1] \
                and inf[1] > 80:
            why.append("YOLO is NOT running on the GPU (GPU load ~0 %, "
                       f"{inf[1]:.0f} ms per frame). Check: python -c "
                       "\"import torch; print(torch.__version__, "
                       "torch.cuda.is_available())\" - if it prints False, "
                       "reinstall the CUDA build: python -m pip install "
                       "--force-reinstall torch torchvision --index-url "
                       "https://download.pytorch.org/whl/cu124")
        temp = res.get("gpu_temp_c", (float("nan"),) * 2)[1]
        hot = any("thermal" in s for s in limits) or (temp == temp
                                                       and temp >= 78)
        if hot:
            why.append("the GPU is too HOT and slowed itself down. Plug the "
                       "laptop in, raise it for airflow, use OR_QUALITY="
                       "'low'.")
        elif worse("gpu_clock_mhz", 0.12, False) or any(
                "power" in s for s in limits):
            why.append("the GPU is NOT hot - Windows / the NVIDIA driver "
                       "lowered its clock to SAVE POWER (battery or "
                       "'Balanced' mode, light GPU load). Fix: plug in; "
                       "Windows power mode 'Best performance'; NVIDIA "
                       "Control Panel > Manage 3D settings > Program "
                       "settings > python.exe > Power management mode = "
                       "'Prefer maximum performance'.")
        if worse("rss_mb", 0.30):
            why.append("memory keeps growing (a leak) - send me "
                       f"{self.path}.")
        if worse("infer_ms", 0.25) and not why:
            why.append("YOLO got slower: usually GPU heat; check that "
                       "GPU temperature / clock lines above.")
        if worse("panel_ms", 0.40):
            why.append("the OR panel got slower to draw (GPU busy/hot); "
                       "OR_QUALITY='low' or lower OR_MAX_FPS.")
        if worse("loop_fps", 0.15, False) and not why:
            why.append("the console loop slowed without a GPU or memory "
                       f"cause - send me {self.path}.")
        if why:
            print("  likely cause:")
            for w in why:
                print(f"   - {w}")
        else:
            print("  no slowdown detected.")
