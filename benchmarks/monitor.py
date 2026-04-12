"""
Background system monitor — samples CPU/GPU temps and load at a fixed interval.
Stores a time-series and benchmark markers for graph generation.
"""
import threading
import time
import subprocess
import json
import psutil


class SystemMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples = []   # [{t, cpu_temp, gpu_temp, cpu_load, gpu_util}]
        self.markers = []   # [{t, label, tier}]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        psutil.cpu_percent()  # discard first dummy reading
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def mark(self, label: str, tier: str = ""):
        with self._lock:
            self.markers.append({"t": time.time(), "label": label, "tier": tier})

    def latest(self) -> dict:
        with self._lock:
            return self.samples[-1].copy() if self.samples else {}

    def save_graph(self, path: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        with self._lock:
            samples = list(self.samples)
            markers = list(self.markers)

        if not samples:
            return

        t0 = samples[0]["t"]
        ts = [s["t"] - t0 for s in samples]

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle("System Metrics During Benchmark Run", fontsize=13)

        tier_colors = {"small": "#4CAF50", "medium": "#FF9800", "large": "#F44336", "": "#9E9E9E"}
        bench_colors = {}
        palette = ["#2196F3", "#9C27B0", "#00BCD4", "#FF5722"]
        bench_names = []
        for m in markers:
            if m["label"] not in bench_names:
                bench_names.append(m["label"])
        for i, name in enumerate(bench_names):
            bench_colors[name] = palette[i % len(palette)]

        def plot_metric(ax, key, label, color, unit=""):
            vals = [s.get(key) for s in samples]
            valid = [(t, v) for t, v in zip(ts, vals) if v is not None]
            if valid:
                xt, xv = zip(*valid)
                ax.plot(xt, xv, color=color, linewidth=1.5, label=label)
            ax.set_ylabel(f"{label}{' (' + unit + ')' if unit else ''}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)

        plot_metric(axes[0], "cpu_temp", "CPU Temp", "#E53935", "°C")
        plot_metric(axes[0], "gpu_temp", "GPU Temp", "#FB8C00", "°C")

        plot_metric(axes[1], "cpu_load", "CPU Load", "#1E88E5", "%")
        plot_metric(axes[1], "gpu_util", "GPU Util", "#8E24AA", "%")
        axes[1].set_ylim(0, 105)

        # shade benchmark regions
        marker_times = [(m["t"] - t0, m["label"], m["tier"]) for m in markers]
        for i in range(0, len(marker_times) - 1, 2):
            t_start, label, tier = marker_times[i]
            t_end = marker_times[i + 1][0] if i + 1 < len(marker_times) else ts[-1]
            color = bench_colors.get(label, "#888")
            alpha = {"small": 0.10, "medium": 0.15, "large": 0.20}.get(tier, 0.10)
            for ax in axes[:2]:
                ax.axvspan(t_start, t_end, alpha=alpha, color=color)
            for ax in axes[:2]:
                ax.axvline(t_start, color=color, linewidth=0.8, alpha=0.6)

        # bottom panel: benchmark timeline
        ax = axes[2]
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_title("Benchmark Timeline", fontsize=9)
        for i in range(0, len(marker_times) - 1, 2):
            t_start, label, tier = marker_times[i]
            t_end = marker_times[i + 1][0] if i + 1 < len(marker_times) else ts[-1]
            color = bench_colors.get(label, "#888")
            ax.barh(0.5, t_end - t_start, left=t_start, height=0.5,
                    color=color, alpha=0.7, align="center")
            if t_end - t_start > 1:
                ax.text((t_start + t_end) / 2, 0.5, f"{label}\n{tier}",
                        ha="center", va="center", fontsize=7, color="white", fontweight="bold")

        # legend
        patches = [mpatches.Patch(color=c, label=n) for n, c in bench_colors.items()]
        fig.legend(handles=patches, loc="upper right", fontsize=8, ncol=2)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    # ── internals ─────────────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop.is_set():
            s = self._sample()
            with self._lock:
                self.samples.append(s)
            self._stop.wait(self.interval)

    def _sample(self) -> dict:
        import glob
        sample = {"t": time.time()}

        # CPU load (non-blocking — uses interval since last call)
        sample["cpu_load"] = psutil.cpu_percent()

        # CPU temperature — prefer k10temp Tctl
        try:
            temps = psutil.sensors_temperatures()
            if "k10temp" in temps:
                for entry in temps["k10temp"]:
                    if entry.label in ("Tctl", "Tccd1", ""):
                        sample["cpu_temp"] = round(entry.current, 1)
                        break
            if "cpu_temp" not in sample and "acpitz" in temps:
                sample["cpu_temp"] = round(temps["acpitz"][0].current, 1)
        except Exception:
            sample["cpu_temp"] = None

        # GPU utilization — sysfs (AMD) then nvidia-smi (NVIDIA)
        sample["gpu_util"] = None
        try:
            paths = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
            if paths:
                sample["gpu_util"] = float(open(paths[0]).read().strip())
        except Exception:
            pass
        if sample["gpu_util"] is None:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2
                )
                sample["gpu_util"] = float(out.stdout.strip())
            except Exception:
                pass

        # GPU temperature — rocm-smi (AMD) then nvidia-smi (NVIDIA)
        sample["gpu_temp"] = None
        try:
            out = subprocess.run(
                ["rocm-smi", "--showtemp", "--json"],
                capture_output=True, text=True, timeout=2
            )
            card = next(iter(json.loads(out.stdout).values()))
            temp_str = (card.get("Temperature (Sensor edge) (C)")
                        or card.get("Temperature (Sensor junction) (C)"))
            sample["gpu_temp"] = float(temp_str) if temp_str else None
        except Exception:
            pass
        if sample["gpu_temp"] is None:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2
                )
                sample["gpu_temp"] = float(out.stdout.strip())
            except Exception:
                pass

        return sample
