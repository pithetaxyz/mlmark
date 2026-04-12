"""
mlmark — main benchmark runner
Round-robin by size tier (small → medium → large), Rich TUI, system monitor.

Usage:
    python run_all.py -o <results_dir>
"""
import argparse
import json
import os
import signal
import subprocess
import time
import threading


def _rocm_override_if_needed():
    """Set HSA_OVERRIDE_GFX_VERSION when the detected GPU arch isn't in PyTorch's
    compiled target list.  No-op on CUDA, macOS, and CPU-only machines."""
    if "HSA_OVERRIDE_GFX_VERSION" in os.environ:
        return  # user already set it

    try:
        out = subprocess.run(
            ["rocm_agent_enumerator"], capture_output=True, text=True, timeout=5
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return  # not a ROCm system

    agents = [l.strip() for l in out.splitlines()
              if l.strip().startswith("gfx") and l.strip() != "gfx000"]
    if not agents:
        return

    arch = agents[0]  # e.g. "gfx1035"

    # Architectures that PyTorch ROCm wheels normally include
    PYTORCH_TARGETS = {
        "gfx900", "gfx906", "gfx908", "gfx90a",
        "gfx940", "gfx941", "gfx942",
        "gfx1030", "gfx1100", "gfx1101", "gfx1102",
    }
    if arch in PYTORCH_TARGETS:
        return  # natively supported, no override needed

    # Map unsupported arch prefix → nearest supported HSA version string
    FAMILY_FALLBACK = {
        "gfx90":  "9.0.4",   # Vega20 variants
        "gfx94":  "9.4.2",   # CDNA3 variants
        "gfx103": "10.3.0",  # RDNA2 iGPU variants (gfx1031/1035/1036)
        "gfx110": "11.0.0",  # RDNA3 variants (gfx1103)
        "gfx115": "11.5.0",  # RDNA4 variants
    }
    for prefix, version in FAMILY_FALLBACK.items():
        if arch.startswith(prefix):
            os.environ["HSA_OVERRIDE_GFX_VERSION"] = version
            return


_rocm_override_if_needed()

import torch
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

import bench_matmul
import bench_cnn
import bench_transformer
import bench_memory
from monitor import SystemMonitor

# ── Config ────────────────────────────────────────────────────────────────────

TIERS = ["small", "medium", "large"]
TIER_STYLE = {"small": "green", "medium": "yellow", "large": "red"}

Bench = namedtuple("Bench", "label module specs result_name")
BENCHMARKS = [
    Bench("matmul",      bench_matmul,      ["cpu/fp32", "gpu/fp32", "gpu/fp16", "gpu/fp8", "gpu/fp4"], "matmul"),
    Bench("cnn",         bench_cnn,         ["cpu/fp32", "gpu/fp32", "gpu/fp16"],                        "cnn_resnet50"),
    Bench("transformer", bench_transformer, ["cpu/fp32", "gpu/fp32", "gpu/fp16"],                        "transformer_gpt2"),
    Bench("memory",      bench_memory,      ["gpu/fp32"],                                               "memory_bandwidth"),
]

# Pre-computed: display label → result benchmark name (used in hot loop)
RESULT_NAME = {b.label: b.result_name for b in BENCHMARKS}

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "fp8":  getattr(torch, "float8_e4m3fn", None),
    "fp4":  getattr(torch, "float4_e2m1fn_x2", None),
}
HAS_GPU = torch.cuda.is_available()

# Spec → results-table column name
SPEC_TO_COL = {
    "cpu/fp32": "CPU FP32", "gpu/fp32": "GPU FP32",
    "gpu/fp16": "GPU FP16", "gpu/fp8":  "GPU FP8", "gpu/fp4": "GPU FP4",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

console = Console()


def device_label(spec: str):
    """Return (device_str, dtype) for a spec, or None if not runnable."""
    dev, dt = spec.split("/")
    if dev == "gpu" and not HAS_GPU:
        return None
    dtype = DTYPE_MAP.get(dt)
    if dtype is None:
        return None
    return ("cuda" if dev == "gpu" else "cpu"), dtype


def fmt_metric(val, unit="", na="N/A"):
    if val is None:
        return f"[dim]{na}[/dim]"
    return f"{val}{unit}"


# ── TUI panels ────────────────────────────────────────────────────────────────

def make_header(device_name: str, tier: str, tier_idx: int) -> Panel:
    txt = Text()
    txt.append("  mlmark  ", style="bold white on dark_blue")
    txt.append(f"  {device_name}  ", style="dim")
    txt.append("  Tier: ", style="white")
    txt.append(f"{tier.upper()} ({tier_idx+1}/{len(TIERS)})  ", style=f"bold {TIER_STYLE.get(tier, 'white')}")
    txt.append("  Ctrl+C to stop after current tier  ", style="dim italic")
    return Panel(txt, box=box.SIMPLE)


def make_progress_panel(
    jobs: list,
    done: set,
    unavailable: set,
    current: tuple,
    completed: int,
    total: int,
    start_time: float,
) -> Panel:
    elapsed = time.time() - start_time
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    elapsed_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    pct = int(completed / total * 100) if total else 0
    bar_filled = int(pct / 5)

    summary = Text()
    summary.append(f" {completed}/{total}  ")
    summary.append("█" * bar_filled, style="green")
    summary.append("░" * (20 - bar_filled), style="dim")
    summary.append(f"  {pct}%  {elapsed_str}\n\n")

    if current != ():
        tier, bench, spec = current
        summary.append("▶ ", style="bold yellow")
        summary.append(bench, style="bold")
        summary.append(f"  [{tier}]  {spec.upper()}\n\n")
    else:
        summary.append("idle\n\n", style="dim")

    tbl = Table(box=None, show_header=True, padding=(0, 1), expand=True, header_style="bold dim")
    tbl.add_column("", width=13, no_wrap=True)
    for tier in TIERS:
        tbl.add_column(Text(tier, style=f"bold {TIER_STYLE[tier]}"), justify="center")

    for b in BENCHMARKS:
        row_cells = [Text(b.label, style="bold")]
        for tier in TIERS:
            cell_text = Text()
            for i, spec in enumerate(b.specs):
                if i > 0:
                    cell_text.append("  ")
                key   = (tier, b.label, spec)
                label = spec.replace("cpu/", "C·").replace("gpu/", "G·").upper()
                if key in unavailable:
                    cell_text.append("✗", style="dim red");  cell_text.append(label, style="dim red")
                elif key in done:
                    cell_text.append("✓", style="green");    cell_text.append(label, style="green dim")
                elif key == current:
                    cell_text.append("▶", style="bold yellow"); cell_text.append(label, style="bold yellow")
                else:
                    cell_text.append("○", style="dim");      cell_text.append(label, style="dim")
            row_cells.append(cell_text)
        tbl.add_row(*row_cells)

    return Panel(Group(summary, tbl), title="[bold]Progress[/bold]", box=box.ROUNDED)


def make_metrics_panel(monitor: SystemMonitor) -> Panel:
    s = monitor.latest()
    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column(style="dim", width=12)
    tbl.add_column(justify="right", width=10)

    cpu_t = s.get("cpu_temp")
    gpu_t = s.get("gpu_temp")
    cpu_l = s.get("cpu_load")
    gpu_u = s.get("gpu_util")

    def temp_style(v):
        if v is None: return "dim"
        if v >= 85:   return "bold red"
        if v >= 70:   return "yellow"
        return "green"

    tbl.add_row("CPU Temp", f"[{temp_style(cpu_t)}]{fmt_metric(cpu_t, '°C')}[/]")
    tbl.add_row("GPU Temp", f"[{temp_style(gpu_t)}]{fmt_metric(gpu_t, '°C')}[/]")
    tbl.add_row("CPU Load", fmt_metric(f"{cpu_l:.0f}" if cpu_l is not None else None, "%"))
    tbl.add_row("GPU Util", fmt_metric(f"{gpu_u:.0f}" if gpu_u is not None else None, "%"))
    tbl.add_row("Fan RPM",  "[dim]N/A[/dim]")

    return Panel(tbl, title="[bold]System[/bold]", box=box.ROUNDED, width=26)


def _result_key(r: dict) -> str:
    """Short primary metric string for a result row."""
    bench = r.get("benchmark", "")
    if bench == "matmul":          return f"{r['tflops']:.2f} TFLOPS"
    if bench == "cnn_resnet50":    return f"{r['images_per_sec']:.0f} img/s"
    if bench == "transformer_gpt2":return f"{r['tokens_per_sec']:.0f} tok/s"
    if bench == "memory_bandwidth":
        return " / ".join(f"{k} {v:.1f}" for k, v in r.get("transfers", {}).items())
    return "?"


def make_results_table(table_cells: dict) -> Panel:
    """Pivot table driven by table_cells: (result_name, tier, col) → display string.
    Cells are pre-populated at startup and updated in-place as jobs finish."""

    COLS = ["CPU FP32", "GPU FP32", "GPU FP16", "GPU FP8", "GPU FP4"]

    def _cell(v: str) -> Text:
        if v == "hw unsupported": return Text("hw unsupported", style="dim red")
        if v == "ctx error":      return Text("ctx error",      style="dim red")
        if v == "unavailable":    return Text("unavailable",    style="dim")
        if v == "—":              return Text("—",              style="dim")
        return Text(v)

    tbl = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", expand=True)
    tbl.add_column("Benchmark", style="bold", width=14, no_wrap=True)
    tbl.add_column("Tier",      width=7,      no_wrap=True)
    for col in COLS:
        tbl.add_column(col, justify="right", min_width=12, no_wrap=True)

    prev_result_name = None
    for b in BENCHMARKS:
        for tier in TIERS:
            label = b.label if b.result_name != prev_result_name else ""
            prev_result_name = b.result_name

            if b.result_name == "memory_bandwidth":
                # Memory row: transfer speeds span the result columns (dtype-independent)
                v = table_cells.get((b.result_name, tier, "GPU FP32"), "—")
                span = Text(v, style="dim" if v == "—" else "")
                tbl.add_row(label, Text(tier, style=TIER_STYLE.get(tier, "")),
                            span, Text(""), Text(""), Text(""), Text(""))
                continue

            tbl.add_row(
                label,
                Text(tier, style=TIER_STYLE.get(tier, "")),
                *[_cell(table_cells.get((b.result_name, tier, col), "—")) for col in COLS],
            )

    footnote = Text("\n H2D = Host→GPU  D2H = GPU→Host  D2D = GPU internal copy", style="dim italic")
    return Panel(Group(tbl, footnote), title="[bold]Results[/bold]", box=box.ROUNDED)


# ── Report ───────────────────────────────────────────────────────────────────

def generate_report(results: list, device_name: str, out_path: Path):
    lines = [
        "# mlmark — Benchmark Report", "",
        f"**Device:** {device_name}  ",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        "", "---", "",
    ]

    matmul = [r for r in results if r.get("benchmark") == "matmul"]
    cnn    = [r for r in results if r.get("benchmark") == "cnn_resnet50"]
    tfm    = [r for r in results if r.get("benchmark") == "transformer_gpt2"]
    mem    = [r for r in results if r.get("benchmark") == "memory_bandwidth"]

    def dtype_label(r):
        return r.get("dtype_label", "FP32")

    def _lookup(valid, dev, dt):
        return next((r for r in valid if r["device"] == dev and dtype_label(r) == dt), None)

    if matmul:
        lines += ["## Matrix Multiplication (TFLOPS)", "",
                  "| Tier | Size | CPU FP32 | GPU FP32 | GPU FP16 |",
                  "|------|------|----------|----------|----------|"]
        for tier in TIERS:
            valid = [r for r in matmul if r.get("tier") == tier and "error" not in r]
            size  = next((r["size"] for r in valid), "")
            cpu32, gpu32, gpu16 = _lookup(valid,"cpu","FP32"), _lookup(valid,"cuda","FP32"), _lookup(valid,"cuda","FP16")
            c = f"{cpu32['tflops']:.3f}" if cpu32 else "—"
            g = f"{gpu32['tflops']:.3f}" if gpu32 else "—"
            h = f"{gpu16['tflops']:.3f}" if gpu16 else "—"
            lines.append(f"| {tier} | {size} | {c} | {g} | {h} |")
        lines.append("")

    if cnn:
        lines += ["## CNN Inference — ResNet-50 (img/s)", "",
                  "| Tier | Batch | CPU FP32 | GPU FP32 | GPU FP16 |",
                  "|------|-------|----------|----------|----------|"]
        for tier in TIERS:
            valid = [r for r in cnn if r.get("tier") == tier and "error" not in r]
            batch = next((r["batch_size"] for r in valid), "")
            cpu32, gpu32, gpu16 = _lookup(valid,"cpu","FP32"), _lookup(valid,"cuda","FP32"), _lookup(valid,"cuda","FP16")
            c = f"{cpu32['images_per_sec']:.1f}" if cpu32 else "—"
            g = f"{gpu32['images_per_sec']:.1f}" if gpu32 else "—"
            h = f"{gpu16['images_per_sec']:.1f}" if gpu16 else "—"
            lines.append(f"| {tier} | {batch} | {c} | {g} | {h} |")
        lines.append("")

    if tfm:
        lines += ["## Transformer Inference — GPT-2 scale (tokens/s)", "",
                  "| Tier | Seq len | CPU FP32 | GPU FP32 | GPU FP16 |",
                  "|------|---------|----------|----------|----------|"]
        for tier in TIERS:
            valid = [r for r in tfm if r.get("tier") == tier and "error" not in r]
            seq   = next((r["seq_len"] for r in valid), "")
            cpu32, gpu32, gpu16 = _lookup(valid,"cpu","FP32"), _lookup(valid,"cuda","FP32"), _lookup(valid,"cuda","FP16")
            c = f"{cpu32['tokens_per_sec']:.0f}" if cpu32 else "—"
            g = f"{gpu32['tokens_per_sec']:.0f}" if gpu32 else "—"
            h = f"{gpu16['tokens_per_sec']:.0f}" if gpu16 else "—"
            lines.append(f"| {tier} | {seq} | {c} | {g} | {h} |")
        lines.append("")

    if mem:
        lines += ["## Memory Bandwidth (GB/s)", "",
                  "| Tier | Size | H2D | D2H | D2D |",
                  "|------|------|-----|-----|-----|"]
        for tier in TIERS:
            r = next((x for x in mem if x.get("tier") == tier and "error" not in x), None)
            if r:
                t = r.get("transfers", {})
                lines.append(f"| {tier} | {r['size_mb']} MB"
                             f" | {t.get('H2D','—')} | {t.get('D2H','—')} | {t.get('D2D','—')} |")
        lines += ["",
                  "> **H2D** Host→GPU  **D2H** GPU→Host  **D2D** GPU internal — "
                  "on iGPU all share physical RAM, so H2D/D2H are CPU-controller bound "
                  "(~2-3 GB/s) while D2D is a fast memcopy (~290 GB/s).", ""]

    lines += ["## Peak Results", ""]
    peaks = []
    valid_matmul = [r for r in matmul if "error" not in r]
    if valid_matmul:
        best = max(valid_matmul, key=lambda r: r.get("tflops", 0))
        dev = "GPU" if best["device"] == "cuda" else "CPU"
        peaks.append(f"- **Matmul:** {best['tflops']:.3f} TFLOPS  ({dev} {dtype_label(best)}, N={best['size']})")

    valid_cnn = [r for r in cnn if "error" not in r]
    if valid_cnn:
        best = max(valid_cnn, key=lambda r: r.get("images_per_sec", 0))
        dev = "GPU" if best["device"] == "cuda" else "CPU"
        peaks.append(f"- **CNN ResNet-50:** {best['images_per_sec']:.1f} img/s  ({dev} {dtype_label(best)}, batch={best['batch_size']})")

    valid_tfm = [r for r in tfm if "error" not in r]
    if valid_tfm:
        best = max(valid_tfm, key=lambda r: r.get("tokens_per_sec", 0))
        dev = "GPU" if best["device"] == "cuda" else "CPU"
        peaks.append(f"- **Transformer:** {best['tokens_per_sec']:.0f} tok/s  ({dev} {dtype_label(best)}, seq={best['seq_len']})")

    valid_mem = [r for r in mem if "error" not in r]
    if valid_mem:
        best = max(valid_mem, key=lambda r: r.get("transfers", {}).get("D2D", 0))
        peaks.append(f"- **Memory D2D:** {best['transfers']['D2D']:.1f} GB/s  (GPU, {best['size_mb']} MB)")

    lines += peaks
    lines.append("")
    out_path.write_text("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="mlmark benchmark runner")
    parser.add_argument("-o", "--output", required=True, help="Results output directory")
    parser.add_argument(
        "--tiers", choices=["small", "medium", "all"], default="all",
        help="Which size tiers to run: small | medium (small+medium) | all (default)",
    )
    args = parser.parse_args()

    # Filter active tiers based on --tiers flag
    global TIERS
    if args.tiers == "small":
        TIERS = ["small"]
    elif args.tiers == "medium":
        TIERS = ["small", "medium"]
    # else: keep ["small", "medium", "large"]

    out_dir = Path(args.output) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = torch.cuda.get_device_name(0) if HAS_GPU else "CPU only"

    # ── Pre-flight dtype probes ───────────────────────────────────────────────
    # Run a tiny op for each GPU dtype before any benchmark.
    # Marks all affected (bench, tier) cells immediately — no wasted job time.
    hw_unsupported_specs: set = set()

    def _probe_dtype(dtype) -> bool:
        """Return True if dtype is usable for GPU compute."""
        try:
            if dtype in (getattr(torch, "float8_e4m3fn", None),
                         getattr(torch, "float8_e5m2",   None)):
                a = torch.zeros(16, 16, device="cuda").to(dtype)
                s = torch.tensor(1.0, device="cuda")
                torch._scaled_mm(a, a.t(), scale_a=s, scale_b=s,
                                 out_dtype=torch.float16)
            else:
                torch.zeros(8, 8, device="cuda").to(dtype)
            torch.cuda.synchronize()
            return True
        except Exception:
            try:
                torch.cuda.empty_cache()
                torch.zeros(1, device="cuda")
                torch.cuda.synchronize()
            except Exception:
                pass
            return False

    if HAS_GPU:
        for spec in ("gpu/fp8", "gpu/fp4"):
            dtype = DTYPE_MAP.get(spec.split("/")[1])
            if dtype is None:
                continue
            if not _probe_dtype(dtype):
                hw_unsupported_specs.add(spec)
                console.print(f"[dim]Pre-flight: {spec.upper()} → hw unsupported[/dim]")

    monitor = SystemMonitor(interval=0.5)
    monitor.start()

    all_results = []

    # Pre-populate display table: every cell starts as "—", "hw unsupported", or "unavailable"
    table_cells: dict = {}
    for b in BENCHMARKS:
        for tier in TIERS:
            for spec in b.specs:
                key = (b.result_name, tier, SPEC_TO_COL[spec])
                if device_label(spec) is None:
                    table_cells[key] = "unavailable"
                elif spec in hw_unsupported_specs:
                    table_cells[key] = "hw unsupported"
                else:
                    table_cells[key] = "—"

    # Round-robin job list: all small → all medium → all large
    jobs = [
        (tier, b.label, b.module, spec)
        for tier in TIERS
        for b in BENCHMARKS
        for spec in b.specs
    ]

    current_tier     = TIERS[0]
    current_job      = ()
    done_jobs        = set()
    unavailable_jobs = {
        (tier, b.label, spec)
        for tier in TIERS
        for b in BENCHMARKS
        for spec in b.specs
        if device_label(spec) is None or spec in hw_unsupported_specs
    }
    completed  = 0
    total_jobs = sum(
        1 for tier, _, _, spec in jobs
        if device_label(spec) is not None
    )
    start_time  = time.time()
    interrupted = False

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=2),
        Layout(name="results", ratio=3),
    )
    layout["body"].split_row(
        Layout(name="metrics", ratio=1),
        Layout(name="progress", ratio=3),
    )

    def refresh_layout():
        tier_idx = TIERS.index(current_tier)
        layout["header"].update(make_header(device_name, current_tier, tier_idx))
        layout["metrics"].update(make_metrics_panel(monitor))
        layout["body"]["progress"].update(
            make_progress_panel(jobs, done_jobs, unavailable_jobs, current_job,
                                completed, total_jobs, start_time)
        )
        layout["results"].update(make_results_table(table_cells))

    gpu_dead = False
    smi_snapshot_done = False

    def _save_smi_snapshot(path: Path):
        """Capture rocm-smi or nvidia-smi output to file while GPU is under load."""
        lines = []
        for cmd in (["rocm-smi"], ["nvidia-smi"]):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if out.returncode == 0 and out.stdout.strip():
                    lines.append(f"# {' '.join(cmd)}\n{out.stdout}")
                    break
            except (FileNotFoundError, subprocess.SubprocessError):
                continue
        if lines:
            path.write_text("\n".join(lines))

    def probe_gpu() -> bool:
        try:
            torch.zeros(1, device="cuda"); torch.cuda.synchronize(); return True
        except Exception:
            return False

    def reset_gpu() -> bool:
        try:
            torch.cuda.empty_cache(); torch.cuda.synchronize(); return probe_gpu()
        except Exception:
            return False

    try:
        with Live(layout, refresh_per_second=4, screen=False):
            for tier, bench_label, mod, spec in jobs:
                parsed = device_label(spec)
                if parsed is None:
                    continue
                device, dtype = parsed
                dt_label     = spec.split("/")[1].upper()
                result_name  = RESULT_NAME[bench_label]

                if spec in hw_unsupported_specs:
                    done_jobs.add((tier, bench_label, spec)); completed += 1; refresh_layout(); continue

                if gpu_dead and device == "cuda":
                    table_cells[(result_name, tier, SPEC_TO_COL[spec])] = "ctx error"
                    done_jobs.add((tier, bench_label, spec)); completed += 1; refresh_layout(); continue

                current_tier = tier
                current_job  = (tier, bench_label, spec)
                refresh_layout()
                monitor.mark(bench_label, tier)

                result_box = [None]
                error_box  = [None]
                def _run(mod=mod, tier=tier, device=device, dtype=dtype):
                    try:
                        result_box[0] = mod.run_one(tier, device, dtype)
                    except Exception as e:
                        error_box[0] = e
                t = threading.Thread(target=_run, daemon=True)
                t.start()
                job_start = time.time()
                while t.is_alive():
                    time.sleep(0.1); refresh_layout()
                    if (not smi_snapshot_done and device == "cuda"
                            and tier == TIERS[-1]
                            and time.time() - job_start > 1.0):
                        threading.Thread(
                            target=_save_smi_snapshot,
                            args=(out_dir / "gpu_snapshot.txt",),
                            daemon=True,
                        ).start()
                        smi_snapshot_done = True

                if error_box[0] is not None:
                    err_str = str(error_box[0])
                    # Write error into display table
                    table_cells[(result_name, tier, SPEC_TO_COL[spec])] = "ctx error" if "ctx" in err_str else "error"
                    if device == "cuda" and "ctx" not in err_str:
                        # First runtime failure for this spec: mark all pending cells hw unsupported
                        hw_unsupported_specs.add(spec)
                        col = SPEC_TO_COL[spec]
                        for _tier in TIERS:
                            for b in BENCHMARKS:
                                if spec in b.specs:
                                    k = (b.result_name, _tier, col)
                                    if table_cells.get(k) == "—":
                                        table_cells[k] = "hw unsupported"
                    if device == "cuda" and not probe_gpu() and not reset_gpu():
                        gpu_dead = True
                else:
                    result = result_box[0]
                    result["dtype_label"] = dt_label
                    all_results.append(result)
                    table_cells[(result_name, tier, SPEC_TO_COL[spec])] = _result_key(result)

                monitor.mark(f"{bench_label}_end", tier)
                done_jobs.add((tier, bench_label, spec)); completed += 1; refresh_layout()

    except KeyboardInterrupt:
        interrupted = True
        console.print("\n[yellow]Interrupted — saving partial results...[/yellow]")

    finally:
        # Block further Ctrl+C while saving to prevent exception corruption
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        monitor.stop()

        if not all_results:
            console.print("[dim]No results to save.[/dim]")
        else:
            status = "partial" if interrupted else "complete"
            console.print(f"\n[bold]Run {status}:[/bold] {len(all_results)} results")

            results_path = out_dir / "results.json"
            results_path.write_text(json.dumps(all_results, indent=2))
            console.print(f"[green]Results:[/green] {results_path}")

            graph_path = out_dir / "metrics.png"
            try:
                monitor.save_graph(str(graph_path))
                console.print(f"[green]Graph:  [/green] {graph_path}")
            except Exception as e:
                console.print(f"[yellow]Graph failed:[/yellow] {e}")

            report_path = out_dir / "report.md"
            try:
                generate_report(all_results, device_name, report_path)
                console.print(f"[green]Report: [/green] {report_path}")
            except Exception as e:
                console.print(f"[yellow]Report failed:[/yellow] {e}")

            snap_path = out_dir / "gpu_snapshot.txt"
            if snap_path.exists():
                console.print(f"[green]GPU snap:[/green] {snap_path}")

        if interrupted:
            # os._exit bypasses Python thread cleanup, preventing ROCm crash
            # when the daemon benchmark thread is killed mid-GPU-op
            os._exit(0)


if __name__ == "__main__":
    main()
