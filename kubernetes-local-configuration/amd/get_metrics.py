#!/usr/bin/env python3
import argparse
import collections
import curses
import datetime as dt
import json
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class DashboardConfig:
    namespace: str
    target: str
    base_url: str
    metrics_url: str
    model_name: str
    rocm_smi_path: str
    interval: float
    window: int


def parse_args() -> DashboardConfig:
    parser = argparse.ArgumentParser(description="Live vLLM dashboard (dashboard mode only).")
    parser.add_argument("--namespace", default="llm-amd")
    parser.add_argument("--target", default="deploy/qwen3-0-8b-vllm")
    parser.add_argument("--base-url", default="", help="Direct vLLM base URL, e.g. http://10.43.167.147:8000")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--rocm-smi-path", default="/opt/rocm/bin/rocm-smi")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=60)
    args = parser.parse_args()

    if args.interval <= 0:
        raise RuntimeError("--interval must be > 0")
    if args.window < 2:
        raise RuntimeError("--window must be >= 2")

    return DashboardConfig(
        namespace=args.namespace,
        target=args.target,
        base_url=args.base_url,
        metrics_url=args.metrics_url,
        model_name=args.model_name,
        rocm_smi_path=args.rocm_smi_path,
        interval=args.interval,
        window=args.window,
    )


def fetch_raw_metrics(cfg: DashboardConfig) -> str:
    if cfg.base_url:
        direct_url = f"{cfg.base_url.rstrip('/')}/metrics"
        with urllib.request.urlopen(direct_url, timeout=4) as response:
            return response.read().decode("utf-8", errors="replace")

    cmd = [
        "kubectl",
        "-n",
        cfg.namespace,
        "exec",
        cfg.target,
        "--",
        "sh",
        "-lc",
        f"wget -qO- {cfg.metrics_url}",
    ]
    return subprocess.check_output(cmd, text=True)


def parse_vllm_metrics(raw: str, model_name: str) -> dict:
    totals = {
        "waiting": 0.0,
        "running": 0.0,
        "kv_cache_usage_perc": 0.0,
        "prompt_tokens_total": 0.0,
        "generation_tokens_total": 0.0,
        "request_success_total": 0.0,
    }

    found_queue_metric = False

    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        if model_name and f'model_name="{model_name}"' not in line:
            continue

        value = line.rsplit(" ", 1)[-1]
        try:
            numeric = float(value)
        except ValueError:
            continue

        if line.startswith("vllm:num_requests_waiting{"):
            totals["waiting"] += numeric
            found_queue_metric = True
        elif line.startswith("vllm:num_requests_running{"):
            totals["running"] += numeric
            found_queue_metric = True
        elif line.startswith("vllm:kv_cache_usage_perc{"):
            totals["kv_cache_usage_perc"] += numeric
        elif line.startswith("vllm:prompt_tokens_total{"):
            totals["prompt_tokens_total"] += numeric
        elif line.startswith("vllm:generation_tokens_total{"):
            totals["generation_tokens_total"] += numeric
        elif line.startswith("vllm:request_success_total{"):
            totals["request_success_total"] += numeric

    if not found_queue_metric:
        raise RuntimeError("No vLLM queue metrics found for this target/model filter.")

    return totals


def fetch_gpu_stats(rocm_smi_path: str) -> tuple[dict | None, str]:
    cmd = [
        rocm_smi_path,
        "--showuse",
        "--showmemuse",
        "--showmeminfo",
        "vram",
        "--showtemp",
        "--showpower",
        "--json",
    ]

    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE, timeout=2.5)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)

    json_start = output.find("{")
    if json_start < 0:
        return None, "rocm-smi returned no JSON payload"

    try:
        payload = json.loads(output[json_start:])
    except json.JSONDecodeError:
        return None, "failed to parse rocm-smi JSON"

    if not payload:
        return None, "rocm-smi JSON had no GPU entries"

    card = next(iter(payload.values()))
    used_b = float(card.get("VRAM Total Used Memory (B)", "0") or 0)
    total_b = float(card.get("VRAM Total Memory (B)", "0") or 0)

    stats = {
        "gpu_use_pct": float(card.get("GPU use (%)", "0") or 0),
        "vram_pct": float(card.get("GPU Memory Allocated (VRAM%)", "0") or 0),
        "power_w": float(card.get("Average Graphics Package Power (W)", "0") or 0),
        "temp_c": float(card.get("Temperature (Sensor junction) (C)", "0") or 0),
        "vram_used_gb": used_b / (1024**3) if used_b else 0.0,
        "vram_total_gb": total_b / (1024**3) if total_b else 0.0,
    }
    return stats, ""


def init_color_pairs() -> dict:
    pairs = {"queue": 0, "running": 0, "prompt": 0, "generation": 0}
    if not curses.has_colors():
        return pairs

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_MAGENTA, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)

    pairs["queue"] = curses.color_pair(1)
    pairs["running"] = curses.color_pair(2)
    pairs["prompt"] = curses.color_pair(3)
    pairs["generation"] = curses.color_pair(4)
    return pairs


def safe_rate(current: float, previous: float | None, dt_s: float) -> float:
    if previous is None or dt_s <= 0:
        return 0.0
    return max((current - previous) / dt_s, 0.0)


def resample(values: collections.deque, width: int) -> list[float]:
    if width <= 0:
        return []
    if not values:
        return [0.0] * width

    seq = list(values)
    if len(seq) <= width:
        return [0.0] * (width - len(seq)) + seq

    step = len(seq) / width
    return [seq[int(i * step)] for i in range(width)]


def make_box(title: str, lines: list[str], width: int) -> list[str]:
    inside = max(width - 2, 1)
    top = "+" + "-" * inside + "+"
    marker = f" {title} "
    if len(marker) < inside:
        left = (inside - len(marker)) // 2
        top = "+" + "-" * left + marker + "-" * (inside - left - len(marker)) + "+"

    out = [top]
    for line in lines:
        clipped = line[:inside]
        out.append("|" + clipped + " " * (inside - len(clipped)) + "|")
    out.append("+" + "-" * inside + "+")
    return out


def graph_box(title: str, history: collections.deque, width: int, height: int) -> list[str]:
    values = resample(history, width)
    vmax = max(max(values, default=0.0), 1.0)
    rows = []
    for level in range(height, 0, -1):
        threshold = vmax * (level / height)
        rows.append("".join("#" if value >= threshold else " " for value in values))
    rows.append(f"max={vmax:.2f} now={values[-1] if values else 0.0:.2f}")
    return make_box(title, rows, width + 2)


def draw_panel_row(
    stdscr,
    start_y: int,
    left_lines: list[str],
    right_lines: list[str],
    left_attr: int,
    right_attr: int,
) -> int:
    rows, cols = stdscr.getmaxyx()
    left_width = max((len(line) for line in left_lines), default=0)
    gap = 2

    for idx, (left, right) in enumerate(zip(left_lines, right_lines)):
        y = start_y + idx
        if y >= rows - 1:
            break
        stdscr.addnstr(y, 0, left, max(cols - 1, 1), left_attr)
        right_x = left_width + gap
        if right_x < cols - 1:
            stdscr.addnstr(y, right_x, right, max(cols - right_x - 1, 1), right_attr)

    return start_y + min(len(left_lines), len(right_lines))


def run_dashboard(cfg: DashboardConfig) -> None:
    if not sys.stdout.isatty():
        raise RuntimeError("Dashboard mode requires an interactive terminal (TTY).")

    stop = False

    def _handle_stop(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    wait_hist = collections.deque(maxlen=cfg.window)
    run_hist = collections.deque(maxlen=cfg.window)
    prompt_tps_hist = collections.deque(maxlen=cfg.window)
    gen_tps_hist = collections.deque(maxlen=cfg.window)

    prev_totals = None
    prev_ts = None
    last_metrics_error = ""
    last_gpu = None
    last_gpu_error = ""
    last_gpu_sample_ts = 0.0

    def _loop(stdscr):
        nonlocal prev_totals, prev_ts, last_metrics_error, last_gpu, last_gpu_error, last_gpu_sample_ts, stop
        curses.curs_set(0)
        stdscr.nodelay(True)
        colors = init_color_pairs()

        while not stop:
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                break

            now = time.time()
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

            totals = None
            try:
                raw = fetch_raw_metrics(cfg)
                totals = parse_vllm_metrics(raw, cfg.model_name)
                last_metrics_error = ""
            except Exception as exc:  # noqa: BLE001
                last_metrics_error = str(exc)

            if totals is None:
                totals = {
                    "waiting": wait_hist[-1] if wait_hist else 0.0,
                    "running": run_hist[-1] if run_hist else 0.0,
                    "kv_cache_usage_perc": prev_totals["kv_cache_usage_perc"] if prev_totals else 0.0,
                    "prompt_tokens_total": prev_totals["prompt_tokens_total"] if prev_totals else 0.0,
                    "generation_tokens_total": prev_totals["generation_tokens_total"] if prev_totals else 0.0,
                    "request_success_total": prev_totals["request_success_total"] if prev_totals else 0.0,
                }

            dt_s = (now - prev_ts) if prev_ts else 0.0
            prompt_tps = safe_rate(totals["prompt_tokens_total"], prev_totals["prompt_tokens_total"] if prev_totals else None, dt_s)
            gen_tps = safe_rate(
                totals["generation_tokens_total"],
                prev_totals["generation_tokens_total"] if prev_totals else None,
                dt_s,
            )
            req_rps = safe_rate(
                totals["request_success_total"],
                prev_totals["request_success_total"] if prev_totals else None,
                dt_s,
            )

            wait_hist.append(totals["waiting"])
            run_hist.append(totals["running"])
            prompt_tps_hist.append(prompt_tps)
            gen_tps_hist.append(gen_tps)

            if now - last_gpu_sample_ts >= 1.0:
                gpu, gpu_err = fetch_gpu_stats(cfg.rocm_smi_path)
                last_gpu_sample_ts = now
                if gpu is not None:
                    last_gpu = gpu
                    last_gpu_error = ""
                else:
                    last_gpu_error = gpu_err

            rows, cols = stdscr.getmaxyx()
            graph_width = max((cols // 2) - 4, 28)
            graph_height = 8

            header = [
                "vLLM Dashboard (q to quit)",
                f"time={stamp} interval={cfg.interval:.2f}s window={cfg.window}",
                f"source={'direct' if cfg.base_url else 'kubectl'} model={cfg.model_name or 'all'}",
                (
                    f"load: waiting={totals['waiting']:.2f} running={totals['running']:.2f} "
                    f"kv_cache={totals['kv_cache_usage_perc'] * 100.0:.1f}%"
                ),
                (
                    f"rates: req_rps={req_rps:.2f} prompt_tps={prompt_tps:.2f} "
                    f"generation_tps={gen_tps:.2f}"
                ),
            ]
            if last_gpu is not None:
                header.append(
                    (
                        f"gpu: use={last_gpu['gpu_use_pct']:.1f}% vram={last_gpu['vram_pct']:.1f}% "
                        f"({last_gpu['vram_used_gb']:.2f}/{last_gpu['vram_total_gb']:.2f} GiB) "
                        f"power={last_gpu['power_w']:.1f}W temp={last_gpu['temp_c']:.1f}C"
                    )
                )
            else:
                header.append(f"gpu: unavailable ({last_gpu_error or 'rocm-smi not readable'})")
            if last_metrics_error:
                header.append(f"metrics_error={last_metrics_error}")

            left_top = graph_box("Queue Waiting", wait_hist, graph_width, graph_height)
            right_top = graph_box("Requests Running", run_hist, graph_width, graph_height)
            left_bottom = graph_box("Prompt Tokens/s", prompt_tps_hist, graph_width, graph_height)
            right_bottom = graph_box("Generation Tokens/s", gen_tps_hist, graph_width, graph_height)

            stdscr.erase()
            y = 0
            for line in header:
                if y >= rows - 1:
                    break
                stdscr.addnstr(y, 0, line, max(cols - 1, 1))
                y += 1
            if y < rows - 1:
                y += 1

            y = draw_panel_row(stdscr, y, left_top, right_top, colors["queue"], colors["running"])
            if y < rows - 1:
                y += 1
            draw_panel_row(stdscr, y, left_bottom, right_bottom, colors["prompt"], colors["generation"])

            stdscr.refresh()
            prev_totals = totals
            prev_ts = now
            time.sleep(cfg.interval)

    curses.wrapper(_loop)


def main() -> None:
    try:
        cfg = parse_args()
        run_dashboard(cfg)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        if exc.output:
            print(exc.output.strip(), file=sys.stderr)
        else:
            print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
