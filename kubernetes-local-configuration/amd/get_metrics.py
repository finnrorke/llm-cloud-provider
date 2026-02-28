#!/usr/bin/env python3
import argparse
import collections
import curses
import datetime as dt
import json
import shutil
import signal
import subprocess
import sys
import time
import urllib.request


def parse_args():
    p = argparse.ArgumentParser(description="Fetch and visualize vLLM queue/load metrics.")
    p.add_argument("--namespace", default="llm-amd")
    p.add_argument("--target", default="deploy/qwen3-0-8b-vllm")
    p.add_argument("--base-url", default="", help="Direct vLLM base URL, e.g. http://10.43.167.147:8000")
    p.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    p.add_argument("--model-name", default="")
    p.add_argument("--rocm-smi-path", default="/opt/rocm/bin/rocm-smi")
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--snapshot", action="store_true", help="One-time snapshot output.")
    p.add_argument("--dashboard", action="store_true", help="Btop-style dashboard mode (default).")
    return p.parse_args()


def fetch_raw_metrics(namespace, target, metrics_url, base_url):
    if base_url:
        direct_url = f"{base_url.rstrip('/')}/metrics"
        with urllib.request.urlopen(direct_url, timeout=5) as resp:
            return resp.read().decode("utf-8", errors="replace")

    cmd = [
        "kubectl",
        "-n",
        namespace,
        "exec",
        target,
        "--",
        "sh",
        "-lc",
        f"wget -qO- {metrics_url}",
    ]
    return subprocess.check_output(cmd, text=True)


def parse_metrics(raw, model_name):
    waiting = 0.0
    running = 0.0
    kv_cache_usage_perc = 0.0
    prompt_tokens_total = 0.0
    generation_tokens_total = 0.0
    req_success_total = 0.0
    queue_lines = []

    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        if model_name and f'model_name="{model_name}"' not in line:
            continue

        if line.startswith("vllm:num_requests_waiting{"):
            waiting += float(line.rsplit(" ", 1)[-1])
            queue_lines.append(line)
        elif line.startswith("vllm:num_requests_running{"):
            running += float(line.rsplit(" ", 1)[-1])
            queue_lines.append(line)
        elif line.startswith("vllm:prompt_tokens_total{"):
            prompt_tokens_total += float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:generation_tokens_total{"):
            generation_tokens_total += float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:kv_cache_usage_perc{"):
            kv_cache_usage_perc += float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:request_success_total{"):
            req_success_total += float(line.rsplit(" ", 1)[-1])

    if not queue_lines:
        raise RuntimeError("No vLLM queue metrics found for this target/model filter.")

    return {
        "queue_lines": queue_lines,
        "waiting": waiting,
        "running": running,
        "kv_cache_usage_perc": kv_cache_usage_perc,
        "prompt_tokens_total": prompt_tokens_total,
        "generation_tokens_total": generation_tokens_total,
        "req_success_total": req_success_total,
    }


def fetch_gpu_stats(rocm_smi_path):
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
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE, timeout=2.0)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)

    start = raw.find("{")
    if start < 0:
        return None, "rocm-smi returned no JSON payload"
    try:
        data = json.loads(raw[start:])
    except json.JSONDecodeError:
        return None, "failed to parse rocm-smi JSON"
    if not data:
        return None, "rocm-smi JSON had no GPU entries"

    first_card = next(iter(data.values()))
    used_b = float(first_card.get("VRAM Total Used Memory (B)", "0") or 0)
    total_b = float(first_card.get("VRAM Total Memory (B)", "0") or 0)
    return {
        "gpu_use_pct": float(first_card.get("GPU use (%)", "0") or 0),
        "vram_pct": float(first_card.get("GPU Memory Allocated (VRAM%)", "0") or 0),
        "power_w": float(first_card.get("Average Graphics Package Power (W)", "0") or 0),
        "temp_c": float(first_card.get("Temperature (Sensor junction) (C)", "0") or 0),
        "vram_used_gb": (used_b / (1024**3)) if used_b else 0.0,
        "vram_total_gb": (total_b / (1024**3)) if total_b else 0.0,
    }, ""


def draw_lines_curses(stdscr, lines):
    rows, cols = stdscr.getmaxyx()
    stdscr.erase()
    max_lines = max(rows - 1, 1)
    for i, line in enumerate(lines[:max_lines]):
        stdscr.addnstr(i, 0, line, max(cols - 1, 1))
    stdscr.refresh()


def init_color_pairs():
    color_pairs = {
        "default": 0,
        "queue": 0,
        "running": 0,
        "prompt": 0,
        "generation": 0,
    }
    if not curses.has_colors():
        return color_pairs

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_MAGENTA, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)
    color_pairs["queue"] = curses.color_pair(1)
    color_pairs["running"] = curses.color_pair(2)
    color_pairs["prompt"] = curses.color_pair(3)
    color_pairs["generation"] = curses.color_pair(4)
    return color_pairs


def draw_two_panels(stdscr, start_y, left_lines, right_lines, left_attr, right_attr):
    rows, cols = stdscr.getmaxyx()
    left_width = max(len(x) for x in left_lines) if left_lines else 0
    gap = 2
    for i, (l, r) in enumerate(zip(left_lines, right_lines)):
        y = start_y + i
        if y >= rows - 1:
            break
        stdscr.addnstr(y, 0, l, min(len(l), cols - 1), left_attr)
        rx = left_width + gap
        if rx < cols - 1:
            stdscr.addnstr(y, rx, r, min(len(r), cols - 1 - rx), right_attr)
    return start_y + min(len(left_lines), len(right_lines))


def resample(values, width):
    if not values:
        return [0.0] * width
    if len(values) <= width:
        return [0.0] * (width - len(values)) + list(values)
    step = len(values) / width
    out = []
    for i in range(width):
        out.append(values[int(i * step)])
    return out


def render_bar(value, vmax, width):
    if vmax <= 0:
        vmax = 1.0
    filled = int((value / vmax) * width)
    filled = max(0, min(width, filled))
    return "#" * filled + "." * (width - filled)


def make_box(title, lines, width):
    inside = max(width - 2, 1)
    top = "+" + "-" * inside + "+"
    t = f" {title} "
    if len(t) < inside:
        pad_l = (inside - len(t)) // 2
        top = "+" + "-" * pad_l + t + "-" * (inside - pad_l - len(t)) + "+"
    out = [top]
    for line in lines:
        line = line[:inside]
        out.append("|" + line + " " * (inside - len(line)) + "|")
    out.append("+" + "-" * inside + "+")
    return out


def graph_panel(title, values, width, height):
    vals = resample(values, width)
    vmax = max(max(vals), 1.0)
    lines = []
    for r in range(height, 0, -1):
        threshold = vmax * (r / height)
        row = "".join("#" if v >= threshold else " " for v in vals)
        lines.append(row)
    lines.append(f"max={vmax:.2f} now={vals[-1]:.2f}")
    return make_box(title, lines, width + 2)


def safe_rate(curr, prev, dt_s):
    if prev is None or dt_s <= 0:
        return 0.0
    return max((curr - prev) / dt_s, 0.0)


def print_snapshot(args):
    raw = fetch_raw_metrics(args.namespace, args.target, args.metrics_url, args.base_url)
    metrics = parse_metrics(raw, args.model_name)
    gpu, gpu_err = fetch_gpu_stats(args.rocm_smi_path)
    for line in metrics["queue_lines"]:
        print(line)
    print()
    print(f"total_waiting={metrics['waiting']}")
    print(f"total_running={metrics['running']}")
    print(f"kv_cache_usage_pct={metrics['kv_cache_usage_perc'] * 100.0:.2f}")
    if gpu is not None:
        print(f"gpu_use_pct={gpu['gpu_use_pct']:.1f}")
        print(f"vram_pct={gpu['vram_pct']:.1f}")
        print(f"vram_used_gb={gpu['vram_used_gb']:.2f}")
        print(f"vram_total_gb={gpu['vram_total_gb']:.2f}")
        print(f"gpu_power_w={gpu['power_w']:.1f}")
        print(f"gpu_temp_c={gpu['temp_c']:.1f}")
    else:
        print(f"gpu_stats_unavailable={gpu_err}")


def run_dashboard(args):
    if args.window < 2:
        raise RuntimeError("--window must be >= 2")
    if args.interval <= 0:
        raise RuntimeError("--interval must be > 0")

    if not sys.stdout.isatty():
        raise RuntimeError("--dashboard requires an interactive terminal (TTY), not redirected output.")
    stop = False

    def _stop(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    history_wait = collections.deque(maxlen=args.window)
    history_run = collections.deque(maxlen=args.window)
    history_prompt_tps = collections.deque(maxlen=args.window)
    history_gen_tps = collections.deque(maxlen=args.window)
    prev = None
    prev_ts = None
    last_error = ""
    last_gpu = None
    last_gpu_error = ""

    def _loop(stdscr):
        nonlocal prev, prev_ts, last_error, stop
        curses.curs_set(0)
        stdscr.nodelay(True)
        colors = init_color_pairs()
        while not stop:
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                break
            now = time.time()
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
            data = None
            try:
                raw = fetch_raw_metrics(args.namespace, args.target, args.metrics_url, args.base_url)
                data = parse_metrics(raw, args.model_name)
                last_error = ""
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

            if data is None:
                data = {
                    "waiting": history_wait[-1] if history_wait else 0.0,
                    "running": history_run[-1] if history_run else 0.0,
                    "prompt_tokens_total": prev["prompt_tokens_total"] if prev else 0.0,
                    "generation_tokens_total": prev["generation_tokens_total"] if prev else 0.0,
                    "req_success_total": prev["req_success_total"] if prev else 0.0,
                }

            dt_s = (now - prev_ts) if prev_ts else 0.0
            prompt_tps = safe_rate(data["prompt_tokens_total"], prev["prompt_tokens_total"] if prev else None, dt_s)
            gen_tps = safe_rate(data["generation_tokens_total"], prev["generation_tokens_total"] if prev else None, dt_s)
            req_rps = safe_rate(data["req_success_total"], prev["req_success_total"] if prev else None, dt_s)

            history_wait.append(data["waiting"])
            history_run.append(data["running"])
            history_prompt_tps.append(prompt_tps)
            history_gen_tps.append(gen_tps)
            gpu, gpu_err = fetch_gpu_stats(args.rocm_smi_path)
            if gpu is not None:
                last_gpu = gpu
                last_gpu_error = ""
            elif gpu_err:
                last_gpu_error = gpu_err

            cols = shutil.get_terminal_size((120, 40)).columns
            graph_width = max((cols // 2) - 4, 30)
            graph_height = 8

            header = [
                "vLLM Dashboard (press q to exit)",
                f"time={stamp}  interval={args.interval:.1f}s  window={args.window}",
                f"namespace={args.namespace} target={args.target}",
                f"model={args.model_name or 'all'} metrics_url={args.metrics_url}",
                "",
                (
                    f"current: waiting={data['waiting']:.2f} running={data['running']:.2f} "
                    f"req_rps={req_rps:.2f} prompt_tps={prompt_tps:.2f} gen_tps={gen_tps:.2f} "
                    f"kv_cache={data['kv_cache_usage_perc'] * 100.0:.1f}%"
                ),
            ]
            if last_gpu is not None:
                header.append(
                    (
                        f"gpu_use={last_gpu['gpu_use_pct']:.1f}% vram={last_gpu['vram_pct']:.1f}% "
                        f"({last_gpu['vram_used_gb']:.2f}/{last_gpu['vram_total_gb']:.2f} GiB) "
                        f"power={last_gpu['power_w']:.1f}W temp={last_gpu['temp_c']:.1f}C"
                    )
                )
            else:
                header.append(f"gpu_stats_unavailable={last_gpu_error or 'rocm-smi not readable'}")
            if last_error:
                header.append(f"last_error={last_error}")

            left_top = graph_panel("Queue Waiting", history_wait, graph_width, graph_height)
            right_top = graph_panel("Requests Running", history_run, graph_width, graph_height)
            left_bottom = graph_panel("Prompt Tokens/s", history_prompt_tps, graph_width, graph_height)
            right_bottom = graph_panel("Generation Tokens/s", history_gen_tps, graph_width, graph_height)
            stdscr.erase()
            rows, cols = stdscr.getmaxyx()
            y = 0
            for line in header:
                if y >= rows - 1:
                    break
                stdscr.addnstr(y, 0, line, max(cols - 1, 1))
                y += 1
            if y < rows - 1:
                y += 1

            y = draw_two_panels(
                stdscr,
                y,
                left_top,
                right_top,
                colors["queue"],
                colors["running"],
            )
            if y < rows - 1:
                y += 1
            y = draw_two_panels(
                stdscr,
                y,
                left_bottom,
                right_bottom,
                colors["prompt"],
                colors["generation"],
            )
            stdscr.refresh()
            prev = data
            prev_ts = now
            time.sleep(args.interval)

    curses.wrapper(_loop)


def main():
    args = parse_args()
    try:
        if args.snapshot:
            print_snapshot(args)
        else:
            run_dashboard(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(exc.output.strip() if exc.output else str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
