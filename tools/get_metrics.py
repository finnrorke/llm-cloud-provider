#!/usr/bin/env python3
import collections
import curses
import datetime as dt
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from common import load_json, require_run_config_path, run_config_args


COLOR_NAME_TO_CURSES = {
    "black": curses.COLOR_BLACK,
    "red": curses.COLOR_RED,
    "green": curses.COLOR_GREEN,
    "yellow": curses.COLOR_YELLOW,
    "blue": curses.COLOR_BLUE,
    "magenta": curses.COLOR_MAGENTA,
    "cyan": curses.COLOR_CYAN,
    "white": curses.COLOR_WHITE,
}


@dataclass
class MetricsConfig:
    namespace: str
    target: str
    base_url: str
    metrics_url: str
    model_name: str
    interval: float
    window: int
    color_queue: str
    color_running: str
    color_prompt: str
    color_generation: str


DEFAULTS = {
    "namespace": "llm-amd",
    "target": "deploy/qwen3-0-8b-vllm",
    "base_url": "",
    "metrics_url": "http://127.0.0.1:8000/metrics",
    "model_name": "",
    "interval": 0.5,
    "window": 60,
    "color_queue": "yellow",
    "color_running": "green",
    "color_prompt": "magenta",
    "color_generation": "cyan",
}


def load_dashboard_config(argv=None):
    run_cfg_path = require_run_config_path(argv, script_name="get_metrics.py")

    run_cfg = load_json(run_cfg_path, required=True)
    run_args = run_config_args(run_cfg)
    defaults = dict(DEFAULTS)
    defaults.update({key: value for key, value in run_args.items() if key in defaults})
    interval = float(defaults["interval"])
    window = int(defaults["window"])

    if interval <= 0:
        raise RuntimeError("interval must be > 0")
    if window < 2:
        raise RuntimeError("window must be >= 2")

    return MetricsConfig(
        namespace=str(defaults["namespace"]),
        target=str(defaults["target"]),
        base_url=str(defaults["base_url"]),
        metrics_url=str(defaults["metrics_url"]),
        model_name=str(defaults["model_name"]),
        interval=interval,
        window=window,
        color_queue=str(defaults["color_queue"]),
        color_running=str(defaults["color_running"]),
        color_prompt=str(defaults["color_prompt"]),
        color_generation=str(defaults["color_generation"]),
    )


def fetch_raw_metrics(cfg: MetricsConfig):
    if cfg.base_url:
        url = f"{cfg.base_url.rstrip('/')}/metrics"
        with urllib.request.urlopen(url, timeout=4) as response:
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


def parse_vllm_metrics(raw, model_name):
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

        try:
            value = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue

        if line.startswith("vllm:num_requests_waiting{"):
            totals["waiting"] += value
            found_queue_metric = True
        elif line.startswith("vllm:num_requests_running{"):
            totals["running"] += value
            found_queue_metric = True
        elif line.startswith("vllm:kv_cache_usage_perc{"):
            totals["kv_cache_usage_perc"] += value
        elif line.startswith("vllm:prompt_tokens_total{"):
            totals["prompt_tokens_total"] += value
        elif line.startswith("vllm:generation_tokens_total{"):
            totals["generation_tokens_total"] += value
        elif line.startswith("vllm:request_success_total{"):
            totals["request_success_total"] += value

    if not found_queue_metric:
        raise RuntimeError("No vLLM queue metrics found for this target/model filter.")

    return totals


def safe_rate(current, previous, dt_s):
    if previous is None or dt_s <= 0:
        return 0.0
    return max((current - previous) / dt_s, 0.0)


def resample(history, width):
    if width <= 0:
        return []
    values = list(history)
    if not values:
        return [0.0] * width

    if len(values) <= width:
        return [0.0] * (width - len(values)) + values

    total = len(values)
    bucketed = []
    for idx in range(width):
        start = (idx * total) // width
        end = ((idx + 1) * total) // width
        if end <= start:
            end = min(start + 1, total)
        bucketed.append(max(values[start:end]))
    return bucketed


def make_box(title, lines, width):
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


def graph_box(title, history, width, height):
    raw_values = list(history)
    values = resample(raw_values, width)
    raw_max = max(raw_values, default=0.0)
    now_value = raw_values[-1] if raw_values else 0.0
    vmax = max(raw_max, 1.0)

    rows = []
    for level in range(height, 0, -1):
        threshold = vmax * (level / height)
        rows.append("".join("#" if value >= threshold else " " for value in values))
    rows.append(f"max={raw_max:.2f} now={now_value:.2f}")
    return make_box(title, rows, width + 2)


def draw_panel_row(stdscr, start_y, left_lines, right_lines, left_attr, right_attr):
    rows, cols = stdscr.getmaxyx()
    left_width = max((len(line) for line in left_lines), default=0)
    right_x = left_width + 2

    for idx, (left, right) in enumerate(zip(left_lines, right_lines)):
        y = start_y + idx
        if y >= rows - 1:
            break
        stdscr.addnstr(y, 0, left, max(cols - 1, 1), left_attr)
        if right_x < cols - 1:
            stdscr.addnstr(y, right_x, right, max(cols - right_x - 1, 1), right_attr)

    return start_y + min(len(left_lines), len(right_lines))


def init_color_pairs(cfg: MetricsConfig):
    pairs = {"queue": 0, "running": 0, "prompt": 0, "generation": 0}
    if not curses.has_colors():
        return pairs

    curses.start_color()
    curses.use_default_colors()

    wanted = {
        "queue": cfg.color_queue.lower(),
        "running": cfg.color_running.lower(),
        "prompt": cfg.color_prompt.lower(),
        "generation": cfg.color_generation.lower(),
    }

    idx = 1
    for key, color_name in wanted.items():
        color_val = COLOR_NAME_TO_CURSES.get(color_name)
        if color_val is None:
            continue
        curses.init_pair(idx, color_val, -1)
        pairs[key] = curses.color_pair(idx)
        idx += 1

    return pairs


def run_dashboard(cfg: MetricsConfig):
    if not sys.stdout.isatty():
        raise RuntimeError("Dashboard mode requires an interactive terminal (TTY).")

    stop = False

    def _stop(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    wait_hist = collections.deque(maxlen=cfg.window)
    run_hist = collections.deque(maxlen=cfg.window)
    prompt_hist = collections.deque(maxlen=cfg.window)
    generation_hist = collections.deque(maxlen=cfg.window)

    prev_totals = None
    prev_ts = None
    last_metrics_error = ""

    def _loop(stdscr):
        nonlocal prev_totals, prev_ts, last_metrics_error, stop
        curses.curs_set(0)
        stdscr.nodelay(True)
        colors = init_color_pairs(cfg)

        while not stop:
            if stdscr.getch() in (ord("q"), ord("Q")):
                break

            now = time.time()
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

            totals = None
            try:
                totals = parse_vllm_metrics(fetch_raw_metrics(cfg), cfg.model_name)
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
            prompt_tps = safe_rate(
                totals["prompt_tokens_total"],
                prev_totals["prompt_tokens_total"] if prev_totals else None,
                dt_s,
            )
            generation_tps = safe_rate(
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
            prompt_hist.append(prompt_tps)
            generation_hist.append(generation_tps)

            rows, cols = stdscr.getmaxyx()
            graph_width = max((cols // 2) - 4, 28)
            graph_height = 8

            source = "direct" if cfg.base_url else "kubectl"
            header = [
                "vLLM Dashboard (q to quit)",
                f"time={stamp} interval={cfg.interval:.2f}s window={cfg.window} source={source}",
                (
                    f"load: waiting={totals['waiting']:.2f} running={totals['running']:.2f} "
                    f"kv_cache={totals['kv_cache_usage_perc'] * 100.0:.1f}%"
                ),
                (
                    f"rates: req_rps={req_rps:.2f} prompt_tps={prompt_tps:.2f} "
                    f"generation_tps={generation_tps:.2f}"
                ),
            ]

            if cfg.model_name:
                header.append(f"model_filter={cfg.model_name}")
            if last_metrics_error:
                header.append(f"metrics_error={last_metrics_error}")

            top_left = graph_box("Queue Waiting", wait_hist, graph_width, graph_height)
            top_right = graph_box("Requests Running", run_hist, graph_width, graph_height)
            bottom_left = graph_box("Prompt Tokens/s", prompt_hist, graph_width, graph_height)
            bottom_right = graph_box("Generation Tokens/s", generation_hist, graph_width, graph_height)

            stdscr.erase()
            y = 0
            for line in header:
                if y >= rows - 1:
                    break
                stdscr.addnstr(y, 0, line, max(cols - 1, 1))
                y += 1
            if y < rows - 1:
                y += 1

            y = draw_panel_row(stdscr, y, top_left, top_right, colors["queue"], colors["running"])
            if y < rows - 1:
                y += 1
            draw_panel_row(stdscr, y, bottom_left, bottom_right, colors["prompt"], colors["generation"])

            stdscr.refresh()
            prev_totals = totals
            prev_ts = now
            time.sleep(cfg.interval)

    curses.wrapper(_loop)


def main(argv=None):
    cfg = load_dashboard_config(argv)
    run_dashboard(cfg)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print((exc.output or str(exc)).strip(), file=sys.stderr)
        sys.exit(1)
