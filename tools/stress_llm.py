#!/usr/bin/env python3
import datetime as dt
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

try:
    from .common import load_json, require_run_config_path, resolve_path, run_config_args
except ImportError:
    from common import load_json, require_run_config_path, resolve_path, run_config_args


DEFAULT_BASE_URL_PROFILES = {
    "localhost": "http://127.0.0.1:8000",
    "cluster-ip": "http://10.43.167.147:8000",
    "k8s-service": "http://qwen3-0-8b-vllm.llm-amd.svc.cluster.local:8000",
}


@dataclass
class StressConfig:
    namespace: str
    base_url_profile: str
    base_url: str
    pod: str
    select_pod: bool
    list_pods: bool
    runs_dir: str
    status: str
    list_runs: bool
    model: str
    duration: int
    messages: int
    burst: bool
    concurrency: int
    qps: float
    timeout: float
    max_tokens: int
    prompt: str


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.issued = 0
        self.total = 0
        self.ok = 0
        self.fail = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.latencies = []

    def try_issue(self, max_messages, stop_at):
        with self.lock:
            if max_messages > 0 and self.issued >= max_messages:
                return False
            if max_messages == 0 and time.time() >= stop_at:
                return False
            self.issued += 1
            return True

    def mark_dispatched(self):
        with self.lock:
            self.issued += 1
            return self.issued

    def record(self, ok, prompt_tokens, completion_tokens, total_tokens, latency_s):
        with self.lock:
            self.total += 1
            if ok:
                self.ok += 1
            else:
                self.fail += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.latencies.append(latency_s)

    def snapshot(self):
        with self.lock:
            return {
                "issued": self.issued,
                "total": self.total,
                "ok": self.ok,
                "fail": self.fail,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "latencies": list(self.latencies),
            }


def percentile(values, pct):
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1 - weight) + values[high] * weight


def classify_latency(p95_s):
    if p95_s <= 2.0:
        return "low"
    if p95_s <= 8.0:
        return "moderate"
    return "high (likely queued)"


def iso_utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


DEFAULTS = {
    "namespace": "llm-amd",
    "base_url_profile": "cluster-ip",
    "base_url": "",
    "pod": "",
    "select_pod": False,
    "list_pods": False,
    "model": "Qwen/Qwen3-0.6B",
    "duration": 120,
    "messages": 0,
    "burst": False,
    "concurrency": 8,
    "qps": 4.0,
    "timeout": 120.0,
    "max_tokens": 128,
    "prompt": "Explain one practical use of queue-depth autoscaling in two sentences.",
    "runs_dir": "../../stress-runs",
    "status": "",
    "list_runs": False,
}


def _as_bool(value, field_name):
    if isinstance(value, bool):
        return value
    raise RuntimeError(f"{field_name} must be true or false")


def load_stress_config(argv=None):
    run_cfg_path = require_run_config_path(argv, script_name="stress_llm.py")

    run_cfg = load_json(run_cfg_path, required=True)
    run_args = run_config_args(run_cfg)
    base_url_profiles = dict(DEFAULT_BASE_URL_PROFILES)
    defaults = dict(DEFAULTS)
    defaults.update({key: value for key, value in run_args.items() if key in defaults})
    if defaults["base_url_profile"] not in base_url_profiles:
        known = ", ".join(sorted(base_url_profiles))
        raise RuntimeError(f"Unknown base_url_profile '{defaults['base_url_profile']}'. Known profiles: {known}")

    run_cfg_dir = Path(run_cfg_path).resolve().parent
    cfg = StressConfig(
        namespace=str(defaults["namespace"]),
        base_url_profile=str(defaults["base_url_profile"]),
        base_url=str(defaults["base_url"]),
        pod=str(defaults["pod"] or ""),
        select_pod=_as_bool(defaults["select_pod"], "select_pod"),
        list_pods=_as_bool(defaults["list_pods"], "list_pods"),
        runs_dir=str(resolve_path(defaults["runs_dir"], base_dir=run_cfg_dir)),
        status=str(defaults["status"] or ""),
        list_runs=_as_bool(defaults["list_runs"], "list_runs"),
        model=str(defaults["model"]),
        duration=int(defaults["duration"]),
        messages=int(defaults["messages"]),
        burst=_as_bool(defaults["burst"], "burst"),
        concurrency=int(defaults["concurrency"]),
        qps=float(defaults["qps"]),
        timeout=float(defaults["timeout"]),
        max_tokens=int(defaults["max_tokens"]),
        prompt=str(defaults["prompt"]),
    )

    if cfg.messages < 0:
        raise RuntimeError("messages must be >= 0")
    if cfg.burst and cfg.messages <= 0:
        raise RuntimeError("burst mode requires messages > 0")
    if cfg.duration <= 0:
        raise RuntimeError("duration must be > 0")
    if cfg.concurrency <= 0:
        raise RuntimeError("concurrency must be > 0")
    if cfg.timeout <= 0:
        raise RuntimeError("timeout must be > 0")
    if cfg.max_tokens <= 0:
        raise RuntimeError("max_tokens must be > 0")
    if not cfg.burst and cfg.qps <= 0:
        raise RuntimeError("qps must be > 0 in paced mode")

    return cfg, base_url_profiles


def ensure_runs_dir(path_value):
    path = Path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_path(runs_dir, run_id):
    return runs_dir / f"{run_id}.json"


def save_run_status(runs_dir, run_id, payload):
    path = run_path(runs_dir, run_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def list_saved_runs(runs_dir):
    files = sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print(f"No saved runs in {runs_dir}")
        return
    for fpath in files:
        try:
            doc = json.loads(fpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"{fpath.stem} invalid-json")
            continue
        print(
            f"{fpath.stem} status={doc.get('status', 'unknown')} "
            f"started_at={doc.get('started_at', '')} completed_at={doc.get('completed_at', '')} "
            f"total={doc.get('total_requests', 0)}"
        )


def show_saved_run(runs_dir, run_id):
    path = run_path(runs_dir, run_id)
    if not path.exists():
        raise RuntimeError(f"Run ID '{run_id}' not found in {runs_dir}")
    print(path.read_text(encoding="utf-8").strip())


def discover_compatible_pods(namespace):
    cmd = ["kubectl", "-n", namespace, "get", "pods", "-o", "json"]
    try:
        payload = subprocess.check_output(cmd, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"Failed to query pods from namespace '{namespace}': {exc}") from exc

    doc = json.loads(payload)
    pods = []
    for item in doc.get("items", []):
        metadata = item.get("metadata", {})
        status = item.get("status", {})
        spec = item.get("spec", {})

        name = metadata.get("name", "")
        ip = status.get("podIP")
        if not name or not ip or status.get("phase") != "Running":
            continue

        conditions = status.get("conditions", [])
        ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)
        if not ready:
            continue

        has_8000 = False
        for container in spec.get("containers", []):
            for port in container.get("ports", []):
                if port.get("containerPort") == 8000:
                    has_8000 = True
                    break
            if has_8000:
                break

        if has_8000:
            pods.append({"name": name, "ip": ip})

    return sorted(pods, key=lambda x: x["name"])


def select_pod_interactively(pods):
    if not pods:
        raise RuntimeError("No compatible pods found.")

    print("Compatible pods:")
    for idx, pod in enumerate(pods, start=1):
        print(f"{idx}. {pod['name']} ({pod['ip']})")

    while True:
        raw = input("Select pod number: ").strip()
        try:
            selected = int(raw)
            if 1 <= selected <= len(pods):
                return pods[selected - 1]
        except ValueError:
            pass
        print("Invalid selection. Try again.")


def resolve_base_url(args, base_url_profiles):
    pods = []
    if args.list_pods or args.select_pod or args.pod:
        pods = discover_compatible_pods(args.namespace)

    if args.list_pods:
        if not pods:
            print(f"No compatible pods found in namespace '{args.namespace}'.")
            sys.exit(1)
        for pod in pods:
            print(f"{pod['name']} {pod['ip']}")
        sys.exit(0)

    base_url = args.base_url or base_url_profiles[args.base_url_profile]
    selected_pod = None

    if args.pod:
        selected = next((p for p in pods if p["name"] == args.pod), None)
        if not selected:
            raise RuntimeError(f"Pod '{args.pod}' is not compatible or not running in namespace '{args.namespace}'.")
        base_url = f"http://{selected['ip']}:8000"
        selected_pod = selected["name"]
    elif args.select_pod:
        selected = select_pod_interactively(pods)
        base_url = f"http://{selected['ip']}:8000"
        selected_pod = selected["name"]

    return base_url, selected_pod


def send_request(endpoint, body_bytes, timeout_s):
    request = urllib.request.Request(
        endpoint,
        data=body_bytes,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        },
        method="POST",
    )

    start = time.time()
    ok = False
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload_raw = response.read()
            ok = 200 <= response.status < 300
            if ok:
                try:
                    usage = json.loads(payload_raw.decode("utf-8")).get("usage", {})
                    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
                except (ValueError, TypeError, json.JSONDecodeError):
                    prompt_tokens = 0
                    completion_tokens = 0
                    total_tokens = 0
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        ok = False

    return ok, prompt_tokens, completion_tokens, total_tokens, time.time() - start


def build_base_status(args, run_id, started_at, base_url, selected_pod, endpoint):
    return {
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "base_url": base_url,
        "selected_pod": selected_pod,
        "endpoint": endpoint,
        "model": args.model,
        "duration_s": args.duration,
        "messages_target": args.messages,
        "burst_mode": args.burst,
        "concurrency": args.concurrency,
        "target_qps": args.qps,
        "timeout_s": args.timeout,
        "max_tokens": args.max_tokens,
        "total_requests": 0,
        "success": 0,
        "fail": 0,
        "success_rate_pct": 0.0,
        "achieved_rps": 0.0,
        "elapsed_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_per_s": 0.0,
        "completion_tokens_per_s": 0.0,
        "total_tokens_per_s": 0.0,
        "latency_avg_s": 0.0,
        "latency_p50_s": 0.0,
        "latency_p95_s": 0.0,
        "latency_p99_s": 0.0,
        "updated_at": iso_utc_now(),
    }


def status_from_state(base_status, snapshot, elapsed_s, status, completed_at=None):
    total = snapshot["total"]
    ok = snapshot["ok"]
    fail = snapshot["fail"]
    prompt_tokens = snapshot["prompt_tokens"]
    completion_tokens = snapshot["completion_tokens"]
    total_tokens = snapshot["total_tokens"]
    latencies = sorted(snapshot["latencies"])

    safe_elapsed = max(elapsed_s, 1e-6)
    success_rate = (ok / total * 100.0) if total else 0.0

    return {
        **base_status,
        "status": status,
        "completed_at": completed_at,
        "total_requests": total,
        "success": ok,
        "fail": fail,
        "success_rate_pct": round(success_rate, 2),
        "achieved_rps": round(total / safe_elapsed, 3),
        "elapsed_s": round(safe_elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens_per_s": round(prompt_tokens / safe_elapsed, 3),
        "completion_tokens_per_s": round(completion_tokens / safe_elapsed, 3),
        "total_tokens_per_s": round(total_tokens / safe_elapsed, 3),
        "latency_avg_s": round(statistics.mean(latencies) if latencies else 0.0, 3),
        "latency_p50_s": round(percentile(latencies, 0.50), 3),
        "latency_p95_s": round(percentile(latencies, 0.95), 3),
        "latency_p99_s": round(percentile(latencies, 0.99), 3),
        "updated_at": iso_utc_now() if status == "running" else completed_at,
    }


def print_report(args, base_url, selected_pod_name, endpoint, started_at, completed_at, final_status):
    print(f"base_url={base_url}")
    if selected_pod_name:
        print(f"selected_pod={selected_pod_name}")
    print(f"endpoint={endpoint}")
    print(f"model={args.model}")
    print(f"duration_s={args.duration}")
    print(f"concurrency={args.concurrency}")
    print(f"target_qps={args.qps}")
    print(f"total_requests={final_status['total_requests']}")
    print(f"success={final_status['success']}")
    print(f"fail={final_status['fail']}")
    print(f"success_rate_pct={final_status['success_rate_pct']:.2f}")
    print(f"achieved_rps={final_status['achieved_rps']:.2f}")
    print(f"prompt_tokens={final_status['prompt_tokens']}")
    print(f"completion_tokens={final_status['completion_tokens']}")
    print(f"total_tokens={final_status['total_tokens']}")
    print(f"prompt_tokens_per_s={final_status['prompt_tokens_per_s']:.2f}")
    print(f"completion_tokens_per_s={final_status['completion_tokens_per_s']:.2f}")
    print(f"total_tokens_per_s={final_status['total_tokens_per_s']:.2f}")
    print(f"latency_avg_s={final_status['latency_avg_s']:.3f}")
    print(f"avg_reply_time_s={final_status['latency_avg_s']:.3f}")
    print(f"latency_p50_s={final_status['latency_p50_s']:.3f}")
    print(f"latency_p95_s={final_status['latency_p95_s']:.3f}")
    print(f"latency_p99_s={final_status['latency_p99_s']:.3f}")
    print(f"started_at={started_at}")
    print(f"completed_at={completed_at}")
    print()
    print("summary:")
    print(
        f"- requests: {final_status['total_requests']} total "
        f"({final_status['success']} success, {final_status['fail']} fail, {final_status['success_rate_pct']:.2f}% success)"
    )
    print(f"- time: {final_status['elapsed_s']:.2f}s ({started_at} -> {completed_at})")
    print(
        f"- throughput: {final_status['achieved_rps']:.2f} req/s, "
        f"{final_status['total_tokens_per_s']:.2f} tok/s"
    )
    print(
        f"- latency: avg {final_status['latency_avg_s']:.3f}s, "
        f"p50 {final_status['latency_p50_s']:.3f}s, "
        f"p95 {final_status['latency_p95_s']:.3f}s, "
        f"p99 {final_status['latency_p99_s']:.3f}s"
    )
    print(f"- latency_profile: {classify_latency(final_status['latency_p95_s'])}")


def main(argv=None):
    args, base_url_profiles = load_stress_config(argv)
    runs_dir = ensure_runs_dir(args.runs_dir)

    if args.list_runs:
        list_saved_runs(runs_dir)
        return
    if args.status:
        show_saved_run(runs_dir, args.status)
        return

    base_url, selected_pod_name = resolve_base_url(args, base_url_profiles)
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    started_at = iso_utc_now()
    started_ts = time.time()
    stop_at = started_ts + args.duration

    base_status = build_base_status(args, run_id, started_at, base_url, selected_pod_name, endpoint)
    save_run_status(runs_dir, run_id, base_status)
    print(f"run_id={run_id}")
    print(f"run_status_file={run_path(runs_dir, run_id)}")

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
    }
    body_bytes = json.dumps(payload).encode("utf-8")

    state = SharedState()
    stop_progress = threading.Event()
    start_burst = threading.Event()
    dispatched_burst = threading.Event()

    per_worker_qps = max(args.qps / max(args.concurrency, 1), 0.001)
    per_worker_interval = 1.0 / per_worker_qps
    start_reference = [started_ts]

    def paced_worker():
        next_fire = time.time() + random.uniform(0, per_worker_interval)
        while state.try_issue(args.messages, stop_at):
            now = time.time()
            if now < next_fire:
                time.sleep(next_fire - now)
            next_fire += per_worker_interval
            state.record(*send_request(endpoint, body_bytes, args.timeout))

    def burst_worker(total_messages):
        start_burst.wait()
        dispatched = state.mark_dispatched()
        if dispatched >= total_messages:
            dispatched_burst.set()
        state.record(*send_request(endpoint, body_bytes, args.timeout))

    def progress_writer():
        while not stop_progress.is_set():
            elapsed = max(time.time() - start_reference[0], 0.001)
            running_status = status_from_state(base_status, state.snapshot(), elapsed, status="running")
            save_run_status(runs_dir, run_id, running_status)
            stop_progress.wait(1.0)

    progress_thread = threading.Thread(target=progress_writer, daemon=True)
    progress_thread.start()

    if args.burst:
        threads = [threading.Thread(target=burst_worker, args=(args.messages,), daemon=True) for _ in range(args.messages)]
    else:
        threads = [threading.Thread(target=paced_worker, daemon=True) for _ in range(args.concurrency)]

    for thread in threads:
        thread.start()

    if args.burst:
        start_reference[0] = time.time()
        print(f"burst_releasing_requests={args.messages}")
        start_burst.set()
        dispatched_burst.wait(timeout=10.0)
        print(f"all_requests_dispatched={args.messages}")

    for thread in threads:
        thread.join()

    stop_progress.set()
    progress_thread.join(timeout=2.0)

    completed_at = iso_utc_now()
    elapsed_total = max(time.time() - start_reference[0], 1e-6)
    final_status = status_from_state(base_status, state.snapshot(), elapsed_total, status="completed", completed_at=completed_at)
    save_run_status(runs_dir, run_id, final_status)

    print_report(args, base_url, selected_pod_name, endpoint, started_at, completed_at, final_status)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
