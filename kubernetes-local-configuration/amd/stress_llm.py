#!/usr/bin/env python3
import argparse
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
from pathlib import Path

BASE_URL_PROFILES = {
    "localhost": "http://127.0.0.1:8000",
    "cluster-ip": "http://10.43.167.147:8000",
    "k8s-service": "http://qwen3-0-8b-vllm.llm-amd.svc.cluster.local:8000",
}


def parse_args():
    p = argparse.ArgumentParser(description="Stress test an OpenAI-compatible vLLM endpoint.")
    p.add_argument(
        "--base-url-profile",
        choices=sorted(BASE_URL_PROFILES.keys()),
        default="cluster-ip",
        help="Named base URL profile.",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="Manual base URL override (without /v1). Takes precedence over --base-url-profile.",
    )
    p.add_argument("--namespace", default="llm-amd", help="Kubernetes namespace used for pod discovery.")
    p.add_argument("--pod", default=None, help="Pod name to target directly (builds base URL from pod IP).")
    p.add_argument(
        "--select-pod",
        action="store_true",
        help="Interactively select a compatible pod and target its pod IP:8000.",
    )
    p.add_argument(
        "--list-pods",
        action="store_true",
        help="List compatible pods in --namespace and exit.",
    )
    p.add_argument("--runs-dir", default=None, help="Directory where run status/results are saved.")
    p.add_argument("--status", default=None, help="Show a saved run status by run ID and exit.")
    p.add_argument("--list-runs", action="store_true", help="List saved runs and exit.")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name.")
    p.add_argument("--duration", type=int, default=120, help="Test duration in seconds.")
    p.add_argument("--messages", type=int, default=0, help="Fixed number of requests to send (0 disables).")
    p.add_argument("--burst", action="store_true", help="Send all --messages requests at once.")
    p.add_argument("--concurrency", type=int, default=8, help="Number of worker threads.")
    p.add_argument("--qps", type=float, default=4.0, help="Global request rate cap.")
    p.add_argument("--timeout", type=float, default=120.0, help="HTTP request timeout seconds.")
    p.add_argument("--max-tokens", type=int, default=128, help="Generation max tokens.")
    p.add_argument("--prompt", default="Explain one practical use of queue-depth autoscaling in two sentences.")
    return p.parse_args()


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


def default_runs_dir():
    return Path(__file__).resolve().parent / "stress-runs"


def iso_utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ensure_runs_dir(path_value):
    path = Path(path_value) if path_value else default_runs_dir()
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
        status = doc.get("status", "unknown")
        started = doc.get("started_at", "")
        completed = doc.get("completed_at", "")
        total = doc.get("total_requests", 0)
        print(f"{fpath.stem} status={status} started_at={started} completed_at={completed} total={total}")


def show_saved_run(runs_dir, run_id):
    path = run_path(runs_dir, run_id)
    if not path.exists():
        raise RuntimeError(f"Run ID '{run_id}' not found in {runs_dir}")
    print(path.read_text(encoding="utf-8").strip())


def discover_compatible_pods(namespace):
    cmd = ["kubectl", "-n", namespace, "get", "pods", "-o", "json"]
    try:
        data = subprocess.check_output(cmd, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"Failed to query pods from namespace '{namespace}': {exc}") from exc

    doc = json.loads(data)
    compatible = []
    for item in doc.get("items", []):
        metadata = item.get("metadata", {})
        status = item.get("status", {})
        spec = item.get("spec", {})
        pod_name = metadata.get("name", "")
        pod_ip = status.get("podIP")
        if not pod_name or not pod_ip:
            continue
        if status.get("phase") != "Running":
            continue

        conditions = status.get("conditions", [])
        is_ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)
        if not is_ready:
            continue

        has_8000 = False
        for container in spec.get("containers", []):
            for port in container.get("ports", []):
                if port.get("containerPort") == 8000:
                    has_8000 = True
                    break
            if has_8000:
                break
        if not has_8000:
            continue

        compatible.append({"name": pod_name, "ip": pod_ip})

    compatible.sort(key=lambda x: x["name"])
    return compatible


def select_pod_interactively(pods):
    if not pods:
        raise RuntimeError("No compatible pods found.")
    print("Compatible pods:")
    for idx, pod in enumerate(pods, start=1):
        print(f"{idx}. {pod['name']} ({pod['ip']})")
    while True:
        raw = input("Select pod number: ").strip()
        try:
            choice = int(raw)
            if 1 <= choice <= len(pods):
                return pods[choice - 1]
        except ValueError:
            pass
        print("Invalid selection. Try again.")


def main():
    args = parse_args()
    runs_dir = ensure_runs_dir(args.runs_dir)

    if args.list_runs:
        list_saved_runs(runs_dir)
        sys.exit(0)

    if args.status:
        show_saved_run(runs_dir, args.status)
        sys.exit(0)

    compatible_pods = []
    if args.list_pods or args.select_pod or args.pod:
        compatible_pods = discover_compatible_pods(args.namespace)

    if args.list_pods:
        if not compatible_pods:
            print(f"No compatible pods found in namespace '{args.namespace}'.")
            sys.exit(1)
        for pod in compatible_pods:
            print(f"{pod['name']} {pod['ip']}")
        sys.exit(0)

    resolved_base_url = args.base_url or BASE_URL_PROFILES[args.base_url_profile]
    selected_pod_name = None

    if args.pod:
        selected = next((p for p in compatible_pods if p["name"] == args.pod), None)
        if not selected:
            raise RuntimeError(f"Pod '{args.pod}' is not compatible or not running in namespace '{args.namespace}'.")
        selected_pod_name = selected["name"]
        resolved_base_url = f"http://{selected['ip']}:8000"
    elif args.select_pod:
        selected = select_pod_interactively(compatible_pods)
        selected_pod_name = selected["name"]
        resolved_base_url = f"http://{selected['ip']}:8000"

    endpoint = f"{resolved_base_url.rstrip('/')}/v1/chat/completions"
    start_ts = time.time()
    stop_at = time.time() + args.duration
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    started_at = iso_utc_now()
    max_messages = max(args.messages, 0)
    if args.messages < 0:
        raise RuntimeError("--messages must be >= 0")
    if args.burst and max_messages <= 0:
        raise RuntimeError("--burst requires --messages > 0")

    latencies = []
    lock = threading.Lock()
    total = 0
    ok = 0
    fail = 0
    issued = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    # Per-worker throttle. Approximate global QPS by splitting budget across workers.
    per_worker_qps = max(args.qps / max(args.concurrency, 1), 0.001)
    per_worker_interval = 1.0 / per_worker_qps

    initial_status = {
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "base_url": resolved_base_url,
        "selected_pod": selected_pod_name,
        "endpoint": endpoint,
        "model": args.model,
        "duration_s": args.duration,
        "messages_target": max_messages,
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
    save_run_status(runs_dir, run_id, initial_status)
    print(f"run_id={run_id}")
    print(f"run_status_file={run_path(runs_dir, run_id)}")

    def send_request():
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0.7,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer EMPTY",
            },
            method="POST",
        )

        t0 = time.time()
        local_ok = False
        pt = 0
        ct = 0
        tt = 0
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                payload_raw = resp.read()
                local_ok = (200 <= resp.status < 300)
                if local_ok:
                    try:
                        payload_json = json.loads(payload_raw.decode("utf-8"))
                        usage = payload_json.get("usage", {})
                        pt = int(usage.get("prompt_tokens", 0) or 0)
                        ct = int(usage.get("completion_tokens", 0) or 0)
                        tt = int(usage.get("total_tokens", pt + ct) or (pt + ct))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        pt = 0
                        ct = 0
                        tt = 0
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            local_ok = False
        elapsed = time.time() - t0
        return local_ok, pt, ct, tt, elapsed

    def worker(worker_id):
        nonlocal total, ok, fail, issued, prompt_tokens, completion_tokens, total_tokens
        next_fire = time.time() + random.uniform(0, per_worker_interval)
        while True:
            with lock:
                if max_messages > 0 and issued >= max_messages:
                    break
                if max_messages == 0 and time.time() >= stop_at:
                    break
                issued += 1

            now = time.time()
            if now < next_fire:
                time.sleep(next_fire - now)
            next_fire += per_worker_interval

            local_ok, pt, ct, tt, elapsed = send_request()

            with lock:
                total += 1
                if local_ok:
                    ok += 1
                else:
                    fail += 1
                latencies.append(elapsed)
                prompt_tokens += pt
                completion_tokens += ct
                total_tokens += tt

    start_burst = threading.Event()
    dispatched_burst = threading.Event()
    dispatched_count = 0

    def burst_worker():
        nonlocal total, ok, fail, issued, prompt_tokens, completion_tokens, total_tokens, dispatched_count
        start_burst.wait()
        with lock:
            issued += 1
            dispatched_count += 1
            if dispatched_count >= max_messages:
                dispatched_burst.set()
        local_ok, pt, ct, tt, elapsed = send_request()
        with lock:
            total += 1
            if local_ok:
                ok += 1
            else:
                fail += 1
            latencies.append(elapsed)
            prompt_tokens += pt
            completion_tokens += ct
            total_tokens += tt

    stop_progress = threading.Event()

    def progress_writer():
        while not stop_progress.is_set():
            with lock:
                elapsed_now = max(time.time() - start_ts, 0.001)
                success_rate_now = (ok / total * 100.0) if total else 0.0
                snapshot = {
                    **initial_status,
                    "status": "running",
                    "total_requests": total,
                    "success": ok,
                    "fail": fail,
                    "success_rate_pct": round(success_rate_now, 2),
                    "achieved_rps": round(total / elapsed_now, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "prompt_tokens_per_s": round(prompt_tokens / elapsed_now, 3),
                    "completion_tokens_per_s": round(completion_tokens / elapsed_now, 3),
                    "total_tokens_per_s": round(total_tokens / elapsed_now, 3),
                    "updated_at": iso_utc_now(),
                }
            save_run_status(runs_dir, run_id, snapshot)
            stop_progress.wait(1.0)

    progress_thread = threading.Thread(target=progress_writer, daemon=True)
    progress_thread.start()
    if args.burst:
        threads = [threading.Thread(target=burst_worker, daemon=True) for _ in range(max_messages)]
    else:
        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.concurrency)]
    for t in threads:
        t.start()
    if args.burst:
        start_ts = time.time()
        print(f"burst_releasing_requests={max_messages}")
        start_burst.set()
        dispatched_burst.wait(timeout=10.0)
        print(f"all_requests_dispatched={max_messages}")
    for t in threads:
        t.join()
    stop_progress.set()
    progress_thread.join(timeout=2.0)

    lat_sorted = sorted(latencies)
    elapsed_total = max(time.time() - start_ts, 1e-6)
    success_rate = (ok / total * 100.0) if total else 0.0
    achieved_rps = total / elapsed_total
    completed_at = iso_utc_now()

    final_status = {
        **initial_status,
        "status": "completed",
        "completed_at": completed_at,
        "total_requests": total,
        "success": ok,
        "fail": fail,
        "success_rate_pct": round(success_rate, 2),
        "achieved_rps": round(achieved_rps, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens_per_s": round(prompt_tokens / elapsed_total, 3),
        "completion_tokens_per_s": round(completion_tokens / elapsed_total, 3),
        "total_tokens_per_s": round(total_tokens / elapsed_total, 3),
        "latency_avg_s": round(statistics.mean(lat_sorted) if lat_sorted else 0.0, 3),
        "latency_p50_s": round(percentile(lat_sorted, 0.50), 3),
        "latency_p95_s": round(percentile(lat_sorted, 0.95), 3),
        "latency_p99_s": round(percentile(lat_sorted, 0.99), 3),
        "updated_at": completed_at,
    }
    save_run_status(runs_dir, run_id, final_status)

    print(f"base_url={resolved_base_url}")
    if selected_pod_name:
        print(f"selected_pod={selected_pod_name}")
    print(f"endpoint={endpoint}")
    print(f"model={args.model}")
    print(f"duration_s={args.duration}")
    print(f"concurrency={args.concurrency}")
    print(f"target_qps={args.qps}")
    print(f"total_requests={total}")
    print(f"success={ok}")
    print(f"fail={fail}")
    print(f"success_rate_pct={success_rate:.2f}")
    print(f"achieved_rps={achieved_rps:.2f}")
    print(f"prompt_tokens={prompt_tokens}")
    print(f"completion_tokens={completion_tokens}")
    print(f"total_tokens={total_tokens}")
    print(f"prompt_tokens_per_s={prompt_tokens / elapsed_total:.2f}")
    print(f"completion_tokens_per_s={completion_tokens / elapsed_total:.2f}")
    print(f"total_tokens_per_s={total_tokens / elapsed_total:.2f}")
    print(f"latency_avg_s={statistics.mean(lat_sorted) if lat_sorted else 0.0:.3f}")
    print(f"avg_reply_time_s={statistics.mean(lat_sorted) if lat_sorted else 0.0:.3f}")
    print(f"latency_p50_s={percentile(lat_sorted, 0.50):.3f}")
    print(f"latency_p95_s={percentile(lat_sorted, 0.95):.3f}")
    print(f"latency_p99_s={percentile(lat_sorted, 0.99):.3f}")
    print(f"started_at={started_at}")
    print(f"completed_at={completed_at}")
    print()
    print("summary:")
    print(f"- requests: {total} total ({ok} success, {fail} fail, {success_rate:.2f}% success)")
    print(f"- time: {elapsed_total:.2f}s total ({started_at} -> {completed_at})")
    print(f"- throughput: {achieved_rps:.2f} req/s, {total_tokens / elapsed_total:.2f} tok/s")
    print(
        f"- latency: avg {statistics.mean(lat_sorted) if lat_sorted else 0.0:.3f}s, "
        f"p50 {percentile(lat_sorted, 0.50):.3f}s, "
        f"p95 {percentile(lat_sorted, 0.95):.3f}s, "
        f"p99 {percentile(lat_sorted, 0.99):.3f}s"
    )
    print(f"- latency_profile: {classify_latency(percentile(lat_sorted, 0.95))}")


if __name__ == "__main__":
    main()
