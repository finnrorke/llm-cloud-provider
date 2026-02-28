# Root Tools

This folder contains the active stress and metrics tools.

## Layout
- `tools/stress_llm.py`
- `tools/get_metrics.py`
- `tools/common.py`
- `tools/configs/stress_llm/*.json`
- `tools/configs/get_metrics/*.json`

## Usage
Each script takes exactly one argument: a run-config path.

```bash
python3 tools/stress_llm.py tools/configs/stress_llm/stress-burst.json
python3 tools/get_metrics.py tools/configs/get_metrics/dashboard-fast.json
```
