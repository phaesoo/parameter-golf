"""Parse experiment logs and output a comparison table sorted by BPB."""

import re
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def parse_log(log_path: Path) -> dict | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")

    # Find the last val_bpb line (final evaluation)
    val_bpb_matches = re.findall(r"val_bpb:([\d.]+)", text)
    if not val_bpb_matches:
        return None

    val_bpb = float(val_bpb_matches[-1])
    val_loss_matches = re.findall(r"val_loss:([\d.]+)", text)
    val_loss = float(val_loss_matches[-1]) if val_loss_matches else None

    # Find step count
    step_matches = re.findall(r"step:(\d+)/", text)
    last_step = int(step_matches[-1]) if step_matches else None

    # Find train time
    time_matches = re.findall(r"train_time:(\d+)ms", text)
    train_time_ms = int(time_matches[-1]) if time_matches else None

    # Find model params
    param_match = re.search(r"model_params:(\d+)", text)
    params = int(param_match.group(1)) if param_match else None

    # Find int8 roundtrip if available
    rt_match = re.search(r"final_int8_zlib_roundtrip.*val_bpb:([\d.]+)", text)
    rt_bpb = float(rt_match.group(1)) if rt_match else None

    return {
        "val_bpb": val_bpb,
        "val_loss": val_loss,
        "rt_bpb": rt_bpb,
        "step": last_step,
        "time_ms": train_time_ms,
        "params": params,
    }


def main():
    if not RESULTS_DIR.exists():
        print(f"No results directory found at {RESULTS_DIR}")
        sys.exit(1)

    logs = sorted(RESULTS_DIR.glob("*.log"))
    if not logs:
        print("No log files found.")
        sys.exit(1)

    results = []
    for log_path in logs:
        parsed = parse_log(log_path)
        if parsed:
            results.append((log_path.stem, parsed))

    if not results:
        print("No parseable results found.")
        sys.exit(1)

    # Sort by val_bpb (lower is better)
    results.sort(key=lambda x: x[1]["val_bpb"])

    # Find baseline for delta calculation
    baseline_bpb = None
    for name, r in results:
        if name == "baseline":
            baseline_bpb = r["val_bpb"]
            break

    # Print table
    header = f"{'Rank':<5} {'Experiment':<30} {'BPB':>8} {'Delta':>8} {'Loss':>8} {'Params':>10} {'Steps':>7} {'Time(s)':>8}"
    print(header)
    print("-" * len(header))

    for i, (name, r) in enumerate(results, 1):
        delta = ""
        if baseline_bpb is not None:
            d = r["val_bpb"] - baseline_bpb
            delta = f"{d:+.4f}"

        params_str = f"{r['params'] / 1e6:.1f}M" if r["params"] else "?"
        time_str = f"{r['time_ms'] / 1000:.1f}" if r["time_ms"] else "?"
        step_str = str(r["step"]) if r["step"] else "?"
        loss_str = f"{r['val_loss']:.4f}" if r["val_loss"] else "?"

        print(
            f"{i:<5} {name:<30} {r['val_bpb']:>8.4f} {delta:>8} {loss_str:>8} "
            f"{params_str:>10} {step_str:>7} {time_str:>8}"
        )


if __name__ == "__main__":
    main()
