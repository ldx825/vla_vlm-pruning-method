import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Summarize BLINK FastV baseline pruning sweep results.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results_dir = run_dir / "results"
    rows = []
    for result_file in sorted(results_dir.glob("fastv*.json")):
        payload = load_json(result_file)
        summary = payload.get("summary", {})
        name = result_file.stem
        rows.append(
            {
                "name": name,
                "successful_rate": summary.get("avg_accuracy"),
                "latency_seconds": summary.get("avg_latency_seconds"),
                "flops_reduction": summary.get("avg_flops_reduction"),
                "avg_last_tokens_after": summary.get("avg_last_tokens_after"),
                "method_overhead_seconds": summary.get("avg_method_overhead_seconds"),
                "sample_count": summary.get("sample_count"),
            }
        )

    baseline = next((row for row in rows if row["name"] == "fastv448"), None)
    baseline_latency = baseline.get("latency_seconds") if baseline else None
    for row in rows:
        latency = row.get("latency_seconds")
        row["speedup_vs_fastv448"] = (
            float(baseline_latency) / float(latency)
            if baseline_latency is not None and latency
            else None
        )

    payload = {"run_dir": str(run_dir), "baseline_for_speedup": "fastv448", "rows": rows}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
