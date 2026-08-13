import csv
import os
from datetime import datetime

import yaml

TIMING_REPORT_COLUMNS = [
    "stage",
    "batch_size",
    "n_samples",
    "mean_ms_per_tx",
    "std_ms_per_tx",
    "min_ms_per_tx",
    "max_ms_per_tx",
    "total_seconds",
]


def _build_tag_suffix(tags: str | None) -> str:
    tag_list = [tag.strip() for tag in (tags or "").split(",") if tag.strip()]
    return "-".join(tag_list)


class TimingReporter:
    """Records per-stage inference timing results for a `time-infer` CLI run and
    persists them to a timestamped folder placed beside the checkpoint folder it timed."""

    def __init__(
        self,
        checkpoint_dir: str,
        tags: str | None,
        dataset_type: str,
        dataset_path: str,
        device: str,
        fold: int,
        warmup_batches: int,
        repeats: int,
    ):
        run_datetime = datetime.now()
        timestamp = run_datetime.strftime("%Y%m%d%H%M")
        tag_suffix = _build_tag_suffix(tags)
        run_name = f"timing_{tag_suffix}_{timestamp}" if tag_suffix else f"timing_{timestamp}"

        run_dir_parent = os.path.dirname(os.path.normpath(checkpoint_dir))
        self.run_dir = os.path.join(run_dir_parent, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        document = {
            "run_info": {
                "out_dir": self.run_dir,
                "run_datetime": run_datetime.isoformat(timespec="seconds"),
                "checkpoint_dir": checkpoint_dir,
                "device": device,
                "fold": fold,
                "warmup_batches": warmup_batches,
                "repeats": repeats,
            },
            "dataset": {
                "type": dataset_type,
                "path": dataset_path,
            },
        }
        with open(os.path.join(self.run_dir, "params.yaml"), "w") as f:
            yaml.dump(document, f, default_flow_style=False, sort_keys=False)

    def save_timing_report(self, rows: list[dict]) -> None:
        path = os.path.join(self.run_dir, "timing_report.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TIMING_REPORT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in TIMING_REPORT_COLUMNS})
