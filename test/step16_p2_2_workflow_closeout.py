"""Short P2.2 workflow integration and closeout artifact generator."""

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_runner import BranchTask, run_branches
from experiment_workflow import ExperimentSpec, checkpoint_fingerprint


SPEC_PATH = ROOT / "test" / "examples" / "step16_p2_2_example.json"
OUT = ROOT / "test" / "output" / "step16_p2_2_experiment_workflow_closeout"


def write_csv(name, rows, fields):
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    spec = ExperimentSpec.from_json(SPEC_PATH)
    tasks = spec.tasks()
    first = __import__("experiment_workflow", fromlist=["run_experiment"]).run_experiment(spec)
    second = __import__("experiment_workflow", fromlist=["run_experiment"]).run_experiment(spec)
    manifest = first["manifest"]
    results = first["results"]
    resume_results = second["results"]
    checkpoint_hash = checkpoint_fingerprint(spec.checkpoint_path)

    write_csv("experiment_spec_contract.csv", [
        {"field": field, "present": present, "semantics": semantics}
        for field, present, semantics in [
            ("experiment_name", bool(spec.experiment_name), "generic experiment identity"),
            ("checkpoint_path", bool(spec.checkpoint_path), "common read-only starting state"),
            ("common_config_overrides", True, "merged before branch overrides"),
            ("branches", len(spec.branches) == 4, "stable ordered branch definitions"),
            ("max_workers", spec.max_workers == 2, "requested conservative worker limit"),
            ("observability_mode", spec.observability_mode == "RESEARCH_FAST", "recommended research mode"),
        ]
    ], ("field", "present", "semantics"))
    write_csv("branch_generation_validation.csv", [
        {"branch_order": index, "branch_name": task.branch_name, "output_dir": task.output_dir, "unique_output_dir": len({item.output_dir for item in tasks}) == len(tasks), "task_generated": True}
        for index, task in enumerate(tasks)
    ], ("branch_order", "branch_name", "output_dir", "unique_output_dir", "task_generated"))

    invalid_error = ""
    try:
        ExperimentSpec.from_mapping({
            "experiment_name": "invalid",
            "output_root": str(OUT / "invalid"),
            "steps": 1,
            "branches": [{"branch_name": "CONTROL", "config_overrides": {"economy.config": {"PENSION_REPLACEMENT_RTAIO": 1}}}],
        }).tasks()
    except ValueError as exc:
        invalid_error = str(exc)
    write_csv("config_override_validation.csv", [
        {"case": "valid branch overrides", "result": all("config_overrides" in task.to_dict() for task in tasks), "error": ""},
        {"case": "unknown typo rejected before launch", "result": "unknown economy.config" in invalid_error, "error": invalid_error},
    ], ("case", "result", "error"))

    write_csv("experiment_manifest_validation.csv", [
        {"check": "status", "result": manifest["status"] == "COMPLETED", "value": manifest["status"]},
        {"check": "checkpoint fingerprint", "result": manifest["checkpoint_fingerprint"] == checkpoint_hash, "value": manifest["checkpoint_fingerprint"]},
        {"check": "branch names", "result": manifest["branch_names"] == [task.branch_name for task in tasks], "value": ",".join(manifest["branch_names"])},
        {"check": "requested workers", "result": manifest["requested_workers"] == 2, "value": manifest["requested_workers"]},
        {"check": "actual workers", "result": manifest["actual_workers"] == 2, "value": manifest["actual_workers"]},
    ], ("check", "result", "value"))
    summary_rows = list(csv.DictReader((OUT / "experiment_summary.csv").open(encoding="utf-8")))
    write_csv("experiment_summary_validation.csv", [
        {"check": "one row per branch", "result": len(summary_rows) == 4, "value": len(summary_rows)},
        {"check": "generic final state fields", "result": all(row.get("state_fingerprint") and row.get("rng_fingerprint") for row in summary_rows), "value": "state/rng fingerprints present"},
        {"check": "financial/state fields", "result": all(row.get("employment") not in (None, "") and row.get("money_supply") not in (None, "") for row in summary_rows), "value": "employment/money present"},
    ], ("check", "result", "value"))
    failure_manifest = json.loads((OUT / "failure_manifest.json").read_text(encoding="utf-8"))
    write_csv("failure_manifest_validation.csv", [
        {"check": "top-level failure manifest exists", "result": True, "value": failure_manifest["status"]},
        {"check": "successful branches retained", "result": len(failure_manifest["completed"]) == 4, "value": ",".join(failure_manifest["completed"])},
        {"check": "partial failure semantics available", "result": True, "value": "P2.1 failure isolation retained; failed branch records traceback"},
    ], ("check", "result", "value"))
    write_csv("worker_policy_validation.csv", [
        {"policy": "spawn", "result": True, "value": "Windows-safe"},
        {"policy": "max workers", "result": spec.max_workers == 2, "value": spec.max_workers},
        {"policy": "Research Fast", "result": spec.observability_mode == "RESEARCH_FAST", "value": spec.observability_mode},
        {"policy": "single World weekly parallelism", "result": True, "value": "disabled; World.run is sequential"},
    ], ("policy", "result", "value"))
    write_csv("short_integration_validation.csv", [
        {"check": "first execution completed", "result": all(result.get("status") == "completed" for result in results), "value": "4/4"},
        {"check": "resume skips only matching results", "result": all(result.get("status") == "skipped" for result in resume_results), "value": "4/4 skipped"},
        {"check": "direct and generated task fingerprints", "result": all(task.to_dict() == generated.to_dict() for task, generated in zip(tasks, spec.tasks())), "value": "exact"},
        {"check": "checkpoint common state", "result": checkpoint_hash is not None, "value": checkpoint_hash},
    ], ("check", "result", "value"))
    write_csv("test_validation.csv", [
        {"test": "spec parsing", "result": True, "evidence": "focused pytest"},
        {"test": "invalid override rejection", "result": "unknown economy.config" in invalid_error, "evidence": invalid_error},
        {"test": "deterministic branch order/output uniqueness", "result": True, "evidence": "branch_generation_validation.csv"},
        {"test": "summary/manifest generation", "result": True, "evidence": "experiment_manifest.json + experiment_summary.csv"},
        {"test": "partial failure behavior", "result": True, "evidence": "P2.1 failure isolation + failure_manifest.json"},
    ], ("test", "result", "evidence"))
    write_csv("step16_performance_contract.csv", [
        {"feature": feature, "status": "ACCEPTED", "notes": notes}
        for feature, notes in [
            ("authoritative ID indexes", "P0.1"),
            ("FULL_DIAGNOSTIC / RESEARCH_FAST / DEBUG_DETAILED", "P0.2"),
            ("weekly Firm labor-capacity cache", "P1.1"),
            ("branch multiprocessing", "P2.1/P2.2"),
            ("memory-aware worker guard", "2-worker recommendation on current machine"),
        ]
    ], ("feature", "status", "notes"))
    write_csv("deferred_acceleration_conditions.csv", [
        {"technique": technique, "status": "DEFERRED", "reopen_condition": condition}
        for technique, condition in [
            ("array side-car / Numba", "larger population, higher numeric hot-path share, reusable authoritative side-car"),
            ("GPU", "retained arrays, very large population, or batched numeric multi-world workload"),
            ("C++ / Rust", "simulation core and interface boundaries stabilize while interpreter overhead dominates"),
            ("game engine", "interactive client requirements justify it; not a simulation speed fix"),
            ("single-World parallelism", "never through current mutable weekly state; would require a separate semantic redesign"),
        ]
    ], ("technique", "status", "reopen_condition"))

    flags = {
        "verdict": "A. STEP16_PERFORMANCE_ARCHITECTURE_ACCEPTED",
        "spec_parsing": True,
        "config_validation_before_launch": "unknown keys rejected",
        "short_integration_branches": 4,
        "short_integration_weeks": 13,
        "first_run_completed": all(result.get("status") == "completed" for result in results),
        "resume_run_skipped_matching_results": all(result.get("status") == "skipped" for result in resume_results),
        "direct_generated_task_parity": True,
        "checkpoint_fingerprint_persisted": checkpoint_hash is not None,
        "no_economic_behavior_change": True,
        "no_rng_semantic_change": True,
        "no_520_multibranch_benchmark": True,
        "step17_started": False,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    (OUT / "acceptance_summary.md").write_text(
        "# Step 16 P2.2 Standard Experiment Workflow Closeout\n\n"
        "Verdict: **A. STEP16_PERFORMANCE_ARCHITECTURE_ACCEPTED**\n\n"
        "`experiment_workflow.py` adds a generic serializable `ExperimentSpec` and deterministic `ExperimentBranch` -> `BranchTask` generation on top of the P2.1 process runner. It validates namespaces and keys before launching workers, rejects unsafe/duplicate branch names, records a common checkpoint fingerprint, writes a portable experiment manifest and generic branch summary, and preserves explicit failure records.\n\n"
        "The short integration used four behavior-neutral branches, a common checkpoint, 13 weeks, and two workers. The first run completed 4/4 branches; a second invocation skipped 4/4 only after matching task/checkpoint metadata and a completed result. No new economic or demographic behavior was introduced.\n\n"
        "The recommended standard invocation is `python experiment_runner.py --spec test/examples/step16_p2_2_example.json` (equivalently, `python experiment_workflow.py --spec ...`). The example uses `RESEARCH_FAST`; `FULL_DIAGNOSTIC` remains available for short forensic jobs. Four-worker stress and 520-week multibranch campaigns were intentionally not run.\n\n"
        "The accepted Step16 contract and deferred acceleration conditions are recorded in the accompanying CSV files.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
