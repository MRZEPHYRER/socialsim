"""Standard experiment-spec workflow built on the P2.1 branch runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from experiment_runner import BranchTask, run_branches


SUPPORTED_NAMESPACES = {
    "world",
    "economy.config",
    "fertility.config",
    "central_bank.config",
}


@dataclass(frozen=True)
class ExperimentBranch:
    branch_name: str
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_name: str
    output_root: str
    steps: int
    branches: tuple[ExperimentBranch, ...]
    checkpoint_path: str | None = None
    seed: int | None = None
    population: int = 5000
    scenario: str = "baseline"
    firm_count: int | None = 5
    max_workers: int = 2
    observability_mode: str = "RESEARCH_FAST"
    diagnostics_mode: str = "compact"
    common_config_overrides: Mapping[str, Any] = field(default_factory=dict)
    persist_diagnostics: bool = False
    diagnostic_cadence: int = 1
    resume: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], base_dir: Path | None = None) -> "ExperimentSpec":
        base_dir = base_dir or Path.cwd()
        raw_branches = payload.get("branches", [])
        branches = tuple(
            item if isinstance(item, ExperimentBranch) else ExperimentBranch(
                branch_name=item["branch_name"],
                config_overrides=item.get("config_overrides", {}),
                seed=item.get("seed"),
                metadata=item.get("metadata", {}),
            )
            for item in raw_branches
        )
        checkpoint = payload.get("checkpoint_path")
        output_root = payload.get("output_root", "experiment_output")
        if checkpoint:
            checkpoint = str(_resolve_path(checkpoint, base_dir))
        output_root = str(_resolve_path(output_root, base_dir))
        return cls(
            experiment_name=payload["experiment_name"],
            output_root=output_root,
            steps=int(payload.get("steps", 13)),
            branches=branches,
            checkpoint_path=checkpoint,
            seed=payload.get("seed"),
            population=int(payload.get("population", 5000)),
            scenario=payload.get("scenario", "baseline"),
            firm_count=payload.get("firm_count", 5),
            max_workers=int(payload.get("max_workers", 2)),
            observability_mode=payload.get("observability_mode", "RESEARCH_FAST"),
            diagnostics_mode=payload.get("diagnostics_mode", "compact"),
            common_config_overrides=payload.get("common_config_overrides", {}),
            persist_diagnostics=bool(payload.get("persist_diagnostics", False)),
            diagnostic_cadence=int(payload.get("diagnostic_cadence", 1)),
            resume=bool(payload.get("resume", True)),
        )

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "ExperimentSpec":
        source = Path(path).resolve()
        return cls.from_mapping(json.loads(source.read_text(encoding="utf-8-sig")), source.parent)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "output_root": self.output_root,
            "steps": self.steps,
            "checkpoint_path": self.checkpoint_path,
            "seed": self.seed,
            "population": self.population,
            "scenario": self.scenario,
            "firm_count": self.firm_count,
            "max_workers": self.max_workers,
            "observability_mode": self.observability_mode,
            "diagnostics_mode": self.diagnostics_mode,
            "common_config_overrides": dict(self.common_config_overrides),
            "persist_diagnostics": self.persist_diagnostics,
            "diagnostic_cadence": self.diagnostic_cadence,
            "resume": self.resume,
            "branches": [
                {
                    "branch_name": branch.branch_name,
                    "config_overrides": dict(branch.config_overrides),
                    "seed": branch.seed,
                    "metadata": dict(branch.metadata),
                }
                for branch in self.branches
            ],
        }

    def tasks(self) -> list[BranchTask]:
        self.validate()
        output_root = Path(self.output_root)
        tasks = []
        for branch in self.branches:
            tasks.append(
                BranchTask(
                    branch_name=branch.branch_name,
                    output_dir=str(output_root / branch.branch_name),
                    steps=self.steps,
                    checkpoint_path=self.checkpoint_path,
                    seed=branch.seed if branch.seed is not None else self.seed,
                    population=self.population,
                    scenario=self.scenario,
                    firm_count=self.firm_count,
                    config_overrides=_merge_overrides(
                        self.common_config_overrides,
                        branch.config_overrides,
                    ),
                    diagnostics_mode=self.diagnostics_mode,
                    observability_mode=self.observability_mode,
                    persist_diagnostics=self.persist_diagnostics,
                    diagnostic_cadence=self.diagnostic_cadence,
                    progress_interval=0,
                )
            )
        return tasks

    def validate(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        if not self.branches:
            raise ValueError("at least one branch is required")
        names = [branch.branch_name for branch in self.branches]
        if len(set(names)) != len(names):
            raise ValueError("branch names must be unique")
        for name in names:
            if not name or name in {".", ".."} or Path(name).name != name:
                raise ValueError(f"branch name must be a single safe path component: {name!r}")
        _validate_override_mapping(self.common_config_overrides)
        for branch in self.branches:
            _validate_override_mapping(branch.config_overrides)
        if self.checkpoint_path and not Path(self.checkpoint_path).is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {self.checkpoint_path}")
        if self.checkpoint_path and any(
            branch.seed is not None and branch.seed != self.seed
            for branch in self.branches
        ):
            raise ValueError("branch seed overrides are not allowed with a common checkpoint")


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _merge_overrides(common: Mapping[str, Any], branch: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(common)))
    for namespace, values in dict(branch or {}).items():
        if namespace not in result:
            result[namespace] = {}
        if not isinstance(result[namespace], dict) or not isinstance(values, Mapping):
            raise ValueError(f"override namespace must map to objects: {namespace}")
        result[namespace].update(dict(values))
    return result


def _validate_override_mapping(overrides: Mapping[str, Any]) -> None:
    for namespace, values in dict(overrides or {}).items():
        if namespace not in SUPPORTED_NAMESPACES:
            raise ValueError(f"unsupported config override namespace: {namespace}")
        if not isinstance(values, Mapping):
            raise ValueError(f"config override namespace must be an object: {namespace}")
        if namespace == "world":
            allowed = {
                "diagnostics_mode",
                "observability_mode",
                "diagnostic_micro_snapshot_cadence",
                "initial_age_phase_mode",
                "scenario_name",
                "payg_pension_enabled",
                "payg_contribution_rate",
                "payg_pension_target_multiplier",
                "intergenerational_wealth_transfer_enabled",
                "intergenerational_reserve_weeks",
                "intergenerational_donor_surplus_share",
                "intergenerational_recipient_target_weeks",
                "recipient_policy_instrumentation_enabled",
                "payg_cash_safety_audit_enabled",
                "household_liquidity_snapshot_enabled",
                "long_horizon_liquidity_enabled",
                "long_horizon_liquidity_snapshot_cadence",
                "long_horizon_household_snapshot_stream_path",
            }
            unknown = sorted(set(values) - allowed)
            if unknown:
                raise ValueError(f"unknown world override(s): {', '.join(unknown)}")
            continue
        module_name = {
            "economy.config": "economy.config",
            "fertility.config": "fertility.config",
            "central_bank.config": "central_bank.config",
        }[namespace]
        module = __import__(module_name, fromlist=["*"])
        unknown = sorted(key for key in values if not hasattr(module, key))
        if unknown:
            raise ValueError(f"unknown {namespace} override(s): {', '.join(unknown)}")


def checkpoint_fingerprint(path: str | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_fingerprint(task: BranchTask, checkpoint_hash: str | None) -> str:
    payload = {
        "task": task.to_dict(),
        "checkpoint_fingerprint": checkpoint_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _portable_path(path: str | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return Path(path).name


def _existing_completed(task: BranchTask, expected_task_hash: str, checkpoint_hash: str | None):
    directory = Path(task.output_dir)
    result_path = directory / "branch_result.json"
    sidecar_path = directory / "task_spec.json"
    if not result_path.is_file() or not sidecar_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if result.get("status") != "completed":
        return None
    if sidecar.get("task_fingerprint") != expected_task_hash:
        return None
    if sidecar.get("checkpoint_fingerprint") != checkpoint_hash:
        return None
    result = dict(result)
    result.update({"status": "skipped", "skip_reason": "matching_completed_branch_result"})
    return result


def _summary_rows(results):
    rows = []
    for result in results:
        rows.append({
            "branch": result.get("branch_name"),
            "status": result.get("status"),
            "runtime_seconds": result.get("elapsed_seconds"),
            "final_population": result.get("final_population"),
            "household_count": result.get("final_households"),
            "firm_count": result.get("final_firms"),
            "employment": result.get("final_employment"),
            "household_cash": result.get("final_household_cash"),
            "firm_cash": result.get("final_firm_cash"),
            "money_supply": result.get("money_supply"),
            "loan_principal": result.get("loan_principal"),
            "invariant_violations": result.get("invariant_violations"),
            "state_fingerprint": result.get("state_fingerprint"),
            "rng_fingerprint": result.get("rng_fingerprint"),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(spec: ExperimentSpec) -> dict[str, Any]:
    spec.validate()
    output_root = Path(spec.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = checkpoint_fingerprint(spec.checkpoint_path)
    tasks = spec.tasks()
    pending = []
    skipped = []
    task_records = {}
    for task in tasks:
        task_hash = task_fingerprint(task, checkpoint_hash)
        task_records[task.branch_name] = task_hash
        existing = _existing_completed(task, task_hash, checkpoint_hash) if spec.resume else None
        if existing is not None:
            skipped.append(existing)
            continue
        directory = Path(task.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "task_spec.json").write_text(json.dumps({
            "task_fingerprint": task_hash,
            "checkpoint_fingerprint": checkpoint_hash,
            "task": task.to_dict(),
        }, indent=2, default=str), encoding="utf-8")
        pending.append(task)

    started = time.perf_counter()
    execution = run_branches(pending, max_workers=spec.max_workers) if pending else {
        "completed": [], "failed": [], "cancelled": [], "max_workers": 0,
    }
    results = skipped + execution["completed"] + execution["failed"]
    result_by_name = {result["branch_name"]: result for result in results}
    results = [result_by_name[task.branch_name] for task in tasks]
    if not pending:
        existing_manifest_path = output_root / "experiment_manifest.json"
        try:
            existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            existing_manifest = None
        if (isinstance(existing_manifest, dict)
                and existing_manifest.get("experiment_name") == spec.experiment_name
                and existing_manifest.get("checkpoint_fingerprint") == checkpoint_hash
                and existing_manifest.get("task_fingerprints") == task_records):
            return {"manifest": existing_manifest, "results": results, "output_root": str(output_root)}
    failures = [result for result in results if result.get("status") == "failed"]
    status = "PARTIAL_FAILURE" if failures else "COMPLETED"
    manifest = {
        "experiment_name": spec.experiment_name,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": _portable_path(spec.checkpoint_path, output_root.parent),
        "checkpoint_fingerprint": checkpoint_hash,
        "common_seed": spec.seed,
        "steps": spec.steps,
        "observability_mode": spec.observability_mode,
        "requested_workers": spec.max_workers,
        "actual_workers": execution.get("max_workers", 0),
        "branch_names": [task.branch_name for task in tasks],
        "branch_status": {result["branch_name"]: result.get("status") for result in results},
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "resume_enabled": spec.resume,
        "task_fingerprints": task_records,
    }
    (output_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_csv(output_root / "experiment_summary.csv", _summary_rows(results))
    _write_csv(output_root / "experiment_effective_config.csv", [
        {
            "branch": result.get("branch_name"),
            "parameter": key,
            "effective_value": json.dumps(value, sort_keys=True, default=str),
            "differs_from_control": result.get("branch_name") != "CONTROL",
        }
        for result in results
        for key, value in result.get("effective_config", {}).items()
        if key in {"scenario", "steps", "task_seed", "world_seed", "observability_mode", "diagnostics_mode", "config_overrides"}
    ])
    (output_root / "failure_manifest.json").write_text(json.dumps({
        "status": status,
        "completed": [r["branch_name"] for r in results if r.get("status") in {"completed", "skipped"}],
        "failed": failures,
        "cancelled": execution.get("cancelled", []),
    }, indent=2, default=str), encoding="utf-8")
    return {"manifest": manifest, "results": results, "output_root": str(output_root)}


def cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run independent SocialSim branches from an ExperimentSpec JSON file.")
    parser.add_argument("--spec", required=True, help="Path to an ExperimentSpec JSON file")
    args = parser.parse_args(argv)
    result = run_experiment(ExperimentSpec.from_json(args.spec))
    print(json.dumps({
        "experiment": result["manifest"]["experiment_name"],
        "status": result["manifest"]["status"],
        "output_root": result["output_root"],
        "branches": result["manifest"]["branch_status"],
    }, indent=2))
    return 0 if result["manifest"]["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(cli())

