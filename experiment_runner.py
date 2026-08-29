"""Process-isolated runner for independent SocialSim experiment branches.

The runner deliberately sits outside ``World.step``.  A worker owns one
loaded World, one RNG state, and one output directory; no World is shared
between workers.
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class BranchTask:
    """Serializable description of one independent simulation branch."""

    branch_name: str
    output_dir: str
    steps: int
    checkpoint_path: str | None = None
    seed: int | None = None
    population: int = 5000
    scenario: str = "baseline"
    firm_count: int | None = 5
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    diagnostics_mode: str = "compact"
    observability_mode: str = "RESEARCH_FAST"
    persist_diagnostics: bool = False
    diagnostic_cadence: int = 1
    progress_interval: int = 0
    save_checkpoint: bool = False

    def __post_init__(self):
        if not self.branch_name:
            raise ValueError("branch_name must not be empty")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")
        if self.persist_diagnostics and self.diagnostic_cadence < 1:
            raise ValueError("diagnostic_cadence must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config_overrides"] = dict(self.config_overrides)
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BranchTask":
        return cls(**dict(value))


def _float(value: Any) -> float:
    try:
        return round(float(value), 12)
    except (TypeError, ValueError):
        return 0.0


def _state_fingerprint(world) -> str:
    """Hash stable final state fields without relying on object addresses."""
    persons = sorted(
        (
            getattr(person, "id", None),
            bool(getattr(person, "alive", False)),
            _float(getattr(person, "age", 0.0)),
            getattr(person, "firm_id", None),
            getattr(person, "household_id", None),
            getattr(person, "settlement_household_id", None),
        )
        for person in getattr(world, "population", [])
    )
    households = sorted(
        (
            getattr(household, "id", None),
            _float(getattr(household, "wealth", 0.0)),
            tuple(sorted(getattr(household, "parents", []) or [])),
            tuple(sorted(getattr(household, "children", []) or [])),
        )
        for household in getattr(world, "households", [])
    )
    firms = sorted(
        (
            getattr(firm, "firm_id", None),
            getattr(firm, "sector_id", None),
            len(getattr(firm, "employee_ids", []) or []),
            tuple(getattr(firm, "employee_ids", []) or []),
            _float(getattr(firm, "cash", 0.0)),
            _float(getattr(firm, "inventory_units", 0.0)),
            _float(getattr(firm, "production", 0.0)),
            _float(getattr(firm, "sales", 0.0)),
            _float(getattr(firm, "loan_balance", 0.0)),
        )
        for firm in [
            *getattr(world, "firms", []),
            *getattr(world, "capital_good_firms", []),
        ]
    )
    central_bank = getattr(getattr(world, "firm_system", None), "central_bank", None)
    payload = {
        "global_step": len(getattr(world, "population_history", [])),
        "population": persons,
        "households": households,
        "firms": firms,
        "money_supply": _float(getattr(central_bank, "money_supply", 0.0)),
        "history_lengths": {
            name: len(getattr(world, name, []) or [])
            for name in (
                "population_history",
                "birth_history",
                "death_history",
                "income_history",
                "consumption_history",
                "saving_history",
            )
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rng_fingerprint(world) -> str:
    from checkpoint import capture_rng_state

    return hashlib.sha256(pickle.dumps(capture_rng_state(), protocol=4)).hexdigest()


def _apply_branch_overrides(world, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply isolated worker-local settings without mutating a parent process."""
    applied = {}
    namespaces = {
        "world": world,
    }
    from central_bank import config as central_bank_config
    from economy import config as economy_config
    from fertility import config as fertility_config

    namespaces.update({
        "economy.config": economy_config,
        "fertility.config": fertility_config,
        "central_bank.config": central_bank_config,
    })
    for namespace, values in dict(overrides or {}).items():
        if namespace not in namespaces:
            raise ValueError(f"unsupported branch override namespace: {namespace}")
        target = namespaces[namespace]
        for key, value in dict(values).items():
            candidates = [key, str(key).lower()]
            attribute = next((name for name in candidates if hasattr(target, name)), None)
            if attribute is None:
                raise ValueError(f"unknown branch override: {namespace}.{key}")
            setattr(target, attribute, value)
            applied[f"{namespace}.{attribute}"] = value
            if namespace == "world":
                world.scenario_overrides[key] = value
    return applied


def _effective_config(world, task: BranchTask, checkpoint_metadata=None) -> dict[str, Any]:
    return {
        "branch_name": task.branch_name,
        "checkpoint_path": task.checkpoint_path,
        "task_seed": task.seed,
        "world_seed": getattr(world, "seed", None),
        "steps": task.steps,
        "scenario": getattr(world, "scenario_name", task.scenario),
        "firm_count": len(getattr(world, "firms", [])),
        "diagnostics_mode": task.diagnostics_mode,
        "observability_mode": task.observability_mode,
        "config_overrides": dict(task.config_overrides),
        "checkpoint_metadata": {
            "global_step": (checkpoint_metadata or {}).get("global_step"),
            "scenario": (checkpoint_metadata or {}).get("scenario"),
            "observability_mode": (checkpoint_metadata or {}).get("observability_mode"),
        },
    }


def _worker(task_payload: Mapping[str, Any]) -> dict[str, Any]:
    task = BranchTask.from_mapping(task_payload)
    output_dir = Path(task.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    peak_rss = None
    try:
        try:
            import psutil

            process = psutil.Process(os.getpid())
            peak_rss = process.memory_info().rss
        except Exception:
            process = None

        checkpoint_metadata = None
        setup_started = time.perf_counter()
        setup_seconds = 0.0
        simulation_seconds = 0.0
        finalize_seconds = 0.0
        if task.checkpoint_path:
            from checkpoint import load_world_checkpoint

            world, checkpoint_metadata = load_world_checkpoint(task.checkpoint_path)
        else:
            from scenarios import apply_scenario
            from world import World

            scenario = apply_scenario(task.scenario)
            world = World(
                initial_population=task.population,
                seed=task.seed,
                scenario_name=scenario["name"],
                scenario_overrides=scenario["overrides"],
                diagnostics_mode=task.diagnostics_mode,
                observability_mode=task.observability_mode,
            )
            if task.firm_count is not None:
                world.split_firms(task.firm_count)

        applied_overrides = _apply_branch_overrides(world, task.config_overrides)
        world.active_social_policy_branch_name = task.branch_name
        world.steps = task.steps
        world.diagnostics_mode = task.diagnostics_mode
        world.observability_mode = task.observability_mode

        if task.persist_diagnostics:
            world.configure_diagnostic_persistence(
                output_dir / "canonical_diagnostics",
                cadence=task.diagnostic_cadence,
                observability_mode=task.observability_mode,
            )

        setup_seconds = time.perf_counter() - setup_started
        simulation_started = time.perf_counter()
        world.run(progress_interval=task.progress_interval)
        simulation_seconds = time.perf_counter() - simulation_started
        finalize_started = time.perf_counter()

        if task.persist_diagnostics:
            world.export_diagnostics_csv(output_dir / "diagnostics.csv")
            world.export_firm_diagnostics_csv(output_dir / "firm_diagnostics.csv")
        if task.save_checkpoint:
            from checkpoint import save_world_checkpoint

            save_world_checkpoint(
                output_dir / "world_final.pkl",
                world,
                extra_metadata={"branch_name": task.branch_name},
            )

        social_snapshot = []
        for household in getattr(world, "households", []):
            members = [world.get_person_by_id(pid) for pid in [*getattr(household, "parents", []), *getattr(household, "children", [])]]
            members = [person for person in members if person is not None and getattr(person, "alive", False)]
            family_neighbors = set()
            for person in members:
                for relative_id in [*getattr(person, "parent_ids", []), *getattr(person, "children_ids", [])]:
                    relative = world.get_person_by_id(relative_id)
                    relative_household = world.get_household(getattr(relative, "household_id", None)) if relative is not None else None
                    if relative_household is not None and relative_household.id != household.id and not getattr(relative_household, "settlement_only", False):
                        family_neighbors.add(relative_household.id)
            try:
                minimum = float(world.needs_system.household_minimum_need_units(household)) * float(world.household_planning_price_this_step())
            except Exception:
                minimum = 0.0
            social_snapshot.append({
                "household_id": household.id,
                "settlement_only": bool(getattr(household, "settlement_only", False)),
                "cash": float(getattr(household, "wealth", 0.0)),
                "minimum_cost": minimum,
                "liquidity_weeks": float(getattr(household, "wealth", 0.0)) / minimum if minimum > 1e-12 else None,
                "elderly_count": sum(float(getattr(person, "age", 0.0)) >= 65.0 for person in members),
                "family_neighbor_ids": ";".join(str(value) for value in sorted(family_neighbors)),
                "income": float(getattr(household, "income_this_step", 0.0)),
                "consumption": float(getattr(household, "consumption_this_step", 0.0)),
                "saving": float(getattr(household, "saving_this_step", 0.0)),
                "payg_contribution": float(getattr(household, "payg_contribution_this_step", 0.0)),
                "pension_income": float(getattr(household, "pension_income_this_step", 0.0)),
                "wealth_transfer_paid": float(getattr(household, "wealth_transfer_paid_this_step", 0.0)),
                "wealth_transfer_received": float(getattr(household, "wealth_transfer_received_this_step", 0.0)),
            })
        (output_dir / "social_household_snapshot.json").write_text(json.dumps(social_snapshot, default=str), encoding="utf-8")
        liquidity_rows = list(getattr(world, "household_liquidity_snapshot_rows", []) or [])
        if getattr(world, "long_horizon_liquidity_enabled", False):
            long_tables = {
                "long_horizon_weekly_liquidity.csv": getattr(world, "long_horizon_weekly_liquidity_rows", []),
                "long_horizon_household_snapshots.csv": getattr(world, "long_horizon_household_snapshot_rows", []),
                "long_horizon_transition.csv": getattr(world, "long_horizon_transition_rows", []),
                "long_horizon_lifecycle.csv": getattr(world, "long_horizon_lifecycle_rows", []),
            }
            stream_path = getattr(world, "long_horizon_household_snapshot_stream_path", None)
            if stream_path and Path(stream_path).exists():
                shutil.copyfile(stream_path, output_dir / "long_horizon_household_snapshots.csv")
            for filename, rows in long_tables.items():
                if filename == "long_horizon_household_snapshots.csv" and stream_path and Path(stream_path).exists():
                    continue
                fields = []
                for row in rows:
                    for key in row:
                        if key not in fields:
                            fields.append(key)
                with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields or ["global_step"], extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
        if getattr(world, "household_liquidity_snapshot_enabled", False):
            fields = []
            for row in liquidity_rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            with (output_dir / "household_liquidity_weekly_snapshot.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fields or ["global_step", "household_id"],
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(liquidity_rows)
        if hasattr(world, "active_social_recipient_events"):
            (output_dir / "recipient_policy_event_window.json").write_text(json.dumps(world.active_social_recipient_events, default=str), encoding="utf-8")
        payg_audit_rows = list(getattr(world, "payg_cash_safety_audit_rows", []) or [])
        if getattr(world, "payg_cash_safety_audit_enabled", False):
            negative_rows = [row for row in payg_audit_rows if float(row.get("cash_after_consumption", 0.0)) < -1e-8]
            first_negative_week = min((int(row.get("week", 0)) for row in negative_rows), default=None)
            if first_negative_week is None:
                window_rows = []
            else:
                window_rows = [
                    row for row in payg_audit_rows
                    if first_negative_week - 3 <= int(row.get("week", 0)) <= first_negative_week + 2
                ]
            contribution_rows = [row for row in payg_audit_rows if float(row.get("scheduled_payg_contribution", 0.0)) > 1e-8]
            pension_rows = [row for row in payg_audit_rows if float(row.get("pension_received", 0.0)) > 1e-8]
            private_rows = [row for row in payg_audit_rows if float(row.get("private_transfer_paid", 0.0)) > 1e-8]
            negative_values = [float(row.get("cash_after_consumption", 0.0)) for row in negative_rows]
            audit_summary = {
                "row_count": len(payg_audit_rows),
                "negative_household_weeks": len(negative_rows),
                "negative_unique_households": len({row.get("household_id") for row in negative_rows}),
                "minimum_cash_after_consumption": min((float(row.get("cash_after_consumption", 0.0)) for row in payg_audit_rows), default=0.0),
                "cash_constraint_binding_household_weeks": sum(bool(row.get("consumption_cash_constraint_binding", False)) for row in payg_audit_rows),
                "planned_consumption_total": sum(float(row.get("planned_consumption", 0.0)) for row in payg_audit_rows),
                "actual_consumption_total": sum(float(row.get("actual_consumption", 0.0)) for row in payg_audit_rows),
                "consumption_suppressed_total": sum(float(row.get("cash_constraint_suppressed_consumption", 0.0)) for row in payg_audit_rows),
                "mean_suppressed_per_binding": (sum(float(row.get("cash_constraint_suppressed_consumption", 0.0)) for row in payg_audit_rows) / max(1, sum(bool(row.get("consumption_cash_constraint_binding", False)) for row in payg_audit_rows))),
                "median_negative_cash": sorted(negative_values)[len(negative_values) // 2] if negative_values else 0.0,
                "first_negative_week": first_negative_week,
                "first_negative_household_ids": sorted({row.get("household_id") for row in negative_rows if first_negative_week is not None and int(row.get("week", 0)) == first_negative_week}),
                "last_negative_week": max((int(row.get("week", 0)) for row in negative_rows), default=None),
                "opening_negative_household_weeks": sum(float(row.get("cash_opening", 0.0)) < -1e-8 for row in payg_audit_rows),
                "after_contribution_negative_count": sum(float(row.get("cash_after_payg_contribution", 0.0)) < -1e-8 for row in payg_audit_rows),
                "contributor_household_weeks": len(contribution_rows),
                "scheduled_contribution_total": sum(float(row.get("scheduled_payg_contribution", 0.0)) for row in contribution_rows),
                "actual_contribution_total": sum(float(row.get("actual_payg_contribution", 0.0)) for row in contribution_rows),
                "contribution_cash_violations": sum(float(row.get("actual_payg_contribution", 0.0)) > max(0.0, float(row.get("cash_after_legacy_private_support", 0.0))) + 1e-8 for row in contribution_rows),
                "contribution_post_cash_violations": sum(float(row.get("cash_after_legacy_private_support", 0.0)) >= -1e-8 and float(row.get("cash_after_payg_contribution", 0.0)) < -1e-8 for row in contribution_rows),
                "preexisting_negative_before_contribution_count": sum(float(row.get("cash_after_legacy_private_support", 0.0)) < -1e-8 for row in contribution_rows),
                "minimum_cash_after_contribution": min((float(row.get("cash_after_payg_contribution", 0.0)) for row in contribution_rows), default=0.0),
                "pension_recipient_household_weeks": len(pension_rows),
                "pension_received_total": sum(float(row.get("pension_received", 0.0)) for row in pension_rows),
                "pension_cash_decrease_count": sum(float(row.get("cash_after_pension", 0.0)) + 1e-8 < float(row.get("cash_after_payg_contribution", 0.0)) for row in pension_rows),
                "minimum_pension_cash_delta": min((float(row.get("cash_after_pension", 0.0)) - float(row.get("cash_after_payg_contribution", 0.0)) for row in pension_rows), default=0.0),
                "private_donor_household_weeks": len(private_rows),
                "private_transfer_paid_total": sum(float(row.get("private_transfer_paid", 0.0)) for row in private_rows),
                "private_reserve_violations": sum(float(row.get("cash_after_private_transfer", 0.0)) < 13.0 * float(row.get("minimum_consumption", 0.0)) - 1e-8 for row in private_rows),
                "later_negative_after_safe_private_transfer": sum(float(row.get("cash_after_private_transfer", 0.0)) >= 13.0 * float(row.get("minimum_consumption", 0.0)) - 1e-8 and float(row.get("cash_after_consumption", 0.0)) < -1e-8 for row in private_rows),
            }
            (output_dir / "payg_cash_safety_event_window.json").write_text(json.dumps(window_rows, default=str), encoding="utf-8")
            (output_dir / "payg_cash_safety_summary.json").write_text(json.dumps(audit_summary, default=str), encoding="utf-8")
        if getattr(world, "payg_weekly_history", None):
            (output_dir / "payg_weekly_summary.json").write_text(json.dumps(world.payg_weekly_history, default=str), encoding="utf-8")
        if getattr(world, "wealth_transfer_weekly_history", None):
            (output_dir / "wealth_transfer_weekly_summary.json").write_text(json.dumps(world.wealth_transfer_weekly_history, default=str), encoding="utf-8")
        if process is not None:
            peak_rss = max(peak_rss or 0, process.memory_info().rss)
        finalize_seconds = time.perf_counter() - finalize_started

        result = {
            "status": "completed",
            "branch_name": task.branch_name,
            "output_dir": str(output_dir),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "setup_seconds": round(setup_seconds, 6),
            "simulation_seconds": round(simulation_seconds, 6),
            "finalize_seconds": round(finalize_seconds, 6),
            "final_global_step": len(getattr(world, "population_history", [])),
            "final_population": len(getattr(world, "population", [])),
            "final_households": len(getattr(world, "households", [])),
            "final_firms": len(getattr(world, "firms", [])),
            "final_capital_good_firms": len(getattr(world, "capital_good_firms", [])),
            "final_employment": sum(len(getattr(firm, "employee_ids", []) or []) for firm in [*getattr(world, "firms", []), *getattr(world, "capital_good_firms", [])]),
            "final_household_cash": sum(float(getattr(household, "wealth", 0.0)) for household in getattr(world, "households", [])),
            "min_household_cash": min((float(getattr(household, "wealth", 0.0)) for household in getattr(world, "households", [])), default=0.0),
            "final_firm_cash": sum(float(getattr(firm, "cash", 0.0)) for firm in [*getattr(world, "firms", []), *getattr(world, "capital_good_firms", [])]),
            "money_supply": float(getattr(getattr(getattr(world, "firm_system", None), "central_bank", None), "money_supply", 0.0)),
            "loan_principal": sum(float(getattr(firm, "loan_balance", getattr(firm, "principal", 0.0))) for firm in [*getattr(world, "firms", []), *getattr(world, "capital_good_firms", [])]),
            "state_fingerprint": _state_fingerprint(world),
            "rng_fingerprint": _rng_fingerprint(world),
            "peak_rss_bytes": peak_rss,
            "invariant_violations": len(getattr(world, "invariant_violations", [])),
            "max_abs_monetary_accounting_gap": max((abs(float(row.get("monetary_accounting_gap", 0.0))) for row in getattr(world, "diagnostics_rows", [])), default=0.0),
            "max_abs_money_delta_gap": max((abs(float(row.get("money_delta_gap", 0.0))) for row in getattr(world, "diagnostics_rows", [])), default=0.0),
            "max_abs_food_conservation_gap": max((abs(float(row.get("food_conservation_gap", 0.0))) for row in getattr(world, "diagnostics_rows", [])), default=0.0),
            "final_fund_cash": float(getattr(getattr(world, "social_insurance_fund", None), "cash", 0.0)),
            "recipient_event_count": len(getattr(world, "active_social_recipient_events", []) or []),
            "payg_cash_safety_audit_row_count": len(payg_audit_rows),
            "household_liquidity_snapshot_row_count": len(liquidity_rows),
            "long_horizon_weekly_row_count": len(getattr(world, "long_horizon_weekly_liquidity_rows", []) or []),
            "long_horizon_household_snapshot_row_count": len(getattr(world, "long_horizon_household_snapshot_rows", []) or []),
            "payg_cash_safety_first_negative_week": min((int(row.get("week", 0)) for row in payg_audit_rows if float(row.get("cash_after_consumption", 0.0)) < -1e-8), default=None),
            "applied_overrides": applied_overrides,
            "effective_config": _effective_config(world, task, checkpoint_metadata),
        }
        (output_dir / "branch_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        (output_dir / "effective_config.json").write_text(json.dumps(result["effective_config"], indent=2, default=str), encoding="utf-8")
        return result
    except Exception as exc:
        result = {
            "status": "failed",
            "branch_name": task.branch_name,
            "output_dir": str(output_dir),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "setup_seconds": round(setup_seconds, 6),
            "simulation_seconds": round(simulation_seconds, 6),
            "finalize_seconds": round(finalize_seconds, 6),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "branch_failure.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


def _validate_tasks(tasks: list[BranchTask]) -> None:
    names = [task.branch_name for task in tasks]
    if len(set(names)) != len(names):
        raise ValueError("branch_name values must be unique")
    directories = [os.path.abspath(task.output_dir) for task in tasks]
    if len(set(directories)) != len(directories):
        raise ValueError("branch output directories must be unique")


def run_branches(tasks, max_workers: int = 2) -> dict[str, Any]:
    """Run independent tasks with Windows-safe process isolation.

    Results are returned in input order. Worker exceptions are represented in
    the ``failed`` list so successful branches are never silently discarded.
    """
    normalized = [
        task if isinstance(task, BranchTask) else BranchTask.from_mapping(task)
        for task in tasks
    ]
    if not normalized:
        return {"completed": [], "failed": [], "cancelled": [], "max_workers": 0}
    _validate_tasks(normalized)
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    max_workers = min(int(max_workers), len(normalized))
    context = mp.get_context("spawn")
    completed = []
    failed = []
    cancelled = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(_worker, task.to_dict()): task
            for task in normalized
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "branch_name": task.branch_name,
                    "output_dir": task.output_dir,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                }
            if result.get("status") == "completed":
                completed.append(result)
            else:
                failed.append(result)
    ordered = {item["branch_name"]: item for item in [*completed, *failed]}
    return {
        "completed": [ordered[task.branch_name] for task in normalized if task.branch_name in ordered and ordered[task.branch_name].get("status") == "completed"],
        "failed": [ordered[task.branch_name] for task in normalized if task.branch_name in ordered and ordered[task.branch_name].get("status") != "completed"],
        "cancelled": cancelled,
        "max_workers": max_workers,
    }



if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    from experiment_workflow import cli

    raise SystemExit(cli())

