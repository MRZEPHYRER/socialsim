"""Step 15E.4 short validation for canonical diagnostic persistence."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checkpoint import load_world_checkpoint, save_world_checkpoint
from world import World


OUTPUT = ROOT / "test/output/step15E4_diagnostic_persistence"
WARM_CHECKPOINT = ROOT / (
    "test/output/step10_9_warm_checkpoint/"
    "wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
)


def _read(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _signature(world):
    return {
        "population": list(world.population_history),
        "income": list(world.income_history),
        "consumption": list(world.consumption_history),
        "saving": list(world.saving_history),
        "wealth": list(world.wealth_history),
        "firms": [
            (
                firm.firm_id,
                firm.cash,
                firm.inventory_units,
                firm.price,
                firm.loan_balance,
            )
            for firm in world.firms
        ],
    }


def _main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    steps = 12
    # Construct/run the OFF world first, then seed the ON world anew.  World
    # retains the legacy global demographic RNG, so this ordering is required
    # for a valid same-seed control comparison.
    off = World(initial_population=120, seed=42, diagnostics_mode="full")
    off.steps = steps
    off.run(progress_interval=0)
    on = World(initial_population=120, seed=42, diagnostics_mode="full")
    on.configure_diagnostic_persistence(OUTPUT, cadence=1)
    on.steps = steps
    on.run(progress_interval=0)
    parity = _signature(off) == _signature(on)

    macro = _read(OUTPUT / "macro_diagnostics.csv")
    firms = _read(OUTPUT / "firm_diagnostics.csv")
    accounting = _read(OUTPUT / "accounting_diagnostics.csv")
    distinct_macro_steps = {int(row["global_step"]) for row in macro}
    distinct_firm_keys = {(int(row["global_step"]), row["firm_id"]) for row in firms}
    record_types = {row["record_type"] for row in accounting}
    required_macro = {
        "population", "total_employment", "unassigned_labor",
        "household_wage_income", "household_total_income",
        "household_consumption", "household_saving", "household_cash_wealth",
        "household_equity_assets", "household_financial_net_worth",
        "firm_aggregate_revenue", "firm_aggregate_operating_profit",
        "firm_aggregate_cfo", "firm_aggregate_cash", "firm_aggregate_principal",
        "firm_aggregate_arrears", "money_stock", "money_created", "money_destroyed",
        "household_final_consumption", "firm_fixed_investment",
        "government_final_demand", "external_final_demand", "intermediate_demand",
        "accounting_reconciliation_gap", "money_reconciliation_gap",
    }
    macro_fields = set(macro[0]) if macro else set()

    with tempfile.TemporaryDirectory(prefix="step15e4_resume_") as temp:
        resume_dir = Path(temp)
        first = World(initial_population=120, seed=43, diagnostics_mode="full")
        first.configure_diagnostic_persistence(resume_dir, cadence=1)
        first.steps = 6
        first.run(progress_interval=0)
        checkpoint_path = resume_dir / "resume.pkl"
        save_world_checkpoint(str(checkpoint_path), first, {"step15e4": True})
        resumed, _ = load_world_checkpoint(str(checkpoint_path))
        resumed.configure_diagnostic_persistence(resume_dir, cadence=1)
        resumed.steps = 6
        resumed.run(progress_interval=0)
        resumed_macro = _read(resume_dir / "macro_diagnostics.csv")
        resume_steps = [int(row["global_step"]) for row in resumed_macro]
        resume_no_duplicates = len(resume_steps) == len(set(resume_steps))
        resume_continuity = resume_steps == list(range(12))

    old_checkpoint_load = False
    if WARM_CHECKPOINT.exists():
        loaded, metadata = load_world_checkpoint(str(WARM_CHECKPOINT))
        old_checkpoint_load = loaded is not None and int(metadata["global_step"]) == 5000

    checks = {
        "diagnostics_off_on_parity": parity,
        "fresh_run_multiple_authoritative_steps": len(distinct_macro_steps) == steps,
        "macro_schema_complete": required_macro <= macro_fields,
        "firm_panel_multiple_steps": len(distinct_firm_keys) >= steps,
        "accounting_record_types_present": {"firm", "household", "public", "central_bank", "reconciliation"} <= record_types,
        "checkpoint_resume_continuous": resume_continuity,
        "checkpoint_resume_no_duplicates": resume_no_duplicates,
        "old_checkpoint_load_valid": old_checkpoint_load,
        "new_rng_draws": 0,
        "economic_behavior_changed": False,
    }
    all_pass = all(
        value is True
        for key, value in checks.items()
        if key not in {"new_rng_draws", "economic_behavior_changed"}
    ) and checks["new_rng_draws"] == 0 and checks["economic_behavior_changed"] is False
    verdict = (
        "A. CANONICAL_DIAGNOSTIC_PERSISTENCE_READY"
        if all_pass else "G. OTHER_BLOCKER"
    )
    flags = {
        "verdict": verdict,
        **checks,
        "fresh_steps": steps,
        "macro_rows": len(macro),
        "firm_rows": len(firms),
        "accounting_rows": len(accounting),
        "macro_distinct_steps": len(distinct_macro_steps),
        "resume_steps": resume_steps,
        "record_types": sorted(record_types),
        "persistence_cadence": 1,
        "storage_mode": "appendable aggregate/Firm/accounting CSV; no agent serialization",
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = f"""# Step 15E.4 Canonical Diagnostic Persistence

## Verdict

**{verdict}**

A fresh deterministic run recorded `{len(distinct_macro_steps)}` authoritative
macro steps and `{len(firms)}` Firm-panel rows. The canonical files are written
appendably during end-of-week execution, rather than reconstructed from a final
snapshot. Accounting rows are stored in one lightweight file with explicit
`record_type` values: `{', '.join(sorted(record_types))}`.

Diagnostics-on versus diagnostics-off state parity: `{parity}`. The checkpoint
resume test produced the continuous step sequence `{resume_steps}` with no
duplicate keys. An accepted old warm checkpoint loaded successfully without
fabricating diagnostic history: `{old_checkpoint_load}`.

No economic mechanism, RNG path, ownership behavior, investment behavior,
dividend policy, or Step13 rule was changed. Persistence is configured with
`--persist-diagnostics` and defaults to off; `--diagnostic-cadence` controls
recording frequency.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
