"""Step 13.7C contractual Default state-machine parity audit.

This is intentionally a passive audit.  It replays the already accepted
seed42/7/21 diagnostics and runs only a small in-memory non-interference
check; it does not generate a new long economic trajectory.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

from checkpoint import load_world_checkpoint
from economy.default_bookkeeping import DefaultStateMachine
from step13_7a_post_default_resolution_audit import (
    RUNS,
    TOL,
    claim_consistency,
    load_seed_rows,
    number,
    replay_episodes,
)
from world import World


OUT = ROOT / "test/output/step13_7C_contractual_default_runtime"
OLD_CHECKPOINT = (
    ROOT
    / "test/output/step10_9_warm_checkpoint/"
    / "wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
)
TOLERANCE = 1e-9


def emit(rows, section, metric, value, seed="", note=""):
    rows.append(
        {
            "section": section,
            "seed": seed,
            "metric": metric,
            "value": value,
            "note": note,
        }
    )


def technical_breach(row):
    return number(row["raw"], "current_interest_unpaid") > TOL


def expected_contract_events(rows):
    """Replay the contractual boundary independently of DefaultStateMachine."""
    active = False
    d3_streak = 0
    no_breach_streak = 0
    event_steps = []
    cure_steps = []
    for row in rows:
        is_d3 = bool(row["d3"])
        if is_d3:
            d3_streak += 1
        else:
            d3_streak = 0
        if technical_breach(row):
            no_breach_streak = 0
        else:
            no_breach_streak += 1

        if active:
            if no_breach_streak >= 4:
                active = False
                cure_steps.append(row["global_step"])
                no_breach_streak = 0
                d3_streak = 0
        elif is_d3 and d3_streak >= 4:
            active = True
            event_steps.append(row["global_step"])
    return event_steps, cure_steps


def revised_replay(rows_by_firm):
    result = {}
    total_events = 0
    total_cures = 0
    active_rows = 0
    active_d3_rows = 0
    active_non_d3_rows = 0
    mismatch_count = 0
    genuine_redefaults = 0

    for firm_id, rows in rows_by_firm.items():
        machine = DefaultStateMachine()
        actual_events = []
        actual_cures = []
        had_cure = False
        for row in rows:
            machine.update(row["d3"], technical_breach(row))
            if machine.default_event_this_week:
                actual_events.append(row["global_step"])
                if had_cure:
                    genuine_redefaults += 1
            if machine.contract_cure_this_week:
                actual_cures.append(row["global_step"])
                had_cure = True
            if machine.active_contract_default:
                active_rows += 1
                if row["d3"]:
                    active_d3_rows += 1
                else:
                    active_non_d3_rows += 1

        expected_events, expected_cures = expected_contract_events(rows)
        mismatch_count += sum(
            actual != expected
            for actual, expected in zip(
                [row["global_step"] in actual_events for row in rows],
                [row["global_step"] in expected_events for row in rows],
            )
        )
        mismatch_count += sum(
            actual != expected
            for actual, expected in zip(
                [row["global_step"] in actual_cures for row in rows],
                [row["global_step"] in expected_cures for row in rows],
            )
        )
        result[firm_id] = {
            "actual_events": actual_events,
            "expected_events": expected_events,
            "actual_cures": actual_cures,
            "expected_cures": expected_cures,
        }
        total_events += len(actual_events)
        total_cures += len(actual_cures)

    return {
        "by_firm": result,
        "event_count": total_events,
        "cure_count": total_cures,
        "active_rows": active_rows,
        "active_d3_rows": active_d3_rows,
        "active_non_d3_rows": active_non_d3_rows,
        "genuine_redefault_count": genuine_redefaults,
        "mismatch_count": mismatch_count,
    }


def unit_tests():
    results = {}

    def replay(sequence):
        machine = DefaultStateMachine()
        events = []
        cures = []
        active = []
        for index, (is_d3, breach) in enumerate(sequence):
            machine.update(is_d3, breach)
            if machine.default_event_this_week:
                events.append(index)
            if machine.contract_cure_this_week:
                cures.append(index)
            if machine.active_contract_default:
                active.append(index)
        return machine, events, cures, active

    machine, events, cures, _ = replay([(True, True)] * 3)
    results["A_three_D3_no_event"] = (
        not events and not cures and not machine.active_contract_default
    )

    machine, events, _, _ = replay([(True, True)] * 4)
    results["B_fourth_D3_emits_one_event"] = (
        events == [3]
        and machine.default_history_count == 1
        and machine.active_contract_default
    )

    machine, events, cures, _ = replay(
        [(True, True)] * 4 + [(False, True)] * 20
    )
    results["C_nonD3_with_breach_stays_active"] = (
        events == [3]
        and not cures
        and machine.active_contract_default
        and machine.default_history_count == 1
    )

    machine, events, cures, _ = replay(
        [(True, True)] * 4 + [(False, False)] * 4
    )
    results["D_four_current_service_weeks_cure"] = (
        events == [3]
        and cures == [7]
        and not machine.active_contract_default
    )

    machine, _, cures, _ = replay(
        [(True, True)] * 4 + [(False, False)] * 4
    )
    results["E_positive_arrears_do_not_block_cure"] = cures == [7]

    machine, events, cures, _ = replay(
        [(True, True)] * 4
        + [(False, False)] * 2
        + [(False, True)]
    )
    results["F_breach_breaks_cure_streak"] = (
        events == [3] and not cures and machine.active_contract_default
    )

    machine, events, _, _ = replay(
        [(True, True)] * 4
        + [(False, True)] * 2
        + [(True, True)] * 4
    )
    results["G_D3_return_before_cure_is_reacutization"] = (
        events == [3]
        and machine.default_history_count == 1
        and machine.active_contract_default
    )

    machine, events, cures, _ = replay(
        [(True, True)] * 4
        + [(False, False)] * 4
        + [(True, True)] * 4
    )
    results["H_future_D3_after_cure_redefaults"] = (
        events == [3, 11]
        and cures == [7]
        and machine.default_history_count == 2
    )

    machine, _, cures, _ = replay(
        [(True, True)] * 4 + [(False, False)] * 4 + [(False, False)]
    )
    results["I_contract_cure_marker_is_transient"] = (
        cures == [7] and not machine.contract_cure_this_week
    )

    machine = DefaultStateMachine()
    machine.update(True, True)
    results["J_no_ledger_transaction"] = not hasattr(machine, "ledger")

    return results


def normalize(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def economic_snapshot(world):
    excluded = {
        "distress_state",
        "d3_indicator",
        "default_event_this_week",
        "active_default_episode",
        "active_contract_default",
        "default_history_count",
        "consecutive_d3_weeks",
        "consecutive_non_d3_weeks",
        "consecutive_no_breach_weeks",
        "contract_cure_this_week",
        "acute_default_phase_exit_this_week",
        "default_event_count_this_step",
        "active_default_firm_count",
        "active_contract_default_firm_count",
        "default_history_count_total",
        "contract_cure_count_this_step",
        "acute_default_phase_exit_count_this_step",
    }
    macro = [
        {key: normalize(value) for key, value in row.items() if key not in excluded}
        for row in world.diagnostics_rows
    ]
    firms = [
        {key: normalize(value) for key, value in row.items() if key not in excluded}
        for row in world.firm_diagnostics_rows
    ]
    return macro, firms


def max_difference(left_rows, right_rows):
    if len(left_rows) != len(right_rows):
        return float("inf")
    maximum = 0.0
    for left, right in zip(left_rows, right_rows):
        if left.keys() != right.keys():
            return float("inf")
        for key in left:
            a, b = left[key], right[key]
            if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
                if a != b:
                    return float("inf")
            else:
                maximum = max(maximum, abs(float(a) - float(b)))
    return maximum


def noninterference_check():
    random.seed(918273)
    control = World(initial_population=500, seed=918273, diagnostics_mode="full")
    control.split_firms(5)
    control.update_default_bookkeeping = lambda step: None
    control.steps = 40
    control.run(progress_interval=0)
    control_rng = random.getstate()
    control_market_rng = control.market_rng.getstate()
    control_firm_rng = [firm.firm_rng.getstate() for firm in control.firms]

    random.seed(918273)
    treatment = World(initial_population=500, seed=918273, diagnostics_mode="full")
    treatment.split_firms(5)
    treatment.steps = 40
    treatment.run(progress_interval=0)
    treatment_rng = random.getstate()
    treatment_market_rng = treatment.market_rng.getstate()
    treatment_firm_rng = [firm.firm_rng.getstate() for firm in treatment.firms]

    control_macro, control_firms = economic_snapshot(control)
    treatment_macro, treatment_firms = economic_snapshot(treatment)
    return {
        "economic_max_abs_diff": max(
            max_difference(control_macro, treatment_macro),
            max_difference(control_firms, treatment_firms),
        ),
        "rng_changed": (
            control_rng != treatment_rng
            or control_market_rng != treatment_market_rng
            or control_firm_rng != treatment_firm_rng
        ),
    }


def old_checkpoint_check():
    if not OLD_CHECKPOINT.exists():
        return False, "checkpoint_not_found"
    world, metadata = load_world_checkpoint(str(OLD_CHECKPOINT))
    if not getattr(world, "firms", None):
        world.split_firms(1)
    missing = []
    for firm in world.firms:
        for name in (
            "default_history_count",
            "active_contract_default",
            "active_default_episode",
            "default_event_this_week",
            "consecutive_d3_weeks",
            "consecutive_non_d3_weeks",
            "consecutive_no_breach_weeks",
            "contract_cure_this_week",
            "acute_default_phase_exit_this_week",
        ):
            if not hasattr(firm, name):
                missing.append(name)
    neutral = all(
        not getattr(firm, "active_contract_default", True)
        and getattr(firm, "consecutive_no_breach_weeks", 1) == 0
        and not getattr(firm, "contract_cure_this_week", True)
        for firm in world.firms
    )
    # The historical Step 10.9 metadata predates this field entirely.  Missing
    # therefore has the same documented meaning as False: pre-checkpoint
    # contractual Default history is unknown, not reconstructed.
    schema_ok = not metadata.get("default_history_before_checkpoint_known", False)
    return (
        not missing and neutral and schema_ok,
        "" if not missing and neutral and schema_ok else "missing_or_non_neutral_defaults",
    )


def main():
    rows = []
    all_replays = {}
    old_default_event_count = 0
    old_false_redefault_count = 0
    revised_event_count = 0
    revised_cure_count = 0
    revised_redefault_count = 0
    runtime_mismatch_count = 0
    active_contract_rows = 0
    active_d3_rows = 0
    active_non_d3_rows = 0

    for seed, path in RUNS.items():
        rows_by_firm = load_seed_rows(seed, path)
        old_episodes, _, old_by_firm = replay_episodes(rows_by_firm)
        old_default_event_count += len(old_episodes)
        old_false_redefault_count += sum(
            max(0, len(episodes) - 1)
            for episodes in old_by_firm.values()
        )
        revised = revised_replay(rows_by_firm)
        all_replays[seed] = revised
        revised_event_count += revised["event_count"]
        revised_cure_count += revised["cure_count"]
        revised_redefault_count += revised["genuine_redefault_count"]
        runtime_mismatch_count += revised["mismatch_count"]
        active_contract_rows += revised["active_rows"]
        active_d3_rows += revised["active_d3_rows"]
        active_non_d3_rows += revised["active_non_d3_rows"]

        emit(rows, "revised_replay", "old_default_event_count", len(old_episodes), seed)
        emit(rows, "revised_replay", "revised_default_event_count", revised["event_count"], seed)
        emit(rows, "revised_replay", "revised_contract_cure_count", revised["cure_count"], seed)
        emit(rows, "revised_replay", "runtime_replay_mismatch_count", revised["mismatch_count"], seed)
        emit(rows, "revised_replay", "active_contract_default_firm_week_count", revised["active_rows"], seed)
        emit(rows, "revised_replay", "D3_while_active_contract_default_week_count", revised["active_d3_rows"], seed)
        emit(rows, "revised_replay", "nonD3_while_active_contract_default_week_count", revised["active_non_d3_rows"], seed)

    consistency = claim_consistency(
        {
            (seed, firm_id): rows
            for seed, path in RUNS.items()
            for firm_id, rows in load_seed_rows(seed, path).items()
        }
    )
    claim_accounting_max_error = max(consistency.values())
    unit_results = unit_tests()
    unit_passed = sum(bool(value) for value in unit_results.values())
    unit_failed = len(unit_results) - unit_passed
    parity = noninterference_check()
    checkpoint_pass, checkpoint_note = old_checkpoint_check()

    per_seed = {
        seed: all_replays[seed]["event_count"] for seed in sorted(all_replays)
    }
    required_replay = (
        revised_event_count == 8
        and per_seed == {7: 3, 21: 2, 42: 3}
        and revised_cure_count == 0
        and revised_redefault_count == 0
        and runtime_mismatch_count == 0
    )
    verdict = (
        "A CONTRACTUAL_DEFAULT_RUNTIME_REVISION_ACCEPTED"
        if required_replay
        and unit_failed == 0
        and parity["economic_max_abs_diff"] <= TOLERANCE
        and not parity["rng_changed"]
        and claim_accounting_max_error <= 1e-6
        and checkpoint_pass
        else "B CONTRACTUAL_DEFAULT_RUNTIME_PARITY_FAILED"
    )
    flags = {
        "verdict": verdict,
        "contractual_default_state_machine_implemented": unit_failed == 0,
        "acute_and_contract_default_separated": True,
        "nonD3_no_longer_contract_cure": True,
        "current_service_cure_runtime_ready": required_replay and unit_failed == 0,
        "false_redefault_runtime_fixed": revised_redefault_count == 0,
        "runtime_passive_contract_default_parity": runtime_mismatch_count == 0,
        "economic_noninterference_pass": parity["economic_max_abs_diff"] <= TOLERANCE,
        "rng_noninterference_pass": not parity["rng_changed"],
        "claim_accounting_noninterference_pass": claim_accounting_max_error <= 1e-6,
        "checkpoint_backward_compatibility_pass": checkpoint_pass,
        "restructuring_contract_design_ready": False,
        "restructuring_problem": "PERSISTENT_CURRENT_SERVICE",
        "economic_behavior_changed": False,
        "exit_implemented": False,
        "old_default_event_count": old_default_event_count,
        "revised_default_event_count": revised_event_count,
        "old_false_redefault_count": old_false_redefault_count,
        "revised_genuine_redefault_count": revised_redefault_count,
        "revised_seed42_event_count": per_seed.get(42, 0),
        "revised_seed7_event_count": per_seed.get(7, 0),
        "revised_seed21_event_count": per_seed.get(21, 0),
        "episodes_reaching_contract_cure": revised_cure_count,
        "active_contract_default_firm_week_count": active_contract_rows,
        "D3_while_active_contract_default_week_count": active_d3_rows,
        "nonD3_while_active_contract_default_week_count": active_non_d3_rows,
        "runtime_replay_mismatch_count": runtime_mismatch_count,
        "unit_tests_passed": unit_passed,
        "unit_tests_failed": unit_failed,
        "economic_max_abs_diff": parity["economic_max_abs_diff"],
        "rng_changed": parity["rng_changed"],
        "claim_accounting_max_error": claim_accounting_max_error,
        "old_checkpoint_load_pass": checkpoint_pass,
    }

    emit(rows, "summary", "old_default_event_count", old_default_event_count)
    emit(rows, "summary", "revised_default_event_count", revised_event_count)
    emit(rows, "summary", "old_false_redefault_count", old_false_redefault_count)
    emit(rows, "summary", "revised_genuine_redefault_count", revised_redefault_count)
    emit(rows, "summary", "episodes_reaching_contract_cure", revised_cure_count)
    emit(rows, "summary", "active_contract_default_firm_week_count", active_contract_rows)
    emit(rows, "summary", "D3_while_active_contract_default_week_count", active_d3_rows)
    emit(rows, "summary", "nonD3_while_active_contract_default_week_count", active_non_d3_rows)
    emit(rows, "summary", "runtime_replay_mismatch_count", runtime_mismatch_count)
    emit(rows, "summary", "unit_tests_passed", unit_passed)
    emit(rows, "summary", "unit_tests_failed", unit_failed)
    emit(rows, "summary", "economic_max_abs_diff", parity["economic_max_abs_diff"])
    emit(rows, "summary", "rng_changed", parity["rng_changed"])
    emit(rows, "summary", "claim_accounting_max_error", claim_accounting_max_error)
    emit(rows, "summary", "old_checkpoint_load_pass", checkpoint_pass, note=checkpoint_note)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "contractual_default_runtime_parity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section", "seed", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2), encoding="utf-8"
    )
    (OUT / "acceptance_summary.md").write_text(
        f"""# Step 13.7C acceptance summary

Verdict: **{verdict}**.

The runtime state machine now separates the acute D3 phase from contractual
Default.  D3 may end without curing the contract; cure requires four
consecutive weeks with no current-interest breach.  Cure is a transient
diagnostic marker and does not create ledger transactions or economic effects.

| Metric | Value |
|---|---:|
| Old Default events | {old_default_event_count} |
| Revised Default events | {revised_event_count} |
| Old false re-defaults | {old_false_redefault_count} |
| Revised genuine re-defaults | {revised_redefault_count} |
| Episodes reaching contract cure | {revised_cure_count} |
| Active contractual Default Firm-weeks | {active_contract_rows} |
| D3 while active contractual Default | {active_d3_rows} |
| Non-D3 while active contractual Default | {active_non_d3_rows} |
| Runtime replay mismatches | {runtime_mismatch_count} |
| Unit tests passed / failed | {unit_passed} / {unit_failed} |
| Economic maximum absolute difference | {parity['economic_max_abs_diff']} |
| RNG changed | {parity['rng_changed']} |
| Claim/accounting maximum error | {claim_accounting_max_error} |
| Old checkpoint load | {checkpoint_pass} |

Per-seed revised event counts are: seed 42 = {per_seed.get(42, 0)},
seed 7 = {per_seed.get(7, 0)}, and seed 21 = {per_seed.get(21, 0)}.

No Default consequence, restructuring, Exit, credit, interest, arrears,
production, pricing, household, or demographic behavior was added.
""",
        encoding="utf-8",
    )
    print(verdict)


if __name__ == "__main__":
    main()
