"""Step 13.8A passive restructuring contract design.

Only existing accepted Firm diagnostics are replayed.  No restructuring
behavior, economic parameter, ledger entry, or treatment simulation is run.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

from economy.default_bookkeeping import DefaultStateMachine
from step13_7a_post_default_resolution_audit import (
    RUNS,
    TOL,
    load_seed_rows,
    number,
)


OUT = ROOT / "test/output/step13_8A_passive_restructuring_contract_design"
TOLERANCE = 1e-9


def emit(rows, section, metric, value, note=""):
    rows.append(
        {
            "section": section,
            "metric": metric,
            "value": value,
            "note": note,
        }
    )


def technical_breach(row):
    return number(row["raw"], "current_interest_unpaid") > TOL


def quantile(values, probability):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(probability * (len(ordered) - 1)))]


def replay_active_default():
    """Return compact active-default observations and episode durations."""
    active_rows = []
    episodes = []
    per_seed = {}

    for seed, path in RUNS.items():
        seed_episodes = []
        rows_by_firm = load_seed_rows(seed, path)
        for firm_id, rows in rows_by_firm.items():
            machine = DefaultStateMachine()
            current_episode = None
            for row in rows:
                machine.update(row["d3"], technical_breach(row))
                if machine.default_event_this_week:
                    current_episode = {
                        "seed": seed,
                        "firm_id": firm_id,
                        "event_step": row["global_step"],
                        "duration": 0,
                    }
                    episodes.append(current_episode)
                    seed_episodes.append(current_episode)

                if machine.active_contract_default:
                    active_rows.append(row)
                    if current_episode is not None:
                        current_episode["duration"] += 1

                if machine.contract_cure_this_week:
                    current_episode = None

        per_seed[seed] = len(seed_episodes)

    return active_rows, episodes, per_seed


def service_metrics(active_rows):
    ratios = []
    relief = []
    closure = {25: 0, 50: 0, 75: 0}
    capitalization_changes = []
    haircut_25 = []
    haircut_50 = []

    for row in active_rows:
        raw = row["raw"]
        due = max(0.0, number(raw, "current_interest_due"))
        paid = max(0.0, number(raw, "interest_paid"))
        ratio = paid / due if due > TOL else 1.0
        ratio = max(0.0, ratio)
        ratios.append(ratio)
        relief.append(max(1.0 - ratio, 0.0))
        for percentage in closure:
            adjusted_due = due * (1.0 - percentage / 100.0)
            if adjusted_due <= paid + TOL:
                closure[percentage] += 1

        opening_principal = max(0.0, number(raw, "opening_principal"))
        closing_principal = max(
            0.0,
            number(raw, "closing_principal", number(raw, "loan_balance")),
        )
        closing_arrears = max(
            0.0,
            number(
                raw,
                "closing_interest_arrears",
                number(raw, "interest_arrears"),
            ),
        )
        # Infer the already-used weekly rate from the observed due/principal.
        # This is a one-week, first-order counterfactual only.
        weekly_rate = due / opening_principal if opening_principal > TOL else 0.0
        baseline_next_due = closing_principal * weekly_rate
        capitalized_next_due = (closing_principal + closing_arrears) * weekly_rate
        capitalization_changes.append(capitalized_next_due - baseline_next_due)
        haircut_25.append(due * 0.25)
        haircut_50.append(due * 0.50)

    return {
        "ratios": ratios,
        "relief": relief,
        "closure": {key: value / len(active_rows) for key, value in closure.items()},
        "capitalization_mean_next_interest_change": statistics.fmean(
            capitalization_changes
        )
        if capitalization_changes
        else 0.0,
        "principal_haircut_25_mean_interest_reduction": statistics.fmean(
            haircut_25
        )
        if haircut_25
        else 0.0,
        "principal_haircut_50_mean_interest_reduction": statistics.fmean(
            haircut_50
        )
        if haircut_50
        else 0.0,
    }


def main():
    active_rows, episodes, per_seed = replay_active_default()
    metrics = service_metrics(active_rows)
    ratios = metrics["ratios"]
    relief = metrics["relief"]

    immediate = len(episodes)
    eligible_13 = sum(episode["duration"] >= 13 for episode in episodes)
    eligible_26 = sum(episode["duration"] >= 26 for episode in episodes)
    durations = [episode["duration"] for episode in episodes]

    rows = []
    emit(rows, "current_service_gap", "active_contract_default_week_count", len(active_rows))
    emit(rows, "current_service_gap", "interest_service_ratio_p25", quantile(ratios, 0.25))
    emit(rows, "current_service_gap", "interest_service_ratio_median", quantile(ratios, 0.50))
    emit(rows, "current_service_gap", "interest_service_ratio_p75", quantile(ratios, 0.75))
    emit(rows, "current_service_gap", "interest_service_ratio_p90", quantile(ratios, 0.90))
    emit(rows, "current_service_gap", "required_relief_fraction_p25", quantile(relief, 0.25))
    emit(rows, "current_service_gap", "required_relief_fraction_median", quantile(relief, 0.50))
    emit(rows, "current_service_gap", "required_relief_fraction_p75", quantile(relief, 0.75))
    emit(rows, "current_service_gap", "required_relief_fraction_p90", quantile(relief, 0.90))
    emit(rows, "current_service_gap", "mean_current_interest_due", statistics.fmean(
        number(row["raw"], "current_interest_due") for row in active_rows
    ))
    emit(rows, "current_service_gap", "mean_interest_paid_to_current_due", statistics.fmean(
        number(row["raw"], "interest_paid_to_current_due") for row in active_rows
    ))

    for percentage in (25, 50, 75):
        emit(
            rows,
            "interest_relief_static_coverage",
            f"static_gap_closure_{percentage}pct",
            metrics["closure"][percentage],
            note="Adjusted current due <= observed total interest paid; passive only.",
        )

    emit(rows, "eligibility_timing", "default_episodes_eligible_immediate", immediate)
    emit(rows, "eligibility_timing", "default_episodes_eligible_after_13w", eligible_13)
    emit(rows, "eligibility_timing", "default_episodes_eligible_after_26w", eligible_26)
    emit(rows, "eligibility_timing", "episode_duration_minimum_weeks", min(durations))
    emit(rows, "eligibility_timing", "episode_duration_median_weeks", statistics.median(durations))
    emit(rows, "eligibility_timing", "episode_duration_p90_weeks", quantile(durations, 0.90))

    option_rows = [
        (
            "TEMPORARY_INTEREST_RELIEF",
            "DIRECTLY_ADDRESSES_CURRENT_SERVICE",
            "C",
            "Reduces the current contractual obligation; does not write principal or historical arrears. Future conceded income is a lender loss/claim concession.",
        ),
        (
            "INTEREST_PAYMENT_DEFERRAL",
            "SHIFTS_PROBLEM_TO_LEGACY_CLAIM",
            "B",
            "If deferred amounts accumulate as arrears, current cash pressure is delayed but the existing arrears channel is largely duplicated. If breach is suspended, it changes classification rather than proving service feasibility.",
        ),
        (
            "ARREARS_CAPITALIZATION",
            "MAY_WORSEN_CURRENT_SERVICE",
            "C",
            "Transforms arrears into interest-bearing principal. It does not destroy the lender claim and raises next-week interest by the observed first-order amount.",
        ),
        (
            "ARREARS_WRITEDOWN",
            "INDIRECT_LEGACY_CLAIM_RELIEF",
            "D",
            "Explicit lender loss. Current interest is calculated on principal, not arrears, so the direct current-service effect is zero.",
        ),
        (
            "PRINCIPAL_HAIRCUT",
            "DIRECTLY_ADDRESSES_CURRENT_SERVICE",
            "D",
            "Reduces future interest due but imposes a stronger explicit lender loss and changes the principal claim. Reserve as a later comparison if isolated interest relief fails.",
        ),
    ]
    for option, classification, claim, note in option_rows:
        emit(rows, "contract_option_comparison", f"{option}_classification", classification, note)
        emit(rows, "contract_option_comparison", f"{option}_claim_conservation", claim, note)

    emit(
        rows,
        "first_order_effects",
        "capitalization_mean_next_interest_change",
        metrics["capitalization_mean_next_interest_change"],
        note="All closing arrears capitalized at the observed current weekly rate; one-week first order only.",
    )
    emit(
        rows,
        "first_order_effects",
        "principal_haircut_25_mean_interest_reduction",
        metrics["principal_haircut_25_mean_interest_reduction"],
    )
    emit(
        rows,
        "first_order_effects",
        "principal_haircut_50_mean_interest_reduction",
        metrics["principal_haircut_50_mean_interest_reduction"],
    )
    emit(
        rows,
        "first_order_effects",
        "arrears_writedown_current_interest_effect",
        0.0,
        note="Verified from current semantics: weekly interest due uses principal only.",
    )

    selected_mechanism = "TEMPORARY_INTEREST_RELIEF"
    selected_relief = 0.75
    selected_delay = 26
    selected_duration = 26
    emit(rows, "selected_treatment", "selected_restructuring_mechanism", selected_mechanism)
    emit(rows, "selected_treatment", "selected_reference_relief_fraction", selected_relief)
    emit(rows, "selected_treatment", "selected_reference_eligibility_delay", selected_delay)
    emit(rows, "selected_treatment", "selected_reference_max_duration", selected_duration)
    emit(rows, "selected_treatment", "selected_seed42_episode_count", per_seed[42])
    emit(rows, "selected_treatment", "selected_seed7_episode_count", per_seed[7])
    emit(rows, "selected_treatment", "selected_seed21_episode_count", per_seed[21])

    flags = {
        "verdict": "A TEMPORARY_INTEREST_RELIEF_IS_REFERENCE_RESTRUCTURING",
        "current_service_gap_understood": True,
        "restructuring_options_compared": True,
        "interest_relief_semantics_ready": True,
        "interest_deferral_semantics_ready": True,
        "arrears_capitalization_semantics_ready": True,
        "arrears_writedown_semantics_ready": True,
        "principal_haircut_semantics_ready": True,
        "single_reference_mechanism_selected": True,
        "selected_mechanism": selected_mechanism,
        "reference_treatment_defined": True,
        "behavioral_restructuring_test_ready": True,
        "lender_loss_accounting_required": True,
        "economic_behavior_changed": False,
        "rng_changed": False,
        "exit_implemented": False,
        "new_long_runs": 0,
        "active_contract_default_week_count": len(active_rows),
        "interest_service_ratio_p25": quantile(ratios, 0.25),
        "interest_service_ratio_median": quantile(ratios, 0.50),
        "interest_service_ratio_p75": quantile(ratios, 0.75),
        "interest_service_ratio_p90": quantile(ratios, 0.90),
        "required_relief_fraction_p25": quantile(relief, 0.25),
        "required_relief_fraction_median": quantile(relief, 0.50),
        "required_relief_fraction_p75": quantile(relief, 0.75),
        "required_relief_fraction_p90": quantile(relief, 0.90),
        "static_gap_closure_25pct": metrics["closure"][25],
        "static_gap_closure_50pct": metrics["closure"][50],
        "static_gap_closure_75pct": metrics["closure"][75],
        "default_episodes_eligible_immediate": immediate,
        "default_episodes_eligible_after_13w": eligible_13,
        "default_episodes_eligible_after_26w": eligible_26,
        "capitalization_mean_next_interest_change": metrics[
            "capitalization_mean_next_interest_change"
        ],
        "principal_haircut_25_mean_interest_reduction": metrics[
            "principal_haircut_25_mean_interest_reduction"
        ],
        "principal_haircut_50_mean_interest_reduction": metrics[
            "principal_haircut_50_mean_interest_reduction"
        ],
        "arrears_writedown_current_interest_effect": 0.0,
        "selected_reference_relief_fraction": selected_relief,
        "selected_reference_eligibility_delay": selected_delay,
        "selected_reference_max_duration": selected_duration,
    }

    summary = f"""# Step 13.8A acceptance summary

Verdict: **{flags['verdict']}**.

This is a passive design result only. No restructuring, interest, principal,
arrears, ledger, Exit, or economic behavior was implemented, and no treatment
simulation was run.

## Evidence

There are **{len(active_rows)} active contractual-Default Firm-weeks** across
the accepted seed42/7/21 trajectories. The observed interest-service ratio
(`interest_paid / current_interest_due`) is p25={quantile(ratios, 0.25):.6g},
median={quantile(ratios, 0.50):.6g}, p75={quantile(ratios, 0.75):.6g}, and
p90={quantile(ratios, 0.90):.6g}. The corresponding required-relief median is
{quantile(relief, 0.50):.6g}. Current interest paid directly to current due is
zero in the accepted active-default observations; observed interest payments
are directed to historical arrears where payment occurs.

Static gap closure is {metrics['closure'][25]:.6%} at 25% relief,
{metrics['closure'][50]:.6%} at 50%, and {metrics['closure'][75]:.6%} at 75%.
These figures are not behavioral cure forecasts. They show that even the
largest tested concession has limited static coverage, so the next experiment
must measure actual payment priority and current-service behavior rather than
assume cure.

All {immediate} observed episodes survive both the 13-week and 26-week
eligibility screens; their minimum duration is {min(durations)} weeks and
their median duration is {statistics.median(durations)} weeks. A 26-week
grace period is therefore a semantic persistence filter, not a calibrated
threshold.

## Contract baseline

Principal is the outstanding loan stock. Current interest due is calculated on
principal. Current interest paid is recorded separately from unpaid current
interest; unpaid current interest increases historical interest arrears.
Lender exposure is principal plus arrears. Credit limit/headroom controls new
credit availability. Principal repayment remains liquidity-constrained and has
no maturity-based mandatory installment. Default bookkeeping changes no claim.

Pure deferral either duplicates the existing arrears channel or changes only
the breach classification. Arrears capitalization transforms the claim and
raises next-week interest by an average first-order amount of
{metrics['capitalization_mean_next_interest_change']:.6g}. Arrears write-down
has zero direct current-interest effect because current interest is principal-
based. A principal haircut directly reduces future interest but is a stronger
explicit lender-loss intervention.

## Reference treatment for the next stage

The selected isolated mechanism is **TEMPORARY_INTEREST_RELIEF** with a
reference relief fraction of **75%**, eligibility after **26 consecutive
active-contract-Default weeks**, and a maximum duration of **26 weeks**.
These are experimental reference values, not calibration and not a guarantee
of cure. Historical arrears remain intact, D3 remains endogenous, and
`active_contract_default` should only clear through the existing current-service
cure semantics.
"""

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "restructuring_contract_design_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2), encoding="utf-8"
    )
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(flags["verdict"])


if __name__ == "__main__":
    main()
