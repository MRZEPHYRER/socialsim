"""Step 13.7A passive post-Default resolution audit.

Only the accepted seed42/7/21 firm diagnostics are replayed.  This script
does not run the model and does not introduce restructuring or Exit behavior.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

from economy.default_bookkeeping import DefaultStateMachine
from step13_5_passive_distress_semantics import build_features, read
from step13_5c_flow_first_distress_redesign import add_flow_features, classify


OUT = ROOT / "test/output/step13_7A_post_default_resolution"
RUNS = {
    42: ROOT / "test/output/pre_step13_5K_household_denominator_recovery_hazard/n5000_targeted/firm_diagnostics.csv",
    7: ROOT / "test/output/step13_5D_runs/seed7/firm_diagnostics.csv",
    21: ROOT / "test/output/step13_5D_runs/seed21/firm_diagnostics.csv",
}
TOL = 1e-9


def number(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def boolean(row, key):
    value = row.get(key, False)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def median(values):
    return statistics.median(values) if values else 0.0


def p90(values):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(0.9 * (len(values) - 1)))]


def mean(values):
    return statistics.fmean(values) if values else 0.0


def ocf(row):
    return (
        number(row, "sales_revenue")
        - number(row, "executed_wage_bill", number(row, "wage_payment"))
        + number(row, "public_sector_cash_inflow")
        - number(row, "public_sector_cash_outflow")
    )


def exposure_utilization(row):
    limit = number(row, "credit_limit")
    opening = number(row, "opening_lender_exposure")
    if limit > TOL and math.isfinite(limit):
        return opening / limit
    return 0.0


def load_seed_rows(seed, path):
    raw = read(path)
    features = build_features(raw)
    add_flow_features(features)
    raw_map = {
        (
            int(float(row["firm_id"])),
            int(float(row.get("global_step", row.get("step", 0)))),
        ): row
        for row in raw
    }
    by_firm = defaultdict(list)
    for feature in features:
        firm_id = feature["firm_id"]
        step = feature["global_step"]
        raw_row = raw_map[(firm_id, step)]
        by_firm[firm_id].append(
            {
                "seed": seed,
                "firm_id": firm_id,
                "global_step": step,
                "d3": classify(feature, "A") == "D3",
                "state": classify(feature, "A"),
                "raw": raw_row,
            }
        )
    for rows in by_firm.values():
        rows.sort(key=lambda row: row["global_step"])
    return by_firm


def replay_episodes(rows_by_firm):
    """Replay the accepted runtime state machine without exporting episodes."""
    episodes = []
    all_rows = []
    for firm_id, rows in rows_by_firm.items():
        machine = DefaultStateMachine()
        current = None
        for item in rows:
            raw = item["raw"]
            item = dict(item)
            item["ocf"] = ocf(raw)
            all_rows.append(item)
            was_active = machine.active_default_episode
            machine.update(item["d3"])
            event = machine.default_event_this_week
            if event:
                current = {
                    "seed": item["seed"],
                    "firm_id": firm_id,
                    "event_step": item["global_step"],
                    "event_row": item,
                    "rows": [],
                    "outcome": None,
                }
                episodes.append(current)

            exposure = was_active or event
            if current is not None and exposure:
                current["rows"].append(item)

            if (
                current is not None
                and was_active
                and not item["d3"]
                and not machine.active_default_episode
            ):
                current["cure_step"] = item["global_step"]
                current["cure_row"] = item
                current["outcome"] = "CURED"
                current = None

        if current is not None:
            current["outcome"] = "ACTIVE_AT_HORIZON"
            current["horizon_step"] = rows[-1]["global_step"]
            current["horizon_row"] = rows[-1]

    episodes.sort(key=lambda episode: (episode["seed"], episode["event_step"], episode["firm_id"]))
    by_firm = defaultdict(list)
    for episode in episodes:
        by_firm[(episode["seed"], episode["firm_id"])].append(episode)
    for key, firm_episodes in by_firm.items():
        for index, episode in enumerate(firm_episodes):
            episode["episode_index"] = index + 1
            episode["redefaulted_later"] = index < len(firm_episodes) - 1
            if episode["outcome"] == "CURED" and episode["redefaulted_later"]:
                episode["outcome"] = "CURED_THEN_REDEFAULTED"
    return episodes, all_rows, by_firm


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


def claim_consistency(all_rows_by_firm):
    max_principal_gap = 0.0
    max_arrears_gap = 0.0
    max_exposure_gap = 0.0
    max_existing_credit_gap = 0.0
    for rows in all_rows_by_firm.values():
        rows = sorted(rows, key=lambda row: row["global_step"])
        previous = None
        for item in rows:
            raw = item["raw"]
            principal = number(raw, "loan_balance")
            arrears = number(raw, "closing_interest_arrears", number(raw, "interest_arrears"))
            exposure = number(raw, "lender_exposure")
            max_exposure_gap = max(max_exposure_gap, abs(exposure - principal - arrears))
            max_existing_credit_gap = max(
                max_existing_credit_gap,
                abs(number(raw, "credit_bridge_gap")),
            )
            arrears_flow = (
                arrears
                - number(raw, "opening_interest_arrears")
                - number(raw, "current_interest_unpaid")
                + number(raw, "interest_paid_to_opening_arrears")
            )
            max_arrears_gap = max(max_arrears_gap, abs(arrears_flow))
            if previous is not None:
                previous_raw = previous["raw"]
                principal_flow = (
                    principal
                    - number(previous_raw, "loan_balance")
                    - number(raw, "loan_issued")
                    + number(raw, "loan_repaid")
                )
                max_principal_gap = max(max_principal_gap, abs(principal_flow))
            previous = item
    return {
        "max_principal_bridge_gap": max_principal_gap,
        "max_arrears_flow_gap": max_arrears_gap,
        "max_lender_exposure_gap": max_exposure_gap,
        "max_existing_credit_bridge_gap": max_existing_credit_gap,
    }


def endpoint_metrics(episodes):
    cured = [episode for episode in episodes if episode["outcome"] != "ACTIVE_AT_HORIZON"]
    active = [episode for episode in episodes if episode["outcome"] == "ACTIVE_AT_HORIZON"]
    first_non_d3 = []
    completed_cure = []
    for episode in cured:
        non_d3 = [row for row in episode["rows"] if not row["d3"]]
        if non_d3:
            first_non_d3.append(non_d3[0]["global_step"] - episode["event_step"])
        completed_cure.append(episode["cure_step"] - episode["event_step"])
    active_duration = [
        episode["horizon_step"] - episode["event_step"] + 1
        for episode in active
    ]
    return first_non_d3, completed_cure, active_duration


def arrears_metrics(cured):
    decrease_weeks = 0
    positive_to_zero = 0
    repayment = 0.0
    positive_at_event = 0
    zero_at_cure = 0
    current_full = 0
    no_new_growth = 0
    positive_at_cure = 0
    zero_and_full = 0
    endpoint_rows = []
    for episode in cured:
        event_raw = episode["event_row"]["raw"]
        cure_raw = episode["cure_row"]["raw"]
        event_arrears = number(event_raw, "closing_interest_arrears", number(event_raw, "interest_arrears"))
        cure_arrears = number(cure_raw, "closing_interest_arrears", number(cure_raw, "interest_arrears"))
        positive_at_event += int(event_arrears > TOL)
        zero_at_cure += int(cure_arrears <= TOL)
        positive_at_cure += int(cure_arrears > TOL)
        fully_serviced = number(cure_raw, "current_interest_unpaid") <= TOL
        current_full += int(fully_serviced)
        no_growth = number(cure_raw, "closing_interest_arrears") - number(cure_raw, "opening_interest_arrears") <= TOL
        no_new_growth += int(no_growth)
        zero_and_full += int(cure_arrears <= TOL and fully_serviced)
        prior = None
        for item in episode["rows"]:
            value = number(item["raw"], "closing_interest_arrears", number(item["raw"], "interest_arrears"))
            if prior is not None:
                if value < prior - TOL:
                    decrease_weeks += 1
                    repayment += prior - value
                if prior > TOL and value <= TOL:
                    positive_to_zero += 1
            prior = value
        endpoint_rows.append((event_arrears, cure_arrears))
    return {
        "arrears_at_event": [row[0] for row in endpoint_rows],
        "arrears_at_cure": [row[1] for row in endpoint_rows],
        "positive_at_event": positive_at_event,
        "zero_at_cure": zero_at_cure,
        "positive_at_cure": positive_at_cure,
        "current_full": current_full,
        "no_new_growth": no_new_growth,
        "zero_and_full": zero_and_full,
        "decrease_weeks": decrease_weeks,
        "positive_to_zero": positive_to_zero,
        "repayment": repayment,
    }


def recovery_metrics(cured):
    fields = (
        "credit_denied",
        "executed_credit",
        "credit_headroom",
        "exposure_utilization",
        "current_interest_due",
        "interest_paid",
        "current_interest_unpaid",
        "arrears_growth",
        "closing_interest_arrears",
        "principal",
        "lender_exposure",
        "sales_revenue",
        "executed_wage_bill",
        "public_sector_cash_inflow",
        "public_sector_cash_outflow",
        "cash",
        "payroll_funding_ratio",
        "funded_productive_capacity",
        "scheduled_productive_capacity",
    )
    values = defaultdict(list)
    credit_during_cure = 0
    headroom_at_first_non_d3 = 0
    credit_at_first_non_d3 = 0
    for episode in cured:
        non_d3 = [row for row in episode["rows"] if not row["d3"]]
        if not non_d3:
            continue
        first = non_d3[0]["raw"]
        headroom_at_first_non_d3 += int(number(first, "credit_headroom") > TOL)
        credit_at_first_non_d3 += int(number(first, "executed_credit") > TOL)
        if any(number(row["raw"], "executed_credit") > TOL for row in non_d3):
            credit_during_cure += 1
        for label, raw in (
            ("event", episode["event_row"]["raw"]),
            ("first_nonD3", first),
            ("cure", episode["cure_row"]["raw"]),
        ):
            for field in fields:
                if field == "exposure_utilization":
                    value = exposure_utilization(raw)
                elif field == "arrears_growth":
                    value = number(raw, "closing_interest_arrears") - number(raw, "opening_interest_arrears")
                elif field == "principal":
                    value = number(raw, "loan_balance")
                else:
                    value = number(raw, field)
                values[f"{label}_{field}"].append(value)
            values[f"{label}_ocf"].append(ocf(raw))
            values[f"{label}_funded_capacity_ratio"].append(
                number(raw, "funded_productive_capacity")
                / number(raw, "scheduled_productive_capacity")
                if number(raw, "scheduled_productive_capacity") > TOL
                else 0.0
            )
    return values, credit_during_cure, headroom_at_first_non_d3, credit_at_first_non_d3


def persistent_metrics(active):
    categories = {
        "current_service_failure": 0,
        "continued_arrears_growth": 0,
        "credit_exhaustion": 0,
        "negative_ocf": 0,
        "payroll_underfunding": 0,
    }
    for episode in active:
        raw = episode["horizon_row"]["raw"]
        categories["current_service_failure"] += int(number(raw, "current_interest_unpaid") > TOL)
        categories["continued_arrears_growth"] += int(
            number(raw, "closing_interest_arrears") - number(raw, "opening_interest_arrears") > TOL
        )
        categories["credit_exhaustion"] += int(
            number(raw, "denied_credit") > TOL or number(raw, "credit_headroom") <= TOL
        )
        categories["negative_ocf"] += int(ocf(raw) < 0)
        categories["payroll_underfunding"] += int(number(raw, "payroll_funding_ratio", 1.0) < 1.0 - TOL)
    return categories


def redefault_metrics(episodes, all_rows_by_firm):
    intervals = []
    positive_throughout = 0
    redefault_events = 0
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[(episode["seed"], episode["firm_id"])].append(episode)
    for key, firm_episodes in grouped.items():
        for before, after in zip(firm_episodes, firm_episodes[1:]):
            redefault_events += 1
            intervals.append(after["event_step"] - before["cure_step"])
            positive = True
            rows = [
                row
                for row in all_rows_by_firm[key]
                if before["cure_step"] <= row["global_step"] <= after["event_step"]
            ]
            for row in rows:
                if number(row["raw"], "closing_interest_arrears") <= TOL:
                    positive = False
            positive_throughout += int(positive)
    return redefault_events, intervals, positive_throughout


def main():
    all_episodes = []
    all_rows = []
    per_seed = {}
    rows_by_seed_firm = {}
    for seed, path in RUNS.items():
        rows_by_firm = load_seed_rows(seed, path)
        rows_by_seed_firm.update({(seed, firm_id): rows for firm_id, rows in rows_by_firm.items()})
        episodes, raw_rows, by_firm = replay_episodes(rows_by_firm)
        all_episodes.extend(episodes)
        all_rows.extend(raw_rows)
        per_seed[seed] = {
            "episodes": episodes,
            "firms": by_firm,
        }

    all_episodes.sort(key=lambda episode: (episode["seed"], episode["event_step"], episode["firm_id"]))
    cured = [episode for episode in all_episodes if episode["outcome"] != "ACTIVE_AT_HORIZON"]
    active = [episode for episode in all_episodes if episode["outcome"] == "ACTIVE_AT_HORIZON"]
    first_non_d3, cure_times, active_times = endpoint_metrics(all_episodes)
    arrears = arrears_metrics(cured)
    recovery, credit_during_cure, headroom_first, credit_first = recovery_metrics(cured)
    persistent = persistent_metrics(active)
    all_by_firm = defaultdict(list)
    for item in all_rows:
        all_by_firm[(item["seed"], item["firm_id"])].append(item)
    consistency = claim_consistency(all_by_firm)
    # Keep the reconstructed episode grouping in memory only; no raw episode
    # table is exported by this audit.
    grouped_episodes = defaultdict(list)
    for episode in all_episodes:
        grouped_episodes[(episode["seed"], episode["firm_id"])].append(episode)
    redefault_events, redefault_intervals, redefault_positive = redefault_metrics(all_episodes, all_by_firm)

    output = []
    emit(output, "episode_outcomes", "default_episode_count", len(all_episodes))
    natural_cure = [episode for episode in all_episodes if episode["outcome"] == "CURED"]
    redefaulted_cure = [episode for episode in all_episodes if episode["outcome"] == "CURED_THEN_REDEFAULTED"]
    emit(output, "episode_outcomes", "natural_cure_episode_count", len(natural_cure))
    emit(output, "episode_outcomes", "active_at_horizon_episode_count", len(active))
    emit(output, "episode_outcomes", "cured_then_redefaulted_episode_count", len(redefaulted_cure))
    emit(output, "episode_outcomes", "cured_episode_count_including_redefaulted", len(cured))
    emit(output, "episode_outcomes", "firms_ever_defaulted", len({(episode["seed"], episode["firm_id"]) for episode in all_episodes}))
    emit(output, "episode_outcomes", "firms_ever_naturally_cured", len({(episode["seed"], episode["firm_id"]) for episode in cured}))
    emit(output, "episode_outcomes", "firms_ever_redefaulted", len({key for key, values in grouped_episodes.items() if len(values) > 1}))
    emit(output, "time_to_resolution", "median_default_to_first_nonD3_weeks", median(first_non_d3))
    emit(output, "time_to_resolution", "median_default_to_completed_cure_weeks", median(cure_times))
    emit(output, "time_to_resolution", "p90_default_to_completed_cure_weeks", p90(cure_times))
    emit(output, "time_to_resolution", "max_default_to_completed_cure_weeks", max(cure_times, default=0))
    emit(output, "time_to_resolution", "median_active_observed_duration_weeks", median(active_times))
    emit(output, "time_to_resolution", "max_active_observed_duration_weeks", max(active_times, default=0))

    emit(output, "legacy_arrears_resolution", "cured_with_positive_arrears_count", arrears["positive_at_cure"])
    emit(output, "legacy_arrears_resolution", "cured_with_zero_arrears_count", arrears["zero_at_cure"])
    emit(output, "current_service_recovery", "cured_with_current_full_service_count", arrears["current_full"])
    emit(output, "current_service_recovery", "cured_with_no_new_arrears_growth_count", arrears["no_new_growth"])
    emit(output, "legacy_arrears_resolution", "cured_with_zero_arrears_but_full_service_count", arrears["zero_and_full"])
    emit(output, "legacy_arrears_resolution", "arrears_decrease_episode_week_count", arrears["decrease_weeks"])
    emit(output, "legacy_arrears_resolution", "positive_arrears_to_zero_transition_count", arrears["positive_to_zero"])
    emit(output, "legacy_arrears_resolution", "total_observed_arrears_repayment", arrears["repayment"])
    emit(output, "legacy_arrears_resolution", "median_arrears_at_default_event", median(arrears["arrears_at_event"]))
    emit(output, "legacy_arrears_resolution", "median_arrears_at_cure", median(arrears["arrears_at_cure"]))

    for label in ("event", "first_nonD3", "cure"):
        for field in (
            "credit_denied",
            "executed_credit",
            "credit_headroom",
            "exposure_utilization",
            "current_interest_due",
            "interest_paid",
            "current_interest_unpaid",
            "arrears_growth",
            "closing_interest_arrears",
            "principal",
            "lender_exposure",
            "ocf",
            "cash",
            "payroll_funding_ratio",
            "funded_capacity_ratio",
        ):
            values = recovery.get(f"{label}_{field}", [])
            emit(output, "credit_operating_recovery", f"{label}_{field}_mean", mean(values))
    emit(output, "credit_operating_recovery", "credit_execution_during_cure_count", credit_during_cure)
    emit(output, "credit_operating_recovery", "renewed_headroom_at_first_nonD3_count", headroom_first)
    emit(output, "credit_operating_recovery", "renewed_credit_execution_at_first_nonD3_count", credit_first)

    for metric, value in persistent.items():
        emit(output, "persistent_default_unresolved_problem", metric + "_at_horizon_count", value)
    emit(output, "redefault", "redefault_event_count", redefault_events)
    emit(output, "redefault", "median_cure_to_redefault_weeks", median(redefault_intervals))
    emit(output, "redefault", "redefault_with_positive_arrears_throughout_count", redefault_positive)

    for seed in RUNS:
        seed_episodes = per_seed[seed]["episodes"]
        seed_cured = [episode for episode in seed_episodes if episode["outcome"] != "ACTIVE_AT_HORIZON"]
        seed_active = [episode for episode in seed_episodes if episode["outcome"] == "ACTIVE_AT_HORIZON"]
        emit(output, "episode_outcomes_by_seed", "episode_count", len(seed_episodes), seed)
        emit(output, "episode_outcomes_by_seed", "natural_cure_count", sum(episode["outcome"] == "CURED" for episode in seed_episodes), seed)
        emit(output, "episode_outcomes_by_seed", "cured_episode_count_including_redefaulted", len(seed_cured), seed)
        emit(output, "episode_outcomes_by_seed", "active_at_horizon_count", len(seed_active), seed)

    restructuring_need = "R3_BOTH" if active and arrears["positive_at_cure"] else "R1_PERSISTENT_CURRENT_SERVICE_FAILURE" if active else "R2_LEGACY_CLAIM_OVERHANG" if arrears["positive_at_cure"] else "R0_NO_RESTRUCTURING_NEED"
    emit(output, "restructuring_need", "classification", restructuring_need, note="Need category only; no eligibility threshold or restructuring terms selected")
    emit(output, "restructuring_need", "persistent_episode_count", len(active))
    emit(output, "restructuring_need", "legacy_overhang_cured_episode_count", arrears["positive_at_cure"])
    emit(output, "claim_consistency", "max_principal_bridge_gap", consistency["max_principal_bridge_gap"])
    emit(output, "claim_consistency", "max_arrears_flow_gap", consistency["max_arrears_flow_gap"])
    emit(output, "claim_consistency", "max_lender_exposure_gap", consistency["max_lender_exposure_gap"])
    emit(output, "claim_consistency", "max_existing_credit_bridge_gap", consistency["max_existing_credit_bridge_gap"])

    flags = {
        "verdict": "G_ACCOUNTING_OR_SEMANTIC_INCONSISTENCY_FOUND" if max(consistency.values()) > 1e-6 else (
            "D_BOTH_PERSISTENT_DEFAULT_AND_LEGACY_OVERHANG_CONFIRMED" if active and arrears["positive_at_cure"] else (
                "C_PERSISTENT_DEFAULT_RESTRUCTURING_NEED_CONFIRMED" if active else (
                    "B_LEGACY_ARREARS_RESTRUCTURING_NEED_CONFIRMED" if arrears["positive_at_cure"] else "A_NATURAL_DEFAULT_RESOLUTION_SUFFICIENT"
                )
            )
        ),
        "default_episode_count": len(all_episodes),
        "natural_cure_episode_count": len(natural_cure),
        "active_at_horizon_episode_count": len(active),
        "firms_ever_defaulted": len({(episode["seed"], episode["firm_id"]) for episode in all_episodes}),
        "firms_ever_naturally_cured": len({(episode["seed"], episode["firm_id"]) for episode in cured}),
        "firms_ever_redefaulted": len({key for key, values in grouped_episodes.items() if len(values) > 1}),
        "median_default_to_first_nonD3_weeks": median(first_non_d3),
        "median_default_to_completed_cure_weeks": median(cure_times),
        "p90_default_to_completed_cure_weeks": p90(cure_times),
        "cured_with_positive_arrears_count": arrears["positive_at_cure"],
        "cured_with_zero_arrears_count": arrears["zero_at_cure"],
        "cured_with_current_full_service_count": arrears["current_full"],
        "cured_with_no_new_arrears_growth_count": arrears["no_new_growth"],
        "arrears_decrease_episode_week_count": arrears["decrease_weeks"],
        "positive_arrears_to_zero_transition_count": arrears["positive_to_zero"],
        "total_observed_arrears_repayment": arrears["repayment"],
        "credit_execution_during_cure_count": credit_during_cure,
        "redefault_event_count": redefault_events,
        "median_cure_to_redefault_weeks": median(redefault_intervals),
        "current_service_recovered_after_default": arrears["current_full"] > 0,
        "accumulated_arrears_materially_declined": arrears["decrease_weeks"] > 0,
        "accumulated_arrears_returned_to_zero": arrears["positive_to_zero"] > 0,
        "operational_cure_with_legacy_arrears": arrears["positive_at_cure"] > 0,
        "credit_access_recovered_during_cure": credit_during_cure > 0 or credit_first > 0,
        "natural_cure_and_later_redefault": redefault_events > 0,
        "restructuring_need_category": restructuring_need,
        "exit_implemented": False,
        "restructuring_implemented": False,
        "economic_behavior_changed": False,
        "rng_changed": False,
        "max_claim_consistency_gap": max(consistency.values()),
        "accounting_consistency_pass": max(consistency.values()) <= 1e-6,
        "seed42_episode_count": len(per_seed[42]["episodes"]),
        "seed42_natural_cure_count": sum(e["outcome"] == "CURED" for e in per_seed[42]["episodes"]),
        "seed42_active_at_horizon_count": len([e for e in per_seed[42]["episodes"] if e["outcome"] == "ACTIVE_AT_HORIZON"]),
        "seed7_episode_count": len(per_seed[7]["episodes"]),
        "seed7_natural_cure_count": sum(e["outcome"] == "CURED" for e in per_seed[7]["episodes"]),
        "seed7_active_at_horizon_count": len([e for e in per_seed[7]["episodes"] if e["outcome"] == "ACTIVE_AT_HORIZON"]),
        "seed21_episode_count": len(per_seed[21]["episodes"]),
        "seed21_natural_cure_count": sum(e["outcome"] == "CURED" for e in per_seed[21]["episodes"]),
        "seed21_active_at_horizon_count": len([e for e in per_seed[21]["episodes"] if e["outcome"] == "ACTIVE_AT_HORIZON"]),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "post_default_resolution_core_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "seed", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(output)
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    (OUT / "acceptance_summary.md").write_text(
        f"""# Step 13.7A acceptance summary

Verdict: **{flags['verdict']}**.

The accepted seed42/7/21 weekly Firm histories were replayed without running
new simulations.  The audit reconstructed {len(all_episodes)} canonical
Default episodes: {len(natural_cure)} were natural cures without a later
re-Default, {len(redefaulted_cure)} cured and later re-Defaulted, and
{len(active)} remained active at the observed horizon.

Current-service recovery and legacy-claim resolution are reported separately.
At cure, {arrears['current_full']} episodes had fully serviced current interest,
while {arrears['positive_at_cure']} retained positive historical arrears.
Observed arrears decreases totaled {arrears['repayment']:.17g}, across
{arrears['decrease_weeks']} episode-weeks; positive-arrears-to-zero transitions
were {arrears['positive_to_zero']}.

The restructuring classification is **{restructuring_need}**.  This is a need
category only.  No restructuring terms, debt write-off, interest change, credit
change, Exit or economic behavior was implemented.

Claim consistency maximum gap: {max(consistency.values()):.17g}.
""",
        encoding="utf-8",
    )
    print(json.dumps(flags, indent=2))


if __name__ == "__main__":
    main()
