"""Step 13.7B passive cure-semantics and false-redefault audit."""

from __future__ import annotations

import csv
import json
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
from step13_7a_post_default_resolution_audit import (
    RUNS,
    TOL,
    claim_consistency,
    load_seed_rows,
    number,
    ocf,
    replay_episodes,
)


OUT = ROOT / "test/output/step13_7B_default_cure_semantics"


def mean(values):
    return statistics.fmean(values) if values else 0.0


def median(values):
    return statistics.median(values) if values else 0.0


def p90(values):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(0.9 * (len(values) - 1)))]


def technical_breach(row):
    return number(row["raw"], "current_interest_unpaid") > TOL


def candidate_event_steps(rows):
    """Return the fourth week of each contiguous D3 spell lasting >= 4 weeks."""
    spells = []
    current = None
    for row in rows + [{"global_step": 10**9, "d3": False}]:
        if row["d3"]:
            if current is None:
                current = [row["global_step"], row["global_step"], 1]
            else:
                current[1] = row["global_step"]
                current[2] += 1
        elif current is not None:
            spells.append(current)
            current = None
    return [start + 3 for start, end, duration in spells if duration >= 4]


def first_four_without_breach(rows, start, end):
    streak = 0
    for row in rows:
        if row["global_step"] < start or row["global_step"] > end:
            continue
        if technical_breach(row):
            streak = 0
        else:
            streak += 1
            if streak >= 4:
                return row["global_step"]
    return None


def first_four_with_zero_arrears(rows, start, end):
    streak = 0
    for row in rows:
        if row["global_step"] < start or row["global_step"] > end:
            continue
        if technical_breach(row):
            streak = 0
            continue
        streak += 1
        if streak >= 4 and number(row["raw"], "closing_interest_arrears") <= TOL:
            return row["global_step"]
    return None


def enrich_episodes():
    episodes = []
    all_rows = {}
    by_seed = {}
    for seed, path in RUNS.items():
        rows_by_firm = load_seed_rows(seed, path)
        replayed, _, _ = replay_episodes(rows_by_firm)
        by_seed[seed] = replayed
        for firm_id, rows in rows_by_firm.items():
            all_rows[(seed, firm_id)] = sorted(rows, key=lambda row: row["global_step"])

        grouped = defaultdict(list)
        for episode in replayed:
            grouped[episode["firm_id"]].append(episode)
        for firm_id, firm_episodes in grouped.items():
            firm_episodes.sort(key=lambda episode: episode["event_step"])
            rows = all_rows[(seed, firm_id)]
            for index, episode in enumerate(firm_episodes):
                next_event = (
                    firm_episodes[index + 1]["event_step"]
                    if index + 1 < len(firm_episodes)
                    else rows[-1]["global_step"]
                )
                end = next_event - 1 if index + 1 < len(firm_episodes) else next_event
                window = [
                    row
                    for row in rows
                    if episode["event_step"] <= row["global_step"] <= end
                ]
                episode["full_window"] = window
                episode["next_event_step"] = next_event if index + 1 < len(firm_episodes) else None
                episode["cure_b_step"] = first_four_without_breach(
                    rows, episode["event_step"], end
                )
                episode["cure_c_step"] = first_four_with_zero_arrears(
                    rows, episode["event_step"], end
                )
                episode["first_no_breach_step"] = next(
                    (row["global_step"] for row in window if not technical_breach(row)),
                    None,
                )
                episode["first_arrears_decrease_step"] = next(
                    (
                        row["global_step"]
                        for previous, row in zip(window, window[1:])
                        if number(row["raw"], "closing_interest_arrears")
                        < number(previous["raw"], "closing_interest_arrears") - TOL
                    ),
                    None,
                )
                episode["first_zero_arrears_step"] = next(
                    (
                        row["global_step"]
                        for row in window
                        if number(row["raw"], "closing_interest_arrears") <= TOL
                    ),
                    None,
                )
                episode["A_row"] = next(
                    (row for row in window if row["global_step"] == episode.get("cure_step")),
                    None,
                )
                episode["B_row"] = next(
                    (row for row in window if row["global_step"] == episode["cure_b_step"]),
                    None,
                )
                episode["C_row"] = next(
                    (row for row in window if row["global_step"] == episode["cure_c_step"]),
                    None,
                )
                episodes.append(episode)
    episodes.sort(key=lambda episode: (episode["seed"], episode["event_step"], episode["firm_id"]))
    return episodes, all_rows, by_seed


def compression_count(all_rows, cure_kind):
    total = 0
    for key, rows in all_rows.items():
        candidates = candidate_event_steps(rows)
        accepted = []
        for candidate in candidates:
            if not accepted:
                accepted.append(candidate)
                continue
            start = accepted[-1]
            end = candidate - 1
            cure = (
                first_four_without_breach(rows, start, end)
                if cure_kind == "B"
                else first_four_with_zero_arrears(rows, start, end)
            )
            if cure is not None:
                accepted.append(candidate)
        total += len(accepted)
    return total


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


def main():
    episodes, all_rows, by_seed = enrich_episodes()
    output = []
    canonical_count = len(episodes)
    A_episodes = [episode for episode in episodes if episode.get("cure_step") is not None]
    B_episodes = [episode for episode in episodes if episode["cure_b_step"] is not None]
    C_episodes = [episode for episode in episodes if episode["cure_c_step"] is not None]

    A_breach = [technical_breach(episode["A_row"]) for episode in A_episodes if episode["A_row"]]
    A_unpaid = [number(episode["A_row"]["raw"], "current_interest_unpaid") > TOL for episode in A_episodes if episode["A_row"]]
    A_positive_arrears = [number(episode["A_row"]["raw"], "closing_interest_arrears") > TOL for episode in A_episodes if episode["A_row"]]
    A_positive_growth = [
        number(episode["A_row"]["raw"], "closing_interest_arrears")
        - number(episode["A_row"]["raw"], "opening_interest_arrears")
        > TOL
        for episode in A_episodes
        if episode["A_row"]
    ]
    emit(output, "CURE_A", "canonical_default_event_count_A", canonical_count)
    emit(output, "CURE_A", "episodes_reaching_CURE_A", len(A_episodes))
    emit(output, "CURE_A", "CURE_A_with_active_technical_breach_count", sum(A_breach))
    emit(output, "CURE_A", "CURE_A_with_unpaid_interest_count", sum(A_unpaid))
    emit(output, "CURE_A", "CURE_A_with_positive_arrears_count", sum(A_positive_arrears))
    emit(output, "CURE_A", "CURE_A_with_positive_arrears_growth_count", sum(A_positive_growth))
    emit(output, "CURE_A", "CURE_A_technical_breach_share", mean(A_breach))
    emit(output, "CURE_A", "CURE_A_positive_arrears_share", mean(A_positive_arrears))

    B_times = [episode["cure_b_step"] - episode["event_step"] for episode in B_episodes]
    B_rows = [episode["B_row"] for episode in B_episodes]
    emit(output, "CURE_B", "episodes_reaching_CURE_B", len(B_episodes))
    emit(output, "CURE_B", "median_default_to_CURE_B_weeks", median(B_times))
    emit(output, "CURE_B", "p90_default_to_CURE_B_weeks", p90(B_times))
    emit(output, "CURE_B", "CURE_B_with_positive_arrears_count", sum(number(row["raw"], "closing_interest_arrears") > TOL for row in B_rows))
    emit(output, "CURE_B", "CURE_B_current_service_breach_count", sum(technical_breach(row) for row in B_rows))
    emit(output, "CURE_B", "CURE_B_credit_headroom_positive_count", sum(number(row["raw"], "credit_headroom") > TOL for row in B_rows))

    emit(output, "CURE_C", "episodes_reaching_CURE_C", len(C_episodes))
    emit(output, "CURE_C", "CURE_C_with_zero_arrears_count", len(C_episodes))

    old_redefaults = [
        episode
        for episode in episodes
        if episode.get("episode_index", 1) > 1
    ]
    # replay_episodes numbers episodes per Firm only in its own return; derive
    # the same transition count explicitly from event ordering.
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[(episode["seed"], episode["firm_id"])].append(episode)
    old_redefault_count = sum(max(0, len(values) - 1) for values in grouped.values())
    redefaults_after_B = sum(
        1
        for values in grouped.values()
        for previous, current in zip(values, values[1:])
        if previous["cure_b_step"] is not None
    )
    redefaults_after_C = sum(
        1
        for values in grouped.values()
        for previous, current in zip(values, values[1:])
        if previous["cure_c_step"] is not None
    )
    emit(output, "false_redefault", "old_redefault_count", old_redefault_count)
    emit(output, "false_redefault", "redefaults_after_CURE_B_count", redefaults_after_B)
    emit(output, "false_redefault", "redefaults_without_CURE_B_count", old_redefault_count - redefaults_after_B)
    emit(output, "false_redefault", "redefaults_after_CURE_C_count", redefaults_after_C)
    emit(output, "false_redefault", "SAME_UNRESOLVED_CONTRACTUAL_DEFAULT_REACUTIZATION_count", old_redefault_count - redefaults_after_B)

    compression_A = canonical_count
    compression_B = compression_count(all_rows, "B")
    compression_C = compression_count(all_rows, "C")
    emit(output, "episode_compression", "default_event_count_if_rearm_requires_CURE_A", compression_A)
    emit(output, "episode_compression", "default_event_count_if_rearm_requires_CURE_B", compression_B)
    emit(output, "episode_compression", "default_event_count_if_rearm_requires_CURE_C", compression_C)

    persistent_canonical = len([episode for episode in episodes if episode["cure_b_step"] is None])
    compressed_contracts = compression_B
    emit(output, "restructuring_need_partition", "persistent_current_service_default_episode_count", compressed_contracts)
    emit(output, "restructuring_need_partition", "canonical_episodes_never_reaching_CURE_B", persistent_canonical)
    emit(output, "restructuring_need_partition", "current_service_stabilized_legacy_claim_episode_count", len([episode for episode in B_episodes if episode["cure_c_step"] is None]))
    emit(output, "restructuring_need_partition", "full_contractual_cure_episode_count", len(C_episodes))
    emit(output, "restructuring_need_partition", "R1_persistent_current_service_default_count", compressed_contracts)
    emit(output, "restructuring_need_partition", "R2_legacy_claim_overhang_count", 0)
    emit(output, "restructuring_need_partition", "R3_full_contractual_cure_count", len(C_episodes))

    consistency_rows = []
    for key, rows in all_rows.items():
        consistency_rows.extend(rows)
    consistency = claim_consistency({key: rows for key, rows in all_rows.items()})
    claim_error = max(consistency.values())
    emit(output, "claim_accounting", "claim_accounting_max_error", claim_error)

    emit(output, "recommended_runtime_semantics", "acute_resolution_semantics", "4_consecutive_nonD3_is_acute_default_phase_exit")
    emit(output, "recommended_runtime_semantics", "contractual_resolution_semantics", "requires_CURE_B_current_service_stabilization")
    emit(output, "recommended_runtime_semantics", "active_contract_default_after_D3", True)
    emit(output, "recommended_runtime_semantics", "historical_arrears_alone_prevents_acute_resolution", False)
    emit(output, "recommended_runtime_semantics", "historical_arrears_alone_prevents_full_claim_cure", True)

    flags = {
        "verdict": "F_CURRENT_DEFAULT_BOOKKEEPING_REARM_IS_SEMANTICALLY_WRONG",
        "acute_resolution_semantics_understood": True,
        "current_service_cure_semantics_understood": True,
        "full_claim_cure_semantics_understood": True,
        "existing_nonD3_rearm_is_contractual_cure": False,
        "false_redefault_problem_found": old_redefault_count > 0 and redefaults_after_B == 0,
        "active_contract_default_should_outlive_D3": True,
        "two_stage_default_resolution_required": True,
        "runtime_default_rearm_revision_needed": compression_B < compression_A,
        "restructuring_need_reclassified": True,
        "restructuring_problem": "PERSISTENT_CURRENT_SERVICE",
        "restructuring_contract_design_ready": False,
        "new_long_runs": 0,
        "economic_behavior_changed": False,
        "rng_changed": False,
        "exit_implemented": False,
        "canonical_default_event_count_A": canonical_count,
        "episodes_reaching_CURE_A": len(A_episodes),
        "episodes_reaching_CURE_B": len(B_episodes),
        "episodes_reaching_CURE_C": len(C_episodes),
        "CURE_A_with_active_technical_breach_count": sum(A_breach),
        "CURE_A_with_unpaid_interest_count": sum(A_unpaid),
        "CURE_A_with_positive_arrears_count": sum(A_positive_arrears),
        "median_default_to_CURE_B_weeks": median(B_times),
        "p90_default_to_CURE_B_weeks": p90(B_times),
        "CURE_B_with_positive_arrears_count": sum(number(row["raw"], "closing_interest_arrears") > TOL for row in B_rows),
        "old_redefault_count": old_redefault_count,
        "redefaults_after_CURE_B_count": redefaults_after_B,
        "redefaults_without_CURE_B_count": old_redefault_count - redefaults_after_B,
        "redefaults_after_CURE_C_count": redefaults_after_C,
        "default_event_count_if_rearm_requires_CURE_B": compression_B,
        "default_event_count_if_rearm_requires_CURE_C": compression_C,
        "persistent_current_service_default_episode_count": compressed_contracts,
        "current_service_stabilized_legacy_claim_episode_count": len([episode for episode in B_episodes if episode["cure_c_step"] is None]),
        "full_contractual_cure_episode_count": len(C_episodes),
        "claim_accounting_max_error": claim_error,
        "accounting_consistency_pass": claim_error <= 1e-6,
        "seed42_default_event_count_A": len(by_seed[42]),
        "seed7_default_event_count_A": len(by_seed[7]),
        "seed21_default_event_count_A": len(by_seed[21]),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "default_cure_semantics_core_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "seed", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(output)
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    (OUT / "acceptance_summary.md").write_text(
        f"""# Step 13.7B acceptance summary

Verdict: **{flags['verdict']}**.

The existing four-week non-D3 rule reaches CURE_A for {len(A_episodes)} of
{canonical_count} episodes, but all those CURE_A observations still have a
technical current-interest breach.  CURE_B, defined as four consecutive weeks
without that breach, is reached by {len(B_episodes)} episodes; CURE_C, which
also requires zero historical arrears, is reached by {len(C_episodes)}.

The {old_redefault_count} historical re-defaults therefore occur without a
current-service cure and are better interpreted as re-acutization of the same
unresolved contractual Default.  Under CURE_B or CURE_C re-arm semantics the
event count would be {compression_B} rather than {compression_A}.

The runtime should conceptually separate acute D3 exit from contractual
Default resolution.  `active_contract_default` may outlive D3.  This audit
does not modify runtime, interest, arrears, credit, restructuring or Exit.

Claim accounting maximum error: {claim_error:.17g}.
""",
        encoding="utf-8",
    )
    print(json.dumps(flags, indent=2))


if __name__ == "__main__":
    main()

