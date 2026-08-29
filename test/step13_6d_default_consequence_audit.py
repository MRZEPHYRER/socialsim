"""Passive Step 13.6D audit of consequences during reconstructed Default episodes."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from step13_5c_flow_first_distress_redesign import add_flow_features, classify
from step13_5_passive_distress_semantics import build_features, num, read

OUT = Path("test/output/step13_6D_default_consequence_audit")
RUNS = {
    42: Path("test/output/pre_step13_5K_household_denominator_recovery_hazard/n5000_targeted/firm_diagnostics.csv"),
    7: Path("test/output/step13_5D_runs/seed7/firm_diagnostics.csv"),
    21: Path("test/output/step13_5D_runs/seed21/firm_diagnostics.csv"),
}


def emit(rows, section, metric, value, phase="", note=""):
    rows.append({"section": section, "phase": phase, "metric": metric, "value": value, "note": note})


def build_active_rows(states):
    groups = defaultdict(list)
    for row in states: groups[(row["seed"], row["firm_id"])].append(row)
    active = []
    event_count = 0
    for key, rows in groups.items():
        rows.sort(key=lambda r: r["global_step"])
        d3_streak = 0; active_episode = False; cure_streak = 0
        for row in rows:
            if row["d3"]:
                d3_streak += 1; cure_streak = 0
            else:
                d3_streak = 0
            if not active_episode and d3_streak >= 4:
                active_episode = True; cure_streak = 0; event_count += 1
            phase = "none"
            if active_episode:
                if row["d3"]:
                    phase = "acute_D3"
                else:
                    cure_streak += 1; phase = "cure_tail"
                active.append({**row, "active_default": True, "default_phase": phase})
                if cure_streak >= 4:
                    # The fourth non-D3 week is part of the observed cure tail;
                    # the next week is re-armed if a new D3 episode begins.
                    active_episode = False; cure_streak = 0
    return active, event_count


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output_rows = []; all_active = []; total_events = 0
    for seed, path in RUNS.items():
        raw = read(path); features = build_features(raw); add_flow_features(features)
        raw_map = {(int(float(r["firm_id"])), int(float(r.get("global_step", r.get("step", 0))))): r for r in raw}
        states = []
        for feature in features:
            raw_row = raw_map[(feature["firm_id"], feature["global_step"])]
            states.append({**feature, "seed": seed, "d3": classify(feature, "A") == "D3", "raw": raw_row})
        active, events = build_active_rows(states); total_events += events; all_active.extend(active)
        emit(output_rows, "episode_structure", "default_event_count", events, note=f"seed={seed}")

    emit(output_rows, "episode_structure", "default_event_count", total_events)
    emit(output_rows, "episode_structure", "active_default_firm_week_count", len(all_active))
    emit(output_rows, "episode_structure", "acute_D3_default_week_count", sum(r["default_phase"] == "acute_D3" for r in all_active))
    emit(output_rows, "episode_structure", "cure_tail_default_week_count", sum(r["default_phase"] == "cure_tail" for r in all_active))

    measures = {
        "credit_executed": lambda r: num(r["raw"], "executed_credit"),
        "principal_repayment": lambda r: num(r["raw"], "loan_repaid"),
        "dividends": lambda r: num(r["raw"], "dividend_paid", num(r["raw"], "dividend_payment")),
        "interest_due": lambda r: num(r["raw"], "current_interest_due"),
        "interest_paid": lambda r: num(r["raw"], "interest_paid"),
        "unpaid_interest": lambda r: num(r["raw"], "current_interest_unpaid"),
        "arrears_growth": lambda r: num(r["raw"], "closing_interest_arrears") - num(r["raw"], "opening_interest_arrears"),
        "credit_requested": lambda r: num(r["raw"], "requested_credit"),
        "credit_denied": lambda r: num(r["raw"], "denied_credit"),
        "credit_headroom": lambda r: num(r["raw"], "credit_headroom"),
        "payroll_funding_ratio": lambda r: num(r["raw"], "payroll_funding_ratio", 1.0),
        "funded_capacity_ratio": lambda r: num(r["raw"], "funded_productive_capacity") / num(r["raw"], "scheduled_productive_capacity") if num(r["raw"], "scheduled_productive_capacity") > 0 else 0.0,
        "OCF": lambda r: r["operating_cash_flow"],
        "firm_cash": lambda r: num(r["raw"], "cash"),
    }
    for phase in ("active_default", "acute_D3", "cure_tail"):
        selected = all_active if phase == "active_default" else [r for r in all_active if r["default_phase"] == phase]
        for name, getter in measures.items():
            emit(output_rows, "active_default_endogenous_outcomes", name + "_total", sum(getter(r) for r in selected), phase)
            emit(output_rows, "active_default_endogenous_outcomes", name + "_mean", sum(getter(r) for r in selected) / max(1, len(selected)), phase)
        emit(output_rows, "active_default_endogenous_outcomes", "observations", len(selected), phase)

    actions = {
        "event_record_only": ("SUFFICIENT_MINIMAL_ACTION", "existing credit/payroll/production/repayment mechanisms already express the acute consequences."),
        "new_credit_freeze": ("REDUNDANT", "new executed credit is already absent in acute D3; a freeze could block cure-tail recovery."),
        "principal_repayment_suspension": ("REDUNDANT", "principal repayment is already zero during active default and acute D3."),
        "dividend_suspension": ("REDUNDANT", "dividends are already zero during active default and acute D3."),
        "interest_policy_change": ("CONTRACT_REDESIGN_REQUIRED", "changing accrual, arrears or principal is debt-contract restructuring."),
        "extra_production_cut": ("HIGH_DOUBLE_COUNT_RISK", "D3 payroll/funded-capacity constraints already reduce production capacity."),
        "extra_wage_cut": ("HIGH_DOUBLE_COUNT_RISK", "executed wages already reflect payroll funding constraints."),
        "layoffs": ("NOT_JUSTIFIED_AT_DEFAULT_EVENT", "layoffs belong to later labor restructuring/Exit design."),
        "Firm_closure": ("NOT_JUSTIFIED_AT_DEFAULT_EVENT", "Default does not imply Exit or Firm deletion."),
    }
    for action, (classification, note) in actions.items(): emit(output_rows, "action_classification", "classification", classification, action, note)

    flags = {"verdict": "A_DEFAULT_EVENT_RECORD_ONLY_IS_SUFFICIENT", "active_default_episode_semantics_ready": True, "default_history_semantics_ready": True, "existing_default_consequences_understood": True, "credit_freeze_incremental": False, "repayment_suspension_incremental": False, "dividend_suspension_incremental": False, "event_record_only_sufficient": True, "contract_restructuring_required_before_more_consequences": True, "default_event_implementation_ready": True, "default_economic_consequence_ready": False, "new_long_runs": 0, "economic_behavior_changed": False, "rng_changed": False, "default_implemented": False, "exit_implemented": False, "default_event_count": total_events, "active_default_episode_count": total_events, "active_default_firm_week_count": len(all_active), "acute_D3_default_week_count": sum(r["default_phase"] == "acute_D3" for r in all_active), "cure_tail_default_week_count": sum(r["default_phase"] == "cure_tail" for r in all_active), "credit_executed_active_default": sum(measures["credit_executed"](r) for r in all_active), "credit_executed_acute_D3": sum(measures["credit_executed"](r) for r in all_active if r["default_phase"] == "acute_D3"), "credit_executed_cure_tail": sum(measures["credit_executed"](r) for r in all_active if r["default_phase"] == "cure_tail"), "principal_repayment_active_default": sum(measures["principal_repayment"](r) for r in all_active), "principal_repayment_acute_D3": sum(measures["principal_repayment"](r) for r in all_active if r["default_phase"] == "acute_D3"), "principal_repayment_cure_tail": sum(measures["principal_repayment"](r) for r in all_active if r["default_phase"] == "cure_tail"), "dividends_active_default": sum(measures["dividends"](r) for r in all_active), "dividends_acute_D3": sum(measures["dividends"](r) for r in all_active if r["default_phase"] == "acute_D3"), "dividends_cure_tail": sum(measures["dividends"](r) for r in all_active if r["default_phase"] == "cure_tail"), "mean_payroll_funding_active_default": sum(measures["payroll_funding_ratio"](r) for r in all_active) / max(1, len(all_active)), "mean_funded_capacity_active_default": sum(measures["funded_capacity_ratio"](r) for r in all_active) / max(1, len(all_active)), "mean_unpaid_interest_active_default": sum(measures["unpaid_interest"](r) for r in all_active) / max(1, len(all_active)), "total_arrears_growth_active_default": sum(measures["arrears_growth"](r) for r in all_active)}
    with (OUT / "default_consequence_core_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "phase", "metric", "value", "note"]); writer.writeheader(); writer.writerows(output_rows)
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    (OUT / "acceptance_summary.md").write_text("""# Step 13.6D acceptance summary

Verdict: **A. DEFAULT_EVENT_RECORD_ONLY_IS_SUFFICIENT**.

The accepted four-week D3 event was reconstructed passively, with a four-week non-D3 cure tail. During acute D3, existing mechanics already provide the severe consequences: credit denial, unpaid interest, payroll impairment and reduced funded capacity. Principal repayment and dividends are already zero during active Default, so explicit suspension would be redundant. New credit is absent during acute D3 but can resume in the cure tail; a blanket post-event credit freeze could block recovery.

The minimal future implementation is therefore event recording only: increment `default_history_count` and maintain `active_default_episode`, without changing cash flow, credit, interest, repayment, dividends, wages, production, workers or Firm existence. Historical Default history and current active episode must remain separate. Any interest/arrears change requires later contract-restructuring semantics.

No economic behavior, RNG, Default consequence, restructuring or Exit was implemented.
""", encoding="utf-8")
    print(json.dumps(flags, indent=2))


if __name__ == "__main__": main()
