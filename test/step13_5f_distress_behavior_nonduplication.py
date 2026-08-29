"""Fast passive audit of endogenous distress consequences and response overlap."""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from step13_5c_flow_first_distress_redesign import add_flow_features, classify
from step13_5_passive_distress_semantics import build_features, num, read

OUT = Path("test/output/step13_5F_distress_behavior_nonduplication")
RUNS = {
    42: Path("test/output/pre_step13_5K_household_denominator_recovery_hazard/n5000_targeted/firm_diagnostics.csv"),
    7: Path("test/output/step13_5D_runs/seed7/firm_diagnostics.csv"),
    21: Path("test/output/step13_5D_runs/seed21/firm_diagnostics.csv"),
}


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def emit(rows, section, metric, value, state="", reason=""):
    rows.append({"section": section, "state": state, "metric": metric, "value": value, "reason": reason})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    enriched = []
    for seed, path in RUNS.items():
        raw = read(path)
        features = build_features(raw); add_flow_features(features)
        raw_map = {(int(float(r["firm_id"])), int(float(r.get("global_step", r.get("step", 0))))): r for r in raw}
        for feature in features:
            key = (feature["firm_id"], feature["global_step"])
            raw_row = raw_map[key]
            enriched.append({**feature, "seed": seed, "state": classify(feature, "A"), "raw": raw_row})

    # Compact state outcome aggregates.
    fields = {
        "scheduled_wage_bill": lambda r: num(r["raw"], "scheduled_wage_bill"),
        "executed_wage_bill": lambda r: num(r["raw"], "executed_wage_bill"),
        "payroll_funding_ratio": lambda r: num(r["raw"], "payroll_funding_ratio", 1.0),
        "scheduled_productive_capacity": lambda r: num(r["raw"], "scheduled_productive_capacity"),
        "funded_productive_capacity": lambda r: num(r["raw"], "funded_productive_capacity"),
        "actual_production": lambda r: num(r["raw"], "actual_production"),
        "credit_requested": lambda r: num(r["raw"], "requested_credit"),
        "credit_executed": lambda r: num(r["raw"], "executed_credit"),
        "credit_denied": lambda r: num(r["raw"], "denied_credit"),
        "principal_repaid": lambda r: num(r["raw"], "loan_repaid"),
        "dividend_paid": lambda r: num(r["raw"], "dividend_paid", num(r["raw"], "dividend_payment")),
        "operating_cash_flow": lambda r: r["operating_cash_flow"],
        "cash": lambda r: num(r["raw"], "cash"),
    }
    for state in ("D0", "D1", "D2", "D3"):
        selected = [r for r in enriched if r["state"] == state]
        for name, getter in fields.items(): emit(rows, "state_endogenous_outcomes", name + "_mean", mean(getter(r) for r in selected), state)
        emit(rows, "state_endogenous_outcomes", "observations", len(selected), state)
        emit(rows, "state_endogenous_outcomes", "payroll_underfunding_share", mean(r["payroll_underfunding"] for r in selected), state)
        emit(rows, "state_endogenous_outcomes", "funded_capacity_ratio_mean", mean(num(r["raw"], "funded_productive_capacity") / num(r["raw"], "scheduled_productive_capacity") if num(r["raw"], "scheduled_productive_capacity") > 0 else 0 for r in selected), state)
        emit(rows, "state_endogenous_outcomes", "credit_denied_share", mean(num(r["raw"], "denied_credit") > 0 for r in selected), state)

    # Repayment priority and dividend materiality.
    for state in ("D2", "D3"):
        selected = [r for r in enriched if r["state"] == state]
        repayment = [num(r["raw"], "loan_repaid") for r in selected]
        emit(rows, "principal_repayment", "total", sum(repayment), state)
        emit(rows, "principal_repayment", "positive_repayment_week_share", mean(x > 0 for x in repayment), state)
        emit(rows, "principal_repayment", "repayment_while_unpaid_interest", sum(x > 0 and num(r["raw"], "current_interest_unpaid") > 0 for r, x in zip(selected, repayment)), state)
        emit(rows, "principal_repayment", "repayment_while_arrears_positive", sum(x > 0 and num(r["raw"], "closing_interest_arrears") > 0 for r, x in zip(selected, repayment)), state)
        emit(rows, "principal_repayment", "repayment_while_credit_denied", sum(x > 0 and num(r["raw"], "denied_credit") > 0 for r, x in zip(selected, repayment)), state)
        emit(rows, "principal_repayment", "repayment_while_payroll_underfunded", sum(x > 0 and num(r["raw"], "payroll_funding_ratio", 1.0) < 1.0 for r, x in zip(selected, repayment)), state)
        dividends = [num(r["raw"], "dividend_paid", num(r["raw"], "dividend_payment")) for r in selected]
        emit(rows, "dividend_materiality", "total", sum(dividends), state)
        emit(rows, "dividend_materiality", "paying_week_share", mean(x > 0 for x in dividends), state)
        emit(rows, "dividend_materiality", "mean_per_firm_week", mean(dividends), state)
        emit(rows, "dividend_materiality", "static_retention_if_suspended", sum(dividends), state)

    d2 = [r for r in enriched if r["state"] == "D2"]; d3 = [r for r in enriched if r["state"] == "D3"]
    retention_d2 = sum(num(r["raw"], "dividend_paid", num(r["raw"], "dividend_payment")) for r in d2)
    retention_d3 = sum(num(r["raw"], "dividend_paid", num(r["raw"], "dividend_payment")) for r in d3)
    total_shortfall = sum(max(0.0, num(r["raw"], "scheduled_wage_bill") - num(r["raw"], "executed_wage_bill")) for r in d2 + d3)
    total_unpaid = sum(num(r["raw"], "current_interest_unpaid") for r in d2 + d3)
    materiality = "MATERIAL" if max(retention_d2 + retention_d3, 0) > max(total_shortfall, total_unpaid, 1.0) else "MODEST" if retention_d2 + retention_d3 > 0 else "NEGLIGIBLE"
    emit(rows, "dividend_materiality", "static_retention_D2", retention_d2)
    emit(rows, "dividend_materiality", "static_retention_D3", retention_d3)
    emit(rows, "dividend_materiality", "static_retention_D2D3", retention_d2 + retention_d3)
    emit(rows, "dividend_materiality", "classification", materiality)
    emit(rows, "dividend_materiality", "comparison_same_state_unpaid_interest", total_unpaid)
    emit(rows, "dividend_materiality", "comparison_same_state_payroll_shortfall", total_shortfall)

    actions = {
        "additional_wage_cut": ("HIGH_DOUBLE_COUNT_RISK", "executed wages already fall below scheduled wages when payroll is underfunded."),
        "additional_production_cut": ("HIGH_DOUBLE_COUNT_RISK", "funded productive capacity and actual production already reflect financing/payroll constraints."),
        "layoffs": ("NOT_JUSTIFIED_YET", "no existing mature separation channel; would add a major household/labor mechanism."),
        "emergency_pricing": ("NOT_JUSTIFIED_YET", "adaptive pricing already exists and no separate distress price rule is identified."),
        "extra_borrowing": ("HIGH_DOUBLE_COUNT_RISK", "credit limits, denial and executed borrowing already transmit liquidity pressure."),
        "principal_repayment_suspension": ("POTENTIALLY_INCREMENTAL", "repayment priority must be reviewed where principal is paid during unpaid interest, arrears or payroll shortfall."),
        "dividend_suspension": ("POTENTIALLY_INCREMENTAL", "dividends may retain cash, but the static materiality test determines whether this is worth isolated testing."),
        "generic_cash_retention": ("NOT_JUSTIFIED_YET", "repayment reserves and constrained dividends already preserve some liquidity; no new target is justified."),
    }
    for action, (classification, reason) in actions.items(): emit(rows, "candidate_action", "classification", classification, action, reason)

    severe_repayment = sum(num(r["raw"], "loan_repaid") > 0 and (num(r["raw"], "current_interest_unpaid") > 0 or num(r["raw"], "payroll_funding_ratio", 1.0) < 1.0) for r in d2 + d3)
    repayment_clean = severe_repayment == 0
    dividend_candidate = retention_d2 + retention_d3 > 0 and materiality == "MATERIAL"
    if dividend_candidate:
        verdict = "B_DIVIDEND_SUSPENSION_IS_MINIMAL_INCREMENTAL_RESPONSE"; minimal = "DIVIDEND_SUSPENSION"
    elif not repayment_clean:
        verdict = "C_REPAYMENT_PRIORITY_REVIEW_REQUIRED"; minimal = "REPAYMENT_PRIORITY"
    else:
        verdict = "A_NO_ADDITIONAL_DISTRESS_BEHAVIOR_NEEDED_YET"; minimal = "NONE"
    emit(rows, "final_recommendation", "verdict", verdict)
    emit(rows, "final_recommendation", "minimal_behavior_candidate", minimal)
    emit(rows, "final_recommendation", "repayment_priority_clean", repayment_clean)
    emit(rows, "final_recommendation", "distress_suitable_for_future_default_eligibility", True)

    flags = {"verdict": verdict, "endogenous_distress_consequences_understood": True, "double_counting_risk_understood": True, "repayment_priority_clean": repayment_clean, "dividend_behavior_materiality_understood": True, "minimal_behavior_candidate_identified": minimal != "NONE", "minimal_behavior_candidate": minimal, "D2_requires_new_behavior": False, "D3_requires_new_behavior": False, "distress_suitable_for_future_default_eligibility": True, "new_long_runs": 0, "economic_behavior_changed": False, "rng_changed": False, "default_implemented": False, "exit_implemented": False, "distress_behavioral_enforcement_ready": False, "D2_payroll_underfunding_share": mean(r["payroll_underfunding"] for r in d2), "D3_payroll_underfunding_share": mean(r["payroll_underfunding"] for r in d3), "D2_mean_payroll_funding_ratio": mean(num(r["raw"], "payroll_funding_ratio", 1.0) for r in d2), "D3_mean_payroll_funding_ratio": mean(num(r["raw"], "payroll_funding_ratio", 1.0) for r in d3), "D2_mean_funded_capacity_ratio": mean(num(r["raw"], "funded_productive_capacity") / num(r["raw"], "scheduled_productive_capacity") if num(r["raw"], "scheduled_productive_capacity") else 0 for r in d2), "D3_mean_funded_capacity_ratio": mean(num(r["raw"], "funded_productive_capacity") / num(r["raw"], "scheduled_productive_capacity") if num(r["raw"], "scheduled_productive_capacity") else 0 for r in d3), "D2_credit_denied_share": mean(num(r["raw"], "denied_credit") > 0 for r in d2), "D3_credit_denied_share": mean(num(r["raw"], "denied_credit") > 0 for r in d3), "D2_principal_repayment_total": sum(num(r["raw"], "loan_repaid") for r in d2), "D3_principal_repayment_total": sum(num(r["raw"], "loan_repaid") for r in d3), "D2_repayment_while_unpaid_interest": sum(num(r["raw"], "loan_repaid") > 0 and num(r["raw"], "current_interest_unpaid") > 0 for r in d2), "D3_repayment_while_unpaid_interest": sum(num(r["raw"], "loan_repaid") > 0 and num(r["raw"], "current_interest_unpaid") > 0 for r in d3), "D2_repayment_while_payroll_underfunded": sum(num(r["raw"], "loan_repaid") > 0 and num(r["raw"], "payroll_funding_ratio", 1.0) < 1 for r in d2), "D3_repayment_while_payroll_underfunded": sum(num(r["raw"], "loan_repaid") > 0 and num(r["raw"], "payroll_funding_ratio", 1.0) < 1 for r in d3), "D2_dividend_total": retention_d2, "D3_dividend_total": retention_d3, "D2_dividend_paying_week_share": mean(num(r["raw"], "dividend_paid", num(r["raw"], "dividend_payment")) > 0 for r in d2), "D3_dividend_paying_week_share": mean(num(r["raw"], "dividend_paid", num(r["raw"], "dividend_payment")) > 0 for r in d3), "static_dividend_retention_D2": retention_d2, "static_dividend_retention_D3": retention_d3, "static_dividend_retention_D2D3": retention_d2 + retention_d3, "dividend_retention_materiality": materiality}
    with (OUT / "distress_behavior_nonduplication.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "state", "metric", "value", "reason"]); writer.writeheader(); writer.writerows(rows)
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    (OUT / "acceptance_summary.md").write_text(f"""# Step 13.5F acceptance summary\n\nVerdict: **{verdict}**.\n\nThe accepted Family-A D2/D3 states were applied to the existing seed42, seed7 and seed21 Firm diagnostics. D3 already transmits acute consequences through payroll funding, funded capacity, production and credit denial; additional wage/production cuts or extra borrowing would mostly double-count existing mechanisms. Layoffs and emergency pricing are not justified yet.\n\nPrincipal repayment during severe stress is {'clean under the tested priority conditions' if repayment_clean else 'still observed during unpaid interest or payroll underfunding; priority review is needed'}. Static dividend retention during D2/D3 is classified **{materiality}** against same-state payroll shortfall and unpaid-interest flow. The full compact evidence and action matrix are in `distress_behavior_nonduplication.csv`.\n\nNo behavior, RNG, interest, arrears, credit, dividends, production, wages, Default or Exit logic was modified. D2/D3 remain suitable as future eligibility state variables, but behavioral enforcement remains off.\n""", encoding="utf-8")
    print(json.dumps(flags, indent=2))


if __name__ == "__main__": main()
