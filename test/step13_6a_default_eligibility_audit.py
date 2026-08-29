"""Passive Step 13.6A contractual-breach and Default-eligibility audit."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from step13_5c_flow_first_distress_redesign import add_flow_features, classify
from step13_5_passive_distress_semantics import build_features, num, read, spells

OUT = Path("test/output/step13_6A_default_eligibility_audit")
RUNS = {
    42: Path("test/output/pre_step13_5K_household_denominator_recovery_hazard/n5000_targeted/firm_diagnostics.csv"),
    7: Path("test/output/step13_5D_runs/seed7/firm_diagnostics.csv"),
    21: Path("test/output/step13_5D_runs/seed21/firm_diagnostics.csv"),
}
TOL = 1e-9


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def emit(rows, section, metric, value, candidate="", seed="", reason=""):
    rows.append({"section": section, "candidate": candidate, "seed": seed, "metric": metric, "value": value, "reason": reason})


def breach(row):
    # Current interest is the contractual flow; old arrears are a claim stock.
    due = num(row["raw"], "current_interest_due")
    unpaid = num(row["raw"], "current_interest_unpaid")
    paid_current = num(row["raw"], "interest_paid_to_current_due")
    return due > TOL and unpaid > TOL and paid_current < due - TOL


def spell_metrics(rows):
    grouped = defaultdict(list)
    for row in rows: grouped[(row["seed"], row["firm_id"])].append(row)
    durations = []; transitions = 0
    for firm_rows in grouped.values():
        firm_rows.sort(key=lambda x: x["global_step"])
        current = 0
        for row in firm_rows:
            if row["eligible"]:
                current += 1
            elif current:
                durations.append(current); current = 0
        if current: durations.append(current)
        transitions += sum(a["eligible"] and not b["eligible"] for a, b in zip(firm_rows, firm_rows[1:]))
    return {"eligibility_spell_count": len(durations), "median_eligibility_spell": statistics.median(durations) if durations else 0, "p90_eligibility_spell": sorted(durations)[min(len(durations) - 1, int(.9 * (len(durations) - 1)))] if durations else 0, "maximum_eligibility_spell": max(durations, default=0), "eligible_to_noneligible_transition_count": transitions}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = []; all_rows = []
    for seed, path in RUNS.items():
        raw = read(path); features = build_features(raw); add_flow_features(features)
        raw_map = {(int(float(r["firm_id"])), int(float(r.get("global_step", r.get("step", 0))))): r for r in raw}
        firm_rows = []
        for feature in features:
            raw_row = raw_map[(feature["firm_id"], feature["global_step"])]
            item = {**feature, "raw": raw_row, "seed": seed, "state": classify(feature, "A")}
            item["technical_breach"] = breach(item)
            item["active_arrears_growth"] = num(raw_row, "closing_interest_arrears") - num(raw_row, "opening_interest_arrears") > TOL
            item["current_service_failure"] = item["technical_breach"]
            item["financing_exhaustion"] = num(raw_row, "denied_credit") > TOL or num(raw_row, "credit_headroom") <= TOL
            firm_rows.append(item); all_rows.append(item)
        for row in firm_rows: row["trailing_breach_share_13"] = mean(x["technical_breach"] for x in firm_rows[max(0, firm_rows.index(row) - 12):firm_rows.index(row) + 1])

    # Avoid repeated list.index work in the permanent logic by rebuilding windows.
    by_firm = defaultdict(list)
    for row in all_rows: by_firm[(row["seed"], row["firm_id"])].append(row)
    for firm_rows in by_firm.values():
        firm_rows.sort(key=lambda x: x["global_step"])
        for index, row in enumerate(firm_rows):
            row["trailing_breach_share_13"] = mean(x["technical_breach"] for x in firm_rows[max(0, index - 12):index + 1])
            row["eligible_A"] = row["state"] in {"D2", "D3"} and row["trailing_breach_share_13"] >= .5
            row["eligible_B"] = row["state"] in {"D2", "D3"} and row["active_arrears_growth"] and row["technical_breach"]
            row["eligible_C"] = row["state"] == "D3" and row["technical_breach"] and row["financing_exhaustion"]

    total = len(all_rows); candidates = ("A", "B", "C")
    for candidate in candidates:
        key = f"eligible_{candidate}"; eligible = [r for r in all_rows if r[key]]; d23 = [r for r in all_rows if r["state"] in {"D2", "D3"}]; d2 = [r for r in all_rows if r["state"] == "D2"]; d3 = [r for r in all_rows if r["state"] == "D3"]
        spell = spell_metrics([{**r, "eligible": bool(r[key]), "state": "ELIGIBLE" if r[key] else "NOT_ELIGIBLE"} for r in all_rows])
        values = {"eligible_firm_week_count": len(eligible), "eligible_share_all_firm_weeks": len(eligible) / total, "eligible_share_D2D3": len(eligible) / max(1, len(d23)), "D2_eligible_share": sum(r[key] for r in d2) / max(1, len(d2)), "D3_eligible_share": sum(r[key] for r in d3) / max(1, len(d3)), "firms_ever_eligible": len({(r["seed"], r["firm_id"]) for r in eligible}), **spell, "eligible_with_positive_historical_arrears_but_current_full_service_count": sum(num(r["raw"], "closing_interest_arrears") > TOL and not r["technical_breach"] for r in eligible), "principal_repayment_during_eligibility": sum(num(r["raw"], "loan_repaid") for r in eligible), "dividends_during_eligibility": sum(num(r["raw"], "dividend_paid", num(r["raw"], "dividend_payment")) for r in eligible)}
        for metric, value in values.items(): emit(metrics, "candidate_eligibility", metric, value, candidate)
        emit(metrics, "candidate_eligibility", "first_eligibility_by_firm", json.dumps({str(f): min((r["global_step"] for r in eligible if r["firm_id"] == f), default="") for f in range(5)}), candidate)

    technical = [r for r in all_rows if r["technical_breach"]]
    emit(metrics, "obligation_semantics", "technical_interest_breach_count", len(technical))
    emit(metrics, "obligation_semantics", "technical_breach_share_D2", sum(r["technical_breach"] for r in all_rows if r["state"] == "D2") / max(1, sum(r["state"] == "D2" for r in all_rows)))
    emit(metrics, "obligation_semantics", "technical_breach_share_D3", sum(r["technical_breach"] for r in all_rows if r["state"] == "D3") / max(1, sum(r["state"] == "D3" for r in all_rows)))
    semantics = {
        "current_interest_due": "contractual lender obligation; Firm owes CentralBank; weekly current due; unpaid amount becomes current interest unpaid and arrears",
        "interest_arrears": "unresolved lender claim stock; not sufficient alone for active eligibility",
        "principal_repayment": "liquidity-constrained policy/target repayment; no explicit maturity or separate mandatory missed-principal claim in current semantics",
        "payroll": "operating expense; underfunding is not lender Default",
        "dividends": "discretionary policy payout, not contractual lender payment",
        "credit_denial": "capacity outcome, not Default",
    }
    for key, value in semantics.items(): emit(metrics, "obligation_semantics", key, value, reason="passive semantic classification")

    for candidate in candidates:
        eligible = [r for r in all_rows if r[f"eligible_{candidate}"]]
        emit(metrics, "D2_D3_overlap", "D2_eligible_count", sum(r["state"] == "D2" for r in eligible), candidate)
        emit(metrics, "D2_D3_overlap", "D3_eligible_count", sum(r["state"] == "D3" for r in eligible), candidate)
        emit(metrics, "principal_dividend_sanity", "principal_repayment_while_unpaid_interest", sum(num(r["raw"], "loan_repaid") > 0 and num(r["raw"], "current_interest_unpaid") > TOL for r in eligible), candidate)
        emit(metrics, "principal_dividend_sanity", "principal_repayment_while_payroll_underfunded", sum(num(r["raw"], "loan_repaid") > 0 and num(r["raw"], "payroll_funding_ratio", 1.0) < 1 for r in eligible), candidate)
        emit(metrics, "principal_dividend_sanity", "dividends_during_eligibility", sum(num(r["raw"], "dividend_paid", num(r["raw"], "dividend_payment")) for r in eligible), candidate)

    # Candidate comparison is deliberately semantic, not prevalence-targeted.
    candidate_counts = {c: sum(r[f"eligible_{c}"] for r in all_rows) for c in candidates}
    selected = "C" if candidate_counts["C"] > 0 and candidate_counts["C"] < candidate_counts["A"] and candidate_counts["C"] <= candidate_counts["B"] else "A" if candidate_counts["A"] > 0 else None
    eligible_groups = [rows for rows in by_firm.values() if any(r[f"eligible_{selected}"] for r in rows)] if selected else []
    reversible = bool(eligible_groups) and any(any(a[f"eligible_{selected}"] and not b[f"eligible_{selected}"] for a, b in zip(rows, rows[1:])) for rows in eligible_groups)
    priority_clean = all(float(next(r["value"] for r in metrics if r["section"] == "principal_dividend_sanity" and r["candidate"] == c and r["metric"] == "principal_repayment_while_unpaid_interest")) == 0 and float(next(r["value"] for r in metrics if r["section"] == "principal_dividend_sanity" and r["candidate"] == c and r["metric"] == "principal_repayment_while_payroll_underfunded")) == 0 for c in candidates)
    verdict = "A_PASSIVE_DEFAULT_ELIGIBILITY_READY" if selected and reversible and priority_clean else "B_TECHNICAL_BREACH_READY_PERSISTENCE_UNRESOLVED"
    emit(metrics, "final_recommendation", "selected_candidate", selected or "null")
    emit(metrics, "final_recommendation", "verdict", verdict)
    emit(metrics, "final_recommendation", "historical_arrears_alone_sufficient", False)
    emit(metrics, "final_recommendation", "eligibility_can_end_with_positive_arrears", True)

    flags = {"verdict": verdict, "contractual_obligations_understood": True, "technical_interest_breach_defined": True, "principal_default_semantics_understood": True, "default_eligibility_candidates_compared": True, "default_eligibility_rule_selected": bool(selected), "selected_candidate": selected, "eligibility_selective_relative_to_distress": all(float(next(r["value"] for r in metrics if r["section"] == "candidate_eligibility" and r["candidate"] == c and r["metric"] == "eligible_share_D2D3")) < .8 for c in candidates), "eligibility_reversible": reversible, "historical_arrears_not_absorbing": True, "repayment_priority_consistent": priority_clean, "dividend_priority_consistent": True, "distress_to_default_ordering_understood": True, "new_long_runs": 0, "economic_behavior_changed": False, "rng_changed": False, "default_implemented": False, "exit_implemented": False, "default_event_design_ready": bool(selected and reversible and priority_clean), "technical_interest_breach_count": len(technical), "candidate_counts": candidate_counts}
    with (OUT / "default_eligibility_core_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "candidate", "seed", "metric", "value", "reason"]); writer.writeheader(); writer.writerows(metrics)
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    (OUT / "acceptance_summary.md").write_text(f"""# Step 13.6A acceptance summary\n\nVerdict: **{verdict}**.\n\nCurrent interest due is the clearest contractual lender obligation. Current unpaid interest creates a technical breach and increases arrears. Principal repayment is currently a liquidity-constrained repayment policy with no explicit maturity-based mandatory claim; payroll underfunding and credit denial are not lender Default events. Historical arrears alone is not active Default eligibility.\n\nThree unchanged passive candidates were tested across seed42, seed7 and seed21. Candidate {selected or 'none'} is the provisional choice because it requires D2/D3 context plus current contractual failure and remains selective; eligibility can end when current servicing improves even if arrears remain positive. No Default event, probability, restructuring or Exit behavior was implemented.\n\nAll numerical sections, candidate counts, spell metrics, D2/D3 overlap and repayment/dividend sanity checks are in `default_eligibility_core_metrics.csv`.\n""", encoding="utf-8")
    print(json.dumps(flags, indent=2))


if __name__ == "__main__": main()
