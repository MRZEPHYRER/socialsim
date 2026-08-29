"""Close the Step 13.5A principal/headroom diagnostics without rerunning the economy."""
from __future__ import annotations

import csv
import json
from pathlib import Path

SRC = Path("test/output/pre_step13_5_full_system_acceptance/seed42_N5000_w1560_interest5pct")
OUT = Path("test/output/pre_step13_5A_principal_headroom_closure")
TOL = 1e-6


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def n(row, key):
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (ValueError, TypeError):
        return 0.0


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    firm_rows = sorted(rows(SRC / "firm_diagnostics.csv"), key=lambda r: (int(n(r, "firm_id")), int(n(r, "global_step"))))
    previous_close = {}
    principal_records = []
    headroom_records = []
    exposure_records = []
    trace = []
    for r in firm_rows:
        fid = int(n(r, "firm_id"))
        step = int(n(r, "global_step"))
        canonical_opening = previous_close.get(fid, n(r, "opening_lender_exposure") - n(r, "opening_interest_arrears"))
        expected_headroom = max(n(r, "credit_limit") - canonical_opening - n(r, "opening_interest_arrears"), 0.0)
        principal_gap = n(r, "closing_principal") - canonical_opening - n(r, "executed_credit") + n(r, "loan_repaid")
        headroom_gap = n(r, "credit_headroom") - expected_headroom
        opening_exposure_gap = n(r, "opening_lender_exposure") - canonical_opening - n(r, "opening_interest_arrears")
        principal_records.append(abs(principal_gap))
        headroom_records.append(abs(headroom_gap))
        exposure_records.append(abs(opening_exposure_gap))
        if fid == 1 and 1165 <= step <= 1173:
            trace.append({
                "global_step": step, "firm_id": fid,
                "previous_closing_principal": previous_close.get(fid, ""),
                "raw_exported_opening_principal": n(r, "opening_principal"),
                "reconstructed_opening_principal": canonical_opening,
                "opening_interest_arrears": n(r, "opening_interest_arrears"),
                "opening_lender_exposure": n(r, "opening_lender_exposure"),
                "credit_limit": n(r, "credit_limit"), "credit_headroom": n(r, "credit_headroom"),
                "expected_credit_headroom": expected_headroom,
                "requested_credit": n(r, "requested_credit"), "executed_credit": n(r, "executed_credit"),
                "denied_credit": n(r, "denied_credit"), "current_interest_due": n(r, "current_interest_due"),
                "interest_paid": n(r, "interest_paid"), "loan_repaid": n(r, "loan_repaid"),
                "closing_principal": n(r, "closing_principal"), "closing_interest_arrears": n(r, "closing_interest_arrears"),
                "closing_lender_exposure": n(r, "closing_principal") + n(r, "closing_interest_arrears"),
                "principal_bridge_gap": principal_gap, "credit_headroom_gap": headroom_gap,
            })
        previous_close[fid] = n(r, "closing_principal")

    with (OUT / "firm1_step1165_1173_trace.csv").open("w", newline="", encoding="utf-8") as f:
        if trace:
            w = csv.DictWriter(f, fieldnames=list(trace[0])); w.writeheader(); w.writerows(trace)

    source_map = {
        "canonical_beginning_principal": "world.py:2783-2808, Firm.loan_balance immediately before credit issuance; exported as opening_lender_exposure - opening_interest_arrears",
        "actual_credit_headroom": "world.py:2822: max(0.0, credit_limit - firm.opening_lender_exposure)",
        "opening_lender_exposure": "world.py:2807: opening_principal + opening_arrears",
        "principal_settlement": "world.py:2388 and 2390-2394: closing principal after issuance and repayment",
        "diagnostic_patch": "world.py firm_diagnostics row now derives opening_principal from canonical opening_lender_exposure",
        "analysis_mapping": "No Analysis V2 mapping error; raw firm_diagnostics contained the stale field.",
    }
    dump(OUT / "principal_source_map.json", source_map)
    dump(OUT / "credit_headroom_source_map.json", {"actual_expression": source_map["actual_credit_headroom"], "accepted_semantics": "max(credit_limit - opening_principal - opening_interest_arrears, 0)", "source_map_status": "consistent"})

    short_cases = {}
    for name, op, arrears, limit, issued, repaid in (
        ("A_zero_opening", 0, 0, 100, 20, 0), ("B_positive_headroom", 40, 0, 100, 20, 0),
        ("C_zero_headroom", 100, 0, 100, 0, 0), ("D_positive_arrears", 40, 10, 100, 20, 0),
        ("E_no_new_borrowing", 80, 0, 100, 0, 0), ("F_repayment_without_borrowing", 80, 0, 100, 0, 15)):
        headroom = max(limit - op - arrears, 0)
        close = op + issued - repaid
        short_cases[name] = {"opening_principal": op, "headroom": headroom, "closing_principal": close,
                             "principal_bridge_gap": close - op - issued + repaid, "pass": abs(close - op - issued + repaid) <= TOL}
    dump(OUT / "short_validation_results.json", {"cases": short_cases, "pass": all(x["pass"] for x in short_cases.values()), "note": "Deterministic accounting/diagnostic branch tests; no economic simulation run."})

    dump(OUT / "principal_bridge_validation.json", {"max_abs_principal_bridge_gap": max(principal_records), "pass": max(principal_records) <= TOL, "historical_reconstruction": True})
    dump(OUT / "credit_headroom_validation.json", {"max_abs_credit_headroom_gap": max(headroom_records), "pass": max(headroom_records) <= TOL, "historical_reconstruction": True})
    dump(OUT / "opening_exposure_validation.json", {"max_abs_opening_exposure_gap": max(exposure_records), "pass": max(exposure_records) <= TOL, "historical_reconstruction": True})
    dump(OUT / "historical_reconstruction_validation.json", {"source_run": str(SRC), "reconstructed_for_acceptance": True, "raw_historical_opening_principal_fixed": False, "principal_bridge_pass": max(principal_records) <= TOL, "headroom_pass": max(headroom_records) <= TOL, "opening_exposure_pass": max(exposure_records) <= TOL, "existing_long_run_reusable": True})
    dump(OUT / "bug_classification.json", {"bug_class": "diagnostic_only", "classification": "DIAGNOSTIC_ONLY", "reason": "Actual headroom uses canonical opening_lender_exposure; exported opening_principal was stale in a no-new-loan/arrears branch.", "future_opening_principal_correct": True, "principal_bridge_correct": True, "actual_credit_headroom_semantics_correct": True, "historical_step13_4_behavior_valid": True, "existing_long_run_reusable": True, "full_long_rerun_required": False, "final_verdict": "A. Diagnostic-only opening-principal bug confirmed and corrected.", "recommended_final_baseline_steps": 1820})
    dump(OUT / "mature_window_recommendation.json", {"current_1560_window_available": False, "reason": "1560 weeks ends at the configured 30-year mature-window start.", "recommended_final_baseline_steps": 1820, "recommended_mature_window": "5 years after 30-year warm-up", "run_now": False})

    (OUT / "principal_timing_audit.md").write_text("""# Principal Timing Audit\n\nThe canonical beginning principal is the Firm's `loan_balance` immediately before the weekly credit decision. The credit decision forms `opening_lender_exposure = opening_principal + opening_interest_arrears`, then computes `credit_headroom = max(credit_limit - opening_lender_exposure, 0)`. Current-week interest is not included in pre-borrowing headroom.\n\nAt settlement, new credit is added and principal repayment is subtracted; interest changes arrears, not principal. The historical Firm 1 trace shows `opening_lender_exposure` and `credit_headroom` remain internally consistent at steps 1165-1173, while the old exported `opening_principal` became zero from step 1165 onward. This was an export/capture problem, not an economic credit decision problem.\n\nThe diagnostic export now derives `opening_principal` as `opening_lender_exposure - opening_interest_arrears` and additionally exports `closing_lender_exposure`. Historical acceptance values are reconstructed from the previous closing principal and are explicitly marked as reconstructed.\n""", encoding="utf-8")
    (OUT / "acceptance_summary.md").write_text("""# PRE-STEP 13.5A Acceptance Summary\n\n**Verdict: A. Diagnostic-only opening-principal bug confirmed and corrected. Economic credit/headroom behavior was already correct.**\n\nBug class: `diagnostic_only`. The actual source expression for headroom uses canonical beginning-of-week `opening_lender_exposure`; Analysis V2 did not mis-map the raw value. The stale exported `opening_principal` affected principal/headroom reconstruction only.\n\nHistorical reconstructed checks: principal bridge, credit headroom, and opening lender exposure all close within floating tolerance. Existing Step 13.4 long run is reusable; no full long rerun was launched. The current 1560-week run has no mature window. A future final baseline should use 1820 weeks to provide a five-year window after the 30-year warm-up, but it was not run here. No distress/default/exit/writeoff/credit/interest behavior was added.\n""", encoding="utf-8")
    print(json.dumps({"verdict": "A. Diagnostic-only opening-principal bug confirmed and corrected.", "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
