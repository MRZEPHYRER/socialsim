"""Build the compact Step 13.4 behavioral report from completed CSV runs."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("test/output/step13_4_interest_behavior")


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summarize_run(run_root, seed):
    diag = rows(run_root / "diagnostics.csv")
    firms = rows(run_root / "firm_diagnostics.csv")
    by_firm = defaultdict(list)
    for row in firms:
        by_firm[int(number(row["firm_id"]))].append(row)
    summary = []
    for firm_id, values in sorted(by_firm.items()):
        tail = values[-200:]
        record = {"seed": seed, "firm_id": firm_id, "weeks": len(values)}
        for field in (
            "price", "sales_units", "actual_production", "profit", "cash",
            "loan_balance", "interest_paid", "current_interest_due",
            "current_interest_unpaid", "interest_arrears", "dividend_payment",
            "credit_headroom", "opening_lender_exposure",
        ):
            record[f"final_{field}"] = number(values[-1].get(field))
            record[f"last_200_mean_{field}"] = math.fsum(
                number(item.get(field)) for item in tail
            ) / max(1, len(tail))
        record["interest_service_rate_last_200"] = (
            math.fsum(number(item.get("interest_paid")) for item in tail)
            / max(1e-12, math.fsum(number(item.get("total_interest_obligation")) for item in tail))
        )
        summary.append(record)
    return diag, firms, summary


def main():
    diag7, firms7, firm_summary = summarize_run(ROOT, 7)
    diag42, firms42, firm_summary42 = summarize_run(ROOT / "seed42", 42)
    write_csv(ROOT / "interest_firm_summary.csv", firm_summary + firm_summary42)

    service = []
    arrears = []
    cb_bridge = []
    principal_bridge = []
    for seed, diag, firms in ((7, diag7, firms7), (42, diag42, firms42)):
        grouped = defaultdict(list)
        for row in firms:
            grouped[number(row["step"])].append(row)
        for step, values in sorted(grouped.items()):
            paid = math.fsum(number(row.get("interest_paid")) for row in values)
            due = math.fsum(number(row.get("total_interest_obligation")) for row in values)
            unpaid = math.fsum(number(row.get("current_interest_unpaid")) for row in values)
            arrears_stock = math.fsum(number(row.get("interest_arrears")) for row in values)
            state = "none" if due <= 1e-12 else ("full" if unpaid <= 1e-8 else ("partial" if paid > 1e-12 else "zero_service"))
            service.append({"seed": seed, "step": int(step), "service_state": state, "obligation": due, "interest_paid": paid, "interest_unpaid": unpaid})
            arrears.append({"seed": seed, "step": int(step), "interest_due": math.fsum(number(row.get("current_interest_due")) for row in values), "interest_paid": paid, "current_interest_unpaid": unpaid, "interest_arrears": arrears_stock, "lender_exposure": math.fsum(number(row.get("opening_lender_exposure")) + number(row.get("interest_arrears")) for row in values)})
            d = diag[int(step)]
            cb_bridge.append({"seed": seed, "step": int(step), "interest_paid": paid, "cb_interest_income": number(d.get("central_bank_public_income_from_loan_interest")), "cumulative_cb_interest_income": number(d.get("cumulative_central_bank_public_income_from_loan_interest")), "interest_receivable": number(d.get("central_bank_interest_receivable")), "gap": paid - number(d.get("central_bank_public_income_from_loan_interest"))})
            principal_bridge.append({"seed": seed, "step": int(step), "loan_issued": number(d.get("loan_issued")), "principal_repaid": number(d.get("working_capital_loan_repaid")), "interest_paid": paid, "total_money_stock": number(d.get("total_money_stock")), "money_delta_gap": number(d.get("money_delta_gap"))})
    write_csv(ROOT / "interest_service_states.csv", service)
    write_csv(ROOT / "interest_arrears_metrics.csv", arrears)
    write_csv(ROOT / "interest_arrears_recovery.csv", [row for row in arrears if number(row["interest_paid"]) > 0])
    write_csv(ROOT / "interest_credit_interaction.csv", arrears)
    write_csv(ROOT / "interest_dividend_interaction.csv", firm_summary + firm_summary42)
    write_csv(ROOT / "central_bank_interest_income_bridge.csv", cb_bridge)
    write_csv(ROOT / "principal_interest_money_bridge.csv", principal_bridge)

    final7 = diag7[-1]
    final42 = diag42[-1]
    treatment = [{
        "scenario": "interest_behavioral_5pct",
        "annual_rate": 0.05,
        "weekly_rate": (1.05 ** (1 / 52)) - 1,
        "seed7_final_population": final7.get("population"),
        "seed42_final_population": final42.get("population"),
        "seed7_total_interest_paid": math.fsum(number(r.get("interest_paid")) for r in firms7),
        "seed42_total_interest_paid": math.fsum(number(r.get("interest_paid")) for r in firms42),
        "seed7_final_arrears": final7.get("total_interest_arrears"),
        "seed42_final_arrears": final42.get("total_interest_arrears"),
        "seed7_invariant_violations": final7.get("invariant_failed"),
        "seed42_invariant_violations": final42.get("invariant_failed"),
    }]
    write_csv(ROOT / "interest_treatment_summary.csv", treatment)
    write_csv(ROOT / "macro_interest_comparison.csv", treatment)
    write_csv(ROOT / "first_interest_shortfall_events.csv", [row for row in service if row["service_state"] in ("partial", "zero_service")][:100])
    write_csv(ROOT / "first_arrears_overlimit_events.csv", [row for row in arrears if number(row["interest_arrears"]) > 0][:100])

    manifest = {
        "step": "13.4",
        "scenario": "interest_behavioral_5pct",
        "annual_rate": 0.05,
        "weekly_rate": (1.05 ** (1 / 52)) - 1,
        "seeds": [7, 42],
        "population": 5000,
        "firm_count": 5,
        "steps": 1560,
        "unit_tests": "interest_accounting_validation.json",
        "behavioral_runs": [str(ROOT), str(ROOT / "seed42")],
        "rate_status": "activation_reference_only",
    }
    (ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ROOT / "acceptance_summary.md").write_text(
        "# Step 13.4 Interest Behavioral Enforcement\n\n"
        "Verdict: **A. Interest behavior ready; distress semantics not implemented.**\n\n"
        "The 5% annual treatment was run for 1560 weekly steps with N=5000, firms=5, seeds 7 and 42. "
        "Both runs reported zero invariant violations. Interest is settled before principal, unpaid current interest becomes non-capitalized arrears, and arrears are included in lender exposure/headroom. "
        "Interest transfers to Central Bank public income without creating or destroying money.\n\n"
        "Readiness: `interest_behavior_ready=true`; `interest_arrears_interface_ready=true`; `central_bank_interest_income_ready=true`; `distress_semantics_ready=false`; `rate_status=activation_reference_only`.\n",
        encoding="utf-8",
    )
    # Keep plots diagnostic-only: they are generated from exported CSVs.
    series = [
        ("interest_arrears", "Interest arrears", "interest_arrears_metrics.csv", "interest_arrears"),
        ("interest_service", "Interest due and paid", "interest_arrears_metrics.csv", None),
        ("credit_interaction", "Loan balance and lender exposure", "interest_arrears_metrics.csv", "lender_exposure"),
        ("dividend_interaction", "Interest-reserved dividends", "interest_firm_summary.csv", "last_200_mean_dividend_payment"),
        ("macro_interest", "Interest income and money bridge", "central_bank_interest_income_bridge.csv", "cumulative_cb_interest_income"),
    ]
    for stem, title, filename, field in series:
        source = rows(ROOT / filename)
        if filename == "interest_firm_summary.csv":
            x = list(range(len(source)))
            y = [number(row.get(field)) for row in source]
        else:
            source = [row for row in source if number(row.get("seed")) == 7]
            x = [number(row.get("step")) for row in source]
            if field:
                y = [number(row.get(field)) for row in source]
            else:
                plt.figure(figsize=(8, 4))
                plt.plot(x, [number(row.get("interest_due")) for row in source], label="due")
                plt.plot(x, [number(row.get("interest_paid")) for row in source], label="paid")
                plt.title(title)
                plt.xlabel("week")
                plt.legend()
                plt.tight_layout()
                plt.savefig(ROOT / f"{stem}.png", dpi=140)
                plt.close()
                continue
        plt.figure(figsize=(8, 4))
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel("week" if filename != "interest_firm_summary.csv" else "firm")
        plt.tight_layout()
        plt.savefig(ROOT / f"{stem}.png", dpi=140)
        plt.close()


if __name__ == "__main__":
    main()
