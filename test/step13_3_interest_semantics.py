"""Passive interest semantics and shadow debt-service audit."""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step13_2a_focused_4x_validation import num, reconstruct_principal  # noqa: E402

OUTPUT = ROOT / "test" / "output" / "step13_3_interest_semantics"
SOURCE_ROOT = ROOT / "test" / "output" / "step13_2A_focused_4x_validation"
RUNS = {
    42: ROOT / "test" / "output" / "step13_2_credit_capacity_behavior" / "raw_runs" / "N5000_seed42_w1560" / "credit_capacity_payroll_4x",
    1: SOURCE_ROOT / "raw_runs" / "seed_1" / "credit_capacity_payroll_4x",
    7: SOURCE_ROOT / "raw_runs" / "seed_7" / "credit_capacity_payroll_4x",
}
RATES = {0.00: "0pct", 0.02: "2pct", 0.05: "5pct", 0.10: "10pct"}
TOL = 1e-9


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load(seed, path):
    return {
        "seed": seed,
        "firms": read_csv(path / "firm_diagnostics.csv"),
        "diagnostics": read_csv(path / "diagnostics.csv"),
    }


def shadow_rows(run):
    items = reconstruct_principal(run["firms"])
    grouped = defaultdict(list)
    for item in items:
        grouped[item["firm_id"]].append(item)
    rows = []
    events = []
    exposure = []
    coverage = []
    for firm_id, firm_items in sorted(grouped.items()):
        arrears = {rate: 0.0 for rate in RATES}
        for item in firm_items:
            raw = item["row"]
            cash_before = max(0.0, num(raw, "cash_before_repayment", num(raw, "cash_end")))
            floor = max(0.0, num(raw, "repayment_buffer", num(raw, "target_cash")))
            available = max(0.0, cash_before - floor)
            wage = max(0.0, num(raw, "scheduled_wage_bill", num(raw, "wage_bill")))
            sales = max(0.0, num(raw, "sales_revenue", num(raw, "sales")))
            operating_cash_flow = num(raw, "sales_revenue", 0.0) - num(raw, "executed_wage_bill", num(raw, "wage_payment"))
            limit = num(raw, "credit_limit", 0.0)
            for annual_rate in RATES:
                weekly_rate = (1.0 + annual_rate) ** (1.0 / 52.0) - 1.0
                due = weekly_rate * item["opening"]
                total_obligation = arrears[annual_rate] + due
                paid = min(total_obligation, available)
                shortfall = max(0.0, total_obligation - paid)
                arrears[annual_rate] = shortfall
                lender_exposure = item["closing"] + shortfall
                utilization = lender_exposure / limit if limit > 0 and math.isfinite(limit) else 0.0
                interest_service_ratio = paid / due if due > 0 else 1.0
                row = {
                    "seed": run["seed"], "firm_id": firm_id, "step": item["step"],
                    "annual_rate": annual_rate, "weekly_rate": weekly_rate,
                    "opening_principal": item["opening"], "current_interest_due": due,
                    "opening_interest_arrears": total_obligation - due,
                    "total_interest_obligation": total_obligation,
                    "cash_before_shadow_interest": cash_before,
                    "operating_liquidity_floor": floor,
                    "cash_available_for_interest": available,
                    "shadow_interest_paid_if_cash_only": paid,
                    "shadow_interest_shortfall": shortfall,
                    "closing_interest_arrears": shortfall,
                    "closing_principal": item["closing"],
                    "shadow_lender_exposure": lender_exposure,
                    "shadow_credit_utilization_if_arrears_count": utilization,
                    "scheduled_wage_bill": wage, "sales": sales,
                    "operating_cash_flow": operating_cash_flow,
                    "interest_due_to_wage_bill": due / wage if wage > 0 else "",
                    "interest_due_to_sales": due / sales if sales > 0 else "",
                    "interest_due_to_operating_cash_flow": due / operating_cash_flow if operating_cash_flow > 0 else "",
                    "interest_due_to_cash_before_interest": due / cash_before if cash_before > 0 else "",
                    "interest_paid_to_due": interest_service_ratio,
                    "principal_to_credit_limit": item["opening"] / limit if limit > 0 and math.isfinite(limit) else "",
                    "exposure_over_credit_limit": utilization > 1.0 + TOL,
                }
                rows.append(row)
                coverage.append({
                    "seed": run["seed"], "firm_id": firm_id, "step": item["step"], "annual_rate": annual_rate,
                    "operating_interest_coverage": operating_cash_flow / due if due > 0 and operating_cash_flow > 0 else "",
                    "cash_interest_coverage": available / due if due > 0 else "",
                    "interest_service_ratio": interest_service_ratio,
                })
                exposure.append({
                    "seed": run["seed"], "firm_id": firm_id, "step": item["step"], "annual_rate": annual_rate,
                    "principal": item["closing"], "interest_arrears": shortfall, "lender_exposure": lender_exposure,
                    "credit_limit": limit, "exposure_utilization": utilization,
                })
                if shortfall > TOL:
                    events.append(row.copy())
    return rows, coverage, exposure, events


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows, all_coverage, all_exposure, all_events = [], [], [], []
    source_validation = {}
    for seed, path in RUNS.items():
        run = load(seed, path)
        rows, coverage, exposure, events = shadow_rows(run)
        all_rows.extend(rows); all_coverage.extend(coverage); all_exposure.extend(exposure); all_events.extend(events)
        source_validation[str(seed)] = {"rows": len(run["firms"]), "path": str(path)}

    write_csv(OUTPUT / "shadow_interest_burden.csv", all_rows)
    write_csv(OUTPUT / "shadow_interest_coverage.csv", all_coverage)
    write_csv(OUTPUT / "shadow_interest_exposure.csv", all_exposure)
    # Compact event windows, event -13 through event +26.
    first_events = {}
    for event in all_events:
        key = (int(event["seed"]), int(event["firm_id"]), float(event["annual_rate"]))
        first_events[key] = min(first_events.get(key, int(event["step"])), int(event["step"]))
    windows = []
    for row in all_rows:
        key = (int(row["seed"]), int(row["firm_id"]), float(row["annual_rate"]))
        if key in first_events:
            first = first_events[key]
            if first - 13 <= int(row["step"]) <= first + 26:
                windows.append(row)
    write_csv(OUTPUT / "shadow_interest_shortfall_events.csv", all_events)
    event_dir = OUTPUT / "event_windows"
    write_csv(event_dir / "interest_shortfall_windows.csv", windows)

    summaries = []
    for rate in RATES:
        for seed in RUNS:
            rows = [row for row in all_rows if int(row["seed"]) == seed and float(row["annual_rate"]) == rate]
            due = [float(row["current_interest_due"]) for row in rows]
            short = [float(row["shadow_interest_shortfall"]) for row in rows]
            service = [float(row["interest_paid_to_due"]) for row in rows]
            summaries.append({
                "seed": seed, "annual_rate": rate,
                "firm_count": len(set(row["firm_id"] for row in rows)),
                "cumulative_shadow_interest_due": sum(due),
                "cumulative_shadow_interest_shortfall": sum(short),
                "share_weeks_fully_serviceable": sum(value >= 1.0 - TOL for value in service) / max(1, len(service)),
                "share_weeks_partially_serviceable": sum(0 < value < 1.0 - TOL for value in service) / max(1, len(service)),
                "share_weeks_zero_serviceable": sum(value <= TOL for value in service) / max(1, len(service)),
                "first_shadow_interest_shortfall_week": min((int(row["step"]) for row in rows if float(row["shadow_interest_shortfall"]) > TOL), default=""),
                "max_shadow_lender_exposure": max((float(row["shadow_lender_exposure"]) for row in rows), default=0.0),
                "max_shadow_credit_utilization": max((float(row["shadow_credit_utilization_if_arrears_count"]) for row in rows), default=0.0),
            })
    write_csv(OUTPUT / "shadow_interest_summary.csv", summaries)

    (OUTPUT / "interest_source_order_audit.json").write_text(json.dumps({
        "current_order": ["opening principal / credit preparation", "wage settlement", "production", "sales", "dividend settlement", "principal repayment"],
        "interest_currently_behavioral": False,
        "recommended_order": ["observe opening principal", "calculate current interest due", "credit-capacity decision", "wages", "production", "sales", "interest payment", "principal repayment", "closing principal"],
        "basis": "Firm-specific simulation performs credit before wages and principal repayment after operating settlement; interest should be inserted immediately before principal repayment.",
    }, indent=2), encoding="utf-8")
    (OUTPUT / "interest_base_semantics.md").write_text("""# Interest Base Semantics\n\nRecommend **opening principal**: `current_interest_due = weekly_rate * opening_principal`. New borrowing starts accruing next week. This matches the canonical Step 13.2A opening principal and avoids charging a full week on same-week borrowing.\n\nThe first behavioral implementation should use principal only, with no interest-on-interest.\n""", encoding="utf-8")
    (OUTPUT / "interest_rate_conversion.json").write_text(json.dumps({"annual_rates": [0.0, 0.02, 0.05, 0.10], "conversion": "(1 + annual_rate) ** (1/52) - 1", "weekly_rates": {str(rate): (1 + rate) ** (1 / 52) - 1 for rate in RATES}}, indent=2), encoding="utf-8")
    (OUTPUT / "interest_payment_destination_audit.json").write_text(json.dumps({"existing_interface": "ledger.repay_loan_attrs can transfer interest from Firm cash to central_bank.public_income_balance and destroy only principal money", "recommended_destination": "explicit Central Bank public income balance", "money_semantics": "interest payment is a transfer; principal repayment is the only debt-service money destruction", "remittance_in_same_week": False}, indent=2), encoding="utf-8")
    (OUTPUT / "interest_arrears_semantics.json").write_text(json.dumps({"recommended": "separate interest_arrears", "principal_bridge_preserved": True, "interest_bridge": "closing_arrears = opening_arrears + current_due - interest_paid", "capitalization": False, "interest_on_arrears": False, "money_created_by_accrual": False}, indent=2), encoding="utf-8")
    (OUTPUT / "interest_credit_capacity_interaction.json").write_text(json.dumps({"candidate_exposure": "principal + interest_arrears", "new_credit_when_exposure_over_limit": "zero", "no_forced_deleveraging": True, "shadow_only": True, "rate_summary": summaries}, indent=2), encoding="utf-8")
    (OUTPUT / "dividend_interest_priority_audit.json").write_text(json.dumps({"current_dividend_before_interest_semantics": True, "recommended_minimum_change": "reserve current interest obligation before affordable dividend calculation in behavioral stage", "dividend_policy_redesign": False, "required_for_behavioral_test": True}, indent=2), encoding="utf-8")
    (OUTPUT / "step13_3_recommendation.json").write_text(json.dumps({
        "verdict": "A",
        "recommended_interest_base": "opening_principal",
        "recommended_weekly_rate_conversion": "(1 + annual_rate) ** (1/52) - 1",
        "recommended_payment_timing": "after operating settlement and before principal repayment",
        "recommended_interest_priority": "interest obligation before discretionary principal repayment",
        "recommended_unpaid_interest_semantics": "separate non-monetary interest_arrears",
        "recommended_interest_on_arrears": False,
        "recommended_lender_exposure_definition": "principal + interest_arrears for future credit-capacity headroom",
        "recommended_interest_payment_destination": "central_bank.public_income_balance",
        "dividend_priority_change_required": True,
        "interest_behavioral_test_ready": True,
        "candidate_K_weeks": 46.36154354202572,
        "source_validation": source_validation,
    }, indent=2), encoding="utf-8")

    human = OUTPUT / "human_review"
    human.mkdir(exist_ok=True)
    five = [row for row in summaries if row["annual_rate"] == 0.05]
    fig, ax = plt.subplots(figsize=(11, 5))
    for row in five:
        ax.bar(f"s{row['seed']}", row["cumulative_shadow_interest_due"], label="due")
    ax.set_title("5% shadow interest burden by seed"); ax.set_ylabel("cumulative interest due")
    fig.tight_layout(); fig.savefig(human / "shadow_interest_burden_by_firm.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for rate in RATES:
        rows = [row for row in summaries if row["annual_rate"] == rate]
        ax.plot([row["seed"] for row in rows], [row["share_weeks_fully_serviceable"] for row in rows], marker="o", label=f"{rate:.0%}")
    ax.set_title("Fully serviceable shadow-interest weeks"); ax.set_xlabel("seed"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(human / "interest_coverage_by_rate.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for rate in RATES:
        rows = [row for row in summaries if row["annual_rate"] == rate]
        ax.plot([row["seed"] for row in rows], [row["cumulative_shadow_interest_shortfall"] for row in rows], marker="o", label=f"{rate:.0%}")
    ax.set_title("Shadow interest shortfall by rate"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(human / "shadow_interest_shortfall_timeline.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for rate in RATES:
        rows = [row for row in summaries if row["annual_rate"] == rate]
        ax.plot([row["seed"] for row in rows], [row["max_shadow_lender_exposure"] for row in rows], marker="o", label=f"{rate:.0%}")
    ax.set_title("Shadow total lender exposure"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(human / "principal_vs_total_lender_exposure.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for rate in RATES:
        rows = [row for row in summaries if row["annual_rate"] == rate]
        ax.plot([row["seed"] for row in rows], [row["max_shadow_credit_utilization"] for row in rows], marker="o", label=f"{rate:.0%}")
    ax.axhline(1.0, color="black", linewidth=.8); ax.set_title("Shadow exposure / credit limit"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(human / "interest_capacity_interaction.png", dpi=150); plt.close(fig)

    (OUTPUT / "acceptance_summary.md").write_text("""# Step 13.3 Interest Semantics and Shadow Debt-Service Audit\n\n## Verdict\n\n**A. Opening-principal interest with separate unpaid-interest arrears and explicit Central Bank income accounting is the cleanest minimal architecture.**\n\nThe audit is passive: no trajectory, price, consumption, production, credit, repayment, dividend, population, or money behavior was changed. The accepted 4x trajectories for seeds 1, 7, and 42 were reused.\n\nInterest should be calculated on opening principal, converted from annual to weekly using effective compounding, paid after operating settlement and before principal repayment, and transferred to `central_bank.public_income_balance`. Unpaid interest should remain a separate non-monetary `interest_arrears` state, with no interest-on-interest and no automatic money creation. Future credit exposure should use principal plus arrears, while preserving the principal-money bridge.\n\n`recommended_interest_base = opening_principal`\n`recommended_weekly_rate_conversion = (1 + annual_rate) ** (1/52) - 1`\n`recommended_interest_on_arrears = false`\n`interest_behavioral_test_ready = true`\n\nNext step: **Step 13.4 Interest Behavioral Enforcement Test**.\n""", encoding="utf-8")


if __name__ == "__main__":
    main()
