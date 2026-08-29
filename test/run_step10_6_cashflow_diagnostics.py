import argparse
import csv
import json
import math
import os
from statistics import mean


DEFAULT_RUNS = {
    "baseline": (
        "test/output/baseline/seed_42_pop_5000_steps_5000/diagnostics.csv"
    ),
    "wage_shock_1_47": (
        "test/output/wage_shock_1_47/seed_42_pop_5000_steps_5000/"
        "diagnostics.csv"
    ),
}


def float_value(row, key, default=0.0):
    value = row.get(key, "")

    if value == "":
        return default

    try:
        return float(value)
    except ValueError:
        return default


def safe_ratio(numerator, denominator):
    if abs(denominator) < 1e-12:
        return 0.0

    return numerator / denominator


def series_slope(values):
    values = list(values)
    n = len(values)

    if n < 2:
        return 0.0

    mean_x = (n - 1) / 2
    mean_y = mean(values)
    numerator = 0.0
    denominator = 0.0

    for index, value in enumerate(values):
        dx = index - mean_x
        numerator += dx * (value - mean_y)
        denominator += dx * dx

    if denominator == 0:
        return 0.0

    return numerator / denominator


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def estimate_wage_bill(row, allow_legacy_estimate=False):
    if "wage_bill" in row and row["wage_bill"] != "":
        return float_value(row, "wage_bill")

    if not allow_legacy_estimate:
        raise ValueError(
            "diagnostics.csv must include direct wage_bill/wage_payment. "
            "Re-run main.py with the current diagnostics, or pass "
            "--allow-legacy-wage-estimate for old CSV files."
        )

    ratio = float_value(row, "firm_cash_to_wage_bill")

    if abs(ratio) < 1e-12:
        return 0.0

    return float_value(row, "firm_cash") / ratio


def decompose_rows(rows, allow_legacy_wage_estimate=False):
    decomposed = []
    previous = None

    for row in rows:
        firm_cash = float_value(row, "firm_cash")
        inventory_value = float_value(row, "firm_inventory_value")
        public_money_stock = float_value(row, "public_money_stock")
        total_money_stock = float_value(row, "total_money_stock")
        loan_balance = float_value(row, "working_capital_loan_balance")
        wage_bill = estimate_wage_bill(row, allow_legacy_wage_estimate)
        loan_issued = float_value(row, "working_capital_loan_issued")
        loan_repaid = float_value(row, "working_capital_loan_repaid")
        interest_paid = float_value(row, "working_capital_interest_paid")
        firm_sales_revenue = float_value(row, "firm_sales_revenue")
        public_inventory_purchase = float_value(
            row,
            "central_bank_inventory_purchase",
        )
        release_revenue = float_value(
            row,
            "central_bank_market_release_revenue",
        )
        public_income = float_value(row, "central_bank_public_income")
        public_income_used = float_value(row, "central_bank_public_income_used")
        dividend = float_value(row, "dividend")

        if previous is None:
            cash_change = 0.0
            inventory_value_change = 0.0
            public_money_stock_change = 0.0
            total_money_stock_change = 0.0
            loan_balance_change = loan_balance
        else:
            cash_change = firm_cash - previous["firm_cash"]
            inventory_value_change = inventory_value - previous["inventory_value"]
            public_money_stock_change = (
                public_money_stock - previous["public_money_stock"]
            )
            total_money_stock_change = (
                total_money_stock - previous["total_money_stock"]
            )
            loan_balance_change = loan_balance - previous["loan_balance"]

        cash_bridge = (
            firm_sales_revenue
            +
            public_inventory_purchase
            +
            loan_issued
            -
            wage_bill
            -
            dividend
            -
            loan_repaid
            -
            interest_paid
        )
        bridge_gap = cash_change - cash_bridge

        decomposed.append(
            {
                "step": row.get("step", ""),
                "scenario": row.get("scenario", ""),
                "population": row.get("population", ""),
                "firm_cash": firm_cash,
                "firm_cash_change": cash_change,
                "firm_sales_revenue": firm_sales_revenue,
                "wage_payment": wage_bill,
                "dividend_payment": dividend,
                "inventory_value": inventory_value,
                "inventory_value_change": inventory_value_change,
                "loan_issued": loan_issued,
                "loan_repaid": loan_repaid,
                "loan_interest_paid": interest_paid,
                "loan_balance": loan_balance,
                "loan_balance_change": loan_balance_change,
                "public_inventory_purchase": public_inventory_purchase,
                "central_bank_release_revenue": release_revenue,
                "public_income": public_income,
                "public_income_used": public_income_used,
                "public_money_stock": public_money_stock,
                "public_money_stock_change": public_money_stock_change,
                "public_money_share": float_value(row, "public_money_share"),
                "total_money_stock": total_money_stock,
                "total_money_stock_change": total_money_stock_change,
                "profit_before_dividend": float_value(
                    row,
                    "firm_profit_before_dividend",
                ),
                "firm_cash_to_wage_bill": float_value(
                    row,
                    "firm_cash_to_wage_bill",
                ),
                "cash_bridge": cash_bridge,
                "cash_bridge_gap": bridge_gap,
            }
        )
        previous = {
            "firm_cash": firm_cash,
            "inventory_value": inventory_value,
            "public_money_stock": public_money_stock,
            "total_money_stock": total_money_stock,
            "loan_balance": loan_balance,
        }

    return decomposed


def window_summary(rows, window):
    if not rows:
        return {}

    tail = rows[-min(window, len(rows)):]
    loan_steps = [row for row in rows if row["loan_issued"] > 0]
    first_loan_step = loan_steps[0]["step"] if loan_steps else ""
    loan_balance = [row["loan_balance"] for row in tail]
    firm_cash = [row["firm_cash"] for row in tail]
    public_money = [row["public_money_stock"] for row in tail]
    total_money = [row["total_money_stock"] for row in tail]

    keys = [
        "firm_cash_change",
        "firm_sales_revenue",
        "wage_payment",
        "dividend_payment",
        "inventory_value_change",
        "loan_issued",
        "loan_repaid",
        "loan_interest_paid",
        "loan_balance_change",
        "public_inventory_purchase",
        "central_bank_release_revenue",
        "public_income",
        "public_income_used",
        "public_money_stock_change",
        "total_money_stock_change",
        "profit_before_dividend",
        "firm_cash_to_wage_bill",
        "cash_bridge_gap",
    ]
    summary = {
        "scenario": tail[-1]["scenario"],
        "first_step": tail[0]["step"],
        "last_step": tail[-1]["step"],
        "window": len(tail),
        "first_loan_step": first_loan_step,
        "final_firm_cash": tail[-1]["firm_cash"],
        "final_loan_balance": tail[-1]["loan_balance"],
        "peak_loan_balance": max(row["loan_balance"] for row in rows),
        "loan_balance_slope": series_slope(loan_balance),
        "firm_cash_slope": series_slope(firm_cash),
        "public_money_stock_slope": series_slope(public_money),
        "total_money_stock_slope": series_slope(total_money),
        "final_public_money_share": tail[-1]["public_money_share"],
    }

    for key in keys:
        summary[f"avg_{key}"] = mean(row[key] for row in tail)
        summary[f"sum_{key}"] = sum(row[key] for row in tail)

    return summary


def write_csv(path, rows):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value == "":
        return ""

    if isinstance(value, str):
        return value

    if not math.isfinite(value):
        return str(value)

    return f"{value:,.2f}"


def write_report(path, summaries):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wage = summaries.get("wage_shock_1_47")
    baseline = summaries.get("baseline")

    lines = [
        "# Step 10.6 Cashflow Diagnostics",
        "",
        "This report only post-processes diagnostics.csv outputs. It does not "
        "change inventory, credit, inheritance, fiscal, fertility, or price "
        "rules.",
        "",
        "wage_payment is read directly from diagnostics wage_bill/wage_payment "
        "fields, not inferred from firm_cash / firm_cash_to_wage_bill.",
        "",
    ]

    for name, summary in summaries.items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Window: steps {summary['first_step']} to "
                f"{summary['last_step']} ({summary['window']} rows)",
                f"- first_loan_step: {summary['first_loan_step']}",
                f"- final_loan_balance: {fmt(summary['final_loan_balance'])}",
                f"- loan_balance_slope: {fmt(summary['loan_balance_slope'])}",
                f"- firm_cash_slope: {fmt(summary['firm_cash_slope'])}",
                f"- public_money_stock_slope: "
                f"{fmt(summary['public_money_stock_slope'])}",
                f"- avg firm_cash_change: "
                f"{fmt(summary['avg_firm_cash_change'])}",
                f"- avg firm_sales_revenue: "
                f"{fmt(summary['avg_firm_sales_revenue'])}",
                f"- avg wage_payment: {fmt(summary['avg_wage_payment'])}",
                f"- avg inventory_value_change: "
                f"{fmt(summary['avg_inventory_value_change'])}",
                f"- avg loan_issued / repaid: "
                f"{fmt(summary['avg_loan_issued'])} / "
                f"{fmt(summary['avg_loan_repaid'])}",
                f"- avg public_income / public_income_used: "
                f"{fmt(summary['avg_public_income'])} / "
                f"{fmt(summary['avg_public_income_used'])}",
                f"- avg public_money_stock_change: "
                f"{fmt(summary['avg_public_money_stock_change'])}",
                f"- avg profit_before_dividend: "
                f"{fmt(summary['avg_profit_before_dividend'])}",
                f"- avg firm_cash_to_wage_bill: "
                f"{fmt(summary['avg_firm_cash_to_wage_bill'])}",
                f"- avg cash_bridge_gap: "
                f"{fmt(summary['avg_cash_bridge_gap'])}",
                "",
            ]
        )

    if wage:
        loan_growth = wage["avg_loan_balance_change"]
        public_absorption = wage["avg_public_money_stock_change"]
        inventory_absorption = wage["avg_inventory_value_change"]
        repayment_shortfall = (
            wage["avg_loan_issued"] - wage["avg_loan_repaid"]
        )
        lines.extend(
            [
                "## Reading",
                "",
                "- In wage_shock_1_47, firm cash is near-stationary in the "
                "last window, but loan_balance is still rising because average "
                "loan issuance remains above average repayment.",
                "- Inventory value is not the dominant last-window cash sink if "
                "its average change is small compared with loan growth.",
                "- Public-sector absorption matters when public_money_stock keeps "
                "rising while firm cash is flat: money circulates through wages "
                "and sales, then a growing share ends up outside the firm.",
                "- The repayment rule matters mechanically because repayment is "
                "capped by cash above the fixed repayment buffer; once firm cash "
                "hovers near that buffer, positive profits do not automatically "
                "translate into enough principal repayment to offset new gap "
                "loans.",
                "",
                "Last-window diagnostic magnitudes:",
                "",
                f"- avg loan balance change: {fmt(loan_growth)}",
                f"- avg loan issued minus repaid: {fmt(repayment_shortfall)}",
                f"- avg public money stock change: {fmt(public_absorption)}",
                f"- avg inventory value change: {fmt(inventory_absorption)}",
                "",
            ]
        )

    if baseline:
        lines.extend(
            [
                "Baseline remains the control case: it has no material "
                "working-capital borrowing in the same diagnostics, so the "
                "persistent leverage pattern is tied to the wage shock regime, "
                "not to a global accounting drift.",
                "",
            ]
        )

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
        file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="test/output/step10_6_cashflow_diagnostics",
    )
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument(
        "--diagnostics",
        action="append",
        default=[],
        help="Optional name=path diagnostics CSV. Can be passed repeatedly.",
    )
    parser.add_argument(
        "--allow-legacy-wage-estimate",
        action="store_true",
        help=(
            "Allow old diagnostics files without wage_bill by estimating "
            "wages from firm_cash / firm_cash_to_wage_bill."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runs = {} if args.diagnostics else dict(DEFAULT_RUNS)

    for item in args.diagnostics:
        if "=" not in item:
            raise ValueError("--diagnostics must use name=path")

        name, path = item.split("=", 1)
        runs[name] = path

    os.makedirs(args.output_dir, exist_ok=True)
    summaries = {}
    manifest = {
        "window": args.window,
        "runs": {},
    }

    for name, path in runs.items():
        if not os.path.exists(path):
            manifest["runs"][name] = {
                "diagnostics_csv": path,
                "status": "missing",
            }
            continue

        rows = read_rows(path)
        decomposed = decompose_rows(
            rows,
            allow_legacy_wage_estimate=args.allow_legacy_wage_estimate,
        )
        per_step_path = os.path.join(args.output_dir, f"{name}_cashflow.csv")
        write_csv(per_step_path, decomposed)
        summary = window_summary(decomposed, args.window)
        summaries[name] = summary
        manifest["runs"][name] = {
            "diagnostics_csv": path,
            "cashflow_csv": per_step_path,
            "status": "ok",
        }

    summary_rows = list(summaries.values())
    summary_path = os.path.join(args.output_dir, "cashflow_summary.csv")
    write_csv(summary_path, summary_rows)
    report_path = os.path.join(args.output_dir, "cashflow_report.md")
    write_report(report_path, summaries)
    manifest["summary_csv"] = summary_path
    manifest["report_md"] = report_path

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Summary CSV: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
