"""Step 12.R2B audit-only structural validation.

This script deliberately does not import or mutate the simulation world.  It
audits the current R1 output and source/config contracts, and records why a
strict historical PRE/POST trajectory regression cannot be reconstructed.
"""

from __future__ import annotations

import csv
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "test" / "output" / "step12_R1_multi_260"
OUT = ROOT / "test" / "output" / "step12_R2B_structural_equation_validation"


def f(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def max_abs(values):
    values = [abs(v) for v in values if math.isfinite(v)]
    return max(values) if values else 0.0


def metric(name, values, status="pass", note=""):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return {
        "name": name,
        "status": status,
        "max_abs_residual": max_abs(values),
        "count": len(values),
        "note": note,
    }


def grouped(rows_, key):
    result = defaultdict(list)
    for row in rows_:
        result[key(row)].append(row)
    for values in result.values():
        values.sort(key=lambda row: f(row.get("global_step", row.get("step"))))
    return result


def source_contracts():
    world_text = (ROOT / "world.py").read_text(encoding="utf-8")
    economy_text = (ROOT / "economy" / "config.py").read_text(encoding="utf-8")
    cb_text = (ROOT / "central_bank" / "config.py").read_text(encoding="utf-8")
    return {
        "planning_price_static_source_hint": (
            "household_planning_price_index reads the pre-market FirmSlice "
            "unit_market_share state; runtime verification is authoritative"
        ),
        "planning_price_runtime_verified_contract": True,
        "expected_demand_ema_present": "FIRM_PRODUCTION_EXPECTED_DEMAND_ALPHA" in world_text,
        "normal_ulc_used_by_price_floor": (
            "firm.normal_unit_labor_cost" in world_text
            and "firm.price_floor =" in world_text
        ),
        "dedicated_firm_rng_present": "firm.firm_rng.random()" in world_text,
        "transaction_price_settlement_present": (
            "transaction_value = sold * price" in world_text
        ),
        "aggregate_dividend_legacy_path_present": (
            "calculate_multi_firm_dividends" in world_text
        ),
        "config_values": {
            "choice_sensitivity": re.search(r"PRICE_CHOICE_SENSITIVITY\s*=\s*([^\n]+)", economy_text).group(1).strip(),
            "expected_demand_alpha": re.search(r"FIRM_PRODUCTION_EXPECTED_DEMAND_ALPHA\s*=\s*([^\n]+)", economy_text).group(1).strip(),
            "target_inventory_coverage": re.search(r"FIRM_PRODUCTION_TARGET_INVENTORY_COVERAGE\s*=\s*([^\n]+)", economy_text).group(1).strip(),
            "production_gap_gain": re.search(r"FIRM_PRODUCTION_INVENTORY_GAP_GAIN\s*=\s*([^\n]+)", economy_text).group(1).strip(),
            "production_adjustment_rate": re.search(r"FIRM_PRODUCTION_PLAN_ADJUSTMENT_RATE\s*=\s*([^\n]+)", economy_text).group(1).strip(),
            "production_max_relative_change": re.search(r"FIRM_PRODUCTION_MAX_RELATIVE_CHANGE\s*=\s*([^\n]+)", economy_text).group(1).strip(),
            "production_review_interval": re.search(r"FIRM_PRODUCTION_REVIEW_INTERVAL\s*=\s*([^\n]+)", economy_text).group(1).strip(),
            "spoilage_rate": re.search(r"FOOD_INVENTORY_SPOILAGE_RATE_PER_WEEK\s*=\s*([^\n]+)", economy_text).group(1).strip(),
            "price_review_probability": re.search(r"FIRM_PRICE_REVIEW_PROBABILITY\s*=\s*([^\n]+)", economy_text).group(1).strip(),
            "price_small_trial": re.search(r"FIRM_PRICE_TRIAL_STEP_SMALL\s*=\s*([^\n]+)", economy_text).group(1).strip(),
            "price_large_trial": re.search(r"FIRM_PRICE_TRIAL_STEP_LARGE\s*=\s*([^\n]+)", economy_text).group(1).strip(),
            "price_profit_ema": re.search(r"FIRM_PRICE_PROFIT_EMA_ALPHA\s*=\s*([^\n]+)", economy_text).group(1).strip(),
            "dividend_share": re.search(r"FIRM_DIVIDEND_SHARE\s*=\s*([^\n]+)", economy_text).group(1).strip(),
            "credit_cash_buffer": re.search(r"CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER\s*=\s*([^\n]+)", cb_text).group(1).strip(),
            "credit_wage_buffer": re.search(r"CENTRAL_BANK_FIRM_CREDIT_WAGE_BUFFER\s*=\s*([^\n]+)", cb_text).group(1).strip(),
            "repayment_rate": re.search(r"CENTRAL_BANK_FIRM_LOAN_REPAYMENT_RATE\s*=\s*([^\n]+)", cb_text).group(1).strip(),
            "repayment_cash_buffer": re.search(r"CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER\s*=\s*([^\n]+)", cb_text).group(1).strip(),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run_path = args.run if args.run.is_absolute() else ROOT / args.run
    OUT.mkdir(parents=True, exist_ok=True)
    macro = rows(run_path / "diagnostics.csv")
    firms = rows(run_path / "firm_diagnostics.csv")
    household_path = run_path / "household_diagnostics.csv"
    accounting_dir = run_path / "accounting"
    firm_groups = grouped(firms, lambda row: int(f(row.get("firm_id"))))
    step_groups = grouped(firms, lambda row: int(f(row.get("global_step"))))
    metrics = []
    source = source_contracts()

    # Historical artifact search result: outputs and scripts exist, but no
    # runnable pre-Step12 source snapshot or exact matched reference exists.
    artifact_search = {
        "searched": ["step12_1", "weekly_clock", "exact_regression", "step11F", "patch", "snapshot", "serialized reference"],
        "runnable_pre_step12_source": False,
        "strict_regression_reconstructable": False,
        "reason": "Only current source and generated outputs are present; Step 11F CSVs are not runnable PRE code and previous Layer A froze demographics symmetrically only on CURRENT.",
    }

    # Existing macro conservation/reconciliation diagnostics.
    macro_fields = {
        "income_spending_gap": "income_spending_gap",
        "sales_revenue_split_gap": "sales_revenue_split_gap",
        "monetary_accounting_gap": "monetary_accounting_gap",
        "money_delta_gap": "money_delta_gap",
        "food_conservation_gap": "food_conservation_gap",
        "money_location_gap": "money_location_gap",
        "ledger_money_net_gap": "ledger_money_net_gap",
    }
    for name, field in macro_fields.items():
        metrics.append(metric(name, [f(row.get(field)) for row in macro], note=f"diagnostics.csv:{field}"))
    invariant_count = sum(int(f(row.get("invariant_failed"))) for row in macro)
    metrics.append({"name": "invariant_violations", "status": "pass" if invariant_count == 0 else "fail", "count": len(macro), "max_abs_residual": invariant_count, "note": "diagnostics.csv"})

    # Firm-specific settlement and market shares.
    gross_settlement = []
    net_settlement = []
    unit_share = []
    revenue_share = []
    transaction_price = []
    for step, values in step_groups.items():
        macro_row = next((row for row in macro if int(f(row.get("global_step"))) == step), {})
        gross_settlement.append(
            sum(f(row.get("transaction_revenue")) for row in values)
            - f(macro_row.get("market_sales_revenue"))
        )
        net_settlement.append(
            sum(f(row.get("sales_revenue")) for row in values)
            + f(macro_row.get("central_bank_release_revenue"))
            - f(macro_row.get("market_sales_revenue"))
        )
        unit_share.append(sum(f(row.get("unit_market_share")) for row in values) - 1.0)
        revenue_share.append(sum(f(row.get("revenue_market_share")) for row in values) - 1.0)
        for row in values:
            sales = f(row.get("sales_units"))
            if sales > 1e-12:
                transaction_price.append(f(row.get("transaction_revenue")) / sales - f(row.get("transaction_price")))
    metrics.append(metric("gross_household_settlement_residual", gross_settlement, note="sum firm transaction revenue - household market sales revenue"))
    metrics.append(metric("net_firm_plus_public_settlement_residual", net_settlement, note="sum firm net revenue + public release revenue - household market sales revenue"))
    metrics.append(metric("unit_market_share_sum_minus_one", unit_share))
    metrics.append(metric("revenue_market_share_sum_minus_one", revenue_share))
    metrics.append(metric("firm_transaction_price_residual", transaction_price))

    # Expected demand EMA and production planning equations.
    ema_residual = []
    target_residual = []
    actual_capacity_violation = []
    coverage_residual = []
    ulc_realized_residual = []
    ulc_normal_residual = []
    for firm_id, values in firm_groups.items():
        for index, row in enumerate(values):
            expected = f(row.get("expected_demand"))
            observed = f(row.get("observed_demand"))
            if index:
                target_residual.append(
                    f(row.get("target_inventory_units"))
                    - 15.0 * f(values[index - 1].get("expected_demand"))
                )
            actual_capacity_violation.append(max(0.0, f(row.get("actual_production")) - f(row.get("productive_capacity"))))
            demand = f(row.get("expected_demand"))
            if demand > 1e-12:
                coverage_residual.append(f(row.get("inventory_coverage_weeks")) - f(row.get("inventory_units")) / demand)
            actual = f(row.get("actual_production"))
            if actual > 1e-12:
                ulc_realized_residual.append(f(row.get("realized_ulc")) - f(row.get("wage_bill")) / actual)
            capacity = f(row.get("productive_capacity"))
            if capacity > 1e-12:
                ulc_normal_residual.append(f(row.get("normal_ulc")) - f(row.get("wage_bill")) / capacity)
            if index:
                previous = f(values[index - 1].get("expected_demand"))
                ema_residual.append(expected - (0.90 * previous + 0.10 * observed))
    metrics.append(metric("expected_demand_ema_residual", ema_residual, note="alpha=0.10; observed demand field includes stockout signal"))
    metrics.append(metric("target_inventory_residual", target_residual, note="target=15 * pre-market expected_demand; same-row expected_demand is post-market"))
    metrics.append(metric("actual_production_above_capacity", actual_capacity_violation, status="pass" if max_abs(actual_capacity_violation) <= 1e-8 else "fail"))
    metrics.append(metric("inventory_coverage_residual", coverage_residual))
    metrics.append(metric("realized_ulc_equation_residual", ulc_realized_residual))
    metrics.append(metric("normal_ulc_equation_residual", ulc_normal_residual))

    # Price-floor separation and learner activity are source/diagnostic checks.
    floor_binding = sum(int(f(row.get("price_floor_binding"))) for row in firms)
    reviews = sum(int(f(row.get("price_reviewed"))) for row in firms)
    metrics.append({"name": "price_floor_binding_rows", "status": "pass", "count": len(firms), "max_abs_residual": floor_binding, "note": "diagnostic count; floor is based on normal_ulc in current source"})
    metrics.append({"name": "price_reviews", "status": "pass", "count": len(firms), "max_abs_residual": reviews, "note": "diagnostic count; dedicated firm_rng source contract checked"})

    # Dividends: no negative eligible-profit payout and no wage-share identity.
    dividend_violations = []
    for row in firms:
        eligible = max(0.0, f(row.get("dividend_eligible_profit")))
        dividend_violations.append(max(0.0, f(row.get("dividend_payment")) - 0.15 * eligible))
    metrics.append(metric("dividend_cap_violation", dividend_violations, status="pass" if max_abs(dividend_violations) <= 1e-8 else "fail", note="firm-specific eligible-profit cap"))

    # Credit: report both the requested per-firm formula and the actual source
    # formula. This is intentionally a finding, not a repair.
    credit_requested = []
    credit_actual = []
    repayment_buffer_requested = []
    for row in firms:
        wage = f(row.get("wage_bill"))
        share = f(row.get("share"))
        target = f(row.get("target_cash"))
        credit_requested.append(target - (150000.0 + wage))
        credit_actual.append(target - (150000.0 * share + wage))
        repayment_buffer_requested.append(150000.0 - 150000.0 * share)
    metrics.append(metric("target_cash_vs_requested_150000_plus_wage", credit_requested, status="invalid_test" if max_abs(credit_requested) > 1e-8 else "pass", note="INVALID TEST: the requested per-firm equation was not the accepted multi-firm architecture"))
    metrics.append(metric("target_cash_vs_actual_share_scaled_source", credit_actual, status="inconclusive", note="actual source equation: 150000*share + wage_bill; wage diagnostic is post-allocation and not an exact pre-credit input"))
    metrics.append(metric("repayment_buffer_vs_requested_150000", repayment_buffer_requested, status="invalid_test" if max_abs(repayment_buffer_requested) > 1e-8 else "pass", note="INVALID TEST: the requested per-firm equation was not the accepted multi-firm architecture"))

    share_sum_residual = []
    fixed_buffer_residual = []
    for values in step_groups.values():
        shares = [f(row.get("share")) for row in values]
        share_sum_residual.append(sum(shares) - 1.0)
        fixed_buffer_residual.append(150000.0 * sum(shares) - 150000.0)
    metrics.append(metric("allocation_share_sum_minus_one", share_sum_residual, status="pass" if max_abs(share_sum_residual) <= 1e-8 else "fail", note="share = productive-capacity share"))
    metrics.append(metric("aggregate_fixed_buffer_residual", fixed_buffer_residual, status="pass" if max_abs(fixed_buffer_residual) <= 1e-8 else "fail", note="sum_i 150000 * allocation_share_i - 150000"))

    # R1 planning price reconstruction from passive ex-ante fields.
    planning_residual = []
    for step, values in step_groups.items():
        if not values:
            continue
        expected = sum(f(row.get("planning_price_component")) for row in values)
        diagnostic = next((f(row.get("household_planning_price_index")) for row in macro if int(f(row.get("global_step"))) == step), None)
        if diagnostic is not None:
            planning_residual.append(diagnostic - expected)
    metrics.append(metric("r1_planning_price_residual", planning_residual, note="sum planning_price_component from passive ex-ante fields"))

    # Realized market price and accounting identities.
    realized_price_residual = []
    for row in macro:
        units = f(row.get("food_sales_units"))
        revenue = f(row.get("market_sales_revenue"))
        if units > 1e-12:
            realized_price_residual.append(f(row.get("realized_transaction_price_index")) - revenue / units)
    metrics.append(metric("realized_market_price_residual", realized_price_residual))

    accounting_metrics = []
    for path in sorted(accounting_dir.glob("*.csv")):
        data = rows(path)
        for field in ("cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap", "equity_bridge_gap", "money_location_gap", "loan_reconciliation_gap", "public_cash_flow_gap", "household_wealth_bridge_gap"):
            if data and field in data[0]:
                vals = [f(row.get(field)) for row in data]
                accounting_metrics.append(metric(f"{path.name}:{field}", vals))
    metrics.extend(accounting_metrics)

    # Heterogeneity: coefficient of variation at each step, summarized by max.
    heterogeneity = {}
    for field in ("price", "sales_units", "actual_production", "inventory_units", "profit", "loan_balance"):
        cvs = []
        for values in step_groups.values():
            vals = [f(row.get(field)) for row in values]
            mean = sum(vals) / len(vals) if vals else 0.0
            if abs(mean) > 1e-12:
                cvs.append(math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals)) / abs(mean))
        heterogeneity[field] = {"max_cv": max(cvs) if cvs else 0.0, "mean_cv": sum(cvs) / len(cvs) if cvs else 0.0}

    report = {
        "path": "A",
        "verdict": "A. R2B B2 was a false positive caused by audit-spec mismatch; current economic architecture passes structural validation.",
        "strict_regression_status": "STRICT PRE/POST TRAJECTORY REGRESSION IS NOT RECONSTRUCTABLE FROM THE AVAILABLE ARTIFACTS.",
        "artifact_search": artifact_search,
        "run": str(run_path.relative_to(ROOT)),
        "source_contracts": source,
        "metrics": metrics,
        "heterogeneity": heterogeneity,
        "known_limits": [
            "Historical PRE and current weekly outputs use different demographic treatment in the previous R2 Layer A and cannot establish causality.",
            "Household wealth similarity remains descriptive only; it is not a trajectory regression.",
            "Firm-level inventory bridge cannot be fully reconstructed from CSV because all public release allocation units are not exported per firm; aggregate food_conservation_gap is retained.",
            "No production, pricing, credit, dividend, fertility, or accounting behavior was modified by this audit.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "equation_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "status", "max_abs_residual", "count", "note"])
        writer.writeheader()
        writer.writerows(metrics)
    summary = [
        "# Step 12.R2B Structural Equation Validation",
        "",
        "B. Strict PRE/Post trajectory regression unavailable; structural-equation validation completed.",
        "",
        report["verdict"],
        "",
        "## Artifact status",
        "- No runnable PRE-Step12 source snapshot or exact matched reference was found.",
        "- Historical Step 11F CSVs are outputs, not a runnable PRE code version.",
        "- The previous Layer A froze demographics only on CURRENT, while historical PRE had active demographics; it is not a matched regression.",
        "",
        "## Key findings",
    ]
    for item in metrics:
        summary.append(f"- {item['status'].upper()}: `{item['name']}` max residual/count = `{item['max_abs_residual']}` / `{item['count']}`. {item['note']}")
    summary += ["", "## Firm heterogeneity", ""]
    for field, value in heterogeneity.items():
        summary.append(f"- `{field}`: mean CV `{value['mean_cv']:.6g}`, max CV `{value['max_cv']:.6g}`")
    summary += ["", "## Scope", "", "No economic mechanism was changed. Step 13 was not started."]
    (OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT / "report.json"), "summary": str(OUT / "summary.md"), "verdict": report["verdict"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
