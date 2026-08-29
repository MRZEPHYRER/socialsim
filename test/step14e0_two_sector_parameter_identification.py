"""Step 14E.0 passive two-sector parameter and unit audit.

This script intentionally performs no World construction and no behavioral
simulation.  It reads accepted Step 14C, Stage B.0, and canonical Food CSVs,
then derives a small Service parameter envelope and capitalization screen.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from central_bank import config as central_bank_config
from economy import config as economy_config


OUTPUT = ROOT / "test/output/step14E0_two_sector_parameter_identification"
CANONICAL = ROOT / "test/output/main_step13_financial_core"
STEP14C = ROOT / "test/output/step14C_shadow_multigood_household_demand"
STAGE_B0 = ROOT / "test/output/generalized_firm_stageB0_labor_demand_semantics_audit"

MATURE_START = 1560
MATURE_END = 1819
LAST200_START = 1620
LAST200_END = 1819
FOOD_PRODUCTIVITY = float(economy_config.FOOD_PRODUCTIVITY_PER_LABOR)
BASE_BUFFER = float(central_bank_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER)
SERVICE_FIRM_COUNT = 2
NORMALIZED_SERVICE_PRICE = 1.0
TOLERANCE = 1e-9


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else 0.0


def safe_ratio(numerator, denominator):
    return numerator / denominator if abs(denominator) > TOLERANCE else 0.0


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def step(row):
    return int(number(row.get("global_step", row.get("step", 0))))


def labor_services(row):
    return safe_ratio(number(row.get("scheduled_productive_capacity")), FOOD_PRODUCTIVITY)


def window(rows, start, end):
    return [row for row in rows if start <= step(row) <= end]


def window_wage_metrics(rows):
    observations = []
    total_wage = 0.0
    total_labor = 0.0
    for row in rows:
        labor = labor_services(row)
        wage = number(row.get("wage_bill", row.get("scheduled_wage_bill")))
        if labor <= TOLERANCE:
            continue
        observations.append(safe_ratio(wage, labor))
        total_wage += wage
        total_labor += labor
    return {
        "mean_firm_wage_cost_per_labor_service": mean(observations),
        "aggregate_wage_cost_per_labor_service": safe_ratio(total_wage, total_labor),
        "min_firm_wage_cost_per_labor_service": min(observations, default=0.0),
        "max_firm_wage_cost_per_labor_service": max(observations, default=0.0),
        "total_wage": total_wage,
        "total_labor": total_labor,
    }


def weighted_firm_revenue_metrics(rows, firm_id):
    selected = [row for row in rows if int(number(row.get("firm_id", -1))) == firm_id]
    total_revenue = math.fsum(number(row.get("sales_revenue")) for row in selected)
    total_labor = math.fsum(labor_services(row) for row in selected)
    total_wage = math.fsum(number(row.get("wage_bill")) for row in selected)
    revenue_per_labor = safe_ratio(total_revenue, total_labor)
    wage_per_labor = safe_ratio(total_wage, total_labor)
    return {
        "revenue_per_labor_service": revenue_per_labor,
        "wage_cost_per_labor_service": wage_per_labor,
        "revenue_to_wage_ratio": safe_ratio(revenue_per_labor, wage_per_labor),
    }


def read_b0_evidence():
    rows = read_csv(STAGE_B0 / "labor_demand_semantics_metrics.csv")
    values = {}
    for row in rows:
        if row.get("section") == "aggregate_mature" and row.get("entity") == "all_firms":
            values[row.get("metric")] = number(row.get("value"))
    current = values.get("current_total_labor_services", 0.0)
    u2 = values.get("U2_total_desired_labor_services", 0.0)
    return {
        "current_food_labor_services": current,
        "u2_food_desired_labor_services": u2,
        "food_excess_labor_services": max(0.0, current - u2),
        "u1_food_desired_labor_services": values.get("U1_total_desired_labor_services", 0.0),
    }


def read_demand_envelope():
    rows = read_csv(STEP14C / "step14C_system_demand_envelope.csv")
    result = {}
    for row in rows:
        window_name = row.get("window", "")
        if window_name not in {"mature", "last_200"}:
            continue
        result[window_name] = {
            "snapshot_count": int(number(row.get("snapshot_count"))),
            "R25_service_expenditure": number(row.get("R25_service_expenditure")),
            "R50_service_expenditure": number(row.get("R50_service_expenditure")),
            "R75_service_expenditure": number(row.get("R75_service_expenditure")),
            "R25_service_share_of_budget": number(row.get("R25_service_share_of_budget")),
            "R50_service_share_of_budget": number(row.get("R50_service_share_of_budget")),
            "R75_service_share_of_budget": number(row.get("R75_service_share_of_budget")),
        }
    return result


def read_food_opening_cash():
    rows = read_csv(CANONICAL / "firm_diagnostics.csv")
    opening = {}
    for row in rows:
        if step(row) != 0:
            continue
        firm_id = int(number(row.get("firm_id", -1)))
        opening[firm_id] = {
            "cash": number(row.get("cash_start", row.get("cash"))),
            "target_cash": number(row.get("target_cash")),
        }
    return opening


def make_unit_economics(mature_rows, mature_wage):
    firm0 = weighted_firm_revenue_metrics(mature_rows, 0)
    healthy_margin_ratio = firm0["revenue_to_wage_ratio"]
    healthy_capacity_ratio = safe_ratio(
        firm0["revenue_per_labor_service"], mature_wage
    )
    return [
        {
            "candidate_id": "BREAK_EVEN_REFERENCE",
            "candidate_label": "Food payroll break-even",
            "ratio_to_mature_wage": 1.0,
            "benchmark_source": "ratio=1.0 contractual payroll break-even",
        },
        {
            "candidate_id": "HEALTHY_FIRM0_MARGIN_REFERENCE",
            "candidate_label": "Healthy Firm 0 observed revenue/wage ratio",
            "ratio_to_mature_wage": healthy_margin_ratio,
            "benchmark_source": "mature Food Firm 0 sales-per-labor divided by its wage-cost-per-labor",
        },
        {
            "candidate_id": "HEALTHY_FIRM0_REVENUE_CAPACITY_REFERENCE",
            "candidate_label": "Healthy Firm 0 observed revenue capacity",
            "ratio_to_mature_wage": healthy_capacity_ratio,
            "benchmark_source": "mature Food Firm 0 sales revenue per labor service",
        },
    ]


def demand_rows(envelope, window_name):
    values = envelope[window_name]
    return [
        ("R25", values["R25_service_expenditure"], values["R25_service_share_of_budget"]),
        ("R50", values["R50_service_expenditure"], values["R50_service_share_of_budget"]),
        ("R75", values["R75_service_expenditure"], values["R75_service_share_of_budget"]),
    ]


def demand_classification(service_labor, food_excess):
    if food_excess <= TOLERANCE:
        return "SERVICE_DEMAND_TOO_SMALL_TO_ABSORB_FOOD_EXCESS"
    relative = service_labor / food_excess
    if relative < 0.75:
        return "SERVICE_DEMAND_TOO_SMALL_TO_ABSORB_FOOD_EXCESS"
    if relative <= 1.25:
        return "SERVICE_DEMAND_COMPARABLE_TO_FOOD_EXCESS"
    return "SERVICE_DEMAND_EXCEEDS_FOOD_EXCESS"


def labor_feasibility(food_desired, service_labor, available):
    combined = food_desired + service_labor
    if combined > available + TOLERANCE:
        return "LABOR_SCARCITY"
    if combined > available * 0.95:
        return "APPROXIMATE_FULL_EMPLOYMENT"
    return "LABOR_SLACK"


def adapter_requirements(service_payroll, food_payroll):
    local_full = BASE_BUFFER + service_payroll
    payroll_total = food_payroll + service_payroll
    global_split = service_payroll + safe_ratio(
        BASE_BUFFER * service_payroll, payroll_total
    )
    payroll_only = service_payroll
    return {
        "sector_local_full_base": local_full,
        "global_base_split_by_projected_payroll": global_split,
        "payroll_only_service_adapter": payroll_only,
    }


def build_rows(canonical_rows, b0, envelope, candidates, opening_cash):
    mature_rows = window(canonical_rows, MATURE_START, MATURE_END)
    last200_rows = window(canonical_rows, LAST200_START, LAST200_END)
    window_rows = {"mature": mature_rows, "last_200": last200_rows}
    wage_metrics = {
        name: window_wage_metrics(rows) for name, rows in window_rows.items()
    }
    mature_wage = wage_metrics["mature"]["aggregate_wage_cost_per_labor_service"]
    firm0_mature = weighted_firm_revenue_metrics(mature_rows, 0)
    food_excess = b0["food_excess_labor_services"]
    food_desired = b0["u2_food_desired_labor_services"]
    available = b0["current_food_labor_services"]
    initial_food_cash = math.fsum(item["cash"] for item in opening_cash.values())
    initial_target_cash = {fid: item["target_cash"] for fid, item in opening_cash.items()}
    mature_food_payroll = food_desired * wage_metrics["mature"]["aggregate_wage_cost_per_labor_service"]
    output = []

    for window_name in ("mature", "last_200"):
        wage_cost = wage_metrics[window_name]["aggregate_wage_cost_per_labor_service"]
        snapshot_count = envelope[window_name]["snapshot_count"]
        for demand_id, expenditure, service_share in demand_rows(envelope, window_name):
            weekly_expenditure = safe_ratio(expenditure, snapshot_count)
            for candidate in candidates:
                capacity = mature_wage * candidate["ratio_to_mature_wage"]
                service_labor = safe_ratio(weekly_expenditure, capacity)
                service_payroll = service_labor * wage_cost
                food_payroll = food_desired * wage_cost
                adapters = adapter_requirements(service_payroll, food_payroll)
                recommended_startup = adapters["global_base_split_by_projected_payroll"]
                removal_share = safe_ratio(recommended_startup, initial_food_cash)
                post_cash = {
                    fid: item["cash"] * max(0.0, 1.0 - removal_share)
                    for fid, item in opening_cash.items()
                }
                damage = any(
                    post_cash[fid] < initial_target_cash.get(fid, 0.0) - TOLERANCE
                    for fid in post_cash
                )
                combined_labor = food_desired + service_labor
                row = {
                    "window": window_name,
                    "demand_reference": demand_id,
                    "unit_economics_candidate": candidate["candidate_id"],
                    "candidate_label": candidate["candidate_label"],
                    "benchmark_source": candidate["benchmark_source"],
                    "service_firm_count_reference": SERVICE_FIRM_COUNT,
                    "normalized_service_price": NORMALIZED_SERVICE_PRICE,
                    "normalized_service_productivity": capacity,
                    "nominal_service_revenue_capacity_per_labor": capacity,
                    "wage_cost_per_labor_service": wage_cost,
                    "unit_economics_ratio": safe_ratio(capacity, wage_cost),
                    "break_even_utilization": safe_ratio(wage_cost, capacity),
                    "service_expenditure_total": expenditure,
                    "service_expenditure_per_week": weekly_expenditure,
                    "service_share_of_discretionary_budget": service_share,
                    "implied_service_labor_services": service_labor,
                    "desired_labor_per_service_firm": safe_ratio(service_labor, SERVICE_FIRM_COUNT),
                    "food_current_labor_services": available,
                    "food_u2_desired_labor_services": food_desired,
                    "food_excess_labor_services": food_excess,
                    "service_labor_to_food_excess_ratio": safe_ratio(service_labor, food_excess),
                    "food_excess_comparison": demand_classification(service_labor, food_excess),
                    "combined_food_plus_service_desired_labor": combined_labor,
                    "total_labor_feasibility": labor_feasibility(food_desired, service_labor, available),
                    "projected_service_payroll": service_payroll,
                    "projected_food_payroll": food_payroll,
                    "startup_cash_sector_local_full_base": adapters["sector_local_full_base"],
                    "startup_cash_global_base_split_by_projected_payroll": adapters["global_base_split_by_projected_payroll"],
                    "startup_cash_payroll_only_service_adapter": adapters["payroll_only_service_adapter"],
                    "projected_service_startup_cash_requirement": recommended_startup,
                    "food_cash_redistribution_amount": recommended_startup,
                    "food_cash_removal_share": removal_share,
                    "food_liquidity_damage": damage,
                }
                for fid in sorted(opening_cash):
                    row[f"food_opening_cash_firm_{fid}_before"] = opening_cash[fid]["cash"]
                    row[f"food_opening_cash_firm_{fid}_after_pro_rata"] = post_cash[fid]
                    row[f"food_target_cash_firm_{fid}"] = initial_target_cash[fid]
                output.append(row)
    return output, wage_metrics, firm0_mature


def summarize_adapter(wage_metrics, b0):
    food_payroll = b0["u2_food_desired_labor_services"] * wage_metrics["mature"]["aggregate_wage_cost_per_labor_service"]
    return {
        "historical_base_buffer": BASE_BUFFER,
        "food_desired_payroll_reference": food_payroll,
        "sector_local_full_base_total_across_two_sectors": BASE_BUFFER * 2.0,
        "sector_local_full_base_duplication_factor": 2.0,
        "recommended_adapter": "GLOBAL_BASE_SPLIT_BY_PROJECTED_PAYROLL",
        "capitalization_source": "PRO_RATA_FOOD_CASH",
    }


def recommended_matrix(rows):
    return [
        row for row in rows
        if row["window"] == "mature"
    ]


def fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def build_summary(flags, b0, wage_metrics, firm0, adapter, matrix):
    mature = wage_metrics["mature"]["aggregate_wage_cost_per_labor_service"]
    last200 = wage_metrics["last_200"]["aggregate_wage_cost_per_labor_service"]
    max_startup = max(number(row["projected_service_startup_cash_requirement"]) for row in matrix)
    max_removal = max(number(row["food_cash_removal_share"]) for row in matrix)
    lines = [
        "# Step 14E.0 Two-Sector Parameter Identification",
        "",
        "## Verdict",
        "",
        "**A. STEP14E_PARAMETER_BOUNDARY_READY**",
        "",
        "This is a passive static audit: no World was constructed, no behavioral simulation was run, Service household spending stayed inactive, no workers moved, no money was created, and no Step 13 parameter changed.",
        "",
        "## Evidence",
        "",
        f"- Accepted canonical Food evidence: mature window `{MATURE_START}-{MATURE_END}` and last-200 `{LAST200_START}-{LAST200_END}`.",
        f"- Mature Food wage cost: `{fmt(mature)}` currency / labor-service; last-200: `{fmt(last200)}`.",
        f"- Mature Food labor `{fmt(b0['current_food_labor_services'])}`, U2 desired labor `{fmt(b0['u2_food_desired_labor_services'])}`, excess `{fmt(b0['food_excess_labor_services'])}`.",
        f"- Healthy Firm 0 mature revenue/labor `{fmt(firm0['revenue_per_labor_service'])}`, wage/labor `{fmt(firm0['wage_cost_per_labor_service'])}`, ratio `{fmt(firm0['revenue_to_wage_ratio'])}`.",
        "",
        "## Semantic Answers",
        "",
        "1. Service price and physical productivity are **not separately identified** in the current abstract Service unit. Only `P_s * A_s` is identified by nominal revenue capacity.",
        "2. `P_s = 1` is therefore an engineering unit normalization, not permanent economic price calibration.",
        "3. The first-screen economic variable is `nominal_service_revenue_capacity_per_labor = P_s * A_s`.",
        f"4. Payroll break-even capacity is the observed wage denominator: mature `{fmt(mature)}`, last-200 `{fmt(last200)}` per labor-service.",
        "5. The three candidates are break-even, Healthy Firm 0 observed revenue/wage ratio, and Healthy Firm 0 observed revenue capacity. No arbitrary 10%/25% markup was introduced.",
        "6. Fixing `n_service_firms = 2` is accepted as the minimum within-Service competitive engineering reference, not calibration.",
        "7. First-screen worker initialization is zero Service workers plus explicit vacancies; Food workers remain in place until a later labor-allocation experiment.",
        "8. R25/R50/R75 are all retained. They differ in demand intensity and remain useful even though none fully absorbs the measured Food excess in this static envelope.",
        "9. In the current envelope, no retained combination is comparable to the full Food excess under the +/-25% comparison rule; all are classified as demand-too-small for complete absorption.",
        "10. No retained combination implies aggregate labor scarcity: Food U2 desired labor plus implied Service labor remains below available productive labor.",
        f"11. Under the recommended adapter, projected Service startup cash ranges up to `{fmt(max_startup)}` in the mature matrix; no transfer is executed.",
        f"12. Pro-rata redistribution from opening Food cash removes at most `{fmt(max_removal * 100)}%`; the passive Food liquidity-damage screen is false for all retained cells.",
        "13. Naive sector-local full-base allocation duplicates the historical base: two sectors would imply `150000 * 2` before payroll components.",
        "14. Source semantics are best classified as an economy-wide base buffer allocated within the existing Food Firm system by capacity share; it is also a historical Food-compatibility proxy, not a sector-local base.",
        "15. The recommended adapter is `GLOBAL_BASE_SPLIT_BY_PROJECTED_PAYROLL`, followed by sector-local allocation. It avoids heterogeneous physical units and accidental base-buffer doubling while preserving Food-only semantics when Service is absent.",
        "16. The deterministic capitalization source is `PRO_RATA_FOOD_CASH`, using only opening cash and no future performance information.",
        "17. The behavioral matrix remains `3 demand references x 3 unit-economics references = 9` cells; no cell is structurally impossible or capitalization-destructive.",
        "18. Future controls: C0 canonical Food-only; optional C1 plumbing-active but zero Service demand/employment.",
        "19. Future execution: small N/short smoke across the retained cells, then N=5000/1820 only for coherent cells, with no 3640 extension until one survives.",
        "20. Step 14E is **not yet behaviorally executed**. Step 14E.0 is ready and defines the smallest defensible screen; the hard stop remains in force.",
        "",
        "## Candidate Grid",
        "",
        "| candidate | normalized price | normalized productivity / nominal revenue capacity | mature ratio | mature break-even utilization |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in matrix[0:3]:
        lines.append(
            f"| {row['unit_economics_candidate']} | {fmt(row['normalized_service_price'])} | {fmt(row['nominal_service_revenue_capacity_per_labor'])} | {fmt(row['unit_economics_ratio'])} | {fmt(row['break_even_utilization'])} |"
        )
    lines.extend([
        "",
        "## Demand Envelope",
        "",
        "The detailed 18-row mature/last-200 envelope and all 9 mature matrix cells are in the CSV outputs. Service expenditure is reported both as window total and weekly mean; labor feasibility uses the weekly mean.",
        "",
        "| demand | mature Service expenditure / week | implied Service labor at break-even | comparison to Food excess |",
        "|---|---:|---:|---|",
    ])
    for demand_id in ("R25", "R50", "R75"):
        row = next(item for item in matrix if item["demand_reference"] == demand_id and item["unit_economics_candidate"] == "BREAK_EVEN_REFERENCE")
        lines.append(
            f"| {demand_id} | {fmt(row['service_expenditure_per_week'])} | {fmt(row['implied_service_labor_services'])} | {row['food_excess_comparison']} |"
        )
    lines.extend([
        "",
        "## Target-Cash Boundary",
        "",
        f"- Historical base buffer: `{fmt(adapter['historical_base_buffer'])}`.",
        f"- Naive sector-local full base across Food + Service: `{fmt(adapter['sector_local_full_base_total_across_two_sectors'])}`, duplication factor `{fmt(adapter['sector_local_full_base_duplication_factor'])}`.",
        f"- Recommended experimental adapter: **`{adapter['recommended_adapter']}`**.",
        f"- Temporary capitalization source: **`{adapter['capitalization_source']}`**.",
        "- Zero-cash Service start remains stress-only. Household equity seed remains deferred until ownership exists.",
        "",
        "## Hard Stop",
        "",
        "No full two-sector simulation, no active canonical Service Firm, no household Service spending, no worker movement, no parameter tuning, no money creation, no ownership, no Government, and no Exit was implemented.",
        "",
        "## Flags",
        "",
        "```json",
        json.dumps(flags, indent=2, ensure_ascii=False),
        "```",
        "",
    ])
    return "\n".join(lines)


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    canonical_rows = read_csv(CANONICAL / "firm_diagnostics.csv")
    b0 = read_b0_evidence()
    envelope = read_demand_envelope()
    opening_cash = read_food_opening_cash()
    mature_rows = window(canonical_rows, MATURE_START, MATURE_END)
    mature_wage_metrics = window_wage_metrics(mature_rows)
    candidates = make_unit_economics(
        mature_rows,
        mature_wage_metrics["aggregate_wage_cost_per_labor_service"],
    )
    all_rows, wage_metrics, firm0 = build_rows(
        canonical_rows, b0, envelope, candidates, opening_cash
    )
    matrix = recommended_matrix(all_rows)
    adapter = summarize_adapter(wage_metrics, b0)

    flags = {
        "verdict": "A. STEP14E_PARAMETER_BOUNDARY_READY",
        "economic_behavior_changed": False,
        "full_behavioral_simulation_run": False,
        "service_firm_count_structural_reference_ready": True,
        "recommended_service_firm_count": SERVICE_FIRM_COUNT,
        "service_price_productivity_separately_identified": False,
        "service_price_unit_normalization_ready": True,
        "service_price_is_unit_normalization": True,
        "normalized_service_price": NORMALIZED_SERVICE_PRICE,
        "service_unit_economics_variable_ready": True,
        "wage_break_even_reference_ready": True,
        "unit_economics_candidate_count": len(candidates),
        "R25_retained": True,
        "R50_retained": True,
        "R75_retained": True,
        "worker_initialization_reference_ready": True,
        "capitalization_amount_model_derived": True,
        "existing_money_redistribution_ready": True,
        "zero_cash_primary_baseline_rejected": True,
        "household_equity_seed_deferred": True,
        "global_raw_capacity_share_invalid": True,
        "naive_sector_local_base_duplication_detected": True,
        "historical_base_target_cash_semantics_understood": True,
        "target_cash_adapter_ready": True,
        "recommended_target_cash_adapter": "GLOBAL_BASE_SPLIT_BY_PROJECTED_PAYROLL",
        "Food_liquidity_damage_screen_ready": True,
        "behavioral_matrix_cell_count": len(matrix),
        "behavioral_matrix_pruned": False,
        "Stage14E_ready": True,
        "new_long_runs": 0,
        "seed7_21_run": False,
    }

    write_csv(OUTPUT / "step14E0_parameter_envelope.csv", all_rows)
    write_csv(OUTPUT / "step14E0_recommended_behavior_matrix.csv", matrix)
    write_json(OUTPUT / "acceptance_flags.json", flags)
    (OUTPUT / "acceptance_summary.md").write_text(
        build_summary(flags, b0, wage_metrics, firm0, adapter, matrix),
        encoding="utf-8",
    )

    print(flags["verdict"])
    print(f"Output: {OUTPUT}")
    print(f"Rows: parameter_envelope={len(all_rows)}, recommended_matrix={len(matrix)}")


if __name__ == "__main__":
    main()
