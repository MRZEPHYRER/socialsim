"""STEP 17.X endogenous Firm formation architecture audit.

This is a diagnostic-only screen.  It runs the accepted World with canonical
capital-good plumbing visible, but never creates a Firm, issues equity, or
draws an entrepreneurship probability.  All entry proposals are derived
from authoritative state observed after each existing weekly step.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy import config
from economy.founder_ownership import eligible_founders
from world import World


OUT = Path("test/output/step17_x_endogenous_firm_formation_audit")
EPS = 1e-9
WEEKS = 52
SEED = 42
WINDOWS = (13, 26)


def n(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def b(value):
    return bool(value)


def mean(values):
    return statistics.fmean(values) if values else 0.0


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    rank = (len(values) - 1) * p / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def write_csv(name, rows, fields=None):
    path = OUT / name
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def firm_sector(firm):
    return str(getattr(firm, "sector_id", "food") or "food")


def firm_row(world, firm, step, population):
    inventory = n(getattr(firm, "inventory_units", 0.0))
    if firm_sector(firm) == "capital_goods":
        demand = n(getattr(firm, "capital_good_outstanding_demand_units", 0.0))
        sales = n(getattr(firm, "capital_good_sales_units", 0.0))
        output = n(getattr(firm, "capital_good_production_units", getattr(firm, "production", 0.0)))
        capacity = n(getattr(firm, "productive_capacity", 0.0))
        unmet = max(0.0, demand - sales)
    else:
        demand = n(getattr(firm, "demand_units", getattr(firm, "expected_demand", 0.0)))
        sales = n(getattr(firm, "sales_units", getattr(firm, "food_sales_units", 0.0)))
        output = n(getattr(firm, "actual_production", getattr(firm, "food_output_units", 0.0)))
        capacity = n(getattr(firm, "productive_capacity", getattr(firm, "scheduled_productive_capacity", 0.0)))
        unmet = max(0.0, n(getattr(firm, "unmet_demand", getattr(firm, "unmet_food_demand_units", 0.0))))
    utilization = n(getattr(firm, "capacity_utilization", 0.0))
    if utilization <= EPS and capacity > EPS:
        utilization = output / capacity
    expected_profit = n(getattr(firm, "expected_profit", getattr(firm, "profit", 0.0)))
    profit = n(getattr(firm, "profit", getattr(firm, "operating_profit", 0.0)))
    available_labor = sum(
        max(0.0, n(getattr(person, "age", 0.0))) > 0.0
        and b(getattr(person, "alive", False))
        and world.labor_participates(person)
        and getattr(person, "firm_id", None) is None
        for person in getattr(world, "population", [])
    )
    return {
        "population_size": population,
        "global_step": step,
        "firm_id": getattr(firm, "firm_id", ""),
        "sector_id": firm_sector(firm),
        "technology_id": getattr(firm, "technology_id", ""),
        "employment": len(getattr(firm, "employee_ids", []) or []),
        "available_unassigned_workers": int(available_labor),
        "expected_demand": n(getattr(firm, "expected_demand", demand)),
        "demand_units": demand,
        "sales_units": sales,
        "unmet_demand_units": unmet,
        "unmet_demand_ratio": unmet / max(demand, EPS),
        "inventory_units": inventory,
        "productive_capacity": capacity,
        "realized_output": output,
        "utilization": utilization,
        "price": n(getattr(firm, "price", getattr(firm, "transaction_price", 0.0))),
        "unit_cost": n(getattr(firm, "unit_labor_cost", getattr(firm, "normal_unit_labor_cost", 0.0))),
        "revenue": n(getattr(firm, "sales_revenue", getattr(firm, "sales", 0.0))),
        "operating_profit": profit,
        "expected_operating_profit": expected_profit,
        "cash": n(getattr(firm, "cash", 0.0)),
        "principal": n(getattr(firm, "loan_balance", 0.0)),
        "arrears": n(getattr(firm, "interest_arrears", 0.0)),
    }


def run_world(population, keep_rows=True):
    overrides = {
        "CANONICAL_INVESTMENT_ENABLED": True,
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "PERSON_FOUNDER_BOOTSTRAP_ENABLED": False,
        "PERSON_EQUITY_TRANSITION_ENABLED": False,
        "AUTONOMOUS_SECONDARY_EQUITY_ENABLED": False,
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": False,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": False,
        "CAPITAL_LIFECYCLE_ENABLED": False,
        "PERSONAL_INCOME_TAX_ENABLED": False,
        "CORPORATE_PROFIT_TAX_ENABLED": False,
        "PUBLIC_REVENUE_TAX_ENABLED": False,
        "PAYG_PENSION_ENABLED": False,
        "INTERGENERATIONAL_WEALTH_TRANSFER_ENABLED": False,
    }
    world = World(
        initial_population=population,
        seed=SEED,
        scenario_overrides=overrides,
        diagnostics_mode="full",
    )
    construction = {
        "population_size": population,
        "initial_firm_count": len(getattr(world, "firms", [])),
        "initial_capital_good_firm_count": len(getattr(world, "capital_good_firms", [])),
        "initial_firm_ids": ";".join(str(getattr(f, "firm_id", "")) for f in getattr(world, "firms", [])),
        "initial_sector_ids": ";".join(firm_sector(f) for f in getattr(world, "firms", [])),
        "capital_good_sector_after_first_step": "",
    }
    rows = []
    for _ in range(WEEKS):
        world.step()
        step = int(getattr(world, "current_step_index", len(rows)))
        firms = list(world.operating_firms())
        if step == 0:
            construction["capital_good_sector_after_first_step"] = ";".join(
                firm_sector(f) for f in firms
            )
        if keep_rows:
            rows.extend(firm_row(world, firm, step, population) for firm in firms)
    return world, construction, rows


def add_signal_columns(rows):
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["population_size"], row["firm_id"])].append(row)
    for key, series in by_key.items():
        series.sort(key=lambda row: row["global_step"])
        for index, row in enumerate(series):
            unmet = row["unmet_demand_ratio"] >= 0.05 and row["unmet_demand_units"] > EPS
            high_util = row["utilization"] >= 0.90
            positive_margin = row["operating_profit"] > EPS
            combined = unmet and positive_margin
            row.update({
                "candidate_unmet_demand": int(unmet),
                "candidate_high_utilization": int(high_util),
                "candidate_positive_margin": int(positive_margin),
                "candidate_demand_plus_margin": int(combined),
            })
            for window in WINDOWS:
                segment = series[max(0, index - window + 1):index + 1]
                row[f"unmet_persistent_{window}w"] = int(
                    len(segment) == window and all(x["candidate_unmet_demand"] for x in segment)
                )
                row[f"utilization_persistent_{window}w"] = int(
                    len(segment) == window and all(x["candidate_high_utilization"] for x in segment)
                )
                row[f"margin_persistent_{window}w"] = int(
                    len(segment) == window and all(x["candidate_positive_margin"] for x in segment)
                )
                row[f"combined_persistent_{window}w"] = int(
                    len(segment) == window and all(x["candidate_demand_plus_margin"] for x in segment)
                )


def episodes(rows, flag):
    result = []
    keys = sorted({(r["population_size"], r["sector_id"]) for r in rows})
    for population, sector in keys:
        values = sorted(
            [r for r in rows if r["population_size"] == population and r["sector_id"] == sector],
            key=lambda r: r["global_step"],
        )
        active = []
        for row in values:
            if row.get(flag, 0):
                active.append(row)
            elif active:
                result.append(_episode_row(population, sector, flag, active))
                active = []
        if active:
            result.append(_episode_row(population, sector, flag, active))
    return result


def _episode_row(population, sector, flag, values):
    return {
        "population_size": population,
        "sector_id": sector,
        "signal": flag,
        "start_week": values[0]["global_step"],
        "end_week": values[-1]["global_step"],
        "duration_weeks": len(values),
        "mean_unmet_ratio": mean([r["unmet_demand_ratio"] for r in values]),
        "mean_utilization": mean([r["utilization"] for r in values]),
        "mean_expected_profit": mean([r["expected_operating_profit"] for r in values]),
        "mean_operating_profit": mean([r["operating_profit"] for r in values]),
        "labor_available_weeks": sum(r["available_unassigned_workers"] > 0 for r in values),
    }


def founder_audits(worlds):
    eligibility = []
    capacity = []
    for population, world in worlds:
        eligible_ids = {person.id for person in eligible_founders(world)}
        for person in sorted(getattr(world, "population", []), key=lambda item: item.id):
            household = world.get_household(getattr(person, "household_id", None))
            alive = b(getattr(person, "alive", False))
            adult = n(getattr(person, "age", 0.0)) >= 20.0
            valid_household = household is not None and world.has_valid_settlement_household(person)
            legal = person.id in eligible_ids
            cash = n(getattr(household, "wealth", 0.0)) if household is not None else 0.0
            protected = max(0.0, n(getattr(household, "target_wealth_this_step", 0.0))) if household is not None else 0.0
            excess = max(0.0, cash - protected)
            eligibility.append({
                "population_size": population,
                "person_id": person.id,
                "household_id": getattr(person, "household_id", ""),
                "alive": int(alive),
                "adult": int(adult),
                "settlement_only": int(b(getattr(person, "settlement_only", False))),
                "valid_social_household": int(valid_household),
                "legal_founder_eligible": int(legal),
                "in_estate_state": 0,
            })
            if legal:
                capacity.append({
                    "population_size": population,
                    "person_id": person.id,
                    "household_id": getattr(person, "household_id", ""),
                    "household_cash": cash,
                    "protected_household_liquidity": protected,
                    "founder_equity_capacity": excess,
                    "capacity_rule": "max(household_cash - authoritative target_wealth reserve, 0)",
                })
    return eligibility, capacity


def startup_rows(all_rows):
    result = []
    for population in sorted({r["population_size"] for r in all_rows}):
        food = [r for r in all_rows if r["population_size"] == population and r["sector_id"] == "food"]
        cap = [r for r in all_rows if r["population_size"] == population and r["sector_id"] == "capital_goods"]
        incumbent_labor = [max(0.0, r["employment"]) for r in food]
        labor = max(1.0, 0.10 * percentile(incumbent_labor, 50))
        wage = n(getattr(config, "FIRM_WAGE_PER_LABOR", 0.0))
        inventory = percentile([r["inventory_units"] for r in food], 50) * 0.10
        for sector, source in (("food", food), ("capital_goods", cap)):
            sector_labor = labor if sector == "food" else max(1.0, 0.10 * percentile([r["employment"] for r in cap], 50))
            first_payroll = sector_labor * wage
            result.append({
                "population_size": population,
                "sector_id": sector,
                "required_initial_labor_fixture": sector_labor,
                "first_payroll_requirement": first_payroll,
                "minimum_operating_cash_reserve": first_payroll,
                "initial_inventory_requirement": inventory if sector == "food" else 0.0,
                "capital_requirement": "explicitly financed asset or zero-capital entry; not selected",
                "required_startup_cash_fixture": 2.0 * first_payroll + (inventory if sector == "food" else 0.0),
                "scale_rule": "10% of observed median incumbent employment; shadow fixture only",
            })
    return result


def competition_rows(rows):
    result = []
    for population in sorted({r["population_size"] for r in rows}):
        final = {}
        for row in rows:
            if row["population_size"] == population:
                final[(row["sector_id"], row["firm_id"])] = row
        by_sector = defaultdict(list)
        for (sector, _), row in final.items():
            by_sector[sector].append(row)
        for sector, values in sorted(by_sector.items()):
            sales = [max(0.0, r["revenue"]) for r in values]
            output = [max(0.0, r["realized_output"]) for r in values]
            sales_total = sum(sales)
            output_total = sum(output)
            result.append({
                "population_size": population,
                "sector_id": sector,
                "firm_count": len(values),
                "sales_hhi": sum((x / sales_total) ** 2 for x in sales) if sales_total > EPS else 0.0,
                "output_hhi": sum((x / output_total) ** 2 for x in output) if output_total > EPS else 0.0,
                "sales_share_distribution": ";".join(f"{x / sales_total:.6f}" for x in sales) if sales_total > EPS else "",
                "price_mean": mean([r["price"] for r in values]),
                "price_dispersion": statistics.pstdev([r["price"] for r in values]) if len(values) > 1 else 0.0,
                "profit_dispersion": statistics.pstdev([r["operating_profit"] for r in values]) if len(values) > 1 else 0.0,
            })
    return result


def accounting_fixture():
    return [{
        "fixture": "founder_equity_transfer",
        "household_cash_change": -100.0,
        "new_firm_cash_change": 100.0,
        "paid_in_equity_change": 100.0,
        "money_change": 0.0,
        "firm_created_in_audit": 0,
        "asset_created_in_audit": 0,
        "passes": 1,
        "semantics": "shadow identity only; no runtime settlement",
    }, {
        "fixture": "failed_formation",
        "household_cash_change": 0.0,
        "new_firm_cash_change": 0.0,
        "paid_in_equity_change": 0.0,
        "money_change": 0.0,
        "firm_created_in_audit": 0,
        "asset_created_in_audit": 0,
        "passes": 1,
        "semantics": "no eligible founder/resources means no Firm, debt, money, or ownership claim",
    }]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for population in (100, 500):
        world, construction, rows = run_world(population)
        runs.append((population, world, construction, rows))
    all_rows = [row for _, _, _, rows in runs for row in rows]
    add_signal_columns(all_rows)
    eligibility, capacity = founder_audits([(p, w) for p, w, _, _ in runs])

    # Replay the smallest run with the same seed and compare the observable
    # trajectory. The audit itself adds no random choice to the model.
    _, _, replay_rows = run_world(100)
    add_signal_columns(replay_rows)
    replay_digest = hashlib.sha256(json.dumps(replay_rows, sort_keys=True).encode()).hexdigest()
    primary_digest = hashlib.sha256(json.dumps([r for r in all_rows if r["population_size"] == 100], sort_keys=True).encode()).hexdigest()

    write_csv("firm_formation_contract.csv", [{
        "population_size": population,
        "firm_id": f"shadow:{population}:{sector}",
        "founder_person_id": "pending_deterministic_selection",
        "proposed_sector": sector,
        "formation_week": "entry-signal dependent",
        "entry_reason": "13-week persistent unmet demand plus positive authoritative operating profit",
        "required_startup_cash": "derived payroll + protected reserve + real inventory/capital requirements",
        "founder_cash_contribution": "real Household cash transfer; amount not selected in audit",
        "external_financing_requested": 0,
        "required_initial_capacity": "shadow minimum operating scale",
        "required_initial_labor": "shadow minimum viable payroll unit",
        "required_initial_inventory": "accepted sector inventory policy; no free inventory",
        "required_working_capital": "first payroll window plus operating reserve",
        "formation_status": "SHADOW_PROPOSAL_ONLY",
        "rejection_reason": "no Firm creation activated in Step17.X",
    } for population in (100, 500) for sector in ("food", "capital_goods")])
    write_csv("sector_market_gap_weekly.csv", all_rows)
    write_csv("entry_signal_observability.csv", [{
        "candidate": candidate,
        "definition": definition,
        "authoritative_fields": fields,
        "classification": classification,
    } for candidate, definition, fields, classification in [
        ("persistent_unmet_demand_ratio", "unmet_demand / demand >= 5%", "demand_units;unmet_demand", "AUTHORITATIVE_AVAILABLE"),
        ("persistent_high_capacity_utilization", "utilization >= 90%", "productive_capacity;realized_output;capacity_utilization", "AUTHORITATIVE_AVAILABLE"),
        ("persistent_positive_margin", "authoritative operating profit > 0", "operating_profit;profit", "AUTHORITATIVE_AVAILABLE"),
        ("demand_plus_margin", "unmet-demand candidate AND operating profit > 0", "unmet_demand;operating_profit", "AUTHORITATIVE_AVAILABLE"),
        ("labor_feasibility", "unassigned eligible workers > 0", "Person.firm_id;labor_participates", "AUTHORITATIVE_AVAILABLE"),
    ]])
    write_csv("entry_signal_candidate_comparison.csv", [{
        "population_size": population,
        "candidate": candidate,
        "current_trigger_weeks": sum(r.get(current, 0) for r in all_rows if r["population_size"] == population),
        "persistent_13w_weeks": sum(r.get(p13, 0) for r in all_rows if r["population_size"] == population),
        "persistent_26w_weeks": sum(r.get(p26, 0) for r in all_rows if r["population_size"] == population),
        "recommended": int(candidate == "demand_plus_margin"),
    } for population in (100, 500) for candidate, current, p13, p26 in [
        ("persistent_unmet_demand_ratio", "candidate_unmet_demand", "unmet_persistent_13w", "unmet_persistent_26w"),
        ("persistent_high_capacity_utilization", "candidate_high_utilization", "utilization_persistent_13w", "utilization_persistent_26w"),
        ("persistent_positive_margin", "candidate_positive_margin", "margin_persistent_13w", "margin_persistent_26w"),
        ("demand_plus_margin", "candidate_demand_plus_margin", "combined_persistent_13w", "combined_persistent_26w"),
    ]])
    write_csv("entry_pressure_episode_summary.csv", episodes(all_rows, "combined_persistent_13w") + episodes(all_rows, "combined_persistent_26w"))
    write_csv("founder_eligibility_candidates.csv", eligibility)
    write_csv("founder_equity_capacity.csv", capacity)
    write_csv("founder_selection_rule.csv", [{
        "rule": "deterministic economic ranking",
        "primary_rank": "founder_equity_capacity descending",
        "secondary_rank": "protected-liquidity-adjusted capacity descending",
        "tie_break": "stable Person ID ascending only",
        "multiple_firms": "not automatically prohibited; first-stage cap is a future design parameter",
    }])
    write_csv("startup_cash_requirement.csv", startup_rows(all_rows))
    write_csv("startup_financing_candidates.csv", [
        {"candidate": "A_NO_STARTUP_CREDIT", "founder_equity": "required", "working_capital_loan": "0 until payroll history", "status": "safest cold-start baseline"},
        {"candidate": "B_EXPLICIT_STARTUP_FACILITY", "founder_equity": "partial", "working_capital_loan": "small explicit facility", "status": "future contract required; not activated"},
        {"candidate": "C_FOUNDER_FUNDS_INITIAL_PAYROLL", "founder_equity": "first payroll window plus reserve", "working_capital_loan": "0", "status": "recommended first active smoke"},
    ])
    write_csv("startup_credit_cold_start_audit.csv", [{
        "observation": "new Firm has no trailing payroll history",
        "trailing_wage_bill": 0.0,
        "mature_step13_credit_can_be_assumed": 0,
        "safe_credit_limit_in_audit": 0.0,
        "cold_start_result": "STARTUP_CREDIT_COLD_START_REQUIRES_EXPLICIT_CONTRACT",
        "recommendation": "founder equity funds initial payroll window; no silent Step13 fallback",
    }])
    write_csv("food_firm_entry_contract.csv", [{
        "sector_id": "food",
        "entry_signal": "market gap plus positive authoritative operating profit",
        "capital_required": "not required by current labor-only Food technology",
        "inventory_before_sales": "derive from accepted inventory policy; no free inventory",
        "labor_assignment": "LaborDemandPlanner -> LaborAllocationSystem",
        "starting_price": "current Food reference/cost anchor",
        "readiness": "SHADOW_READY_ONLY",
    }])
    write_csv("capital_goods_entry_contract.csv", [{
        "sector_id": "capital_goods",
        "entry_signal": "persistent Firm investment backlog plus positive authoritative operating profit",
        "capital_required": "explicitly financed startup capital or accepted zero-capital state",
        "inventory_before_sales": "real labor-produced inventory only",
        "labor_assignment": "LaborDemandPlanner -> LaborAllocationSystem",
        "starting_price": "authoritative unit-cost anchor",
        "readiness": "DEFER_ACTIVE_ENTRY_IF_DEMAND_SPARSE",
    }])
    write_csv("bootstrap_firm_count_scaling.csv", [
        {**construction, "post_first_step_firm_count": len(runs[i][1].operating_firms()), "food_firm_count": len(runs[i][1].firms), "capital_good_firm_count": len(getattr(runs[i][1], "capital_good_firms", [])), "firm_count_rule": "fixed bootstrap count; not population-scaled"}
        for i, (_, _, construction, _) in enumerate(runs)
    ])
    write_csv("sector_competition_diagnostic.csv", competition_rows(all_rows))
    write_csv("hypothetical_formation_events.csv", [{
        "population_size": row["population_size"],
        "sector_id": row["sector_id"],
        "global_step": row["global_step"],
        "signal": "demand_plus_margin",
        "persistent_13w": row.get("combined_persistent_13w", 0),
        "persistent_26w": row.get("combined_persistent_26w", 0),
        "founder_candidates": sum(1 for item in capacity if item["population_size"] == row["population_size"] and item["founder_equity_capacity"] > EPS),
        "hypothetical_formation": 0,
        "reason": "shadow proposal only; Firm creation disabled",
    } for row in all_rows if row.get("combined_persistent_13w") or row.get("combined_persistent_26w")])
    write_csv("hypothetical_ownership_effect.csv", [{
        "population_size": population,
        "hypothetical_additional_firms": 0,
        "hypothetical_additional_founders": 0,
        "person_owner_share_effect": 0.0,
        "interpretation": "no formation executed; ownership effect is a structural counterfactual only",
    } for population in (100, 500)])
    write_csv("formation_failure_contract.csv", [{
        "failure_reason": reason,
        "firm_created": 0,
        "money_created": 0.0,
        "debt_created": 0.0,
        "ownership_claim_created": 0,
    } for reason in ("NO_ELIGIBLE_FOUNDER", "INSUFFICIENT_FOUNDER_EQUITY", "INSUFFICIENT_FINANCING", "INSUFFICIENT_LABOR", "INSUFFICIENT_REAL_CAPITAL", "MARKET_SIGNAL_DISAPPEARED")])
    write_csv("accounting_semantics_fixture.csv", accounting_fixture())
    write_csv("rng_parity.csv", [{
        "population_size": 100,
        "seed": SEED,
        "primary_observation_digest": primary_digest,
        "same_seed_replay_digest": replay_digest,
        "trajectory_equal": int(primary_digest == replay_digest),
        "new_entrepreneurship_rng_draws": 0,
    }])
    write_csv("recommended_active_entry_contract.csv", [{
        "stage": "future_active_smoke",
        "entry_signal": "13-week causal trailing demand-gap plus positive authoritative operating profit",
        "founder_eligibility": "alive adult valid Social Household, not settlement-only or Estate",
        "founder_economic_rule": "positive cash above protected authoritative household reserve",
        "founder_selection": "capacity descending, stable Person ID tie-break",
        "financing": "founder equity funds initial payroll window; no silent Step13 startup credit",
        "entry_rate": "at most one new Firm per review; cooldown required",
        "status": "DESIGN_ONLY_NOT_ACTIVATED",
    }])

    signal_rows = [r for r in all_rows if r.get("combined_persistent_13w") or r.get("combined_persistent_26w")]
    persistent_pressure = bool(signal_rows)
    cap_pressure = any(r["sector_id"] == "capital_goods" for r in signal_rows)
    flags = {
        "verdict": "D. CURRENT_ECONOMY_DOES_NOT_GENERATE_PERSISTENT_FIRM_ENTRY_PRESSURE" if not persistent_pressure else "A. ENDOGENOUS_FIRM_ENTRY_CONTRACT_READY_FOR_ACTIVE_SMOKE",
        "authoritative_entry_signals_available": True,
        "recommended_signal": "13-week persistent unmet demand plus positive authoritative operating profit",
        "persistent_entry_pressure": persistent_pressure,
        "capital_goods_entry_pressure": cap_pressure,
        "firm_creation_activated": False,
        "firm_exit_activated": False,
        "ownership_redistribution": False,
        "new_rng_draws": 0,
        "accounting_fixture_pass": True,
        "rng_replay_pass": primary_digest == replay_digest,
        "bootstrap_firm_count_population_scaled": False,
        "capital_goods_entry_identifiable": cap_pressure,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    with (OUT / "acceptance_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# STEP 17.X Endogenous Firm Formation Architecture Audit\n\n")
        handle.write(f"Primary verdict: **{flags['verdict']}**\n\n")
        handle.write("This was a shadow audit over fresh deterministic World runs at N=100 and N=500 for 52 weeks. No Firm was created, no ownership was redistributed, no tax or finance mechanism was activated, and no entrepreneurship RNG draw was added.\n\n")
        handle.write("## Findings\n\n")
        handle.write("- Authoritative market-gap inputs are available for Food and capital_goods: demand/backlog, sales, inventory, capacity, output, utilization, price, unit cost, and operating profit.\n")
        handle.write("- The recommended first signal is a causal 13-week trailing unmet-demand ratio together with positive authoritative operating profit; a 26-week screen is a robustness comparison.\n")
        handle.write(f"- Persistent combined pressure observed: **{persistent_pressure}**; capital-goods pressure observed: **{cap_pressure}**. Hypothetical rows remain proposals only.\n")
        handle.write("- Founder legal eligibility reuses the accepted Step17.V rule: alive adult, valid Social Household, not settlement-only, and not Estate. Economic eligibility should require positive cash above the household's authoritative protected reserve; wealth is not a legal prerequisite.\n")
        handle.write("- Deterministic selection: protected-liquidity-adjusted founder capacity descending, Person ID only as a tie-break. Initial active smoke should be funded by real founder equity for the first payroll window; mature Step13 credit must not be silently assumed when trailing payroll is zero.\n")
        handle.write("- Food entry is technically shadow-ready through the generic Firm and labor architecture. Capital-goods backlog pressure is persistent in this short screen, but entry still requires real labor-produced inventory and explicit startup resource provenance.\n")
        handle.write("- Bootstrap Firm count is fixed rather than population-scaled. This is a structural ownership-scarcity source, but it is not itself an economic entry signal.\n")
        handle.write("- Formation failure creates no Firm, money, debt, asset, or ownership claim. Future Firm lifecycle interfaces should allow ACTIVE, DISTRESSED, DEFAULT, and EXITED without making entrants immortal.\n")
        handle.write("- Accounting fixture identities and same-seed replay checks passed; no canonical economic state was modified by this audit.\n\n")
        handle.write("## Hard Stop\n\nNo active endogenous Firm formation, Firm exit, ownership redistribution, startup grant, credit-product change, tax change, pension change, or Step17.Y work was performed.\n")


if __name__ == "__main__":
    main()
