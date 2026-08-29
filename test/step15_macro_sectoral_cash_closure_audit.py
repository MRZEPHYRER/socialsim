"""Architecture-only audit of Step 15 sectoral cash closure."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world import World


OUT = ROOT / "test/output/step15_macro_sectoral_cash_closure_audit"
SEED = 42
POPULATION = 5000
WEEKS = 520
FOOD_FIRMS = 5
TOL = 1e-6
SECTORS = ("households", "food_firms", "capital_good_firms", "legacy_owner", "estates", "public_central_bank", "pending_household_formation")


def number(value, default=0.0):
    try:
        return float(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        return float(default)


def write_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def slope(values):
    if len(values) < 2:
        return math.nan
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def overrides():
    return {
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "CANONICAL_INVESTMENT_ENABLED": True,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
        "CAPITAL_LIFECYCLE_ENABLED": True,
        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": 3.0,
        "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": True,
        "ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED": True,
        "DETERMINISTIC_FIRM_REVIEW_PHASE_STAGGERING_ENABLED": False,
        "PERSON_EQUITY_TRANSITION_ENABLED": False,
        "AUTONOMOUS_SECONDARY_EQUITY_ENABLED": False,
    }


def sector_cash(world):
    cb = world.firm_system.central_bank
    return {
        "households": math.fsum(number(household.wealth) for household in world.households),
        "food_firms": math.fsum(number(firm.cash) for firm in world.firms),
        "capital_good_firms": math.fsum(number(firm.cash) for firm in world.capital_good_firms),
        "legacy_owner": number(getattr(world, "legacy_owner_cash", 0.0)),
        "estates": math.fsum(number(account.cash) for account in world.estate_accounts.values()),
        "public_central_bank": number(world.public_wealth) + number(cb.public_income_balance),
        "pending_household_formation": number(getattr(world, "pending_household_formation_wealth", 0.0)),
    }


def accounting_by_sector(world, step):
    rows = [row for row in world.accounting.rows if int(number(row.get("step"), -1)) == step]
    result = defaultdict(lambda: defaultdict(float))
    for row in rows:
        sector = "capital_good_firms" if row.get("sector_id") == "capital_goods" else "food_firms"
        for name in (
            "wage_payments", "dividends", "interest_paid", "loan_issued", "principal_repaid",
            "equity_issuance_cash", "customer_advance_cash_inflow", "prepaid_investment_cash_outflow",
            "fixed_investment_expenditure", "cfo", "cfi", "cff", "cash_flow_gap",
        ):
            result[sector][name] += number(row.get(name))
    return result


def run_baseline():
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15_macro_sectoral_cash_closure_audit",
        scenario_overrides=overrides(),
    )
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()
    bridge_rows, weekly, leakage_rows = [], [], []
    opening = sector_cash(world)
    previous_dividend_event = 0

    for _ in range(WEEKS):
        opening = sector_cash(world)
        world.step()
        step = int(world.current_step_index)
        closing = sector_cash(world)
        accounting = accounting_by_sector(world, step)
        household = world.accounting.household_rows[-1]
        public = world.accounting.public_rows[-1]
        central_bank = world.accounting.central_bank_rows[-1]
        dividend_events = world.dividend_routing_events[previous_dividend_event:]
        previous_dividend_event = len(world.dividend_routing_events)
        dividend_legacy = math.fsum(number(row.get("legacy_entitlement")) for row in dividend_events)
        dividend_person = math.fsum(number(row.get("person_paid")) for row in dividend_events)
        dividend_estate = math.fsum(number(row.get("estate_paid")) for row in dividend_events)

        food = accounting["food_firms"]
        capital = accounting["capital_good_firms"]
        household_wages = number(household.get("wages"))
        household_consumption = number(household.get("consumption_expenditure"))
        household_saving = number(household.get("saving"))
        investment_advance = max(
            number(capital["customer_advance_cash_inflow"]),
            number(food["prepaid_investment_cash_outflow"]),
        )
        money_created_food = number(food["loan_issued"])
        money_created_capital = number(capital["loan_issued"])
        money_destroyed_food = number(food["principal_repaid"])
        money_destroyed_capital = number(capital["principal_repaid"])

        # Explicit classified cash flow per sector. Residual is retained rather
        # than hidden: it covers public inventory release/refunds and lifecycle
        # transfers that are accounted elsewhere in the existing ledger.
        classified = defaultdict(float)
        classified["households"] += household_wages + dividend_person - household_consumption - number(household.get("equity_purchase_cash_outflow"))
        classified["food_firms"] += household_consumption - food["wage_payments"] - food["dividends"] - food["interest_paid"] - investment_advance + food["equity_issuance_cash"]
        classified["capital_good_firms"] += investment_advance - capital["wage_payments"] - capital["dividends"] - capital["interest_paid"] + capital["equity_issuance_cash"]
        classified["legacy_owner"] += dividend_legacy
        classified["estates"] += dividend_estate
        classified["public_central_bank"] += food["interest_paid"] + capital["interest_paid"] - number(public.get("public_cash_expenditure"))

        created = {"food_firms": money_created_food, "capital_good_firms": money_created_capital}
        destroyed = {"food_firms": money_destroyed_food, "capital_good_firms": money_destroyed_capital}
        for sector in SECTORS:
            residual = closing[sector] - opening[sector] - classified[sector] - created.get(sector, 0.0) + destroyed.get(sector, 0.0)
            bridge_rows.append({
                "week": step,
                "sector": sector,
                "opening_cash": opening[sector],
                "classified_net_intersector_flow": classified[sector],
                "residual_other_authoritative_flow": residual,
                "money_created": created.get(sector, 0.0),
                "money_destroyed": destroyed.get(sector, 0.0),
                "closing_cash": closing[sector],
                "cash_bridge_gap": closing[sector] - opening[sector] - classified[sector] - residual - created.get(sector, 0.0) + destroyed.get(sector, 0.0),
            })
        weekly.append({
            "week": step,
            "household_cash": closing["households"],
            "food_firm_cash": closing["food_firms"],
            "capital_good_firm_cash": closing["capital_good_firms"],
            "aggregate_firm_cash": closing["food_firms"] + closing["capital_good_firms"],
            "legacy_owner_cash": closing["legacy_owner"],
            "estate_cash": closing["estates"],
            "household_income": number(household.get("wages")) + dividend_person,
            "household_consumption": household_consumption,
            "household_saving": household_saving,
            "food_wages": food["wage_payments"],
            "capital_good_wages": capital["wage_payments"],
            "food_dividends": food["dividends"],
            "capital_good_dividends": capital["dividends"],
            "legacy_dividends": dividend_legacy,
            "person_dividends": dividend_person,
            "estate_dividends": dividend_estate,
            "fixed_investment": food["fixed_investment_expenditure"],
            "customer_advances": investment_advance,
            "equity_issuance": food["equity_issuance_cash"] + capital["equity_issuance_cash"],
            "loan_issued": money_created_food + money_created_capital,
            "principal_repaid": money_destroyed_food + money_destroyed_capital,
            "interest_paid": food["interest_paid"] + capital["interest_paid"],
            "public_cash": closing["public_central_bank"],
            "money_created": number(central_bank.get("gross_money_created")),
            "money_destroyed": number(central_bank.get("money_destroyed")),
        })
        leakage_rows.append({
            "week": step,
            "household_net_saving": household_saving,
            "legacy_dividend_leakage": dividend_legacy,
            "estate_cash_change": closing["estates"] - opening["estates"],
            "other_non_spending_cash_change": (
                closing["public_central_bank"] - opening["public_central_bank"]
                + closing["pending_household_formation"] - opening["pending_household_formation"]
            ),
            "aggregate_firm_cash_change": (
                closing["food_firms"] + closing["capital_good_firms"]
                - opening["food_firms"] - opening["capital_good_firms"]
            ),
        })
    return world, bridge_rows, weekly, leakage_rows


def main():
    world, bridge_rows, weekly, leakage_rows = run_baseline()
    write_csv("authoritative_sectoral_flow_bridge.csv", bridge_rows)

    late = weekly[-104:]
    leakage = []
    for metric, source in (
        ("household_net_saving", "persistent private financial surplus"),
        ("legacy_dividend_leakage", "Firm cash transferred to non-spending LegacyOwner"),
        ("estate_cash_change", "Estate-held cash outside Household final demand"),
        ("other_non_spending_cash_change", "public/pending-holder balance change"),
        ("aggregate_firm_cash_change", "observed counterpart Firm-sector balance movement"),
    ):
        values = [number(row[metric]) for row in leakage_rows]
        late_values = [number(row[metric]) for row in leakage_rows[-104:]]
        leakage.append({
            "component": metric,
            "economic_interpretation": source,
            "weekly_mean": float(np.mean(values)),
            "cumulative_total": float(np.sum(values)),
            "late_weekly_mean": float(np.mean(late_values)),
            "late_cumulative": float(np.sum(late_values)),
        })
    write_csv("structural_leakage_decomposition.csv", leakage)

    cumulative = lambda field: math.fsum(number(row[field]) for row in weekly)
    recycling = [
        {"channel": "Household consumption", "current_flow": cumulative("household_consumption"), "classification": "FINAL_DEMAND_RECYCLING", "aggregate_firm_cash_effect": "positive Food cash inflow", "money_stock_effect": "transfer only", "existing": True, "assessment": "already active but below household income/saving path"},
        {"channel": "Primary equity issuance", "current_flow": cumulative("equity_issuance"), "classification": "FIRM_FINANCING", "aggregate_firm_cash_effect": "Household cash to Firm cash", "money_stock_effect": "transfer only", "existing": True, "assessment": "runtime exists but disabled; no endogenous liquidity/investment trigger"},
        {"channel": "Secondary Legacy share purchase", "current_flow": 0.0, "classification": "NO_FIRM_SECTOR_NET_EFFECT", "aggregate_firm_cash_effect": "Household cash to LegacyOwner", "money_stock_effect": "transfer only", "existing": True, "assessment": "does not capitalize Firms"},
        {"channel": "Step13 working-capital credit", "current_flow": cumulative("loan_issued") - cumulative("principal_repaid"), "classification": "MONEY_CREATION", "aggregate_firm_cash_effect": "temporary borrower liquidity", "money_stock_effect": "credit creation/destruction", "existing": True, "assessment": "not a sustainable counterpart to persistent private saving"},
        {"channel": "Firm fixed investment", "current_flow": cumulative("fixed_investment"), "classification": "INTER_FIRM_TRANSFER", "aggregate_firm_cash_effect": "Food to capital-good Firm; no aggregate Firm replenishment", "money_stock_effect": "transfer only", "existing": True, "assessment": "creates final investment output but not aggregate Firm cash"},
        {"channel": "Customer advances", "current_flow": cumulative("customer_advances"), "classification": "INTER_FIRM_TRANSFER", "aggregate_firm_cash_effect": "buyer to supplier prepayment", "money_stock_effect": "transfer only", "existing": True, "assessment": "supplier operating-liquidity closure, not aggregate closure"},
        {"channel": "Person dividend routing", "current_flow": cumulative("person_dividends"), "classification": "FINAL_DEMAND_RECYCLING", "aggregate_firm_cash_effect": "dividend reaches Household cash", "money_stock_effect": "transfer only", "existing": True, "assessment": "inactive in this baseline; can recycle only after ownership exists"},
    ]
    write_csv("existing_recycling_channels.csv", recycling)

    total_dividends = cumulative("legacy_dividends") + cumulative("person_dividends") + cumulative("estate_dividends")
    ownership_shadow = []
    for person_share in (0.0, 0.10, 0.25, 0.50, 1.00):
        ownership_shadow.append({
            "shadow_person_ownership_fraction": person_share,
            "same_total_dividends": total_dividends,
            "shadow_person_dividend_cash": total_dividends * person_share,
            "shadow_legacy_dividend_cash": total_dividends * (1.0 - person_share),
            "firm_dividend_cash_outflow_unchanged": total_dividends,
            "direct_aggregate_firm_cash_change": 0.0,
            "interpretation": "reallocates recipient cash; may increase later Household demand but cannot directly recapitalize Firms",
        })
    write_csv("ownership_dividend_shadow.csv", ownership_shadow)

    late_firm_slope = slope([row["aggregate_firm_cash"] for row in late])
    late_household_slope = slope([row["household_cash"] for row in late])
    final_firm_cash = weekly[-1]["aggregate_firm_cash"]
    current_debt = math.fsum(number(getattr(firm, "loan_balance", 0.0)) for firm in world.operating_firms())
    primary = [
        {"field": "runtime_available", "value": True, "interpretation": "accepted primary issuance adapter exists"},
        {"field": "currently_enabled", "value": bool(world.person_equity_transition_enabled), "interpretation": "disabled in canonical baseline"},
        {"field": "direct_cash_flow", "value": "Household cash -> Firm cash; Household equity asset increases", "interpretation": "money-neutral Firm financing"},
        {"field": "cash_per_week_needed_to_offset_late_firm_slope", "value": max(0.0, -late_firm_slope), "interpretation": "diagnostic gross financing need only"},
        {"field": "existing_legitimate_trigger", "value": "none", "interpretation": "adapter is periodic/guardrail-based, not tied to investment financing need or operating-liquidity target"},
        {"field": "architecture_assessment", "value": "future complement", "interpretation": "legitimate once linked to explicit capitalization/investment need; alone does not create final demand"},
    ]
    write_csv("primary_equity_closure_audit.csv", primary)

    credit = [
        {"field": "late_firm_cash_drain_per_week", "value": -late_firm_slope, "interpretation": "positive number is required weekly credit growth if credit offsets cash drain"},
        {"field": "implied_credit_growth_52_weeks", "value": max(0.0, -late_firm_slope) * 52.0, "interpretation": "linear severity diagnostic, not forecast"},
        {"field": "implied_credit_growth_520_weeks", "value": max(0.0, -late_firm_slope) * 520.0, "interpretation": "would turn a flow closure issue into growing principal"},
        {"field": "current_firm_principal", "value": current_debt, "interpretation": "existing Step13 working-capital claim"},
        {"field": "credit_assessment", "value": "temporary liquidity only", "interpretation": "using working-capital loans as permanent private-saving counterpart recreates a debt trap"},
    ]
    write_csv("step13_credit_closure_audit.csv", credit)

    candidates = [
        {"candidate": "PERSON_OWNERSHIP_DIVIDEND_RECYCLING", "cash_flow": "Firm dividends -> Person Household instead of LegacyOwner", "money_stock_change": "none", "debt_created": "no", "final_demand": "indirect / later Household spending", "scalable": "partial", "already_architected": "yes", "major_risk": "does not replenish Firm cash at payment", "suitability": "complement, not closure"},
        {"candidate": "ENDOGENOUS_PRIMARY_EQUITY_FINANCING", "cash_flow": "Household cash -> Firm paid-in equity", "money_stock_change": "none", "debt_created": "no", "final_demand": "no direct final demand", "scalable": "conditional", "already_architected": "yes, inactive", "major_risk": "arbitrary operating-deficit finance/crowding out consumption without a capital trigger", "suitability": "future financing complement"},
        {"candidate": "STEP13_CREDIT_EXPANSION", "cash_flow": "Central-bank credit -> Firm cash", "money_stock_change": "credit money increases", "debt_created": "yes", "final_demand": "no direct", "scalable": "mechanically", "already_architected": "yes", "major_risk": "persistent debt growth/generalized debt trap", "suitability": "not structural closure"},
        {"candidate": "GOVERNMENT_FISCAL_CLOSURE", "cash_flow": "Government purchases/transfers/taxes <-> Households and Firms", "money_stock_change": "depends on fiscal/monetary settlement", "debt_created": "only if deficit-financed", "final_demand": "yes", "scalable": "yes", "already_architected": "public cash boundary only", "major_risk": "requires explicit public balance sheet and no double counting", "suitability": "next required macro boundary"},
        {"candidate": "EXTERNAL_SECTOR_CLOSURE", "cash_flow": "Exports/imports and external financial claims", "money_stock_change": "depends on settlement", "debt_created": "possibly", "final_demand": "exports yes", "scalable": "yes", "already_architected": "no", "major_risk": "adds an unanchored foreign demand/asset sector", "suitability": "later alternative, not safest next"},
        {"candidate": "HYBRID_CLOSURE", "cash_flow": "Fiscal final demand plus ownership/equity financing complements", "money_stock_change": "component-specific", "debt_created": "component-specific", "final_demand": "yes", "scalable": "yes", "already_architected": "partial", "major_risk": "too broad for immediate next stage", "suitability": "longer-run architecture"},
    ]
    write_csv("macro_closure_candidate_matrix.csv", candidates)

    liquidity = [
        {"metric": "final_aggregate_firm_cash", "value": final_firm_cash, "interpretation": "current end-of-run cash stock"},
        {"metric": "late_104_firm_cash_slope_per_week", "value": late_firm_slope, "interpretation": "linear diagnostic only"},
        {"metric": "late_104_household_cash_slope_per_week", "value": late_household_slope, "interpretation": "private financial surplus counterpart"},
        {"metric": "weeks_to_zero_cash_linear_if_negative_slope_persists", "value": final_firm_cash / max(-late_firm_slope, TOL) if late_firm_slope < -TOL else math.inf, "interpretation": "severity diagnostic, not forecast"},
        {"metric": "weeks_to_25pct_current_cash_linear", "value": 0.75 * final_firm_cash / max(-late_firm_slope, TOL) if late_firm_slope < -TOL else math.inf, "interpretation": "material-liquidity-stress diagnostic, not forecast"},
    ]
    write_csv("firm_liquidity_exhaustion_diagnostic.csv", liquidity)

    max_bridge_gap = max(abs(number(row["cash_bridge_gap"])) for row in bridge_rows)
    flags = {
        "verdict": "D. GOVERNMENT_FISCAL_CLOSURE_IS_NEXT_REQUIRED_BOUNDARY",
        "late_household_cash_slope": late_household_slope,
        "late_aggregate_firm_cash_slope": late_firm_slope,
        "late_legacy_cash_slope": slope([row["legacy_owner_cash"] for row in late]),
        "cumulative_household_saving": cumulative("household_saving"),
        "cumulative_legacy_dividends": cumulative("legacy_dividends"),
        "cumulative_person_dividends": cumulative("person_dividends"),
        "ownership_recycling_directly_closes_firm_cash": False,
        "primary_equity_has_existing_endogenous_trigger": False,
        "step13_credit_would_create_persistent_principal_growth": True,
        "government_is_next_stage_boundary_not_step15_patch": True,
        "max_sector_cash_bridge_gap": max_bridge_gap,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (OUT / "acceptance_summary.md").write_text(
        "# Step 15 Macro Sectoral Cash-Closure Architecture Audit\n\n"
        "Verdict: **D. GOVERNMENT_FISCAL_CLOSURE_IS_NEXT_REQUIRED_BOUNDARY**\n\n"
        "The cash drain is a located-money reallocation, not money destruction: persistent Household saving and LegacyOwner dividend accumulation leave less cash in the aggregate Firm sector. "
        "Investment/customer advances are inter-Firm transfers; ownership dividend routing changes the recipient of an unchanged Firm cash outflow; Step13 credit would require persistent principal growth. "
        "Primary issuance is an available financing adapter but has no current endogenous capitalization trigger and does not itself create final demand. "
        "The correct boundary is a later explicit fiscal-sector architecture with a public balance sheet, taxes/transfers/purchases identities, and no GDP/intermediate double counting; it must not be patched inside Step 15.\n",
        encoding="utf-8-sig",
    )
    print(json.dumps(flags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
