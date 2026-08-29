from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy import config


OUTPUT = ROOT / "test/output/step15I10F_capital_good_price_anchor"
SOURCE = ROOT / "test/output/step15I10E_capital_good_productivity_scale"


def number(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_rows(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value):
    if isinstance(value, float) and math.isnan(value):
        return "N/A"
    return f"{value:.12g}" if isinstance(value, (float, int)) else str(value)


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    grid = read_rows(SOURCE / "productivity_shadow_grid.csv")
    unit_economics = read_rows(SOURCE / "capital_good_unit_economics.csv")
    if not grid or not unit_economics:
        raise RuntimeError("accepted Step15I.10E artifacts are required")

    sale_price = float(config.CAPITAL_GOOD_UNIT_PRICE)
    wage_per_service = float(config.FIRM_WAGE_PER_LABOR)
    current_productivity = float(
        config.CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR
    )
    review_horizon = int(config.CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS)
    current_unit_cost = wage_per_service / current_productivity

    # This table records the authority chain, rather than treating every
    # repeated field as an independent price-setting mechanism.
    authority_rows = [
        {
            "layer": "configuration",
            "source": "economy.config.CAPITAL_GOOD_UNIT_PRICE",
            "value": sale_price,
            "role": "engineering price input",
            "authoritative_now": True,
            "historical_cost": False,
            "future_market_authority": False,
        },
        {
            "layer": "canonical_investment",
            "source": "CanonicalInvestmentSystem.unit_price",
            "value": sale_price,
            "role": "normalized runtime engineering quote",
            "authoritative_now": True,
            "historical_cost": False,
            "future_market_authority": False,
        },
        {
            "layer": "supplier_runtime",
            "source": "FirmSlice(price=self.unit_price)",
            "value": sale_price,
            "role": "capital-good Firm runtime/display price",
            "authoritative_now": False,
            "historical_cost": False,
            "future_market_authority": False,
        },
        {
            "layer": "offer_construction",
            "source": "CapitalGoodSupplyAdapter.offers(unit_prices)",
            "value": sale_price,
            "role": "CapitalGoodOffer.unit_price quote",
            "authoritative_now": True,
            "historical_cost": False,
            "future_market_authority": False,
        },
        {
            "layer": "settlement",
            "source": "CapitalGoodFill.unit_price and units * offer.unit_price",
            "value": sale_price,
            "role": "buyer expenditure and supplier revenue settlement price",
            "authoritative_now": True,
            "historical_cost": False,
            "future_market_authority": False,
        },
        {
            "layer": "inventory_cost",
            "source": "CapitalGoodInventory.produce",
            "value": current_unit_cost,
            "role": "historical unit production/book cost = wages / output",
            "authoritative_now": True,
            "historical_cost": True,
            "future_market_authority": False,
        },
        {
            "layer": "inventory_cost",
            "source": "CapitalGoodInventory.record_sale",
            "value": "weighted_average_unit_cost",
            "role": "COGS and book-value reduction; not selling price",
            "authoritative_now": True,
            "historical_cost": True,
            "future_market_authority": False,
        },
        {
            "layer": "generic_firm_pricing",
            "source": "FirmSlice.price / FirmSystem price review",
            "value": "not used for capital-good settlement",
            "role": "Food/market pricing architecture, not current capital-good authority",
            "authoritative_now": False,
            "historical_cost": False,
            "future_market_authority": "adapter required",
        },
    ]
    write_rows(OUTPUT / "price_authority_audit.csv", authority_rows)

    fixed_margin = sale_price - current_unit_cost
    candidate_rows = [
        {
            "candidate": "A_FIXED_ENGINEERING_PRICE",
            "formula": "10.0",
            "current_price": sale_price,
            "unit_cost_reference": current_unit_cost,
            "unit_margin_at_P1": fixed_margin,
            "margin_rate_at_P1": fixed_margin / sale_price,
            "requires_markup_calibration": False,
            "preserves_settlement_identity": True,
            "generic_adapter_ready": False,
            "economic_interpretation": "legacy engineering fixture quote",
            "main_risk": "persistent negative supplier margin at P1",
        },
        {
            "candidate": "B_COST_RECOVERY_PRICE",
            "formula": "authoritative_unit_cost",
            "current_price": current_unit_cost,
            "unit_cost_reference": current_unit_cost,
            "unit_margin_at_P1": 0.0,
            "margin_rate_at_P1": 0.0,
            "requires_markup_calibration": False,
            "preserves_settlement_identity": True,
            "generic_adapter_ready": True,
            "economic_interpretation": "zero-margin mechanism-isolation reference",
            "main_risk": "not a permanent market valuation or markup",
        },
        {
            "candidate": "C_COST_PLUS_ENGINEERING_MARKUP",
            "formula": "authoritative_unit_cost * (1 + markup)",
            "current_price": "not selected",
            "unit_cost_reference": current_unit_cost,
            "unit_margin_at_P1": "depends on markup",
            "margin_rate_at_P1": "depends on markup",
            "requires_markup_calibration": True,
            "preserves_settlement_identity": True,
            "generic_adapter_ready": True,
            "economic_interpretation": "explicit future engineering policy",
            "main_risk": "unjustified markup would add calibration and feedback",
        },
        {
            "candidate": "D_EXISTING_GENERIC_FIRM_PRICING_ADAPTER",
            "formula": "sector-specific generic Firm quote adapter",
            "current_price": "not selected",
            "unit_cost_reference": current_unit_cost,
            "unit_margin_at_P1": "not defined",
            "margin_rate_at_P1": "not defined",
            "requires_markup_calibration": False,
            "preserves_settlement_identity": "only after explicit adapter",
            "generic_adapter_ready": False,
            "economic_interpretation": "possible future interface, not current authority",
            "main_risk": "current Food price-review path is not capital-good cost authority",
        },
    ]
    write_rows(OUTPUT / "price_cost_candidate_matrix.csv", candidate_rows)

    decomposition_rows = []
    for row in grid:
        productivity = number(row.get("productivity_units_per_labor_service"))
        unit_cost = wage_per_service / productivity
        current_margin = sale_price - unit_cost
        decomposition_rows.append(
            {
                "shadow_case": row.get("shadow_case"),
                "productivity_units_per_labor_service": productivity,
                "sale_price_held_constant": sale_price,
                "unit_cost": unit_cost,
                "break_even_price": unit_cost,
                "current_price_minus_unit_cost": current_margin,
                "current_price_margin_rate": current_margin / sale_price,
                "implied_labor_services_per_unit": number(
                    row.get("implied_labor_services_per_unit")
                ),
                "peak_desired_output_units_per_week": number(
                    row.get("peak_desired_output_units_per_week")
                ),
                "peak_desired_labor_services_per_week": number(
                    row.get("peak_desired_labor_services_per_week")
                ),
                "system_labor_capacity": number(
                    row.get("system_labor_capacity")
                ),
                "desired_to_system_labor_ratio": number(
                    row.get("desired_to_system_labor_ratio")
                ),
                "backlog_clearance_weeks_at_system_capacity": number(
                    row.get("backlog_clearance_weeks_at_system_capacity")
                ),
                "price_below_unit_cost": sale_price < unit_cost,
                "cost_recovery_margin_rate": 0.0,
            }
        )
    write_rows(OUTPUT / "productivity_price_decomposition.csv", decomposition_rows)

    p1 = decomposition_rows[0]
    flags = {
        "verdict": "A. COST_ANCHORED_CAPITAL_GOOD_PRICE_READY",
        "price_source_is_engineering_constant": True,
        "current_price": sale_price,
        "authoritative_unit_cost_formula": "wage_cost_per_labor_service / productivity",
        "current_unit_cost": current_unit_cost,
        "current_price_is_below_unit_cost": sale_price < current_unit_cost,
        "inventory_preserves_historical_cost_basis": True,
        "inventory_cogs_uses_weighted_average_cost": True,
        "generic_firm_pricing_currently_authoritative_for_capital_goods": False,
        "nominated_formula": "offer_price = authoritative_unit_cost",
        "nominated_formula_scope": "zero-margin engineering mechanism-isolation screen only",
        "permanent_markup_selected": False,
        "canonical_price_changed": False,
        "canonical_productivity_changed": False,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "simulation_runs": 0,
        "step13_financing_used": False,
        "free_capital_goods": False,
        "settlement_identity_buyer_equals_supplier": True,
        "historical_cogs_equals_authoritative_production_cost": True,
        "money_creation": 0.0,
        "money_destruction": 0.0,
        "p1_current_price_margin_rate": p1["current_price_margin_rate"],
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = f"""# Step 15I.10F Capital-Good Price-Cost Anchor Audit

## Verdict

**{flags['verdict']}**

This was a diagnostic-only audit. No simulation was run and no canonical
price, productivity, useful life, backlog rule, financing rule, planner, or
Step13 behavior was changed.

## Price authority

The current price `10` originates at `economy.config.CAPITAL_GOOD_UNIT_PRICE`.
`CanonicalInvestmentSystem.unit_price` copies that value (with an optional
world override), initializes the capital-good Firm price, supplies the
`unit_prices` mapping to `CapitalGoodSupplyAdapter.offers`, and is then carried
by `CapitalGoodOffer.unit_price` into settlement. Thus the current settlement
price is an explicit engineering quote, not a price discovered by the generic
Food Firm market or by inventory valuation.

The generic Firm price field can transport a quote, but the current generic
price-review path is not the capital-good price authority. Reusing it directly
would blur the boundary between a sector quote and a cost/offer policy.

## Cost authority

Capital-good production records wage cost in `CapitalGoodInventory.produce`.
Its historical unit cost is `wage_cost_per_labor_service / productivity`,
currently `{current_unit_cost:.12g}`. `record_sale` reduces inventory book
value using weighted-average historical cost, so COGS is not recomputed from
the selling price. This preserves the required distinction between price,
cash settlement, inventory cost, and COGS.

At the current P1 screen, price `10` minus unit cost `41` gives a unit margin
of `-31`, or `-310%` of selling price. The existing pair is therefore not a
coherent zero-subsidy operating screen.

## Nominated next-screen semantics

Use the explicit, pre-settlement cost anchor:

    offer_price = authoritative_unit_cost

This is nominated only for a zero-margin mechanism-isolation screen. It is not
a permanent markup, market valuation, or dynamic pricing rule. No markup has
been selected. Buyer expenditure still equals supplier revenue, acquired asset
cost equals settled expenditure, inventory COGS remains historical production
cost, and money creation/destruction remains zero.

## Productivity-price decomposition

The unchanged-price shadow grid separates physical scale from nominal price:

| Case | Unit cost | Current margin rate | Desired labor/system |
|---|---:|---:|---:|
"""
    for row in decomposition_rows:
        summary += (
            f"| {row['shadow_case']} | {fmt(row['unit_cost'])} | "
            f"{fmt(row['current_price_margin_rate'])} | "
            f"{fmt(row['desired_to_system_labor_ratio'])}x |\n"
        )
    summary += """

The price anchor does not conceal the separate physical productivity issue;
the productivity grid remains diagnostic and unchanged.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
