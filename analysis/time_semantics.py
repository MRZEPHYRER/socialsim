"""Canonical analysis-side weekly time semantics.

This module contains reporting metadata and conversion helpers only. It must
never feed values back into simulation decisions.
"""

import json
import os

from time_system import STEPS_PER_YEAR, steps_to_years, years_to_steps


TIME_SCHEMA = "weekly"
FLOW_FREQUENCY = "weekly"


FIELD_SEMANTICS = {
    "population": ("stock", "week_end", "Population"),
    "active_households": ("stock", "week_end", "Active households"),
    "households": ("stock", "week_end", "Total household objects"),
    "total_household_wealth": ("stock", "week_end", "Household cash wealth"),
    "total_income": ("weekly_flow", "per_week", "Weekly household income"),
    "total_consumption": ("weekly_flow", "per_week", "Weekly consumption"),
    "total_saving": ("weekly_flow", "per_week", "Weekly saving"),
    "saving_rate": ("ratio", "weekly_flow_ratio", "Weekly saving rate"),
    "food_output_units": ("weekly_flow", "per_week", "Weekly production units"),
    "market_sales_revenue": ("weekly_flow", "per_week", "Weekly market sales revenue"),
    "household_planning_price_index": ("ratio", "ex_ante_weekly_market", "Household planning price"),
    "realized_transaction_price_index": ("ratio", "realized_weekly_market", "Realized transaction price"),
    "legacy_food_price": ("ratio", "legacy_compatibility", "Legacy representative food price"),
    "food_inventory_units": ("stock", "week_end", "Food inventory units"),
    "food_spoilage_units": ("weekly_flow", "per_week", "Weekly spoilage units"),
    "firm_cash": ("stock", "week_end", "Firm cash"),
    "firm_profit_before_dividend": ("weekly_flow", "per_week", "Weekly legacy firm profit"),
    "firm_profit": ("weekly_flow", "per_week", "Weekly firm profit"),
    "loan_balance": ("stock", "week_end", "Working-capital loan balance"),
    "loan_issued": ("weekly_flow", "per_week", "Weekly loan issuance"),
    "working_capital_loan_repaid": ("weekly_flow", "per_week", "Weekly principal repayment"),
    "loan_repaid": ("weekly_flow", "per_week", "Weekly principal repayment"),
    "total_money_stock": ("stock", "week_end", "Total money stock"),
    "money_stock": ("stock", "week_end", "Total money stock"),
    "births": ("event", "per_week", "Births this week"),
    "deaths": ("event", "per_week", "Deaths this week"),
    "birth_rate": ("ratio", "weekly_realized_event_rate", "Weekly realized birth event rate"),
    "death_rate": ("ratio", "weekly_realized_event_rate", "Weekly realized death event rate"),
    "inventory_coverage_weeks": ("ratio", "weeks", "Inventory coverage (weeks)"),
    "inventory_coverage": ("ratio", "weeks", "Inventory coverage (weeks)"),
    "capacity_utilization": ("ratio", "weekly_state", "Capacity utilization"),
    "simplified_money_velocity": ("ratio", "weekly_flow_over_stock", "Weekly simplified money velocity"),
    "gdp_nominal_output_value": ("weekly_flow", "per_week", "Weekly nominal output value"),
}


LEGACY_AMBIGUITIES = {
    "households": "Raw object count can spike at the 52-week marriage market; use active_households for macro analysis.",
    "legacy_food_price": "Compatibility state only; not the primary multi-firm market price.",
    "gdp_nominal_output_value": "Legacy field name; value is weekly nominal output, not annual GDP.",
    "birth_rate": "Weekly realized births/population, not an annual demographic rate.",
    "death_rate": "Weekly realized deaths/population, not an annual demographic rate.",
}


def weeks_to_years(weeks):
    return steps_to_years(weeks)


def years_to_weeks(years):
    return years_to_steps(years)


def field_manifest():
    result = []
    for field_name, (category, time_basis, display_label) in FIELD_SEMANTICS.items():
        result.append({
            "field_name": field_name,
            "category": category,
            "time_basis": time_basis,
            "display_label": display_label,
            "source": "diagnostics.csv",
            "legacy_ambiguity": LEGACY_AMBIGUITIES.get(field_name, ""),
            "action_taken": "weekly analysis label/alias" if field_name not in LEGACY_AMBIGUITIES else "retained field; clarified analysis semantics",
        })
    return result


def build_analysis_metadata(start_global_step, end_global_step, source_kind="current_weekly_run"):
    elapsed_weeks = max(0, int(end_global_step) - int(start_global_step))
    return {
        "time_schema": TIME_SCHEMA,
        "steps_per_year": STEPS_PER_YEAR,
        "flow_frequency": FLOW_FREQUENCY,
        "analysis_start_global_step": int(start_global_step),
        "analysis_end_global_step": int(end_global_step),
        "elapsed_weeks": elapsed_weeks,
        "elapsed_years": weeks_to_years(elapsed_weeks),
        "absolute_axis_label": "Simulation week",
        "elapsed_axis_label": "Elapsed week",
        "historical_artifact_policy": "Step 10/11 outputs are LEGACY MIXED-TIME REFERENCE and are not relabeled.",
        "source_kind": source_kind,
        "household_count_primary": "active_households",
        "market_price_primary": ["household_planning_price_index", "realized_transaction_price_index"],
        "legacy_market_price": "legacy_food_price",
        "working_capital_buffer_semantics": "industry-wide fixed 150000 allocated by productive-capacity share, plus each firm's weekly wage bill",
        "principal_repayment_semantics": "35% weekly target/cap, constrained by firm cash above allocated buffer",
        "household_reserve_semantics": "six calendar months = 26 weekly minimum-cost periods",
        "spoilage_semantics": "1% of eligible remaining inventory per week",
        "firm_timing_contract": {
            "expected_demand_ema": "weekly, alpha=0.10",
            "production_review_interval_weeks": 5,
            "inventory_target_coverage_weeks": 15,
            "price_review_probability_per_week": 0.20,
            "price_evaluation_window_weeks": 12,
            "profit_ema": "weekly, alpha=0.20",
            "price_trials_per_review": ["-1%", "-0.5%", "hold", "+0.5%", "+1%"],
            "exploration_probability_per_review": 0.08,
        },
        "field_semantics": field_manifest(),
    }


def export_time_semantics_manifest(path, start_global_step, end_global_step, source_kind="current_weekly_run"):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = build_analysis_metadata(start_global_step, end_global_step, source_kind)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    return path
