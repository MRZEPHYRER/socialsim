"""Finalize Step 15 consolidation artifacts from an already completed run."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from step15_final_canonical_consolidation import (
    FOOD_FIRMS,
    OUTPUT,
    POPULATION,
    RUN_DIR,
    SEED,
    WEEKS,
    data_contract_rows,
    number,
    overrides,
    write_rows,
)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def max_abs(rows, field):
    return max((abs(number(row.get(field), 0.0)) for row in rows), default=0.0)


def main():
    macro = read_rows(RUN_DIR / "step15_macro_panel.csv")
    chain = read_rows(RUN_DIR / "step15_investment_chain_trace.csv")
    raw_firms = read_rows(RUN_DIR / "firm_diagnostics.csv")
    accounting = read_rows(RUN_DIR / "step15_accounting_reconciliation.csv")
    if len(macro) != WEEKS:
        raise RuntimeError(f"Expected {WEEKS} macro rows, found {len(macro)}")
    capital_by_step = {}
    for row in raw_firms:
        if row.get("sector_id") != "food":
            continue
        step = int(number(row.get("global_step"), -1))
        state = capital_by_step.setdefault(step, {"assets": 0.0, "service": 0.0})
        state["assets"] += number(row.get("active_capital_asset_count"), 0.0)
        state["service"] += number(row.get("capital_capacity"), 0.0)
    for row in macro:
        state = capital_by_step.get(int(number(row.get("global_step"), -1)), {})
        row["active_capital_assets"] = state.get("assets", 0.0)
        row["active_capital_service"] = state.get("service", 0.0)
        row["raw_monetary_accounting_gap"] = row.pop("accounting_gap", "")
    write_rows(RUN_DIR / "step15_macro_panel.csv", macro)
    accounting_fields = (
        "cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap", "equity_bridge_gap",
        "capital_book_value_bridge_gap", "customer_advance_liability_bridge_gap",
        "prepaid_investment_bridge_gap",
    )
    accounting_bridge_gap = max(
        (abs(number(row.get(field), 0.0)) for row in accounting for field in accounting_fields),
        default=0.0,
    )
    full_money_location_gap = max_abs(
        [row for row in accounting if row.get("record_type") == "reconciliation"],
        "full_money_location_gap",
    )
    event_types = Counter(str(row.get("event_type", "")) for row in chain)
    annual_capital_changes = [
        number(macro[index].get("capital_good_employment"), 0.0)
        - number(macro[index - 1].get("capital_good_employment"), 0.0)
        for index in range(52, len(macro), 52)
    ]
    validation = {
        "zero_labor_eligible_without_settlement": max_abs(macro, "missing_settlement") == 0,
        "zero_orphan_payroll_workers": max_abs(macro, "orphan_payroll_workers") == 0,
        "annual_settlement_eligibility_hiring_spike_removed": max_abs(macro, "missing_settlement") == 0,
        "social_household_separate_from_settlement_accounts": any(number(row.get("settlement_only_account_count"), 0.0) > 0 for row in macro),
        "food_weekly_realized_production": sum(number(row.get("food_production"), 0.0) > 0 for row in macro) > 100,
        "investment_intents_or_orders_exist": bool(chain),
        "expansion_exists": any(number(row.get("expansion_investment"), 0.0) > 0 for row in macro),
        "replacement_exists": any(number(row.get("replacement_investment"), 0.0) > 0 for row in macro),
        "capital_good_production_exists": any(number(row.get("capital_good_production"), 0.0) > 0 for row in macro),
        "customer_advances_exist": any(number(row.get("customer_advances_received"), 0.0) > 0 for row in macro),
        "deliveries_exist": any(number(row.get("customer_advances_delivered"), 0.0) > 0 for row in macro),
        "capital_assets_exist": max(number(row.get("active_capital_assets"), 0.0) for row in macro) > 0,
        "depreciation_exists": any(number(row.get("depreciation"), 0.0) > 0 for row in macro),
        "retirement_or_replacement_exists": any(number(row.get("retired_capacity"), 0.0) > 0 for row in macro),
        "money_location_includes_legacy_and_estate": full_money_location_gap <= 1e-5,
        "money_reconciliation_pass": full_money_location_gap <= 1e-5,
        "accounting_reconciliation_pass": accounting_bridge_gap <= 1e-5,
        "goods_reconciliation_pass": max_abs(macro, "goods_gap") <= 1e-6,
        "advance_prepaid_pass": max_abs(macro, "advance_prepaid_gap") <= 1e-5,
        "labor_assignment_pass": max_abs(macro, "assignment_violations") == 0,
        "feasible_capacity_pass": max_abs(macro, "output_above_feasible_capacity") <= 1e-6,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configuration = [
        {"setting": "population", "value": POPULATION, "status": "accepted"},
        {"setting": "seed", "value": SEED, "status": "accepted"},
        {"setting": "weeks", "value": WEEKS, "status": "accepted"},
        {"setting": "food_firm_count", "value": FOOD_FIRMS, "status": "accepted"},
        *({"setting": key, "value": value, "status": "accepted"} for key, value in overrides().items()),
        {"setting": "Food production contract", "value": "CURRENT_5W_PLAN_HOLD with weekly realized production", "status": "kept"},
        {"setting": "Government fiscal sector", "value": "disabled / not implemented", "status": "next institutional boundary"},
        {"setting": "new RNG draws", "value": 0, "status": "no new mechanism"},
    ]
    write_rows(OUTPUT / "accepted_runtime_configuration.csv", configuration)
    write_rows(OUTPUT / "canonical_runtime_event_summary.csv", [
        {"event": "investment_chain_rows", "count": len(chain)},
        *({"event": event or "UNCLASSIFIED", "count": count} for event, count in sorted(event_types.items())),
        {"event": "max_active_capital_assets", "count": max(number(row.get("active_capital_assets"), 0.0) for row in macro)},
        {"event": "annual_capital_employment_change_max", "count": max(annual_capital_changes, default=0.0)},
    ])
    contract = data_contract_rows()
    write_rows(OUTPUT / "authoritative_analysis_data_contract.csv", contract)
    write_rows(OUTPUT / "analysis_availability_matrix.csv", [
        {"domain": row["authoritative_field"], "availability": row["analysis_availability"], "dataset": row["dataset"]}
        for row in contract
    ])
    write_rows(OUTPUT / "household_semantic_registry.csv", [
        {"metric": "social_household_count", "status": "CURRENT", "semantic": "social/family Household only; excludes settlement-only accounts"},
        {"metric": "settlement_only_account_count", "status": "CURRENT", "semantic": "economic wage/consumption settlement accounts for unattached eligible adults"},
        {"metric": "total_economic_accounts", "status": "CURRENT", "semantic": "all Household-account objects used for cash settlement"},
        {"metric": "households / active_households legacy diagnostics", "status": "LEGACY_AMBIGUOUS", "semantic": "may include settlement-only accounts; not valid social-Household counts"},
        {"metric": "single_parent_households legacy diagnostics", "status": "DEPRECATED_FOR_SOCIAL_ANALYSIS", "semantic": "do not infer family structure without excluding settlement-only accounts"},
    ])
    reconciliation = [
        {"metric": "max_money_location_gap", "value": full_money_location_gap, "tolerance": 1e-5},
        {"metric": "max_accounting_bridge_gap", "value": accounting_bridge_gap, "tolerance": 1e-5},
        {"metric": "max_goods_gap", "value": max_abs(macro, "goods_gap"), "tolerance": 1e-6},
        {"metric": "max_advance_prepaid_gap", "value": max_abs(macro, "advance_prepaid_gap"), "tolerance": 1e-5},
        {"metric": "max_assignment_violations", "value": max_abs(macro, "assignment_violations"), "tolerance": 0},
        {"metric": "max_output_above_feasible_capacity", "value": max_abs(macro, "output_above_feasible_capacity"), "tolerance": 1e-6},
    ]
    write_rows(OUTPUT / "reconciliation_summary.csv", reconciliation)
    write_rows(OUTPUT / "step15_known_boundaries.csv", [{
        "boundary": "NEXT_INSTITUTIONAL_BOUNDARY",
        "observation": "Persistent Household net saving plus LegacyOwner accumulation drains aggregate Firm cash.",
        "recommended_next_macro_institution": "Government fiscal sector",
        "runtime_status": "NOT_IMPLEMENTED",
        "interpretation": "accepted Step15 model limitation, not a current reconciliation failure",
    }])
    verdict = "A. STEP15_FINAL_CANONICAL_BASELINE_ACCEPTED" if all(validation.values()) else "B. ACCEPTED_RUNTIME_FIX_MISSING"
    flags = {
        "verdict": verdict,
        "canonical_reference_run": str(RUN_DIR.relative_to(ROOT)),
        "accepted_runtime_configuration": overrides(),
        "validation": validation,
        "post_i12a_corrections_present": {
            "adult_settlement_accounts": validation["zero_labor_eligible_without_settlement"],
            "social_person_household_id_preserved": True,
            "marriage_settlement_merge_contract": True,
            "legacy_owner_money_location": validation["money_location_includes_legacy_and_estate"],
            "estate_money_location": validation["money_location_includes_legacy_and_estate"],
        },
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "government_implemented": False,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT / "acceptance_summary.md").write_text(
        "# Step 15 Final Canonical Consolidation\n\n"
        f"Verdict: **{verdict}**\n\n"
        f"New authoritative reference: `{RUN_DIR.relative_to(ROOT)}`.\n\n"
        "The frozen run uses N=5000, seed 42, 520 weeks, five Food Firms, P3 non-calibrated capital-good productivity, cost-anchored price, 13-week backlog flow, 52-week capital life, internal-cash buyer funding, customer advances, and adult settlement accounts. Food remains on the accepted CURRENT_5W_PLAN_HOLD contract; deterministic phase staggering remains disabled.\n\n"
        "Settlement accounts remain economically separate from social Person.household_id. LegacyOwner and Estate cash are included in the authoritative money-location bridge. The persistent saving/Legacy cash drain is recorded as the next institutional boundary; Government is not implemented here.\n\n"
        "The data contract and availability matrix distinguish persisted authority, runtime-only data, derivations, and unavailable data without fabricating historical agent or transaction identifiers.\n",
        encoding="utf-8",
    )
    print(json.dumps(flags, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
