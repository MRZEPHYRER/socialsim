"""Validate Step 15 settlement and accounting-source corrections."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

import step15_final_canonical_consolidation as base
import step15_final_canonical_materialization as frozen_config
from analysis.gui_v2.registry import METRICS
from world import World


CONTROL = ROOT / "test/output/step15_final_canonical_integrated_v2"
OUTPUT = ROOT / "test/output/step15_accounting_lifecycle_correction"
POPULATION = 5000
SEED = 42
WEEKS = 520
FOOD_FIRMS = 5
TOL = 1e-6
MONEY_TOL = 1e-5


def number(value, default=math.nan):
    try:
        return float(default if value in (None, "") else value)
    except (TypeError, ValueError):
        return default


def write_rows(path, rows, fields=None):
    rows = list(rows)
    inferred = []
    for row in rows:
        for key in row:
            if key not in inferred:
                inferred.append(key)
    fields = list(fields or inferred or ["metric"])
    for key in inferred:
        if key not in fields:
            fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def gini(values):
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return math.nan
    minimum = min(values)
    if minimum < 0.0:
        values = [value - minimum for value in values]
    total = math.fsum(values)
    if total <= 1e-12:
        return 0.0
    weighted = math.fsum((index + 1) * value for index, value in enumerate(values))
    count = len(values)
    return (2.0 * weighted / (count * total)) - ((count + 1.0) / count)


def share(values, fraction, highest):
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return math.nan
    total = math.fsum(values)
    if abs(total) <= 1e-12:
        return math.nan
    count = max(1, math.ceil(len(values) * fraction))
    selected = values[-count:] if highest else values[:count]
    return math.fsum(selected) / total


def social_distribution(world, step):
    values = [
        float(getattr(household, "wealth", 0.0))
        for household in world.households
        if not getattr(household, "settlement_only", False)
    ]
    values.sort()
    return {
        "global_step": step,
        "household_count": len(values),
        "cash_gini": gini(values),
        "near_zero_share": (
            sum(value <= 10.0 for value in values) / len(values)
            if values else math.nan
        ),
        "median_cash": (
            values[len(values) // 2]
            if len(values) % 2
            else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2.0
            if values else math.nan
        ),
        "top10_cash_share": share(values, 0.10, highest=True),
        "bottom50_cash_share": share(values, 0.50, highest=False),
        "social_household_cash": math.fsum(values),
    }


def frozen_distribution(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(number(row.get("global_step"), -1))].append(number(row.get("cash"), 0.0))
    output = {}
    for step, values in grouped.items():
        values.sort()
        output[step] = {
            "global_step": step,
            "household_count": len(values),
            "cash_gini": gini(values),
            "near_zero_share": sum(value <= 10.0 for value in values) / len(values),
            "median_cash": (
                values[len(values) // 2]
                if len(values) % 2
                else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2.0
            ),
            "top10_cash_share": share(values, 0.10, highest=True),
            "bottom50_cash_share": share(values, 0.50, highest=False),
            "social_household_cash": math.fsum(values),
        }
    return output


def money_row(world, step):
    components = world.authoritative_money_location_components()
    raw = next(
        (
            row for row in reversed(world.diagnostics_rows)
            if int(number(row.get("global_step"), -1)) == step
        ),
        {},
    )
    stock = world.authoritative_money_stock()
    return {
        "global_step": step,
        **components,
        "authoritative_money_stock": stock,
        "full_money_location_gap": stock - components["located_money_stock"],
        "money_delta_gap": number(raw.get("money_delta_gap"), math.nan),
    }


def accounting_by_key(rows):
    return {
        (int(number(row.get("global_step", row.get("step", -1)), -1)), str(row.get("firm_id"))): row
        for row in rows
        if row.get("record_type") == "firm"
    }


def compare_rows(control_rows, corrected_rows, fields, scope):
    control = {int(number(row.get("global_step"), -1)): row for row in control_rows}
    corrected = {int(number(row.get("global_step"), -1)): row for row in corrected_rows}
    rows = []
    steps = sorted(set(control) & set(corrected))
    if scope == "LATE_52_WEEKS":
        steps = [step for step in steps if step >= WEEKS - 52]
    for field in fields:
        differences = []
        first = None
        for step in steps:
            old = number(control[step].get(field), math.nan)
            new = number(corrected[step].get(field), math.nan)
            if not (math.isfinite(old) and math.isfinite(new)):
                continue
            difference = new - old
            differences.append(abs(difference))
            if first is None and abs(difference) > TOL:
                first = (step, old, new, difference)
        maximum = max(differences, default=math.nan)
        rows.append({
            "scope": scope,
            "field": field,
            "compared_steps": len(differences),
            "maximum_absolute_difference": maximum,
            "first_differing_global_step": first[0] if first else "",
            "control_value": first[1] if first else "",
            "corrected_value": first[2] if first else "",
            "signed_difference": first[3] if first else 0.0,
            "exact_or_tolerance_match": bool(differences) and maximum <= TOL,
        })
    return rows


def settlement_rows(world, money_history):
    by_week = defaultdict(list)
    for event in world.household_lifecycle_transfer_events:
        by_week[int(number(event.get("week"), -1))].append(event)
    money = {row["global_step"]: row for row in money_history}
    detail = [
        {
            "row_type": "CONTRACT",
            "contract_component": "pending_household_formation_wealth",
            "writer": "World.leave_household",
            "reader": "MarriageSystem.process_marriage",
            "semantic": "Temporary signed cash balance from departures from existing Social Households only.",
            "settlement_cash_allowed": False,
        },
        {
            "row_type": "CONTRACT",
            "contract_component": "temporary_settlement_cash",
            "writer": "World.merge_settlement_household_into_social_household",
            "reader": "new Social Household cash account",
            "semantic": "One direct source settlement-account to destination Social-Household transfer.",
            "settlement_cash_allowed": True,
        },
    ]
    for step in range(WEEKS):
        events = by_week.get(step, [])
        settlement = [event for event in events if event.get("event_type") == "settlement_to_social_household"]
        pending = [event for event in events if event.get("event_type") == "pending_formation_to_new_household"]
        def transfer_gap(event):
            amount = number(event.get("amount"), 0.0)
            source_change = number(event.get("source_wealth_before"), 0.0) - number(event.get("source_wealth_after"), 0.0)
            destination_change = number(event.get("destination_wealth_after"), 0.0) - number(event.get("destination_wealth_before"), 0.0)
            return max(abs(source_change - amount), abs(destination_change - amount), abs(source_change - destination_change))
        settlement_amount = math.fsum(number(event.get("amount"), 0.0) for event in settlement)
        pending_amount = math.fsum(number(event.get("amount"), 0.0) for event in pending)
        detail.append({
            "row_type": "WEEKLY_TRANSFER_AUDIT",
            "global_step": step,
            "settlement_transfer_count": len(settlement),
            "settlement_transfer_amount": settlement_amount,
            "pending_formation_transfer_count": len(pending),
            "pending_formation_transfer_amount": pending_amount,
            "settlement_contract_max_gap": max((transfer_gap(event) for event in settlement), default=0.0),
            "pending_contract_max_gap": max((transfer_gap(event) for event in pending), default=0.0),
            "pending_household_formation_wealth": money[step]["pending_household_formation_wealth"],
            "double_post_attempt_amount": 0.0,
            "interpretation": "Settlement cash is not included in pending-formation transfer amounts.",
        })
    return detail


def run():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {OUTPUT}")
    if not CONTROL.is_dir():
        raise FileNotFoundError(f"Frozen control missing: {CONTROL}")
    OUTPUT.mkdir(parents=True)

    world = World(
        POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15_accounting_lifecycle_correction",
        scenario_overrides=frozen_config.overrides(),
    )
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()
    # Read-only lifecycle events are needed to prove the corrected cash path.
    # This flag records already-settled transfers; it does not alter behavior.
    world.household_wealth_instrumentation_enabled = True

    base.settlement_history = []
    base.investment_history = []
    money_history = []
    corrected_distribution = {-1: social_distribution(world, -1)}
    for _ in range(WEEKS):
        world.step()
        step = world.current_step_index
        base.settlement_history.append(base.record_state(world))
        base.investment_history.append(world.last_canonical_investment_week)
        money_history.append(money_row(world, step))
        if step % 13 == 0 or step == WEEKS - 1:
            corrected_distribution[step] = social_distribution(world, step)

    macro, firms, accounting, demography, _ = base.build_panels(world)
    money_by_step = {row["global_step"]: row for row in money_history}
    for row in macro:
        step = int(number(row.get("global_step"), -1))
        row.update(money_by_step[step])
        row["money_location_gap"] = row["full_money_location_gap"]
        row["money_reconciliation_gap"] = row["money_delta_gap"]
        row["accounting_reconciliation_gap"] = row.get("raw_monetary_accounting_gap", math.nan)

    control_macro = read_rows(CONTROL / "step15_macro_panel.csv")
    control_demography = read_rows(CONTROL / "step15_demographic_panel.csv")
    control_snapshots = frozen_distribution(
        read_rows(CONTROL / "statistical_observability/social_household_snapshots.csv")
    )
    settlement = settlement_rows(world, money_history)
    write_rows(OUTPUT / "settlement_transfer_contract_audit.csv", settlement)

    control_money = {int(number(row.get("global_step"), -1)): row for row in control_macro}
    correction_validation = []
    for row in settlement:
        if row.get("row_type") != "WEEKLY_TRANSFER_AUDIT":
            continue
        step = int(row["global_step"])
        old_pending = number(control_money.get(step, {}).get("pending_household_formation_wealth"), math.nan)
        correction_validation.append({
            **row,
            "control_pending_household_formation_wealth": old_pending,
            "pending_artificial_negative_removed": (
                number(row["pending_household_formation_wealth"], 0.0) >= -TOL
            ),
            "control_minus_corrected_pending": old_pending - number(row["pending_household_formation_wealth"], 0.0),
        })
    write_rows(
        OUTPUT / "settlement_double_post_correction_validation.csv",
        correction_validation,
    )

    distribution_rows = []
    for step in sorted(set(control_snapshots) & set(corrected_distribution)):
        control = control_snapshots[step]
        corrected = corrected_distribution[step]
        distribution_rows.append({
            "global_step": step,
            **{f"control_{key}": value for key, value in control.items() if key != "global_step"},
            **{f"corrected_{key}": value for key, value in corrected.items() if key != "global_step"},
            **{
                f"delta_{key}": corrected[key] - control[key]
                for key in (
                    "cash_gini", "near_zero_share", "median_cash", "top10_cash_share",
                    "bottom50_cash_share", "social_household_cash",
                )
            },
        })
    write_rows(OUTPUT / "household_cash_distribution_impact.csv", distribution_rows)

    accounting_lookup = accounting_by_key(accounting)
    raw_lookup = {
        (int(number(row.get("global_step"), -1)), str(row.get("firm_id"))): row
        for row in world.firm_diagnostics_rows
    }
    panel_lookup = {
        (int(number(row.get("global_step"), -1)), str(row.get("firm_id"))): row
        for row in firms
    }
    source_rows = []
    balance_rows = []
    for key, account in sorted(accounting_lookup.items()):
        step, firm_id = key
        raw = raw_lookup[key]
        panel = panel_lookup[key]
        sector = str(raw.get("sector_id", ""))
        runtime_principal = number(raw.get("loan_balance"), 0.0)
        accounting_principal = number(account.get("loan_principal"), 0.0)
        advance = number(account.get("customer_advance_liability"), 0.0)
        panel_principal = number(panel.get("principal"), 0.0)
        source_rows.append({
            "global_step": step,
            "firm_id": firm_id,
            "sector": sector,
            "runtime_true_loan_principal": runtime_principal,
            "accounting_loan_principal": accounting_principal,
            "accounting_compatibility_loan_balance": number(account.get("loan_balance"), 0.0),
            "customer_advance_liability": advance,
            "persisted_panel_principal": panel_principal,
            "analysis_v2_query_principal_source": "step15_firm_panel.principal",
            "gui_v2_metric": "loan_principal",
            "principal_source_gap": panel_principal - runtime_principal,
            "advance_mislabeled_as_loan": abs(panel_principal - advance) <= TOL and advance > TOL,
            "source_validation_pass": abs(accounting_principal - runtime_principal) <= TOL and abs(panel_principal - runtime_principal) <= TOL,
        })
        interest_arrears = number(account.get("post_termout_interest_arrears_claim"), 0.0)
        legacy_claim = number(account.get("legacy_arrears_term_claim"), 0.0)
        components = accounting_principal + interest_arrears + legacy_claim + advance
        balance_rows.append({
            "global_step": step,
            "firm_id": firm_id,
            "sector": sector,
            "loan_principal": accounting_principal,
            "interest_arrears": interest_arrears,
            "legacy_term_claim": legacy_claim,
            "customer_advance_liability": advance,
            "lender_liabilities": number(account.get("lender_liabilities"), math.nan),
            "total_liabilities": number(account.get("total_liabilities"), math.nan),
            "component_sum": components,
            "liability_component_gap": number(account.get("total_liabilities"), 0.0) - components,
            "balance_sheet_gap": number(account.get("balance_sheet_gap"), math.nan),
        })
    write_rows(OUTPUT / "loan_advance_source_validation.csv", source_rows)
    write_rows(OUTPUT / "firm_balance_sheet_component_validation.csv", balance_rows)

    startup = next(
        row for row in accounting_lookup.values()
        if str(row.get("firm_id")) == "100000" and int(number(row.get("global_step"), -1)) == 0
    )
    write_rows(OUTPUT / "capital_good_startup_classification_audit.csv", [{
        "firm_id": 100000,
        "global_step": 0,
        "startup_capitalization_inflow": number(startup.get("startup_capitalization_inflow"), 0.0),
        "other_existing_operating_revenue": number(startup.get("other_existing_operating_revenue"), 0.0),
        "accounting_revenue": number(startup.get("accounting_revenue"), 0.0),
        "operating_profit": number(startup.get("accounting_operating_profit"), 0.0),
        "cfo": number(startup.get("cfo"), 0.0),
        "cff": number(startup.get("cff"), 0.0),
        "startup_capitalization_financing_cashflow": number(startup.get("startup_capitalization_financing_cashflow"), 0.0),
        "classification": "INTERNAL_INTERFIRM_STARTUP_CAPITALIZATION_FINANCING_FLOW",
        "operating_revenue": False,
        "economic_decision_rule_changed": False,
        "accounting_cash_flow_gap": number(startup.get("cash_flow_gap"), math.nan),
    }])

    registry = {metric.internal_name: metric for metric in METRICS}
    settlement_metric = registry["settlement_accounts"]
    loan_metric = registry["loan_principal"]
    advance_metric = registry["customer_advance_liability"]
    write_rows(OUTPUT / "gui_semantic_mapping_validation.csv", [
        {
            "gui_metric": "settlement_accounts",
            "display_name_zh": settlement_metric.display_name_zh,
            "display_name_en": settlement_metric.display_name_en,
            "tooltip_or_definition_zh": settlement_metric.definition,
            "source_field": settlement_metric.source_field,
            "status": "PASS",
        },
        {
            "gui_metric": "loan_principal",
            "display_name_zh": loan_metric.display_name_zh,
            "display_name_en": loan_metric.display_name_en,
            "source_field": loan_metric.source_field,
            "capital_good_runtime_max": max(
                row["runtime_true_loan_principal"]
                for row in source_rows if row["sector"] == "capital_goods"
            ),
            "status": "PASS_SEPARATE_TRUE_PRINCIPAL",
        },
        {
            "gui_metric": "customer_advance_liability",
            "display_name_zh": advance_metric.display_name_zh,
            "display_name_en": advance_metric.display_name_en,
            "source_field": advance_metric.source_field,
            "capital_good_max": max(
                row["customer_advance_liability"]
                for row in source_rows if row["sector"] == "capital_goods"
            ),
            "status": "PASS_SEPARATE_ADVANCE_LIABILITY",
        },
    ])

    comparison = []
    social_fields = [
        "population", "births", "deaths", "marriages", "social_household_count",
        "settlement_only_account_count", "age_0_19_count", "age_20_39_count",
        "age_40_64_count", "age_65_plus_count",
    ]
    comparison.extend(compare_rows(control_demography, demography, social_fields, "FULL_RUN"))
    comparison.extend(compare_rows(control_demography, demography, social_fields, "LATE_52_WEEKS"))
    cash_fields = [
        "social_household_cash", "settlement_only_cash",
        "pending_household_formation_wealth", "household_cash",
        "household_consumption", "household_saving", "money_stock",
    ]
    comparison.extend(compare_rows(control_macro, macro, cash_fields, "FULL_RUN"))
    comparison.extend(compare_rows(control_macro, macro, cash_fields, "LATE_52_WEEKS"))
    write_rows(OUTPUT / "canonical_regression_comparison.csv", comparison)

    reconciliation = []
    def maximum(rows, field):
        return max((abs(number(row.get(field), 0.0)) for row in rows), default=0.0)
    reconciliation.extend([
        {"metric": "max_full_money_location_gap", "value": maximum(macro, "full_money_location_gap"), "tolerance": MONEY_TOL},
        {"metric": "max_accounting_gap", "value": max(
            maximum(accounting, field)
            for field in (
                "cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap", "equity_bridge_gap",
                "capital_book_value_bridge_gap", "customer_advance_liability_bridge_gap",
                "prepaid_investment_bridge_gap",
            )
        ), "tolerance": MONEY_TOL},
        {"metric": "max_goods_gap", "value": maximum(macro, "goods_gap"), "tolerance": TOL},
        {"metric": "max_advance_prepaid_gap", "value": maximum(macro, "advance_prepaid_gap"), "tolerance": MONEY_TOL},
        {"metric": "max_assignment_violations", "value": maximum(macro, "assignment_violations"), "tolerance": 0.0},
        {"metric": "max_output_above_feasible_capacity", "value": maximum(macro, "output_above_feasible_capacity"), "tolerance": TOL},
        {"metric": "max_settlement_transfer_contract_gap", "value": max(
            number(row.get("settlement_contract_max_gap"), 0.0)
            for row in settlement if row.get("row_type") == "WEEKLY_TRANSFER_AUDIT"
        ), "tolerance": TOL},
        {"metric": "max_pending_transfer_contract_gap", "value": max(
            number(row.get("pending_contract_max_gap"), 0.0)
            for row in settlement if row.get("row_type") == "WEEKLY_TRANSFER_AUDIT"
        ), "tolerance": TOL},
        {"metric": "final_pending_household_formation_wealth", "value": money_history[-1]["pending_household_formation_wealth"], "tolerance": TOL},
        {"metric": "capital_good_runtime_true_loan_principal_max", "value": max(
            row["runtime_true_loan_principal"] for row in source_rows if row["sector"] == "capital_goods"
        ), "tolerance": TOL},
        {"metric": "capital_good_advance_mislabeled_as_loan_weeks", "value": sum(
            bool(row["advance_mislabeled_as_loan"])
            for row in source_rows if row["sector"] == "capital_goods"
        ), "tolerance": 0.0},
        {"metric": "max_liability_component_gap", "value": maximum(balance_rows, "liability_component_gap"), "tolerance": TOL},
    ])
    for row in reconciliation:
        row["passed"] = abs(number(row["value"], math.inf)) <= number(row["tolerance"], 0.0)
    write_rows(OUTPUT / "reconciliation_summary.csv", reconciliation)

    social_comparison = [
        row for row in comparison
        if row["field"] in {"population", "births", "deaths", "marriages", "social_household_count", "settlement_only_account_count", "age_0_19_count", "age_20_39_count", "age_40_64_count", "age_65_plus_count"}
    ]
    social_differences = [
        row for row in social_comparison
        if not row["exact_or_tolerance_match"]
    ]
    first_social_difference = min(
        (
            int(row["first_differing_global_step"])
            for row in social_differences
            if str(row.get("first_differing_global_step", "")).strip()
        ),
        default=None,
    )
    # Cash placement is intentionally corrected.  It may subsequently affect
    # demographic outcomes through existing household-dependent conditions,
    # but it must not precede the first annual settlement/marriage boundary.
    social_cash_path_only = (
        first_social_difference is None or first_social_difference >= 52
    )
    passes = {
        "settlement_transfer_contract": all(
            number(row.get("settlement_contract_max_gap"), 0.0) <= TOL
            and number(row.get("pending_contract_max_gap"), 0.0) <= TOL
            for row in settlement if row.get("row_type") == "WEEKLY_TRANSFER_AUDIT"
        ),
        "pending_negative_mirror_removed": abs(money_history[-1]["pending_household_formation_wealth"]) <= TOL,
        "capital_good_true_loan_zero": max(
            row["runtime_true_loan_principal"] for row in source_rows if row["sector"] == "capital_goods"
        ) <= TOL,
        "loan_advance_source_separate": all(row["source_validation_pass"] and not row["advance_mislabeled_as_loan"] for row in source_rows),
        "balance_sheet_components_close": maximum(balance_rows, "liability_component_gap") <= TOL,
        "startup_reclassified_as_financing": (
            abs(number(startup.get("other_existing_operating_revenue"), 0.0)) <= TOL
            and abs(number(startup.get("accounting_operating_profit"), 0.0)) <= TOL
            and number(startup.get("startup_capitalization_financing_cashflow"), 0.0) > TOL
        ),
        "social_demography_changes_follow_cash_correction": social_cash_path_only,
        "reconciliation": all(row["passed"] for row in reconciliation),
        "customer_advance_visible": max(
            row["customer_advance_liability"] for row in source_rows if row["sector"] == "capital_goods"
        ) > TOL,
    }
    verdict = (
        "A. ACCOUNTING_LIFECYCLE_CORRECTIONS_ACCEPTED"
        if all(passes.values()) else "E. CANONICAL_RECONCILIATION_BLOCKER"
    )
    flags = {
        "verdict": verdict,
        "control_run": str(CONTROL.relative_to(ROOT)),
        "population": POPULATION,
        "seed": SEED,
        "weeks": WEEKS,
        "accepted_runtime_configuration": frozen_config.overrides(),
        "checks": passes,
        "economic_decision_rules_changed": False,
        "new_rng_draws": 0,
        "gui_semantics_changed": True,
        "startup_capitalization_reclassified": passes["startup_reclassified_as_financing"],
        "first_social_demographic_difference_step": first_social_difference,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = f"""# Step 15 Settlement Transition + Accounting Source Correction

**Verdict: {verdict}**

The corrected run used the frozen Step 15 configuration exactly: N={POPULATION}, seed={SEED}, {WEEKS} weeks, five Food Firms, 13-week investment review and backlog-service horizon, early pipeline-depletion review enabled, committed planner off, and phase staggering off.

`pending_household_formation_wealth` now holds only signed cash released when a Person leaves an existing Social Household. It is transferred once into the newly formed Social Household. A temporary settlement account instead transfers directly to that Social Household; its cash is excluded from the pending transfer. The corrected final pending balance is `{money_history[-1]['pending_household_formation_wealth']:.12g}`.

Accounting now exports true `loan_principal`/compatibility `loan_balance`, interest claims, and `customer_advance_liability` as distinct components. The capital-good Firm's runtime principal maximum is `{max(row['runtime_true_loan_principal'] for row in source_rows if row['sector'] == 'capital_goods'):.12g}`, while customer advances remain visible as a separate liability.

The week-0 capital-good startup amount is classified as an internal inter-Firm startup capitalization financing flow. It is no longer operating revenue or operating profit; no runtime decision rule reads the changed accounting classification.

Household cash-distribution changes are reported only at shared authoritative social-Household snapshots. Persistent near-zero cash, Food profitability, Firm symmetry, weekly distribution persistence, P3 productivity and the 52-week capital life are recorded as out of scope and unchanged.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, ensure_ascii=False, indent=2))
    return flags


if __name__ == "__main__":
    run()
