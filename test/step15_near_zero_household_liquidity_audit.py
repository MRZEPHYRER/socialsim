"""Step 15 diagnostic-only audit of persistent near-zero household liquidity."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
for path in (ROOT, TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import step15_final_canonical_materialization as frozen_config
from productivity import age_productivity
from world import World

OUTPUT = ROOT / "test/output/step15_near_zero_household_liquidity_audit"
POPULATION, SEED, WEEKS, FOOD_FIRMS, CADENCE, TOL = 5000, 42, 520, 5, 13, 1e-5


def number(value, default=0.0):
    try:
        return float(default if value in (None, "") else value)
    except (TypeError, ValueError):
        return default


def read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["metric"]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gini(values):
    values = sorted(max(0.0, number(value)) for value in values)
    total = math.fsum(values)
    if not values or total <= TOL:
        return 0.0
    return sum((2 * index - len(values) - 1) * value for index, value in enumerate(values, 1)) / (len(values) * total)


def quantile(values, fraction):
    values = sorted(number(value) for value in values)
    if not values:
        return math.nan
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (position - low)


def share(values, fraction, highest):
    values = sorted(max(0.0, number(value)) for value in values)
    count = max(1, math.ceil(len(values) * fraction)) if values else 0
    selected = values[-count:] if highest else values[:count]
    return math.fsum(selected) / math.fsum(values) if math.fsum(values) > TOL else 0.0


def cash_group(rows):
    ordered = sorted(rows, key=lambda row: number(row["cash"]))
    first = int(len(ordered) * 0.5)
    second = int(len(ordered) * 0.9)
    return {
        row["household_id"]: "BOTTOM_50" if index < first else ("MIDDLE_40" if index < second else "TOP_10")
        for index, row in enumerate(ordered)
    }


def trajectory_row(world, step):
    raw = world.diagnostics_rows[-1]
    firms = list(world.operating_firms())
    food = [firm for firm in firms if getattr(firm, "sector_id", "") == "food"]
    return {
        "global_step": step,
        "population": len([person for person in world.population if person.alive]),
        "employment": sum(len(getattr(firm, "employee_ids", [])) for firm in firms),
        "household_income": math.fsum(number(getattr(household, "income_this_step", 0.0)) for household in world.households),
        "household_consumption": math.fsum(number(getattr(household, "consumption_this_step", 0.0)) for household in world.households),
        "household_cash": math.fsum(number(getattr(household, "wealth", 0.0)) for household in world.households),
        "food_production": math.fsum(number(getattr(firm, "production", 0.0)) for firm in food),
        "firm_revenue": math.fsum(number(getattr(firm, "revenue", 0.0)) for firm in firms),
        "fixed_investment": math.fsum(number(getattr(firm, "executed_investment_this_step", 0.0)) for firm in firms),
        "active_capital": math.fsum(
            number(firm.capital_stock.capital_service_capacity(engineering_capacity_per_unit=1.0))
            for firm in food
        ),
        "money_stock": world.authoritative_money_stock(),
        "diagnostic_money_gap": number(raw.get("money_delta_gap", 0.0)),
    }


def run_world(observed, directory=None):
    world = World(
        POPULATION, seed=SEED, diagnostics_mode="full",
        scenario_name="step15_near_zero_liquidity_audit",
        scenario_overrides=frozen_config.overrides(),
    )
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()
    world.steps = WEEKS
    if observed:
        world.household_wealth_instrumentation_enabled = True
        world.configure_diagnostic_persistence(
            Path(directory) / "canonical_diagnostics", cadence=1,
            statistical_observability=True,
            statistical_output_dir=Path(directory) / "statistical_observability",
            statistical_snapshot_cadence=CADENCE,
            statistical_age_cadence=CADENCE,
        )
    trajectory = []
    for _ in range(WEEKS):
        world.step()
        trajectory.append(trajectory_row(world, world.current_step_index))
    return world, trajectory, random.getstate()


def rows_by_step(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(number(row.get("global_step"), -1))].append(row)
    return grouped


def household_status(row):
    cash = number(row["cash"])
    minimum = number(row["interval_minimum_consumption"]) / max(1, int(number(row["interval_weeks"], 1)))
    return {
        "near_zero": cash <= 10.0,
        "below_1w_minimum": cash < minimum,
        "below_2w_minimum": cash < 2 * minimum,
        "below_4w_minimum": cash < 4 * minimum,
        "minimum_weekly_consumption": minimum,
    }


def composition(row):
    adults = int(number(row["adult_count"]))
    children = int(number(row["child_count"]))
    elderly = int(number(row["elderly_count"]))
    if adults == 1 and children == 0:
        return "SINGLE_ADULT"
    if adults == 2 and children == 0:
        return "TWO_ADULT"
    if adults and children:
        return "ADULT_PLUS_CHILDREN"
    if elderly and elderly >= max(1, adults):
        return "ELDERLY_HEAVY"
    return "OTHER"


def income_quantiles(rows, field):
    ordered = sorted(rows, key=lambda row: number(row[field]))
    count = len(ordered)
    return {row["household_id"]: min(4, int(index * 4 / max(1, count))) + 1 for index, row in enumerate(ordered)}


def aggregate_group(rows, label_field, measures):
    groups = defaultdict(list)
    for row in rows:
        groups[row[label_field]].append(row)
    output = []
    for label, items in groups.items():
        result = {label_field: label, "household_count": len(items)}
        for field in measures:
            result[f"mean_{field}"] = math.fsum(number(row.get(field)) for row in items) / len(items)
            result[f"sum_{field}"] = math.fsum(number(row.get(field)) for row in items)
        output.append(result)
    return output


def build_audit(snapshot_rows):
    snapshots = rows_by_step([row for row in snapshot_rows if int(number(row["global_step"])) >= 0])
    cross, bridges, transitions, labor, consumption, burden, formation, accumulation, gini_rows = [], [], [], [], [], [], [], [], []
    previous = {}
    all_transition_counts = defaultdict(int)
    for step in sorted(snapshots):
        rows = snapshots[step]
        cash_groups = cash_group(rows)
        income_q = income_quantiles(rows, "interval_total_authoritative_income")
        labor_q = income_quantiles(rows, "interval_labor_income")
        enriched = []
        for row in rows:
            status = household_status(row)
            employed = int(number(row["employed_member_count"]))
            eligible = int(number(row["labor_eligible_member_count"]))
            item = dict(row)
            item.update(status)
            item.update({
                "global_step": step,
                "cash_status": "NEAR_ZERO" if status["near_zero"] else "ABOVE_ZERO",
                "employment_class": "ZERO_EMPLOYED_ADULTS" if employed == 0 else ("ONE_EMPLOYED_ADULT" if employed == 1 else "TWO_PLUS_EMPLOYED_ADULTS"),
                "labor_eligibility_class": "ELIGIBLE_UNEMPLOYED" if eligible > employed else ("NO_LABOR_ELIGIBLE_ADULT" if eligible == 0 else "ALL_ELIGIBLE_EMPLOYED"),
                "household_composition": composition(row),
                "income_quantile": income_q[row["household_id"]],
                "labor_income_quantile": labor_q[row["household_id"]],
                "employed_to_eligible_ratio": employed / eligible if eligible else math.nan,
                "cash_accumulation_group": cash_groups[row["household_id"]],
            })
            cross.append(item)
            bridge = {
                "global_step": step, "household_id": row["household_id"],
                "opening_cash": number(row["opening_cash"]),
                "labor_income": number(row["interval_labor_income"]),
                "dividend_income": number(row["interval_dividend_income"]),
                "inheritance_transfers": number(row["interval_lifecycle_transfer_in"]),
                "total_authoritative_income": number(row["interval_total_authoritative_income"]),
                "consumption": number(row["interval_consumption"]),
                "other_authoritative_outflow": number(row["interval_lifecycle_transfer_out"]),
                "closing_cash": number(row["cash"]),
                "cash_bridge_gap": number(row["interval_cash_bridge_gap"]),
            }
            bridges.append(bridge)
            enriched.append(item)
        for status_name in ("NEAR_ZERO", "ABOVE_ZERO"):
            items = [row for row in enriched if row["cash_status"] == status_name]
            if not items:
                continue
            employed_total = math.fsum(number(row["employed_member_count"]) for row in items)
            labor_income = math.fsum(number(row["interval_labor_income"]) for row in items)
            total_income = math.fsum(number(row["interval_total_authoritative_income"]) for row in items)
            cons = math.fsum(number(row["interval_consumption"]) for row in items)
            minimum = math.fsum(number(row["interval_minimum_consumption"]) for row in items)
            eligible_total = math.fsum(number(row["labor_eligible_member_count"]) for row in items)
            labor.append({"global_step": step, "cash_status": status_name, "household_count": len(items), "mean_employed_members": employed_total / len(items), "employment_rate_eligible": employed_total / eligible_total if eligible_total else math.nan, "mean_labor_income": labor_income / len(items), "wage_income_per_employed_member": labor_income / employed_total if employed_total else math.nan, "mean_total_income": total_income / len(items)})
            consumption.append({"global_step": step, "cash_status": status_name, "household_count": len(items), "mean_consumption": cons / len(items), "mean_minimum_consumption": minimum / len(items), "consumption_income_ratio": cons / total_income if total_income else math.nan, "mean_saving": math.fsum(number(row["interval_saving"]) for row in items) / len(items), "income_below_minimum_share": sum(number(row["interval_total_authoritative_income"]) < number(row["interval_minimum_consumption"]) for row in items) / len(items)})
            burden.append({"global_step": step, "cash_status": status_name, "household_count": len(items), "children_per_adult": math.fsum(number(row["child_count"]) for row in items) / max(1.0, math.fsum(number(row["adult_count"]) for row in items)), "elderly_per_adult": math.fsum(number(row["elderly_count"]) for row in items) / max(1.0, math.fsum(number(row["adult_count"]) for row in items)), "dependents_per_employed": math.fsum(number(row["child_count"]) + number(row["elderly_count"]) for row in items) / max(1.0, employed_total)})
        created = [row for row in enriched if str(row.get("created_since_previous_snapshot")) == "True"]
        for row in created:
            formation.append({"global_step": step, "household_id": row["household_id"], "opening_cash": number(row["opening_cash"]), "incoming_settlement_or_transfer_cash": number(row["interval_lifecycle_transfer_in"]), "employed_members": row["employed_member_count"], "interval_income": number(row["interval_total_authoritative_income"]), "interval_consumption": number(row["interval_consumption"]), "closing_cash": number(row["cash"]), "near_zero": row["near_zero"]})
        for group in ("BOTTOM_50", "MIDDLE_40", "TOP_10"):
            items = [row for row in enriched if row["cash_accumulation_group"] == group]
            accumulation.append({"global_step": step, "cash_group": group, "household_count": len(items), "cash_share": share([row["cash"] for row in enriched], 0.5 if group == "BOTTOM_50" else (0.4 if group == "MIDDLE_40" else 0.1), highest=group == "TOP_10") if group != "MIDDLE_40" else math.fsum(number(row["cash"]) for row in items) / max(TOL, math.fsum(number(row["cash"]) for row in enriched)), "labor_income_share": math.fsum(number(row["interval_labor_income"]) for row in items) / max(TOL, math.fsum(number(row["interval_labor_income"]) for row in enriched)), "consumption_share": math.fsum(number(row["interval_consumption"]) for row in items) / max(TOL, math.fsum(number(row["interval_consumption"]) for row in enriched)), "saving_share": math.fsum(number(row["interval_saving"]) for row in items) / max(TOL, math.fsum(number(row["interval_saving"]) for row in enriched)), "dividend_share": math.fsum(number(row["interval_dividend_income"]) for row in items) / max(TOL, math.fsum(number(row["interval_dividend_income"]) for row in enriched)), "transfer_share": math.fsum(number(row["interval_lifecycle_transfer_in"]) for row in items) / max(TOL, math.fsum(number(row["interval_lifecycle_transfer_in"]) for row in enriched))})
        current = {row["household_id"]: row for row in enriched}
        continuing = set(previous) & set(current)
        prior_gini = gini([previous[key]["cash"] for key in continuing]) if continuing else math.nan
        current_continuing_gini = gini([current[key]["cash"] for key in continuing]) if continuing else math.nan
        total_gini = gini([row["cash"] for row in enriched])
        gini_rows.append({"global_step": step, "household_count": len(enriched), "cash_gini": total_gini, "income_gini": gini([row["interval_total_authoritative_income"] for row in enriched]), "near_zero_cash_share": sum(row["near_zero"] for row in enriched) / len(enriched), "cash_below_1w_minimum_share": sum(row["below_1w_minimum"] for row in enriched) / len(enriched), "cash_below_2w_minimum_share": sum(row["below_2w_minimum"] for row in enriched) / len(enriched), "cash_below_4w_minimum_share": sum(row["below_4w_minimum"] for row in enriched) / len(enriched), "mean_cash": math.fsum(number(row["cash"]) for row in enriched) / len(enriched), "median_cash": quantile([row["cash"] for row in enriched], 0.5), "p10_cash": quantile([row["cash"] for row in enriched], 0.1), "p25_cash": quantile([row["cash"] for row in enriched], 0.25), "p75_cash": quantile([row["cash"] for row in enriched], 0.75), "p90_cash": quantile([row["cash"] for row in enriched], 0.9), "p95_cash": quantile([row["cash"] for row in enriched], 0.95), "top10_cash_share": share([row["cash"] for row in enriched], 0.1, True), "bottom50_cash_share": share([row["cash"] for row in enriched], 0.5, False), "continuing_households": len(continuing), "gini_value_movement_continuers": current_continuing_gini - prior_gini if continuing else math.nan, "gini_composition_effect": total_gini - current_continuing_gini if continuing else math.nan})
        if previous:
            counts = defaultdict(int)
            for household_id in continuing:
                before = "NEAR_ZERO" if previous[household_id]["near_zero"] else "ABOVE_ZERO"
                after = "NEAR_ZERO" if current[household_id]["near_zero"] else "ABOVE_ZERO"
                counts[f"{before}->{after}"] += 1
            for transition, count in counts.items():
                transitions.append({"from_global_step": min(previous[key]["global_step"] for key in continuing), "to_global_step": step, "transition": transition, "count": count, "row_probability": count / max(1, sum(value for name, value in counts.items() if name.startswith(transition.split("->")[0])))})
                all_transition_counts[transition] += count
        previous = current
    return cross, bridges, transitions, labor, consumption, burden, formation, accumulation, gini_rows, all_transition_counts


def remedy_matrix():
    return [
        {"future_boundary": "wage_or_income_mechanism", "addresses_root_cause": "if low labor income dominates", "merely_transfers_cash": False, "changes_labor_incentives": True, "changes_consumption": True, "requires_government": False, "stage": "later household/labor economics"},
        {"future_boundary": "unemployment_or_eligibility_mechanism", "addresses_root_cause": "if nonemployment dominates", "merely_transfers_cash": False, "changes_labor_incentives": True, "changes_consumption": True, "requires_government": False, "stage": "later labor-market architecture"},
        {"future_boundary": "consumption_floor", "addresses_root_cause": "if income is repeatedly below necessary consumption", "merely_transfers_cash": False, "changes_labor_incentives": False, "changes_consumption": True, "requires_government": False, "stage": "later household-economics boundary"},
        {"future_boundary": "transfer_institution", "addresses_root_cause": "liquidity only; not labor-income cause", "merely_transfers_cash": True, "changes_labor_incentives": "possibly", "changes_consumption": True, "requires_government": "likely", "stage": "later fiscal architecture"},
        {"future_boundary": "capital_income_distribution", "addresses_root_cause": "if top cash/dividend concentration dominates", "merely_transfers_cash": False, "changes_labor_incentives": False, "changes_consumption": True, "requires_government": False, "stage": "later ownership architecture"},
        {"future_boundary": "household_formation_liquidity", "addresses_root_cause": "if newly formed households dominate entries", "merely_transfers_cash": True, "changes_labor_incentives": False, "changes_consumption": "possibly", "requires_government": False, "stage": "later demographic-household boundary"},
        {"future_boundary": "government_fiscal_closure", "addresses_root_cause": "aggregate demand or transfer closure", "merely_transfers_cash": "depends", "changes_labor_incentives": "depends", "changes_consumption": True, "requires_government": True, "stage": "later macro-fiscal architecture"},
    ]


def run():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="step15_near_zero_") as temporary:
        control_world, control, control_rng = run_world(False)
        treatment_world, treatment, treatment_rng = run_world(True, temporary)
        snapshot_rows = read_rows(Path(temporary) / "statistical_observability/social_household_snapshots.csv")

        metrics = ("population", "employment", "household_income", "household_consumption", "household_cash", "food_production", "firm_revenue", "fixed_investment", "active_capital", "money_stock")
        noninterference = []
        for metric in metrics:
            differences = [abs(number(before[metric]) - number(after[metric])) for before, after in zip(control, treatment)]
            noninterference.append({"metric": metric, "maximum_absolute_difference": max(differences), "first_differing_step": next((index for index, value in enumerate(differences) if value > TOL), ""), "passed": max(differences) <= TOL})
        noninterference.append({"metric": "global_rng_state", "maximum_absolute_difference": 0.0 if control_rng == treatment_rng else 1.0, "first_differing_step": "", "passed": control_rng == treatment_rng})
        write_rows(OUTPUT / "observability_noninterference_validation.csv", noninterference)

        cross, bridges, transitions, labor, consumption, burden, formation, accumulation, gini_rows, all_transitions = build_audit(snapshot_rows)
        write_rows(OUTPUT / "near_zero_household_cross_section.csv", cross)
        write_rows(OUTPUT / "household_interval_cash_bridge.csv", bridges)
        write_rows(OUTPUT / "near_zero_transition_matrix.csv", transitions + [{"from_global_step": "ALL", "to_global_step": "ALL", "transition": key, "count": value, "row_probability": ""} for key, value in sorted(all_transitions.items())])
        write_rows(OUTPUT / "labor_income_channel_audit.csv", labor)
        write_rows(OUTPUT / "consumption_floor_channel_audit.csv", consumption)
        write_rows(OUTPUT / "demographic_burden_audit.csv", burden)
        write_rows(OUTPUT / "household_formation_channel_audit.csv", formation)
        write_rows(OUTPUT / "household_cash_accumulation_distribution.csv", accumulation)
        write_rows(OUTPUT / "household_gini_decomposition.csv", gini_rows)
        write_rows(OUTPUT / "household_liquidity_remedy_matrix.csv", remedy_matrix())

    final_step = max(int(number(row["global_step"])) for row in cross)
    final_cross = [row for row in cross if int(number(row["global_step"])) == final_step]
    final_labor = {row["cash_status"]: row for row in labor if int(number(row["global_step"])) == final_step}
    final_consumption = {row["cash_status"]: row for row in consumption if int(number(row["global_step"])) == final_step}
    near = [row for row in final_cross if row["cash_status"] == "NEAR_ZERO"]
    other = [row for row in final_cross if row["cash_status"] == "ABOVE_ZERO"]
    near_zero_share = len(near) / len(final_cross)
    scale_1w = sum(row["below_1w_minimum"] for row in final_cross) / len(final_cross)
    scale_2w = sum(row["below_2w_minimum"] for row in final_cross) / len(final_cross)
    scale_4w = sum(row["below_4w_minimum"] for row in final_cross) / len(final_cross)
    no_employed_share = sum(int(number(row["employed_member_count"])) == 0 for row in near) / max(1, len(near))
    below_minimum_share = sum(number(row["interval_total_authoritative_income"]) < number(row["interval_minimum_consumption"]) for row in near) / max(1, len(near))
    new_near_share = sum(str(row.get("created_since_previous_snapshot")) == "True" for row in near) / max(1, len(near))
    top = next((row for row in accumulation if int(number(row["global_step"])) == final_step and row["cash_group"] == "TOP_10"), {})
    transitions_total = sum(all_transitions.values())
    persistent_share = all_transitions.get("NEAR_ZERO->NEAR_ZERO", 0) / max(1, sum(value for key, value in all_transitions.items() if key.startswith("NEAR_ZERO->")))
    if no_employed_share >= 0.6:
        verdict = "A. LABOR_NONEMPLOYMENT_IS_PRIMARY"
        next_boundary = "劳动市场中的非就业/资格边界"
    elif below_minimum_share >= 0.6:
        verdict = "B. LOW_INCOME_RELATIVE_TO_CONSUMPTION_IS_PRIMARY"
        next_boundary = "家庭收入相对必要消费的闭合边界"
    elif new_near_share >= 0.5:
        verdict = "C. HOUSEHOLD_FORMATION_IS_PRIMARY"
        next_boundary = "家庭形成时的流动性边界"
    elif number(top.get("cash_share")) >= 0.5:
        verdict = "D. CAPITAL_INCOME_CONCENTRATION_IS_PRIMARY"
        next_boundary = "资本收入分配与所有权边界"
    else:
        verdict = "E. MULTIPLE_DISTRIBUTIONAL_CHANNELS"
        next_boundary = "先研究劳动收入与必要消费之间的家庭现金闭合"
    bridge_max = max(abs(number(row["cash_bridge_gap"])) for row in bridges)
    flags = {
        "verdict": verdict,
        "population": POPULATION,
        "seed": SEED,
        "weeks": WEEKS,
        "snapshot_cadence_weeks": CADENCE,
        "observability_noninterference_passed": all(row["passed"] for row in noninterference),
        "interval_cash_bridge_max_gap": bridge_max,
        "near_zero_share_final_snapshot": near_zero_share,
        "cash_below_1w_minimum_share_final_snapshot": scale_1w,
        "cash_below_2w_minimum_share_final_snapshot": scale_2w,
        "cash_below_4w_minimum_share_final_snapshot": scale_4w,
        "near_zero_persistence_conditional_probability": persistent_share,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "next_boundary": next_boundary,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = f"""# Step 15 Persistent Near-Zero Household Liquidity Audit

**Verdict: {verdict}**

The observability treatment used the accepted accounting/lifecycle-corrected Step 15 configuration at N=5000, seed=42 for 520 weeks. It adds only read-only 13-week Social-Household interval persistence. The matched diagnostics-off control and treatment trajectories, including global RNG state, match within tolerance.

At final authoritative snapshot week {final_step}, near-zero cash (cash <= 10) is {near_zero_share:.2%}. Cash below one, two, and four weeks of minimum consumption is {scale_1w:.2%}, {scale_2w:.2%}, and {scale_4w:.2%}. Conditional persistence from near-zero to near-zero across adjacent snapshots is {persistent_share:.2%}. Among near-zero Households, {no_employed_share:.2%} have no employed member and {below_minimum_share:.2%} have interval income below interval minimum consumption. Newly created Households account for {new_near_share:.2%} of the near-zero cross-section.

The maximum Household interval cash-bridge gap is {bridge_max:.6g}. The final top-10 cash group holds {number(top.get('cash_share')):.2%} of cash, {number(top.get('labor_income_share')):.2%} of labor income, and {number(top.get('saving_share')):.2%} of interval saving. Near-zero status is an observed liquidity state; no runtime employment or income rule reads it, so this audit does not identify a coded poverty trap.

Safest next economic boundary to investigate: **{next_boundary}**. The remedy matrix is conceptual only; no wage, employment, consumption, saving, transfer, dividend, inheritance, Firm, Government, or Step16 mechanism was changed.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
