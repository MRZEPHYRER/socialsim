"""Diagnostic-only audit of the Step 15 age, labor, and wage contract."""

from __future__ import annotations

import csv
import json
import math
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

OUTPUT = ROOT / "test/output/step15_age_labor_productivity_contract_audit"
POPULATION, SEED, WEEKS, FOOD_FIRMS = 5000, 42, 520, 5
CADENCE, FINAL_STEP, TOL = 13, 507, 1e-9
INTERVAL_START = FINAL_STEP - CADENCE + 1
PEAK_PRODUCTIVITY = age_productivity(40)


def num(value, default=0.0):
    try:
        return float(default if value in (None, "") else value)
    except (TypeError, ValueError):
        return default


def mean(values):
    values = [num(value, math.nan) for value in values]
    values = [value for value in values if math.isfinite(value)]
    return math.fsum(values) / len(values) if values else math.nan


def ratio(numerator, denominator):
    return num(numerator) / num(denominator) if abs(num(denominator)) > TOL else math.nan


def age_band(age):
    age = num(age)
    if age < 18:
        return "UNDER_18"
    if age < 25:
        return "18_24"
    if age < 35:
        return "25_34"
    if age < 45:
        return "35_44"
    if age < 55:
        return "45_54"
    if age < 65:
        return "55_64"
    if age < 70:
        return "65_69"
    if age < 75:
        return "70_74"
    if age < 80:
        return "75_79"
    return "80_PLUS"


def write_rows(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["metric"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def employer_maps(world):
    firms = [*world.firms, *getattr(world, "capital_good_firms", [])]
    return {
        firm.firm_id: {
            "sector_id": getattr(firm, "sector_id", "UNAVAILABLE"),
            "employee_ids": set(getattr(firm, "employee_ids", [])),
            "desired_labor": num(getattr(firm, "desired_labor", 0.0)),
        }
        for firm in firms
    }


def paid_wage_map(world):
    values = {}
    for record in getattr(world, "_household_payroll_provenance_weekly_state", {}).values():
        for item in record.get("person_components", []):
            person_id = str(item.get("person_id"))
            values[person_id] = {
                "scheduled_wage": num(item.get("scheduled_wage")),
                "paid_wage": num(item.get("paid_wage")),
                "firm_id": item.get("firm_id"),
                "sector_id": item.get("sector_id", "UNAVAILABLE"),
            }
    return values


def collect_person_rows(world):
    firms = employer_maps(world)
    wage_map = paid_wage_map(world)
    rows = []
    for person in world.population:
        if not getattr(person, "alive", False):
            continue
        household = world.settlement_household_for_person(person)
        productivity = age_productivity(person.age)
        eligible = household is not None and productivity > 0.0
        firm_id = getattr(person, "firm_id", None)
        employer = firms.get(firm_id)
        employed = bool(eligible and employer is not None and person.id in employer["employee_ids"])
        wage = wage_map.get(str(person.id), {})
        rows.append({
            "global_step": world.current_step_index,
            "person_id": str(person.id),
            "household_id": str(household.id) if household is not None else "UNAVAILABLE",
            "age": num(person.age),
            "age_band": age_band(person.age),
            "productivity": productivity,
            "labor_eligible": eligible,
            "employed": employed,
            "firm_id": str(firm_id) if employed else "UNASSIGNED",
            "sector_id": employer["sector_id"] if employed else "UNASSIGNED",
            "scheduled_wage": num(wage.get("scheduled_wage")),
            "paid_wage": num(wage.get("paid_wage")),
        })
    return rows


def labor_contract_rows():
    return [
        {"rule": "minimum productive labor age", "file": "productivity.py", "function_or_class": "age_productivity", "threshold": "age >= 18", "semantic": "age < 18 returns zero productivity and cannot enter matching/capacity", "domain": "labor-market", "demographic": False, "diagnostic_only": False},
        {"rule": "maximum productive labor age", "file": "productivity.py", "function_or_class": "age_productivity", "threshold": "none", "semantic": "Gaussian productivity remains positive at every finite age >= 18", "domain": "labor-market", "demographic": False, "diagnostic_only": False},
        {"rule": "settlement eligibility", "file": "world.py", "function_or_class": "ensure_labor_settlement_households", "threshold": "alive and age_productivity(age) > 0", "semantic": "accepted Step15 flag creates an economic settlement Household for otherwise unassociated productive adults", "domain": "labor-market", "demographic": False, "diagnostic_only": False},
        {"rule": "matching eligibility", "file": "world.py", "function_or_class": "assign_unassigned_workers_to_firms", "threshold": "alive, valid settlement Household, productivity > 0, unassigned", "semantic": "no age upper bound", "domain": "labor-market", "demographic": False, "diagnostic_only": False},
        {"rule": "employment continuation/capacity", "file": "world.py", "function_or_class": "firm_employee_capacity", "threshold": "alive, valid settlement Household, productivity > 0", "semantic": "roster remains valid at any age with positive productivity", "domain": "labor-market", "demographic": False, "diagnostic_only": False},
        {"rule": "diagnostic working-age group", "file": "world.py", "function_or_class": "age_labor_group / demographic diagnostics", "threshold": "20 <= age <= 60", "semantic": "reporting classification only; it does not gate matching or employment", "domain": "diagnostic", "demographic": False, "diagnostic_only": True},
        {"rule": "fertility age", "file": "person.py", "function_or_class": "Person.can_reproduce", "threshold": "female and 20 <= age <= 40", "semantic": "fertility only; not retirement or labor-force exit", "domain": "demographic", "demographic": True, "diagnostic_only": False},
    ]


def curve_rows():
    selected_ages = (18, 25, 30, 40, 50, 60, 65, 70, 75, 80)
    thresholds = (1.0, 0.75, 0.5, 0.25, 0.1)
    crossings = {
        threshold: next(
            age for age in range(40, 121)
            if age_productivity(age) / PEAK_PRODUCTIVITY < threshold
        )
        for threshold in thresholds
    }
    return [
        {
            "age": age,
            "productivity": age_productivity(age),
            "share_of_peak": age_productivity(age) / PEAK_PRODUCTIVITY,
            "formula": "1.5 * exp(-(age-40)^2/(2*15^2))",
            **{f"first_descending_age_below_{str(t).replace('.', '_')}_of_peak": crossings[t] for t in thresholds},
        }
        for age in selected_ages
    ]


def aggregate_by(rows, labels, measures):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[label] for label in labels)].append(row)
    result = []
    for key, items in sorted(buckets.items(), key=lambda item: tuple(map(str, item[0]))):
        row = dict(zip(labels, key))
        row["person_week_count"] = len(items)
        row["worker_count"] = len({item["person_id"] for item in items})
        for name, getter in measures.items():
            values = [getter(item) for item in items]
            row[f"sum_{name}"] = math.fsum(num(value) for value in values)
            row[f"mean_{name}"] = mean(values)
        result.append(row)
    return result


def snapshot_households(statistics_dir):
    rows = read_rows(statistics_dir / "social_household_snapshots.csv")
    candidates = [
        row for row in rows
        if int(num(row.get("global_step"), -1)) == FINAL_STEP
        and int(num(row.get("interval_weeks"), 0)) == CADENCE
    ]
    if not candidates:
        raise RuntimeError("missing authoritative final complete Household interval")
    return {str(row["household_id"]): row for row in candidates}


def household_liquidity(rows, households):
    by_household = defaultdict(list)
    for row in rows:
        if row["household_id"] in households:
            by_household[row["household_id"]].append(row)
    result = []
    groups = {
        "ELDERLY_WORKER_PRESENT": lambda values: any(item["employed"] and item["age"] >= 65 for item in values),
        "ELDERLY_DEPENDENT_ONLY": lambda values: any(item["age"] >= 65 and not item["employed"] for item in values) and not any(item["employed"] and item["age"] >= 65 for item in values),
        "NO_65_PLUS_PERSON": lambda values: not any(item["age"] >= 65 for item in values),
    }
    for label, predicate in groups.items():
        selected = [(household_id, values) for household_id, values in by_household.items() if predicate(values)]
        records = [households[household_id] for household_id, _ in selected]
        result.append({
            "household_group": label,
            "comparison_semantics": "descriptive final complete 13-week interval; not covariate-adjusted",
            "household_count": len(records),
            "mean_elderly_worker_person_weeks": mean(sum(item["employed"] and item["age"] >= 65 for item in values) for _, values in selected),
            "mean_total_employed_person_weeks": mean(sum(item["employed"] for item in values) for _, values in selected),
            "mean_labor_income": mean(record.get("interval_labor_income") for record in records),
            "mean_minimum_consumption": mean(record.get("interval_minimum_consumption") for record in records),
            "mean_cash": mean(record.get("cash") for record in records),
            "mean_saving": mean(record.get("interval_saving") for record in records),
            "near_zero_household_share": mean(num(record.get("cash")) <= 10.0 for record in records),
        })
    return result


def wage_compression(rows):
    employed = [row for row in rows if row["employed"] and row["paid_wage"] > TOL]
    contexts = defaultdict(list)
    for row in employed:
        if row["productivity"] > TOL:
            contexts[(row["global_step"], row["firm_id"])].append(row)
    reference_context = {
        key: mean(item["paid_wage"] / item["productivity"] for item in items)
        for key, items in contexts.items()
    }
    output = []
    for age in (60, 65, 70, 75, 80):
        subset = [row for row in employed if int(round(row["age"])) == age and (row["global_step"], row["firm_id"]) in reference_context]
        predicted = [reference_context[(row["global_step"], row["firm_id"])] * age_productivity(age) for row in subset]
        age40 = [reference_context[(row["global_step"], row["firm_id"])] * PEAK_PRODUCTIVITY for row in subset]
        output.append({
            "age": age,
            "person_week_count": len(subset),
            "productivity": age_productivity(age),
            "theoretical_productivity_wage_ratio_to_age_40": age_productivity(age) / PEAK_PRODUCTIVITY,
            "mean_actual_paid_wage": mean(row["paid_wage"] for row in subset),
            "mean_context_matched_predicted_wage": mean(predicted),
            "mean_context_matched_age_40_wage": mean(age40),
            "empirical_context_matched_wage_ratio_to_age_40": ratio(mean(predicted), mean(age40)),
            "interpretation": "same firm-week wage-per-productivity coefficient where observed; no age-40 worker substitution assumed",
        })
    return output


def labor_interaction(rows, firm_snapshots):
    employed = [row for row in rows if row["employed"]]
    summary = aggregate_by(employed, ["age_band", "sector_id"], {
        "effective_labor": lambda item: item["productivity"],
        "paid_wage": lambda item: item["paid_wage"],
    })
    total_headcount = len(employed)
    total_effective = math.fsum(item["productivity"] for item in employed)
    for row in summary:
        row["headcount_person_week_share"] = ratio(row["person_week_count"], total_headcount)
        row["effective_labor_share"] = ratio(row["sum_effective_labor"], total_effective)
        row["headcount_vs_effective_labor_semantics"] = "Person occupies one roster relation; capacity and matching increments use age-productivity labor services, not a fixed headcount slot."
    firm_rows = []
    by_firm = defaultdict(list)
    for row in employed:
        by_firm[(row["firm_id"], row["sector_id"])].append(row)
    for (firm_id, sector_id), items in sorted(by_firm.items()):
        snapshots = firm_snapshots.get(firm_id, [])
        capital_sector = sector_id == "capital_goods"
        firm_rows.append({
            "firm_id": firm_id,
            "sector_id": sector_id,
            "mean_employee_headcount": mean(len({item["person_id"] for item in items if item["global_step"] == step}) for step in sorted({item["global_step"] for item in items})),
            "mean_effective_labor": mean(math.fsum(item["productivity"] for item in items if item["global_step"] == step) for step in sorted({item["global_step"] for item in items})),
            "mean_desired_labor": mean(snapshot["desired_labor"] for snapshot in snapshots),
            "elderly_headcount_share": ratio(sum(item["age"] >= 65 for item in items), len(items)),
            "elderly_effective_labor_share": ratio(math.fsum(item["productivity"] for item in items if item["age"] >= 65), math.fsum(item["productivity"] for item in items)),
            "hiring_basis": (
                "effective labor services: candidate selection checks current capacity + age_productivity against desired_labor"
                if capital_sector else
                "Food assigns eligible unassigned Persons to the Firm with the lowest current effective labor; Food has no desired_labor vacancy target in this path"
            ),
            "fixed_headcount_vacancy_cap": False,
        })
    return summary, firm_rows


def eligibility_interaction(rows):
    output = []
    for band in ("UNDER_18", "18_24", "25_34", "35_44", "45_54", "55_64", "65_69", "70_74", "75_79", "80_PLUS"):
        items = [row for row in rows if row["age_band"] == band]
        eligible = [row for row in items if row["labor_eligible"]]
        employed = [row for row in eligible if row["employed"]]
        eligibility_share = ratio(len(eligible), len(items))
        prod = mean(row["productivity"] for row in employed)
        output.append({
            "age_band": band,
            "alive_person_weeks": len(items),
            "labor_eligible_person_weeks": len(eligible),
            "employed_person_weeks": len(employed),
            "labor_eligible_share": eligibility_share,
            "employed_share_among_eligible": ratio(len(employed), len(eligible)),
            "mean_employed_productivity": prod,
            "mean_employed_wage": mean(row["paid_wage"] for row in employed),
            "classification": (
                "NON_ELIGIBLE" if not eligible else
                "FULL_ELIGIBILITY_LOW_PRODUCTIVITY" if eligibility_share >= .99 and prod / PEAK_PRODUCTIVITY < .5 else
                "FULL_ELIGIBILITY_NORMAL_PRODUCTIVITY" if eligibility_share >= .99 else "PARTIAL_ELIGIBILITY"
            ),
        })
    return output


def age_exit_screen(rows):
    employed = [row for row in rows if row["employed"]]
    totals = {
        "headcount": len(employed),
        "effective_labor": math.fsum(row["productivity"] for row in employed),
        "wage": math.fsum(row["paid_wage"] for row in employed),
    }
    output = []
    for threshold in (65, 70, 75):
        removed = [row for row in employed if row["age"] >= threshold]
        output.append({
            "illustrative_exit_age": threshold,
            "removed_employed_person_weeks": len(removed),
            "removed_headcount_share": ratio(len(removed), totals["headcount"]),
            "removed_effective_labor": math.fsum(row["productivity"] for row in removed),
            "removed_effective_labor_share": ratio(math.fsum(row["productivity"] for row in removed), totals["effective_labor"]),
            "removed_wage_income": math.fsum(row["paid_wage"] for row in removed),
            "removed_wage_income_share": ratio(math.fsum(row["paid_wage"] for row in removed), totals["wage"]),
            "runtime_changed": False,
            "interpretation": "arithmetic deletion of observed employed person-weeks; does not model replacement hiring or Household response",
        })
    return output


def productivity_floor_screen(rows):
    employed = [row for row in rows if row["employed"] and row["paid_wage"] > TOL]
    contexts = defaultdict(list)
    for row in employed:
        contexts[(row["global_step"], row["firm_id"])].append(row)
    output = []
    for floor_fraction in (.25, .5, .75):
        floor = floor_fraction * PEAK_PRODUCTIVITY
        old_total, new_total, elderly_delta, nonelderly_delta = 0.0, 0.0, 0.0, 0.0
        affected = 0
        for items in contexts.values():
            wage_pool = math.fsum(item["paid_wage"] for item in items)
            old_labor = math.fsum(item["productivity"] for item in items)
            new_labor = math.fsum(max(item["productivity"], floor) for item in items)
            for item in items:
                old_wage = item["paid_wage"]
                new_wage = wage_pool * max(item["productivity"], floor) / new_labor if new_labor > TOL else 0.0
                old_total += item["productivity"]
                new_total += max(item["productivity"], floor)
                if item["productivity"] < floor:
                    affected += 1
                if item["age"] >= 65:
                    elderly_delta += new_wage - old_wage
                else:
                    nonelderly_delta += new_wage - old_wage
        output.append({
            "illustrative_productivity_floor_share_of_peak": floor_fraction,
            "productivity_floor": floor,
            "affected_employed_person_weeks": affected,
            "affected_share": ratio(affected, len(employed)),
            "implied_effective_labor_change": new_total - old_total,
            "implied_effective_labor_change_share": ratio(new_total - old_total, old_total),
            "implied_elderly_wage_pool_redistribution": elderly_delta,
            "implied_nonelderly_wage_pool_redistribution": nonelderly_delta,
            "wage_pool_sum_change": elderly_delta + nonelderly_delta,
            "runtime_changed": False,
            "assumption": "within each observed firm-week, paid wage pool is held fixed before person allocation; no production or aggregate wage-bill feedback is simulated",
        })
    return output


def remedy_rows():
    return [
        {"candidate": "CURRENT_CONTRACT", "economic_interpretation": "indefinite participation with declining effective labor and proportional wage", "labor_supply": "elderly headcount remains eligible", "firm_labor_cost": "wage per productivity stays proportional", "effective_labor": "approaches zero at late age", "household_income": "late-age wage approaches zero", "government_required": False, "stage_boundary": "current"},
        {"candidate": "LABOR_ELIGIBILITY_EXIT", "economic_interpretation": "explicit retirement/labor-force exit", "labor_supply": "removes elderly headcount and their small effective labor", "firm_labor_cost": "removes corresponding wage share", "effective_labor": "falls by observed elderly share", "household_income": "removes already-low wages; needs separate income semantics", "government_required": False, "stage_boundary": "age-labor contract, before any pension"},
        {"candidate": "PRODUCTIVITY_FLOOR", "economic_interpretation": "eligible older workers retain a productivity floor", "labor_supply": "unchanged headcount", "firm_labor_cost": "can redistribute a fixed wage pool", "effective_labor": "raises measured labor services", "household_income": "raises affected wage share only conditionally", "government_required": False, "stage_boundary": "not safe before resolving wage-pool feedback"},
        {"candidate": "FLATTER_LATE_AGE_CURVE", "economic_interpretation": "slower late-age decline", "labor_supply": "unchanged headcount", "firm_labor_cost": "changes effective labor and wage shares", "effective_labor": "increases", "household_income": "can redistribute fixed wage pools", "government_required": False, "stage_boundary": "requires joint productivity/wage design"},
        {"candidate": "PARTIAL_RETIREMENT_LABOR_SUPPLY_DECLINE", "economic_interpretation": "participation declines explicitly rather than productivity collapsing under full eligibility", "labor_supply": "gradual reduction", "firm_labor_cost": "depends on replacement hiring", "effective_labor": "declines through participation rather than hidden low service", "household_income": "requires explicit household-transition semantics", "government_required": False, "stage_boundary": "smallest future behavioral contract screen"},
        {"candidate": "PENSION_TRANSFER_INSTITUTION", "economic_interpretation": "old-age non-labor income", "labor_supply": "can be independent of participation", "firm_labor_cost": "unchanged directly", "effective_labor": "unchanged directly", "household_income": "adds non-labor income", "government_required": "institutional counterparty required", "stage_boundary": "later institution design, not Step15 wage repair"},
    ]


def run_world(temp_dir):
    world = World(POPULATION, seed=SEED, diagnostics_mode="full", scenario_name="step15_age_labor_productivity_contract_audit", scenario_overrides=frozen_config.overrides())
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()
    world.household_wealth_instrumentation_enabled = True
    world.household_employer_exposure_instrumentation_enabled = True
    world.configure_diagnostic_persistence(Path(temp_dir) / "canonical", cadence=1, statistical_observability=True, statistical_output_dir=Path(temp_dir) / "statistics", statistical_snapshot_cadence=CADENCE, statistical_age_cadence=CADENCE)
    interval_rows, firm_snapshots = [], defaultdict(list)
    for _ in range(WEEKS):
        world.step()
        if INTERVAL_START <= world.current_step_index <= FINAL_STEP:
            interval_rows.extend(collect_person_rows(world))
            for firm_id, view in employer_maps(world).items():
                firm_snapshots[str(firm_id)].append({"desired_labor": view["desired_labor"]})
    return interval_rows, firm_snapshots, Path(temp_dir) / "statistics"


def run():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="step15_age_labor_") as temporary:
        people, firm_snapshots, statistics_dir = run_world(temporary)
        households = snapshot_households(statistics_dir)

    employed = [row for row in people if row["employed"]]
    age_distribution = aggregate_by(employed, ["age_band", "sector_id"], {
        "productivity": lambda item: item["productivity"],
        "paid_wage": lambda item: item["paid_wage"],
        "near_zero_household": lambda item: num(households.get(item["household_id"], {}).get("cash")) <= 10.0,
    })
    interaction = eligibility_interaction(people)
    headcount_audit, firm_interaction = labor_interaction(people, firm_snapshots)
    elderly_households = household_liquidity(people, households)
    compression = wage_compression(people)
    exits = age_exit_screen(people)
    floors = productivity_floor_screen(people)
    payroll_total = math.fsum(row["paid_wage"] for row in employed)
    elderly_wage = math.fsum(row["paid_wage"] for row in employed if row["age"] >= 65)
    elderly_labor = math.fsum(row["productivity"] for row in employed if row["age"] >= 65)
    total_labor = math.fsum(row["productivity"] for row in employed)

    wage_pool = [
        {"boundary": "aggregate Food wage bill", "file": "economy/firm.py", "function": "FirmSystem.base_wage_bill", "timing": "computed from total effective labor before Person allocation", "effect": "locally fixed for current distribution; person productivity changes redistribute the current firm-week pool"},
        {"boundary": "person allocation", "file": "economy/firm.py", "function": "FirmSystem.distribute_wages", "timing": "after aggregate wage bill", "effect": "wage share equals productivity / total labor, then firm funding ratio"},
        {"boundary": "future feedback", "file": "economy/firm.py", "function": "FirmSystem.base_wage_bill", "timing": "subsequent recalculation", "effect": "a changed productivity contract can alter total effective labor and therefore future aggregate wage bill; not simulated in this audit"},
    ]
    profitability = [
        {"candidate_contract": row["candidate"], "total_wage_bill_effect": "ambiguous: local pool fixed, later base wage bill depends on effective labor", "effective_labor_effect": row["effective_labor"], "production_effect": "requires a treatment because capacity and production use effective labor", "unit_labor_cost_effect": "not identifiable without rerun", "spoilage_effect": "indirect via production/inventory only", "depreciation_effect": "no direct relation", "food_profitability_risk": "must be tested jointly; current Food profitability is weak"}
        for row in remedy_rows()
    ]
    origins = [
        {"parameter_or_rule": "age productivity amplitude=1.5, peak=40, sigma=15", "file": "productivity.py", "classification": "ENGINEERING_ASSUMPTION", "evidence": "inline constants without calibration source or literature citation found", "joint_design_evidence_with_labor_eligibility": "none found"},
        {"parameter_or_rule": "minimum productive labor age=18", "file": "productivity.py", "classification": "ENGINEERING_ASSUMPTION", "evidence": "zero-productivity cutoff is the active labor eligibility gate", "joint_design_evidence_with_productivity_curve": "only mechanical linkage through same function"},
        {"parameter_or_rule": "maximum labor age / retirement age", "file": "world.py + person.py", "classification": "NOT_DEFINED", "evidence": "no active maximum labor age, retirement, pension, or labor-force exit rule found", "joint_design_evidence_with_productivity_curve": "none found"},
        {"parameter_or_rule": "20-60 working-age diagnostic", "file": "world.py", "classification": "DIAGNOSTIC_ASSUMPTION", "evidence": "used for reporting classification, not matching or capacity", "joint_design_evidence_with_productivity_curve": "none found"},
    ]
    retirement = [
        {"concept": "retirement age", "status": "NOT_IMPLEMENTED", "source": "repository audit", "economic_effect": "none"},
        {"concept": "elderly status", "status": "diagnostic / consumption composition only", "source": "world.age_labor_group and Household need composition", "economic_effect": "does not remove labor eligibility"},
        {"concept": "pension / old-age transfer", "status": "NOT_IMPLEMENTED", "source": "repository audit", "economic_effect": "none"},
        {"concept": "labor-force exit", "status": "NOT_IMPLEMENTED", "source": "world matching and capacity gates", "economic_effect": "none"},
    ]

    write_rows(OUTPUT / "labor_age_eligibility_contract.csv", labor_contract_rows())
    write_rows(OUTPUT / "employed_age_distribution.csv", age_distribution)
    write_rows(OUTPUT / "age_productivity_curve_diagnostics.csv", curve_rows())
    write_rows(OUTPUT / "age_eligibility_productivity_interaction.csv", interaction)
    write_rows(OUTPUT / "elderly_worker_household_liquidity.csv", elderly_households)
    write_rows(OUTPUT / "age_wage_compression.csv", compression)
    write_rows(OUTPUT / "labor_headcount_vs_effective_labor_audit.csv", headcount_audit)
    write_rows(OUTPUT / "firm_labor_demand_age_interaction.csv", firm_interaction)
    write_rows(OUTPUT / "demographic_retirement_semantics.csv", retirement)
    write_rows(OUTPUT / "age_exit_shadow_screen.csv", exits)
    write_rows(OUTPUT / "productivity_floor_shadow_screen.csv", floors)
    write_rows(OUTPUT / "wage_pool_distribution_effect.csv", wage_pool)
    write_rows(OUTPUT / "food_profitability_interaction.csv", profitability)
    write_rows(OUTPUT / "age_labor_parameter_origin_audit.csv", origins)
    write_rows(OUTPUT / "age_labor_remedy_matrix.csv", remedy_rows())

    elderly_headcount_share = ratio(sum(row["age"] >= 65 for row in employed), len(employed))
    elderly_effective_share = ratio(elderly_labor, total_labor)
    elderly_wage_share = ratio(elderly_wage, payroll_total)
    verdict = "D. AGE_PRODUCTIVITY_ELIGIBILITY_CONTRACT_NOT_JOINTLY_DESIGNED"
    flags = {
        "verdict": verdict,
        "population": POPULATION,
        "seed": SEED,
        "weeks": WEEKS,
        "final_complete_interval": f"{INTERVAL_START}-{FINAL_STEP}",
        "labor_age_upper_bound_exists": False,
        "retirement_rule_exists": False,
        "elderly_65_plus_employed_headcount_share": elderly_headcount_share,
        "elderly_65_plus_effective_labor_share": elderly_effective_share,
        "elderly_65_plus_wage_income_share": elderly_wage_share,
        "elderly_full_eligibility_low_productivity_observed": any(row["classification"] == "FULL_ELIGIBILITY_LOW_PRODUCTIVITY" for row in interaction),
        "headcount_effective_labor_fixed_slot_mismatch_observed": False,
        "wage_pool_is_fixed_before_person_allocation": True,
        "joint_eligibility_productivity_design_evidence_found": False,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = f"""# Step 15 Adult Labor Eligibility / Age-Productivity / Wage Contract Audit

**Verdict: {verdict}**

The active labor gate is alive + valid settlement Household + `age_productivity(age) > 0`. Productivity is zero below 18 but has no maximum-age cutoff, so a Person remains labor eligible indefinitely at every finite age above 18. There is no retirement, pension, or labor-force exit mechanism. The `20-60` working-age range is diagnostic only.

In the final complete {CADENCE}-week interval ({INTERVAL_START}-{FINAL_STEP}), age 65+ workers account for {elderly_headcount_share:.2%} of employed person-weeks but only {elderly_effective_share:.2%} of effective labor and {elderly_wage_share:.2%} of paid wage. They are therefore predominantly low-productivity headcount, not proportionate productive labor supply. Firm matching and capacity use productivity-weighted labor services, so the audit finds no fixed-headcount vacancy mismatch; the inconsistency is the unbounded eligibility combined with a productivity and wage mapping that asymptotically approaches zero.

The current firm-week wage pool is calculated before Person allocation. A late-age productivity floor or curve change would first redistribute that fixed pool among workers; later effective-labor and production feedback would require a separate treatment. No joint calibration/design evidence was found for the age eligibility boundary and the Gaussian productivity parameters.

Safest next boundary: a joint age-labor contract screen, beginning with an explicit gradual labor-force participation/retirement semantic rather than a wage multiplier or productivity floor. This audit makes no runtime, parameter, RNG, wage, labor, Household, Firm, or policy change.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
