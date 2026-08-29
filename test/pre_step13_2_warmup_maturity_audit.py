import argparse
import csv
import gc
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenarios import apply_runtime_scenario, apply_scenario
from world import World


DEFAULT_OUTPUT = ROOT / "test" / "output" / "pre_step13_2_warmup_maturity_audit"
DEFAULT_SEEDS = (1, 7, 21, 42, 99)
WEEKS_PER_YEAR = 52
CANDIDATE_YEARS = (10, 15, 20, 25, 30, 40)
HOUSEHOLD_SIZE_KEYS = ("size_1_share", "size_2_share", "size_3_share", "size_4_share", "size_5_plus_share")
ROLLING_FIELDS = (
    "population",
    "total_household_wealth",
    "total_consumption",
    "total_income",
    "realized_transaction_price_index",
    "household_planning_price_index",
    "food_output_units",
    "food_inventory_units",
    "inventory_demand_ratio",
    "firm_cash",
    "sum_firm_profit_before_dividend",
    "working_capital_loan_balance",
    "labor",
    "wage_bill",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Observe fresh-world warm-up maturity without changing behavior."
    )
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--firm-count", type=int, default=5)
    parser.add_argument("--weeks", type=int, default=3120)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--detailed-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-interval", type=int, default=260)
    parser.add_argument(
        "--seed42-only",
        action="store_true",
        help="Run only the detailed seed before the multi-seed audit.",
    )
    return parser.parse_args()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(numerator, denominator, default=0.0):
    return numerator / denominator if denominator else default


def percentile(values, quantile):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * quantile
    lower = int(math.floor(rank))
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def cv(values):
    values = [float(value) for value in values]
    if not values:
        return 0.0
    mean = statistics.fmean(values)
    return safe_ratio(statistics.pstdev(values), abs(mean))


def valid_household_assignment(world, person):
    household = world.household_dict.get(person.household_id)
    return household is not None and (
        person.id in household.parents or person.id in household.children
    )


def active_households(world):
    return [household for household in world.households if household.size() > 0]


def household_size_distribution(households):
    sizes = [household.size() for household in households]
    total = len(sizes)
    return {
        "size_1_share": safe_ratio(sum(size == 1 for size in sizes), total),
        "size_2_share": safe_ratio(sum(size == 2 for size in sizes), total),
        "size_3_share": safe_ratio(sum(size == 3 for size in sizes), total),
        "size_4_share": safe_ratio(sum(size == 4 for size in sizes), total),
        "size_5_plus_share": safe_ratio(sum(size >= 5 for size in sizes), total),
    }


def total_variation(left, right):
    return 0.5 * sum(abs(float(left[key]) - float(right[key])) for key in HOUSEHOLD_SIZE_KEYS)


def posted_price_index(firms):
    prices = [max(1e-12, float(firm.price)) for firm in firms]
    shares = [max(0.0, float(getattr(firm, "unit_market_share", 0.0))) for firm in firms]
    if sum(shares) <= 1e-12:
        shares = [max(0.0, float(getattr(firm, "share", 0.0))) for firm in firms]
    if sum(shares) <= 1e-12:
        shares = [1.0 for _ in firms]
    total = sum(shares)
    return sum(price * share / total for price, share in zip(prices, shares))


def firm_cross_section(world):
    firms = list(world.firms)
    shares = [max(0.0, float(getattr(firm, "unit_market_share", 0.0))) for firm in firms]
    if sum(shares) <= 1e-12:
        shares = [max(0.0, float(getattr(firm, "share", 0.0))) for firm in firms]
    share_total = sum(shares)
    normalized_shares = (
        [share / share_total for share in shares]
        if share_total > 0 else [1.0 / len(firms) for _ in firms]
    )
    prices = [float(firm.price) for firm in firms]
    sales = [float(getattr(firm, "sales", 0.0)) for firm in firms]
    production = [float(getattr(firm, "actual_production", 0.0)) for firm in firms]
    inventory = [float(getattr(firm, "inventory_units", 0.0)) for firm in firms]
    profits = [float(getattr(firm, "profit", 0.0)) for firm in firms]
    loans = [float(getattr(firm, "loan_balance", 0.0)) for firm in firms]
    coverages = [float(getattr(firm, "inventory_coverage", 0.0)) for firm in firms]
    return {
        "market_hhi": sum(share * share for share in normalized_shares),
        "price_cv": cv(prices),
        "sales_cv": cv(sales),
        "production_cv": cv(production),
        "inventory_cv": cv(inventory),
        "profit_cv": cv(profits),
        "loan_cv": cv(loans),
        "firm_price_min": min(prices),
        "firm_price_max": max(prices),
        "firm_price_mean": statistics.fmean(prices),
        "inventory_coverage": statistics.fmean(coverages),
        "firm_cash": sum(float(firm.cash) for firm in firms),
        "firm_profit": sum(profits),
        "loan_balance": sum(loans),
        "food_output": sum(production),
        "food_inventory": sum(inventory),
    }


def initial_snapshot(world, initial_household_ids, initial_person_ids):
    return snapshot(world, 0, initial_household_ids, initial_person_ids, None)


def snapshot(
    world,
    completed_week,
    initial_household_ids,
    initial_person_ids,
    latest_macro,
):
    people = list(world.population)
    alive_initial_cohort = sum(
        person.id in initial_person_ids for person in people
    )
    households = active_households(world)
    sizes = [household.size() for household in households]
    assigned = sum(valid_household_assignment(world, person) for person in people)
    wealth = [float(household.wealth) for household in households]
    income = [float(getattr(household, "income_this_step", 0.0)) for household in households]
    saving = [float(getattr(household, "saving_this_step", 0.0)) for household in households]
    security = []
    unmet = []
    for household in households:
        target = float(getattr(household, "target_wealth_this_step", 0.0))
        if target > 0:
            security.append(household.wealth / target)
        gap = float(getattr(household, "food_need_gap_units_this_step", 0.0))
        unmet.append(gap > 1e-12)

    firm = firm_cross_section(world)
    initial_surviving = sum(
        household.id in initial_household_ids for household in households
    )
    current_ids = {household.id for household in households}
    size_distribution = household_size_distribution(households)
    macro = latest_macro or {}
    realized_price = (
        float(macro["realized_transaction_price_index"])
        if macro else None
    )
    planning_price = (
        float(macro["household_planning_price_index"])
        if macro else posted_price_index(world.firms)
    )
    total_money = (
        float(macro["total_money_stock"])
        if macro else (
            sum(household.wealth for household in world.households)
            + firm["firm_cash"]
            + float(getattr(world, "public_wealth", 0.0))
            + float(getattr(world.firm_system.central_bank, "public_income_balance", 0.0))
        )
    )
    children = sum(person.age < 20 for person in people)
    workers = sum(20 <= person.age <= 60 for person in people)
    elderly = len(people) - children - workers
    row = {
        "model_year": completed_week / WEEKS_PER_YEAR,
        "completed_week": completed_week,
        "population": len(people),
        "initial_cohort_alive_count": alive_initial_cohort,
        "initial_cohort_survival_share": safe_ratio(
            alive_initial_cohort, len(initial_person_ids)
        ),
        "share_initial_cohort_alive": safe_ratio(
            alive_initial_cohort, len(people)
        ),
        "active_households": len(households),
        "total_household_objects": len(world.households),
        "assigned_population": assigned,
        "unassigned_population": len(people) - assigned,
        "unassigned_share": safe_ratio(len(people) - assigned, len(people)),
        "mean_household_size": statistics.fmean(sizes) if sizes else 0.0,
        "median_household_size": percentile(sizes, 0.5),
        "married_households": sum(len(household.parents) == 2 for household in households),
        "married_household_share": safe_ratio(sum(len(household.parents) == 2 for household in households), len(households)),
        "single_parent_households": sum(len(household.parents) == 1 for household in households),
        "single_parent_household_share": safe_ratio(sum(len(household.parents) == 1 for household in households), len(households)),
        "households_with_children": sum(bool(household.children) for household in households),
        "households_with_children_share": safe_ratio(sum(bool(household.children) for household in households), len(households)),
        "mean_child_count_per_household": safe_ratio(sum(len(household.children) for household in households), len(households)),
        "child_share": safe_ratio(children, len(people)),
        "worker_share": safe_ratio(workers, len(people)),
        "elderly_share": safe_ratio(elderly, len(people)),
        "total_household_wealth": sum(wealth),
        "median_household_wealth": percentile(wealth, 0.5),
        "p10_household_wealth": percentile(wealth, 0.1),
        "p90_household_wealth": percentile(wealth, 0.9),
        "median_household_income": percentile(income, 0.5),
        "p10_household_income": percentile(income, 0.1),
        "p90_household_income": percentile(income, 0.9),
        "median_saving": percentile(saving, 0.5),
        "p10_saving": percentile(saving, 0.1),
        "p90_saving": percentile(saving, 0.9),
        "saving_nonpositive_share": safe_ratio(sum(value <= 0 for value in saving), len(saving)),
        "unmet_need_household_share": safe_ratio(sum(unmet), len(unmet)),
        "median_security_ratio": percentile(security, 0.5),
        "p10_security_ratio": percentile(security, 0.1),
        "p90_security_ratio": percentile(security, 0.9),
        "total_consumption": float(macro.get("total_consumption", 0.0)),
        "total_income": float(macro.get("total_income", 0.0)),
        "realized_transaction_price": realized_price,
        "household_planning_price": planning_price,
        "food_output": firm["food_output"],
        "food_inventory": firm["food_inventory"],
        "inventory_coverage": firm["inventory_coverage"],
        "firm_cash": firm["firm_cash"],
        "firm_profit": firm["firm_profit"],
        "loan_balance": firm["loan_balance"],
        "total_money_stock": total_money,
        "initial_households_total": len(initial_household_ids),
        "initial_households_surviving": initial_surviving,
        "initial_households_dissolved": len(initial_household_ids) - initial_surviving,
        "initial_household_survival_share": safe_ratio(initial_surviving, len(initial_household_ids)),
        "new_active_households": sum(household.id not in initial_household_ids for household in households),
        "current_initial_household_ids_present": len(current_ids & initial_household_ids),
        **size_distribution,
        **{key: value for key, value in firm.items() if key not in {
            "firm_cash", "firm_profit", "loan_balance", "food_output", "food_inventory", "inventory_coverage"
        }},
    }
    return row


def attach_distribution_distances(yearly):
    for index, row in enumerate(yearly):
        if index == 0:
            row["household_size_tv_vs_prior_year"] = None
            row["household_size_tv_vs_5_years_prior"] = None
            row["demographic_max_share_change_vs_prior_year"] = None
            row["demographic_max_share_change_vs_5_years_prior"] = None
            continue
        prior = yearly[index - 1]
        row["household_size_tv_vs_prior_year"] = total_variation(row, prior)
        row["demographic_max_share_change_vs_prior_year"] = max(
            abs(row[key] - prior[key]) for key in ("child_share", "worker_share", "elderly_share")
        )
        if index >= 5:
            old = yearly[index - 5]
            row["household_size_tv_vs_5_years_prior"] = total_variation(row, old)
            row["demographic_max_share_change_vs_5_years_prior"] = max(
                abs(row[key] - old[key]) for key in ("child_share", "worker_share", "elderly_share")
            )
        else:
            row["household_size_tv_vs_5_years_prior"] = None
            row["demographic_max_share_change_vs_5_years_prior"] = None


def rolling_transient_rows(macro_rows):
    output = []
    for end in range(WEEKS_PER_YEAR - 1, len(macro_rows), WEEKS_PER_YEAR):
        window = macro_rows[end - WEEKS_PER_YEAR + 1:end + 1]
        previous = macro_rows[end - WEEKS_PER_YEAR] if end >= WEEKS_PER_YEAR else None
        current = macro_rows[end]
        year = (end + 1) // WEEKS_PER_YEAR
        for field in ROLLING_FIELDS:
            values = [float(row[field]) for row in window]
            current_value = float(current[field])
            previous_value = float(previous[field]) if previous else None
            output.append({
                "model_year": year,
                "completed_week": end + 1,
                "metric": field,
                "current_value": current_value,
                "rolling_52_mean": statistics.fmean(values),
                "rolling_52_std": statistics.pstdev(values),
                "rolling_52_cv": safe_ratio(statistics.pstdev(values), abs(statistics.fmean(values))),
                "year_over_year_absolute_change": current_value - previous_value if previous_value is not None else None,
                "year_over_year_relative_change": safe_ratio(current_value - previous_value, abs(previous_value), None) if previous_value not in (None, 0.0) else None,
            })
    return output


def lag_correlation(values, lag):
    if len(values) <= lag:
        return None
    left = values[:-lag]
    right = values[lag:]
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def periodicity_windows(macro_rows):
    output = []
    for start_year in range(0, len(macro_rows) // WEEKS_PER_YEAR, 10):
        end_year = min(start_year + 10, len(macro_rows) // WEEKS_PER_YEAR)
        window = macro_rows[start_year * 52:end_year * 52]
        if len(window) < 105:
            continue
        marriage_weeks = [bool(row.get("marriage_market_executed")) for row in window]
        for field in ("workers", "elderly", "labor", "wage_bill"):
            values = [float(row[field]) for row in window]
            deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
            flags = marriage_weeks[1:]
            marriage_abs = [abs(delta) for delta, flag in zip(deltas, flags) if flag]
            other_abs = [abs(delta) for delta, flag in zip(deltas, flags) if not flag]
            output.append({
                "start_year": start_year,
                "end_year": end_year,
                "metric": field,
                "weekly_delta_lag_52_autocorrelation": lag_correlation(deltas, 52),
                "mean_abs_delta_marriage_weeks": statistics.fmean(marriage_abs) if marriage_abs else None,
                "mean_abs_delta_other_weeks": statistics.fmean(other_abs) if other_abs else None,
                "marriage_week_abs_delta_ratio": safe_ratio(
                    statistics.fmean(marriage_abs), statistics.fmean(other_abs), None
                ) if marriage_abs and other_abs and statistics.fmean(other_abs) else None,
            })
    return output


def first_reached_and_stays(yearly, field, threshold):
    for index, row in enumerate(yearly):
        if row[field] < threshold and all(item[field] < threshold for item in yearly[index:]):
            return row["model_year"]
    return None


def first_rolling_condition(yearly, predicate, minimum_year=5, required_years=5):
    for index, row in enumerate(yearly):
        if row["model_year"] < minimum_year:
            continue
        block = yearly[index:index + required_years]
        if len(block) == required_years and all(predicate(item) for item in block):
            return row["model_year"]
    return None


def maturity_metrics(yearly, survival, transient, periodicity, tracker):
    by_year = {int(row["model_year"]): row for row in yearly}
    assignment_thresholds = {
        str(threshold): first_reached_and_stays(yearly, "unassigned_share", threshold)
        for threshold in (0.05, 0.02, 0.01)
    }
    structure_thresholds = {}
    for threshold in (0.10, 0.05, 0.02):
        structure_thresholds[str(threshold)] = first_rolling_condition(
            yearly,
            lambda row, limit=threshold: (
                row["household_size_tv_vs_prior_year"] is not None
                and row["household_size_tv_vs_prior_year"] < limit
                and row["household_size_tv_vs_5_years_prior"] is not None
                and row["household_size_tv_vs_5_years_prior"] < limit * 2
            ),
        )
    demographic_thresholds = {}
    for threshold in (0.02, 0.01, 0.005):
        demographic_thresholds[str(threshold)] = first_rolling_condition(
            yearly,
            lambda row, limit=threshold: (
                row["demographic_max_share_change_vs_prior_year"] is not None
                and row["demographic_max_share_change_vs_prior_year"] < limit
                and row["demographic_max_share_change_vs_5_years_prior"] is not None
                and row["demographic_max_share_change_vs_5_years_prior"] < limit * 5
            ),
        )

    absorbed_times = sorted(tracker["first_absorbed_week"].values())
    initial_count = len(tracker["initial_unassigned_ids"])
    absorption = {
        "initial_unassigned_count": initial_count,
        "absorbed_count": len(absorbed_times),
        "died_before_absorption_count": len(tracker["died_before_absorption"]),
        "still_alive_unassigned_at_end": survival[-1]["remaining_initial_unassigned"],
        "absorption_share": safe_ratio(len(absorbed_times), initial_count),
        "death_before_absorption_share": safe_ratio(len(tracker["died_before_absorption"]), initial_count),
        "absorption_time_among_absorbed_weeks": {
            "median": percentile(absorbed_times, 0.50),
            "p75": percentile(absorbed_times, 0.75),
            "p90": percentile(absorbed_times, 0.90),
            "p95": percentile(absorbed_times, 0.95),
            "p99": percentile(absorbed_times, 0.99),
        },
    }
    return {
        "assignment_threshold_first_year_reached_and_stays": assignment_thresholds,
        "household_structure_candidate_first_5_year_block": structure_thresholds,
        "demographic_composition_candidate_first_5_year_block": demographic_thresholds,
        "initial_unassigned_absorption": absorption,
        "initial_household_survival_share_at_year_20": (
            by_year[20]["initial_household_survival_share"] if 20 in by_year else None
        ),
        "initial_household_survival_share_at_year_40": (
            by_year[40]["initial_household_survival_share"] if 40 in by_year else None
        ),
        "initial_household_survival_share_at_year_60": (
            by_year[60]["initial_household_survival_share"] if 60 in by_year else None
        ),
        "economic_transient_metrics": transient,
        "residual_52_week_periodicity": periodicity,
    }


def tracker_state(world):
    initial_unassigned = {
        person.id for person in world.population
        if not valid_household_assignment(world, person)
    }
    return {
        "initial_unassigned_ids": initial_unassigned,
        "first_absorbed_week": {},
        "absorption_mechanism": {},
        "died_before_absorption": {},
    }


def update_tracker(world, tracker, completed_week):
    for person_id in tracker["initial_unassigned_ids"]:
        if person_id in tracker["first_absorbed_week"] or person_id in tracker["died_before_absorption"]:
            continue
        person = world.person_dict.get(person_id)
        if person is None:
            tracker["died_before_absorption"][person_id] = completed_week
        elif valid_household_assignment(world, person):
            tracker["first_absorbed_week"][person_id] = completed_week
            tracker["absorption_mechanism"][person_id] = (
                "marriage"
                if person.partner_id is not None
                else "other_household_assignment"
            )


def survival_row(world, tracker, completed_week):
    remaining_ids = []
    for person_id in tracker["initial_unassigned_ids"]:
        if person_id in tracker["first_absorbed_week"] or person_id in tracker["died_before_absorption"]:
            continue
        if person_id in world.person_dict:
            remaining_ids.append(person_id)
    initial = len(tracker["initial_unassigned_ids"])
    return {
        "week": completed_week,
        "model_year": completed_week / 52.0,
        "remaining_initial_unassigned": len(remaining_ids),
        "remaining_share": safe_ratio(len(remaining_ids), initial),
        "cumulative_absorbed": len(tracker["first_absorbed_week"]),
        "cumulative_absorbed_share": safe_ratio(len(tracker["first_absorbed_week"]), initial),
        "cumulative_died_before_absorption": len(tracker["died_before_absorption"]),
        "cumulative_died_before_absorption_share": safe_ratio(len(tracker["died_before_absorption"]), initial),
    }


def build_world(seed, population, firm_count):
    scenario = apply_scenario("baseline")
    world = World(
        initial_population=population,
        seed=seed,
        scenario_name=scenario["name"],
        scenario_overrides=scenario["overrides"],
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
    )
    apply_runtime_scenario(scenario, world)
    world.split_firms(firm_count)
    return world


def run_seed(seed, args, detailed=False):
    world = build_world(seed, args.population, args.firm_count)
    initial_person_ids = {person.id for person in world.population}
    initial_household_ids = {household.id for household in world.households}
    tracker = tracker_state(world)
    yearly = [
        initial_snapshot(world, initial_household_ids, initial_person_ids)
    ]
    survival = [survival_row(world, tracker, 0)]
    macro_rows = []

    for _ in range(args.weeks):
        world.step()
        completed = len(world.population_history)
        update_tracker(world, tracker, completed)
        survival.append(survival_row(world, tracker, completed))
        macro_rows.append(world.diagnostics_rows[-1])
        world.household_diagnostics_rows.clear()
        if completed % WEEKS_PER_YEAR == 0:
            yearly.append(
                snapshot(
                    world,
                    completed,
                    initial_household_ids,
                    initial_person_ids,
                    macro_rows[-1],
                )
            )
        if args.progress_interval and completed % args.progress_interval == 0:
            print(
                f"seed={seed} week={completed}/{args.weeks} "
                f"population={len(world.population)} "
                f"unassigned_share={yearly[-1]['unassigned_share']:.3%}"
            )

    attach_distribution_distances(yearly)
    transient = rolling_transient_rows(macro_rows)
    periodicity = periodicity_windows(macro_rows)
    metrics = maturity_metrics(yearly, survival, transient, periodicity, tracker)
    metrics["seed"] = seed
    metrics["population"] = args.population
    metrics["firm_count"] = args.firm_count
    metrics["weeks"] = args.weeks
    metrics["initial_age_phase_mode"] = "distributed"
    metrics["invariant_violations"] = len(world.invariant_violations)
    metrics["max_abs_monetary_accounting_gap"] = max(
        abs(float(row["monetary_accounting_gap"])) for row in macro_rows
    )
    metrics["max_abs_goods_conservation_gap"] = max(
        abs(float(row["food_conservation_gap"])) for row in macro_rows
    )

    absorption_events = []
    for person_id in sorted(tracker["initial_unassigned_ids"]):
        if person_id in tracker["first_absorbed_week"]:
            absorption_events.append({
                "person_id": person_id,
                "outcome": "absorbed",
                "week": tracker["first_absorbed_week"][person_id],
                "model_year": tracker["first_absorbed_week"][person_id] / 52.0,
                "mechanism": tracker["absorption_mechanism"][person_id],
            })
        elif person_id in tracker["died_before_absorption"]:
            absorption_events.append({
                "person_id": person_id,
                "outcome": "died_before_absorption",
                "week": tracker["died_before_absorption"][person_id],
                "model_year": tracker["died_before_absorption"][person_id] / 52.0,
                "mechanism": "death",
            })
        else:
            absorption_events.append({
                "person_id": person_id,
                "outcome": "alive_unassigned_at_horizon",
                "week": args.weeks,
                "model_year": args.weeks / 52.0,
                "mechanism": "unresolved",
            })

    if world.invariant_violations:
        write_csv(
            args.output_dir / f"invariant_violations_seed{seed}.csv",
            world.invariant_violations,
        )

    if detailed:
        write_csv(args.output_dir / f"warmup_yearly_summary_seed{seed}.csv", yearly)
        write_csv(args.output_dir / f"unassigned_survival_curve_seed{seed}.csv", survival)
        write_csv(
            args.output_dir / f"initial_unassigned_absorption_events_seed{seed}.csv",
            absorption_events,
        )
        write_csv(
            args.output_dir / f"household_structure_by_year_seed{seed}.csv",
            [{key: row.get(key) for key in (
                "model_year", "completed_week", "active_households", "total_household_objects",
                "initial_households_surviving", "initial_households_dissolved",
                "initial_household_survival_share", "new_active_households",
                "mean_household_size", "median_household_size", *HOUSEHOLD_SIZE_KEYS,
                "married_household_share", "single_parent_household_share",
                "households_with_children_share", "mean_child_count_per_household",
                "household_size_tv_vs_prior_year", "household_size_tv_vs_5_years_prior",
            )} for row in yearly],
        )
        write_csv(args.output_dir / f"economic_transient_metrics_seed{seed}.csv", transient)
        write_csv(args.output_dir / f"residual_52_week_periodicity_seed{seed}.csv", periodicity)
        write_json(args.output_dir / f"warmup_maturity_metrics_seed{seed}.json", metrics)

    return yearly, survival, metrics


def candidate_rows(yearly_by_seed):
    output = []
    for seed, yearly in yearly_by_seed.items():
        by_year = {int(row["model_year"]): row for row in yearly}
        for year in CANDIDATE_YEARS:
            if year not in by_year:
                continue
            row = by_year[year]
            output.append({
                "seed": seed,
                "candidate_years": year,
                "candidate_weeks": year * 52,
                **{key: row.get(key) for key in (
                    "population", "unassigned_share", "assigned_population",
                    "active_households", "total_household_objects",
                    "initial_household_survival_share", "new_active_households",
                    *HOUSEHOLD_SIZE_KEYS,
                    "household_size_tv_vs_prior_year", "household_size_tv_vs_5_years_prior",
                    "child_share", "worker_share", "elderly_share",
                    "demographic_max_share_change_vs_prior_year",
                    "total_household_wealth", "median_household_wealth",
                    "p10_household_wealth", "p90_household_wealth",
                    "total_consumption", "total_income", "realized_transaction_price",
                    "household_planning_price", "food_output", "food_inventory",
                    "inventory_coverage", "firm_cash", "firm_profit", "loan_balance",
                    "market_hhi", "price_cv", "total_money_stock",
                )},
            })
    return output


def multi_seed_rows(yearly_by_seed):
    output = []
    for seed, yearly in yearly_by_seed.items():
        for row in yearly:
            output.append({
                "seed": seed,
                **{key: row.get(key) for key in (
                    "model_year", "completed_week", "population", "unassigned_share",
                    "initial_cohort_survival_share", "share_initial_cohort_alive",
                    *HOUSEHOLD_SIZE_KEYS, "household_size_tv_vs_prior_year",
                    "household_size_tv_vs_5_years_prior", "total_household_wealth",
                    "total_consumption", "realized_transaction_price", "inventory_coverage",
                    "firm_cash", "loan_balance", "initial_household_survival_share",
                )},
            })
    return output


def source_audit(args):
    return {
        "stage": "Pre-Step13.2 Warm-up / Mature Society Redesign",
        "behavior_changes": False,
        "fresh_configuration": {
            "population": args.population,
            "firm_count": args.firm_count,
            "age_phase_mode": "distributed",
            "audit_horizon_weeks": args.weeks,
            "audit_horizon_years": args.weeks / 52,
        },
        "fresh_household_initialization": {
            "source": "world.py:784-880",
            "adult_threshold": "canonical age >= 20",
            "pairing": "shuffle adult males and females with global initialization RNG; pair min(counts)",
            "created_households": "one two-parent household per pair",
            "initial_children_assigned": False,
            "unmatched_adults_assigned": False,
            "initial_household_wealth": 0.0,
        },
        "unassigned_absorption": {
            "source": "household_manager.py:14-25 and marriage.py:39-229",
            "seeking_status": "all unpartnered people age >=20 are marked seeking weekly",
            "assignment_path": "annual-equivalent marriage creates a new household",
            "marriage_interval_weeks": 52,
            "other_direct_assignment_path_found": False,
            "newborns": "assigned to parents' household at birth",
        },
        "household_scaffold_tracking": {
            "method": "analysis-side frozen initial household ID set",
            "initial_household_survival": "initial ID remains an active nonempty household",
            "new_household": "active household ID not in the frozen initial set",
            "agent_behavior_fields_added": False,
        },
        "firm_bootstrap": {
            "source": "world.py:1114-1199",
            "fresh_multi_firm_bootstrap": "first aggregate market transfers physical supply once",
            "behavior_modified": False,
        },
        "maturity_semantics": {
            "stationarity_required": False,
            "state_based": True,
            "calendar_aware": True,
            "analysis_only": True,
        },
        "prohibited_changes_respected": [
            "population size", "marriage cadence", "fertility", "mortality",
            "age-phase initialization", "household formation", "firm behavior",
            "production", "pricing", "credit", "dividends", "Step 13",
        ],
    }


def plot_outputs(output_dir, yearly_by_seed, detailed_seed, metrics_by_seed):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    detailed = yearly_by_seed[detailed_seed]

    fig, axis = plt.subplots(figsize=(9, 4.5))
    for seed, rows in yearly_by_seed.items():
        axis.plot([row["model_year"] for row in rows], [row["unassigned_share"] for row in rows], label=str(seed))
    axis.set(xlabel="Model year", ylabel="Unassigned population share", title="Unassigned population across seeds")
    axis.legend(title="Seed", ncol=3)
    fig.tight_layout()
    fig.savefig(plot_dir / "unassigned_share_multi_seed.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    for key in HOUSEHOLD_SIZE_KEYS:
        axis.plot([row["model_year"] for row in detailed], [row[key] for row in detailed], label=key.replace("_share", ""))
    axis.set(xlabel="Model year", ylabel="Household share", title=f"Household size structure, seed {detailed_seed}")
    axis.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(plot_dir / "household_size_structure_seed42.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot([row["model_year"] for row in detailed], [row["initial_household_survival_share"] for row in detailed], label="initial household survival")
    axis.plot([row["model_year"] for row in detailed], [row["unassigned_share"] for row in detailed], label="unassigned population")
    axis.set(xlabel="Model year", ylabel="Share", title=f"Initialization memory, seed {detailed_seed}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "initialization_memory_seed42.png", dpi=160)
    plt.close(fig)

    periodicity = metrics_by_seed[detailed_seed]["residual_52_week_periodicity"]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    for field in ("labor", "wage_bill"):
        rows = [row for row in periodicity if row["metric"] == field]
        axis.plot([row["start_year"] for row in rows], [row["weekly_delta_lag_52_autocorrelation"] for row in rows], marker="o", label=field)
    axis.set(xlabel="Window start year", ylabel="Lag-52 autocorrelation of weekly delta", title="Residual annual periodicity by decade")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "residual_52_week_periodicity_seed42.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.weeks % 52:
        raise ValueError("Audit horizon must be a whole number of 52-week model years")

    seeds = [args.detailed_seed] if args.seed42_only else list(dict.fromkeys(args.seeds))
    if args.detailed_seed not in seeds:
        seeds.append(args.detailed_seed)

    write_json(args.output_dir / "source_audit.json", source_audit(args))
    yearly_by_seed = {}
    metrics_by_seed = {}
    for seed in seeds:
        print(f"Starting warm-up audit seed={seed}")
        yearly, _, metrics = run_seed(seed, args, detailed=seed == args.detailed_seed)
        yearly_by_seed[seed] = yearly
        metrics_by_seed[seed] = metrics
        gc.collect()

    write_csv(args.output_dir / "multi_seed_maturity_summary.csv", multi_seed_rows(yearly_by_seed))
    write_csv(args.output_dir / "candidate_warmup_comparison.csv", candidate_rows(yearly_by_seed))
    write_json(args.output_dir / "multi_seed_maturity_metrics.json", metrics_by_seed)
    plot_outputs(args.output_dir, yearly_by_seed, args.detailed_seed, metrics_by_seed)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "seeds": seeds,
        "weeks": args.weeks,
        "behavior_changes": False,
    }, indent=2))


if __name__ == "__main__":
    main()
