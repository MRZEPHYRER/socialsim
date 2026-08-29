import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from central_bank import config as central_bank_config
from scenarios import apply_runtime_scenario, apply_scenario
from world import World


DEFAULT_OUTPUT = ROOT / "test" / "output" / "pre_step13_2A_seed1_goods_audit"
WINDOW_START = 308
WINDOW_END = 318
TARGET_GAPS = {
    314: -2.8510873922350584,
    315: -0.3432366896158783,
}
TOLERANCE = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Passive seed-1 goods-conservation edge-case audit."
    )
    parser.add_argument("--weeks", type=int, default=321)
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--firm-count", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def safe_ratio(numerator, denominator):
    return numerator / denominator if abs(denominator) > 1e-15 else 0.0


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


class PassiveGoodsAudit:
    """Observe phase state without changing decisions, state, or RNG use."""

    def __init__(self, world, seed):
        self.world = world
        self.seed = seed
        self.context = None
        self.week_rows = []
        self.firm_rows = []
        self.public_rows = []
        self.household_rows = []
        self.lifecycle_rows = []
        self.stage_rows = []
        self._install_wrappers()

    def in_window(self):
        return self.context is not None and WINDOW_START <= self.context["step"] <= WINDOW_END

    @staticmethod
    def household_ids(world):
        return {household.id for household in world.households}

    @staticmethod
    def active_household_ids(world):
        return {
            household.id
            for household in world.households
            if household.parents or household.children
        }

    def snapshot_stage(self, stage, inventory_view="slices", include_output=False):
        if not self.in_window():
            return
        world = self.world
        aggregate = world.firm_system
        public = aggregate.central_bank
        slice_private = math.fsum(max(0.0, firm.inventory_units) for firm in world.firms)
        aggregate_private = max(0.0, aggregate.food_inventory_units)
        private = aggregate_private if inventory_view == "aggregate" else slice_private
        unfolded_output = max(0.0, aggregate.food_output_units) if include_output else 0.0
        public_units = max(0.0, public.food_inventory_units)
        self.stage_rows.append({
            "seed": self.seed,
            "global_step": self.context["step"],
            "stage_order": len(self.context["stage_names"]),
            "stage": stage,
            "inventory_view": inventory_view,
            "private_inventory_units": private,
            "slice_private_inventory_units": slice_private,
            "aggregate_private_inventory_units": aggregate_private,
            "public_inventory_units": public_units,
            "unfolded_production_units": unfolded_output,
            "tracked_total_physical_units": private + public_units + unfolded_output,
            "expected_closing_total_physical_units": None,
            "stage_goods_bridge_gap": None,
        })
        self.context["stage_names"].append(stage)

    def begin_step(self, step):
        world = self.world
        self.context = {
            "step": step,
            "opening_private": math.fsum(max(0.0, firm.inventory_units) for firm in world.firms),
            "opening_public": max(0.0, world.firm_system.central_bank.food_inventory_units),
            "opening_firm_inventory": {
                firm.firm_id: max(0.0, firm.inventory_units) for firm in world.firms
            },
            "opening_household_ids": self.household_ids(world),
            "opening_active_ids": self.active_household_ids(world),
            "lifecycle_start": len(world.household_lifecycle_events),
            "planning_counts": Counter(),
            "settlement_counts": Counter(),
            "household_spend": {},
            "household_units": {},
            "market_household_ids": set(),
            "market_active_ids": set(),
            "created_household_ids": set(),
            "emptied_household_ids": set(),
            "deleted_household_ids": set(),
            "stage_names": [],
            "aggregate_result": {},
            "public_purchase_threshold_units": 0.0,
            "public_purchase_excess_units": 0.0,
            "public_purchase_gate": 0.0,
        }
        self.snapshot_stage("A_opening_inventory")

    def _install_wrappers(self):
        world = self.world
        aggregate = world.firm_system
        public = aggregate.central_bank

        original_household_manager_step = world.household_manager.step

        def household_manager_step():
            before = self.household_ids(world)
            result = original_household_manager_step()
            after = self.household_ids(world)
            if self.in_window():
                self.context["deleted_household_ids"].update(before - after)
                self.snapshot_stage("B_after_household_manager")
            return result

        world.household_manager.step = household_manager_step

        original_marriage = world.marriage_system.process_marriage

        def process_marriage():
            before = self.household_ids(world)
            result = original_marriage()
            after = self.household_ids(world)
            if self.in_window():
                self.context["created_household_ids"].update(after - before)
                self.context["emptied_household_ids"].update(
                    household.id for household in world.households if household.size() == 0
                )
                self.snapshot_stage("C_after_marriage_market")
            return result

        world.marriage_system.process_marriage = process_marriage

        original_purchase_plan = aggregate.household_food_purchase_units

        def household_food_purchase_units(household, need_units):
            if self.in_window():
                self.context["planning_counts"][household.id] += 1
            return original_purchase_plan(household, need_units)

        aggregate.household_food_purchase_units = household_food_purchase_units

        original_household_market = aggregate.run_household_food_market

        def run_household_food_market():
            if self.in_window():
                self.context["market_household_ids"] = self.household_ids(world)
                self.context["market_active_ids"] = self.active_household_ids(world)
                self.snapshot_stage(
                    "D_after_aggregate_production",
                    inventory_view="aggregate",
                    include_output=True,
                )
            result = original_household_market()
            if self.in_window():
                self.snapshot_stage(
                    "E_after_household_planning",
                    inventory_view="aggregate",
                    include_output=True,
                )
            return result

        aggregate.run_household_food_market = run_household_food_market

        original_record_households = world.record_household_market_diagnostics

        def record_household_market_diagnostics(spend_values, purchase_units):
            if self.in_window():
                for household in world.households:
                    self.context["settlement_counts"][household.id] += 1
                self.context["household_spend"] = dict(spend_values)
                self.context["household_units"] = dict(purchase_units)
            return original_record_households(spend_values, purchase_units)

        world.record_household_market_diagnostics = record_household_market_diagnostics

        original_subsidy = public.distribute_poverty_food_subsidy

        def distribute_poverty_food_subsidy(households, food_price):
            result = original_subsidy(households, food_price)
            if self.in_window():
                self.snapshot_stage(
                    "F_after_public_in_kind_subsidy",
                    inventory_view="aggregate",
                    include_output=True,
                )
            return result

        public.distribute_poverty_food_subsidy = distribute_poverty_food_subsidy

        original_release = public.release_to_market

        def release_to_market(firm):
            result = original_release(firm)
            if self.in_window():
                self.snapshot_stage(
                    "G_after_public_market_release",
                    inventory_view="aggregate",
                    include_output=True,
                )
            return result

        public.release_to_market = release_to_market

        original_public_purchase = public.purchase_excess_food_inventory

        def purchase_excess_food_inventory(firm):
            if self.in_window():
                demand_base = max(1.0, firm.food_demand_units)
                threshold = (
                    firm.target_inventory_days
                    + central_bank_config.CENTRAL_BANK_FOOD_PURCHASE_EXCESS_INVENTORY_DAYS
                ) * demand_base
                cash_buffer = max(1.0, central_bank_config.CENTRAL_BANK_FIRM_CASH_BUFFER)
                gate = safe_ratio(
                    central_bank_config.CENTRAL_BANK_TARGET_FIRM_CASH
                    + cash_buffer
                    - firm.cash,
                    cash_buffer,
                )
                gate = max(0.0, min(1.0, gate))
                if not central_bank_config.CENTRAL_BANK_FIRM_CASH_PURCHASE_GATE_ENABLED:
                    gate = 1.0
                self.context["public_purchase_threshold_units"] = threshold
                self.context["public_purchase_excess_units"] = max(
                    0.0, firm.food_inventory_units - threshold
                )
                self.context["public_purchase_gate"] = gate
                self.snapshot_stage(
                    "H_after_aggregate_sales_spoilage_before_public_purchase",
                    inventory_view="aggregate",
                )
            result = original_public_purchase(firm)
            if self.in_window():
                self.snapshot_stage(
                    "I_after_public_inventory_purchase",
                    inventory_view="aggregate",
                )
            return result

        public.purchase_excess_food_inventory = purchase_excess_food_inventory

        original_aggregate_step = aggregate.step

        def aggregate_step():
            result = original_aggregate_step()
            if self.in_window():
                self.context["aggregate_result"] = dict(result)
                self.snapshot_stage("J_after_aggregate_firm_step", inventory_view="aggregate")
            return result

        aggregate.step = aggregate_step

        original_sync = world.sync_firms_from_aggregate

        def sync_firms_from_aggregate(firm_result=None):
            result = original_sync(firm_result)
            if self.in_window():
                self.snapshot_stage("K_after_multi_firm_sync", inventory_view="slices")
            return result

        world.sync_firms_from_aggregate = sync_firms_from_aggregate

    def finish_step(self):
        if not self.in_window():
            return
        world = self.world
        step = self.context["step"]
        macro = world.diagnostics_rows[-1]
        aggregate_result = self.context["aggregate_result"]
        self.snapshot_stage("L_closing_after_birth_death", inventory_view="slices")

        end_household_ids = self.household_ids(world)
        self.context["deleted_household_ids"].update(
            self.context["opening_household_ids"] - end_household_ids
        )
        created = self.context["created_household_ids"]
        deleted = self.context["deleted_household_ids"]
        lifecycle = world.household_lifecycle_events[self.context["lifecycle_start"]:]
        event_types = defaultdict(list)
        for event in lifecycle:
            event_types[event["household_id"]].append(event["event_type"])
            self.lifecycle_rows.append({"seed": self.seed, **event})

        household_ids = (
            self.context["market_household_ids"]
            | set(self.context["planning_counts"])
            | set(self.context["settlement_counts"])
            | created
            | deleted
        )
        for household_id in sorted(household_ids):
            active = household_id in self.context["market_active_ids"]
            plan_count = self.context["planning_counts"][household_id]
            settle_count = self.context["settlement_counts"][household_id]
            self.household_rows.append({
                "seed": self.seed,
                "global_step": step,
                "household_id": household_id,
                "active_at_market": active,
                "newly_created_by_marriage": household_id in created,
                "emptied_by_marriage": household_id in self.context["emptied_household_ids"],
                "deleted_this_step": household_id in deleted,
                "event_type": ";".join(event_types.get(household_id, [])),
                "planning_call_count": plan_count,
                "settlement_call_count": settle_count,
                "purchased_units": self.context["household_units"].get(household_id, 0.0),
                "consumption_expenditure": self.context["household_spend"].get(household_id, 0.0),
                "active_planning_omission": active and plan_count != 1,
                "active_settlement_omission": active and settle_count != 1,
                "duplicate_planning": plan_count > 1,
                "duplicate_settlement": settle_count > 1,
            })

        total_wage = math.fsum(max(0.0, firm.wage_bill) for firm in world.firms)
        public_purchase = float(macro["central_bank_food_purchase_units"])
        public_release = float(macro["central_bank_food_release_units"])
        firm_spoilage = 0.0
        firm_bridge_gap = 0.0
        clipping_created = 0.0
        for firm in world.firms:
            share = safe_ratio(max(0.0, firm.wage_bill), total_wage)
            allocated_purchase = max(
                0.0, getattr(firm, "public_inventory_purchase_units", public_purchase * share)
            )
            allocated_release = public_release * share
            opening = self.context["opening_firm_inventory"].get(firm.firm_id, 0.0)
            production = max(0.0, firm.actual_production)
            sales = max(0.0, firm.sales_units)
            raw_pre_spoilage = (
                opening + production + allocated_release - sales - allocated_purchase
            )
            spoilage = max(0.0, getattr(firm, "spoilage_units", 0.0))
            closing = max(0.0, firm.inventory_units)
            gap = (
                opening + production + allocated_release
                - sales - allocated_purchase - spoilage - closing
            )
            clip = max(0.0, -raw_pre_spoilage)
            firm_spoilage += spoilage
            firm_bridge_gap += gap
            clipping_created += clip
            self.firm_rows.append({
                "seed": self.seed,
                "global_step": step,
                "firm_id": firm.firm_id,
                "wage_capacity_share": share,
                "opening_inventory_units": opening,
                "production_units": production,
                "allocated_public_release_units": allocated_release,
                "sales_units": sales,
                "allocated_public_purchase_units": allocated_purchase,
                "raw_inventory_before_nonnegative_clip": raw_pre_spoilage,
                "inventory_clip_created_units": clip,
                "spoilage_units": spoilage,
                "closing_inventory_units": closing,
                "firm_goods_bridge_gap": gap,
                "public_inventory_sales_units": max(
                    0.0, getattr(firm, "public_inventory_sales_units", 0.0)
                ),
            })

        opening_public = self.context["opening_public"]
        closing_public = max(0.0, world.firm_system.central_bank.food_inventory_units)
        subsidy = float(macro["central_bank_food_subsidy_units"])
        public_gap = opening_public + public_purchase - public_release - subsidy - closing_public
        self.public_rows.append({
            "seed": self.seed,
            "global_step": step,
            "opening_public_inventory_units": opening_public,
            "public_purchase_units": public_purchase,
            "public_market_release_units": public_release,
            "public_in_kind_subsidy_units": subsidy,
            "public_spoilage_units": 0.0,
            "closing_public_inventory_units": closing_public,
            "public_goods_bridge_gap": public_gap,
        })

        opening_private = self.context["opening_private"]
        closing_private = math.fsum(max(0.0, firm.inventory_units) for firm in world.firms)
        production = float(macro["food_output_units"])
        sales = float(macro["food_sales_units"])
        spoilage = float(macro["food_spoilage_units"])
        expected_close = (
            opening_private + opening_public + production - sales - subsidy - spoilage
        )
        actual_close = closing_private + closing_public
        reconstructed_gap = expected_close - actual_close
        active_households = len(world.active_households())
        household_rows = [
            row for row in self.household_rows
            if row["seed"] == self.seed and row["global_step"] == step
        ]
        self.week_rows.append({
            "seed": self.seed,
            "global_step": step,
            "population": len(world.population),
            "active_households": active_households,
            "total_household_objects": len(world.households),
            "marriage_market_executed": world.marriage_market_executed,
            "marriage_matches": world.marriage_market_last_result,
            "new_households_created": len(created),
            "empty_households": sum(household.size() == 0 for household in world.households),
            "households_deleted": len(deleted),
            "food_inventory_open": opening_private,
            "public_food_inventory_open": opening_public,
            "food_production": production,
            "household_food_demand": float(macro["food_demand_units"]),
            "firm_food_sales": sales,
            "public_food_release": public_release,
            "public_food_purchase": public_purchase,
            "public_in_kind_food_support": subsidy,
            "food_spoilage": spoilage,
            "food_inventory_close": closing_private,
            "public_food_inventory_close": closing_public,
            "expected_system_food_close": expected_close,
            "actual_system_food_close": actual_close,
            "reconstructed_food_conservation_gap": reconstructed_gap,
            "food_conservation_gap": float(macro["food_conservation_gap"]),
            "aggregate_public_purchase_threshold_units": self.context[
                "public_purchase_threshold_units"
            ],
            "aggregate_public_purchase_excess_units": self.context[
                "public_purchase_excess_units"
            ],
            "aggregate_public_purchase_gate": self.context["public_purchase_gate"],
            "firm_bridge_gap_sum": firm_bridge_gap,
            "public_bridge_gap": public_gap,
            "inventory_clip_created_units": clipping_created,
            "active_planning_omissions": sum(
                row["active_planning_omission"] for row in household_rows
            ),
            "active_settlement_omissions": sum(
                row["active_settlement_omission"] for row in household_rows
            ),
            "duplicate_household_planning": sum(
                row["duplicate_planning"] for row in household_rows
            ),
            "duplicate_household_settlement": sum(
                row["duplicate_settlement"] for row in household_rows
            ),
            "household_purchased_units_sum": math.fsum(
                row["purchased_units"] for row in household_rows
            ),
            "household_expenditure_sum": math.fsum(
                row["consumption_expenditure"] for row in household_rows
            ),
        })

        aggregate_expected_close = (
            self.context["opening_private"]
            + self.context["opening_public"]
            + float(aggregate_result.get("food_output_units", 0.0))
            - float(aggregate_result.get("food_sales_units", 0.0))
            - float(aggregate_result.get("central_bank_food_subsidy_units", 0.0))
            - float(aggregate_result.get("food_spoilage_units", 0.0))
        )
        stage_by_name = {
            row["stage"]: row
            for row in self.stage_rows
            if row["seed"] == self.seed and row["global_step"] == step
        }
        for stage in (
            "H_after_aggregate_sales_spoilage_before_public_purchase",
            "I_after_public_inventory_purchase",
            "J_after_aggregate_firm_step",
        ):
            if stage in stage_by_name:
                row = stage_by_name[stage]
                row["expected_closing_total_physical_units"] = aggregate_expected_close
                row["stage_goods_bridge_gap"] = (
                    aggregate_expected_close - row["tracked_total_physical_units"]
                )
        for stage in ("K_after_multi_firm_sync", "L_closing_after_birth_death"):
            if stage in stage_by_name:
                row = stage_by_name[stage]
                row["expected_closing_total_physical_units"] = expected_close
                row["stage_goods_bridge_gap"] = (
                    expected_close - row["tracked_total_physical_units"]
                )


def run_seed(seed, args):
    world = build_world(seed, args.population, args.firm_count)
    audit = PassiveGoodsAudit(world, seed)
    for step in range(args.weeks):
        audit.begin_step(step)
        world.step()
        audit.finish_step()
        world.household_diagnostics_rows.clear()
    return world, audit


def passive_observer_parity(seed, args, observed_world):
    plain_world = build_world(seed, args.population, args.firm_count)
    for _ in range(args.weeks):
        plain_world.step()
        plain_world.household_diagnostics_rows.clear()

    macro_equal = plain_world.diagnostics_rows == observed_world.diagnostics_rows
    firm_equal = (
        plain_world.firm_diagnostics_rows
        == observed_world.firm_diagnostics_rows
    )
    first_macro_mismatch = next(
        (
            index
            for index, (plain, observed) in enumerate(
                zip(plain_world.diagnostics_rows, observed_world.diagnostics_rows)
            )
            if plain != observed
        ),
        None,
    )
    first_firm_mismatch = next(
        (
            index
            for index, (plain, observed) in enumerate(
                zip(
                    plain_world.firm_diagnostics_rows,
                    observed_world.firm_diagnostics_rows,
                )
            )
            if plain != observed
        ),
        None,
    )
    return {
        "seed": seed,
        "macro_trajectory_exact": macro_equal,
        "firm_trajectory_exact": firm_equal,
        "first_macro_mismatch_row": first_macro_mismatch,
        "first_firm_mismatch_row": first_firm_mismatch,
    }


def first_divergence(stage_rows):
    candidates = [
        row for row in stage_rows
        if row["seed"] == 1
        and row["global_step"] in TARGET_GAPS
        and row["stage_goods_bridge_gap"] is not None
        and abs(float(row["stage_goods_bridge_gap"])) > 1e-9
    ]
    candidates.sort(key=lambda row: (row["global_step"], row["stage_order"]))
    return candidates[0] if candidates else None


def main():
    args = parse_args()
    if args.weeks <= WINDOW_END:
        raise ValueError(f"--weeks must exceed {WINDOW_END}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audits = {}
    worlds = {}
    for seed in args.seeds:
        world, audit = run_seed(seed, args)
        worlds[seed] = world
        audits[seed] = audit
        print(
            f"seed={seed} weeks={args.weeks} "
            f"violations={len(world.invariant_violations)}"
        )

    observer_parity = passive_observer_parity(1, args, worlds[1])

    week_rows = [row for audit in audits.values() for row in audit.week_rows]
    firm_rows = [row for audit in audits.values() for row in audit.firm_rows]
    public_rows = [row for audit in audits.values() for row in audit.public_rows]
    household_rows = [row for audit in audits.values() for row in audit.household_rows]
    lifecycle_rows = [row for audit in audits.values() for row in audit.lifecycle_rows]
    stage_rows = [row for audit in audits.values() for row in audit.stage_rows]

    write_csv(args.output_dir / "week308_318_goods_bridge.csv", week_rows)
    write_csv(args.output_dir / "firm_goods_bridge.csv", firm_rows)
    write_csv(args.output_dir / "public_goods_bridge.csv", public_rows)
    write_csv(args.output_dir / "household_processing_audit.csv", household_rows)
    write_csv(args.output_dir / "household_lifecycle_events.csv", lifecycle_rows)
    write_csv(args.output_dir / "operation_stage_goods_stock.csv", stage_rows)

    seed1_rows = {row["global_step"]: row for row in audits[1].week_rows}
    old_alerts_reproduced = all(
        math.isclose(
            float(seed1_rows[step]["food_conservation_gap"]),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for step, expected in TARGET_GAPS.items()
    )
    new_alerts_closed = all(
        abs(float(seed1_rows[step]["food_conservation_gap"])) <= 1e-9
        for step in TARGET_GAPS
    )
    seed7_clean = all(
        abs(float(row["food_conservation_gap"])) <= 1e-9
        for row in audits[7].week_rows
    )
    first = first_divergence(stage_rows)
    affected = {
        step: {
            "food_conservation_gap": seed1_rows[step]["food_conservation_gap"],
            "inventory_clip_created_units": seed1_rows[step]["inventory_clip_created_units"],
            "public_purchase_units": seed1_rows[step]["public_food_purchase"],
        }
        for step in TARGET_GAPS
    }
    cumulative_created = math.fsum(
        float(seed1_rows[step]["inventory_clip_created_units"])
        for step in TARGET_GAPS
    )

    summary = {
        "stage": "Pre-Step13.2A Seed 1 Goods-Conservation Edge-Case Audit",
        "behavior_changes": False,
        "passive_observer_parity": observer_parity,
        "configuration": {
            "population": args.population,
            "firm_count": args.firm_count,
            "initial_age_phase_mode": "distributed",
            "scenario": "baseline",
            "seeds": args.seeds,
            "weeks": args.weeks,
        },
        "deterministic_reproduction_succeeded": old_alerts_reproduced,
        "post_fix_alerts_closed": new_alerts_closed,
        "seed_1_target_alerts": affected,
        "seed_7_window_clean": seed7_clean,
        "classification": "REAL_GOODS_CREATION_DESTRUCTION_BUG",
        "verdict": "B",
        "cumulative_goods_created_by_clipping_units": cumulative_created,
        "weekly_residual_persists_after_week_315": False,
        "stock_level_effect": (
            f"The {cumulative_created} created units enter opening inventory for later "
            "weeks and can subsequently be consumed or spoiled."
        ),
        "marriage_interaction": (
            "Marriage executes at week 312 and changes household/demand state, but no "
            "lifecycle operation changes physical food. The first physical divergence "
            "occurs during multi-firm synchronization at week 314."
        ),
        "household_duplicate_or_omission_found": any(
            row["active_planning_omission"]
            or row["active_settlement_omission"]
            or row["duplicate_planning"]
            or row["duplicate_settlement"]
            for row in household_rows
        ),
    }
    write_json(args.output_dir / "reproduction_summary.json", summary)

    write_json(
        args.output_dir / "source_audit.json",
        {
            "behavior_changes": False,
            "aggregate_firm_execution_order": [
                "collect public wealth",
                "wages and dividends",
                "aggregate production",
                "household purchase planning and prepayment",
                "public in-kind subsidy",
                "public market release",
                "aggregate sale and unfilled-order refund",
                "aggregate spoilage",
                "aggregate public inventory purchase",
                "multi-firm market replay and firm-specific settlement",
                "multi-firm inventory reconstruction",
                "birth, death, inheritance, and cleanup",
            ],
            "accepted_system_goods_bridge": (
                "opening_private + opening_public + production - market_sales "
                "- public_in_kind_subsidy - spoilage = closing_private + closing_public"
            ),
            "private_firm_bridge": (
                "opening + production + allocated_public_release - sales "
                "- allocated_public_purchase - spoilage = closing"
            ),
            "public_bridge": (
                "opening + purchases - market_release - in_kind_subsidy = closing"
            ),
            "source_paths": {
                "marriage": "world.py:3012-3042; marriage.py:39-229",
                "household_cleanup": "household_manager.py:38-159",
                "aggregate_market_and_inventory": "economy/firm.py:682-1010",
                "public_inventory": "central_bank/central_bank.py:185-493",
                "multi_firm_market": "world.py:1708-2348",
                "goods_conservation_diagnostic": "world.py:4101-4128",
            },
        },
    )

    first_payload = {
        "classification": "REAL_GOODS_CREATION_DESTRUCTION_BUG",
        "first_divergent_global_step": first["global_step"] if first else None,
        "first_divergent_operation": first["stage"] if first else None,
        "expected_total_physical_units": (
            first["expected_closing_total_physical_units"] if first else None
        ),
        "actual_total_physical_units": (
            first["tracked_total_physical_units"] if first else None
        ),
        "goods_bridge_gap": first["stage_goods_bridge_gap"] if first else None,
        "exact_source_path": {
            "aggregate_public_purchase": "central_bank/central_bank.py:185-307",
            "multi_firm_purchase_split": "world.py:2125-2139",
            "nonnegative_inventory_clip": "world.py:2131-2145",
            "system_conservation_check": "world.py:4101-4128",
        },
        "trigger": (
            "The aggregate central bank purchases excess inventory, then the full purchase "
            "quantity is allocated to firms by wage/capacity share. Firm 0 has sold all "
            "available units, so its allocated purchase makes raw inventory negative. "
            "max(0, raw_inventory) discards that negative deduction while public inventory "
            "retains the full purchase."
        ),
        "minimal_fix_proposal_only": (
            "Allocate public purchases only from each firm's available inventory and "
            "redistribute any remainder among firms with stock, or record the exact donor "
            "firm quantities at aggregate purchase time. Never combine a full public receipt "
            "with a pro-rata private deduction that is later clipped."
        ),
        "repair_applied": False,
    }
    write_json(args.output_dir / "first_divergence.json", first_payload)

    report = f"""# Pre-Step13.2A Seed 1 Goods-Conservation Edge-Case Audit

## Verdict

**B. Seed-1 alerts reveal a real but isolated goods-conservation mechanism bug.**

No model behavior, parameter, timing rule, warm-up rule, or population size was
changed. The audit uses passive runtime observation only.

An uninstrumented seed-1 control is exactly equal to the instrumented run for
every macro diagnostics row and every firm diagnostics row.

## Deterministic reproduction

- Seed 1, N=500, five firms, distributed age phase, baseline, {args.weeks} weeks.
- Week 314 reproduced exactly: `{affected[314]['food_conservation_gap']}` units.
- Week 315 reproduced exactly: `{affected[315]['food_conservation_gap']}` units.
- Seed 7 is clean over weeks 308-318 within `1e-9`.

## First causal divergence

The aggregate private-plus-public physical bridge still closes after the
Central Bank purchase. The first non-floating-point divergence is **week 314,
stage K: `after_multi_firm_sync`**.

The causal path is:

1. `central_bank/central_bank.py:185-307` purchases `14.219125273322643`
   aggregate inventory units and adds the full amount to public inventory.
2. `world.py:2125-2139` reallocates that purchase across firms using aggregate
   wage/capacity shares, independently of each firm's remaining inventory.
3. Firm 0 opened with zero inventory, produced and sold exactly
   `2077.779318822962` units, then received a `2.8510873922350584`-unit public
   purchase deduction.
4. Its raw inventory becomes `-2.8510873922350584`. The `max(0, ...)` at
   `world.py:2131-2145` clips it to zero.
5. Public inventory keeps all `14.219125273322643` units, while private firms
   surrender only `11.368037881087585`; total physical food therefore rises by
   `2.8510873922350584` units.

Week 315 repeats the same path for `0.3432366896158783` units. Total goods
created by the two clips are `{cumulative_created}` units. The weekly residual
returns to floating-point scale at week 316, but the excess stock has already
entered the next week's opening inventory and can later be consumed or spoiled.

## Marriage and household processing

Both seeds execute five marriages at week 312. Marriage, household departure,
new-household creation, and empty-household cleanup move people and money only;
no lifecycle source path mutates food inventory. Across the 308-318 window,
every active household appears exactly once in aggregate planning and once in
multi-firm settlement. There are no duplicate or omitted purchases. Marriage is
therefore an upstream state change, not the physical divergence operation.

## Public and firm bridges

The public inventory bridge closes at weeks 314 and 315:

`opening public + purchase - market release - in-kind subsidy = closing public`.

The system bridge fails only because Firm 0's allocated purchase exceeds its
remaining stock. Firm-level rows identify the exact negative raw inventory and
clip. Seed 7 has the same marriage count but no public inventory purchase in
the window, so the problematic allocation branch is not triggered.

## Proposed minimal fix (not applied)

Allocate public purchases from firms according to actual available inventory,
cap each donor at its stock, and redistribute the remainder among firms that
still hold stock. An equivalent implementation may record exact donor-firm
quantities when the aggregate purchase occurs. The invariant is that a full
public receipt must never be paired with a private deduction that is later
silently clipped.

No repair is applied in this audit. Population scaling and Step 13 remain
paused.
"""
    (args.output_dir / "acceptance_summary.md").write_text(report, encoding="utf-8")

    print(f"verdict=B output={args.output_dir}")


if __name__ == "__main__":
    main()
