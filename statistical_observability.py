"""Read-only statistical persistence for the Step 15 analysis contract.

The observer runs after a completed ``World.step`` and copies only the
minimum authoritative state needed for distributional and event analysis.
It must never mutate model state or participate in a simulation decision.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from productivity import age_productivity


def _number(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _step(row):
    return int(_number(row.get("global_step", row.get("step", -1)), -1))


def _latest_firm_rows(world, step):
    return {
        str(row.get("firm_id")): row
        for row in getattr(world, "firm_diagnostics_rows", [])
        if _step(row) == step
    }


class StatisticalObservabilityPersistence:
    """Append-only statistical panels with explicit scope and cadence."""

    VERSION = 3

    AGE_FIELDS = (
        "global_step", "observation_phase", "age_year", "sex",
        "population_count",
    )
    HOUSEHOLD_FIELDS = (
        "global_step", "observation_phase", "household_id", "member_count",
        "adult_count", "child_count", "elderly_count", "employed_member_count",
        "labor_eligible_member_count", "settlement_origin_since_snapshot",
        "current_week_income",
        "current_week_consumption", "current_week_saving", "cash",
        "equity_assets_at_cost", "other_financial_assets", "liabilities",
        "total_financial_assets", "financial_net_worth",
        "interval_weeks", "opening_cash", "interval_labor_income",
        "interval_labor_eligible_person_weeks", "interval_employed_person_weeks",
        "interval_paid_person_weeks", "interval_scheduled_wage",
        "interval_paid_wage", "interval_payroll_shortfall",
        "interval_food_paid_wage", "interval_capital_goods_paid_wage",
        "interval_food_paid_person_weeks",
        "interval_capital_goods_paid_person_weeks",
        "interval_dividend_income", "interval_other_recorded_income",
        "interval_total_authoritative_income", "interval_consumption",
        "interval_saving", "interval_minimum_consumption",
        "interval_lifecycle_transfer_in", "interval_lifecycle_transfer_out",
        "interval_other_authoritative_inflow", "interval_other_authoritative_outflow",
        "interval_cash_bridge_gap", "created_since_previous_snapshot",
    )
    SETTLEMENT_FIELDS = (
        "global_step", "observation_phase", "settlement_account_id",
        "owner_person_id", "cash", "active",
        "transition_to_social_household", "destination_social_household_id",
    )
    LABOR_EVENT_FIELDS = (
        "global_step", "person_id", "event_type", "from_firm_id",
        "to_firm_id", "from_sector", "to_sector", "reason",
    )
    LABOR_DENOMINATOR_FIELDS = (
        "global_step", "population", "working_age_population_20_60",
        "settlement_valid_working_age_20_60", "labor_age_eligible",
        "labor_match_eligible", "employed", "employed_eligible",
        "unassigned_eligible", "eligible_identity_gap",
    )
    CAPITAL_FIELDS = (
        "global_step", "firm_id", "sector_id", "employment",
        "active_capital_service", "capital_asset_count",
        "active_capital_asset_count", "gross_capital_book_value",
        "closing_capital_book_value", "accumulated_depreciation",
        "depreciation_flow", "acquisition_count", "acquisition_expenditure",
        "retirement_count", "retired_service_capacity",
        "replacement_investment", "expansion_investment",
    )

    FILES = {
        "age": ("age_sex_histogram.csv", AGE_FIELDS),
        "households": ("social_household_snapshots.csv", HOUSEHOLD_FIELDS),
        "settlement": ("settlement_account_snapshots.csv", SETTLEMENT_FIELDS),
        "labor_events": ("labor_events.csv", LABOR_EVENT_FIELDS),
        "labor_denominators": ("labor_denominators.csv", LABOR_DENOMINATOR_FIELDS),
        "capital": ("firm_capital_history.csv", CAPITAL_FIELDS),
    }

    def __init__(
        self,
        output_dir,
        snapshot_cadence=13,
        age_cadence=13,
        observability_mode="FULL_DIAGNOSTIC",
    ):
        self.output_dir = Path(output_dir)
        self.snapshot_cadence = max(1, int(snapshot_cadence))
        self.age_cadence = max(1, int(age_cadence))
        self.observability_mode = str(
            observability_mode or "FULL_DIAGNOSTIC"
        ).upper()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._previous_people = {}
        self._previous_settlement = {}
        self._household_intervals = {}
        self._emitted = {name: set() for name in self.FILES}
        self._load_existing_keys()
        self._write_contract()

    def path(self, name):
        return self.output_dir / self.FILES[name][0]

    @staticmethod
    def _append(path, rows, fields):
        rows = list(rows)
        if not rows:
            return
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def _row_key(self, name, row):
        step = int(_number(row.get("global_step"), -1))
        if name == "age":
            return step, str(row.get("age_year")), str(row.get("sex"))
        if name == "households":
            return step, str(row.get("household_id")), str(row.get("observation_phase"))
        if name == "settlement":
            return step, str(row.get("settlement_account_id")), str(row.get("observation_phase"))
        if name == "labor_events":
            return (
                step, str(row.get("person_id")), str(row.get("event_type")),
                str(row.get("from_firm_id")), str(row.get("to_firm_id")),
            )
        if name == "labor_denominators":
            return (step,)
        return step, str(row.get("firm_id"))

    def _load_existing_keys(self):
        for name, (filename, _) in self.FILES.items():
            path = self.output_dir / filename
            if not path.exists() or path.stat().st_size == 0:
                continue
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self._emitted[name].add(self._row_key(name, row))

    def _emit(self, name, rows):
        fields = self.FILES[name][1]
        fresh = []
        for row in rows:
            key = self._row_key(name, row)
            if key in self._emitted[name]:
                continue
            self._emitted[name].add(key)
            fresh.append(row)
        self._append(self.path(name), fresh, fields)

    def _write_contract(self):
        rows = []
        contracts = {
            "age_sex_histogram.csv": {
                "entity": "exact attained age x authoritative Person.sex",
                "grain": "snapshot age-sex cell",
                "kind": "stock",
                "scope": "all alive Persons",
                "cadence": f"initial, every {self.age_cadence} weeks, final",
            },
            "social_household_snapshots.csv": {
                "entity": "social Household",
                "grain": "Household snapshot",
                "kind": "stocks, current-week flows, and authoritative interval flows",
                "scope": "settlement-only accounts excluded",
                "cadence": f"initial, every {self.snapshot_cadence} weeks, final",
            },
            "settlement_account_snapshots.csv": {
                "entity": "settlement-only economic account",
                "grain": "account snapshot or authoritative transition",
                "kind": "stock/event",
                "scope": "settlement-only accounts only",
                "cadence": f"initial, every {self.snapshot_cadence} weeks, final; transitions weekly",
            },
            "labor_events.csv": {
                "entity": "Person labor-state transition",
                "grain": "event",
                "kind": "event",
                "scope": "runtime employment and eligibility transitions",
                "cadence": (
                    "every completed week"
                    if self.observability_mode != "RESEARCH_FAST"
                    else "sparse snapshot cadence; transitions between snapshots are not persisted"
                ),
            },
            "labor_denominators.csv": {
                "entity": "World labor population",
                "grain": "world-week",
                "kind": "stock",
                "scope": "explicit age and settlement predicates",
                "cadence": "every completed week plus initial",
            },
            "firm_capital_history.csv": {
                "entity": "operating Firm",
                "grain": "Firm-week",
                "kind": "mixed stocks and weekly flows",
                "scope": "Food and capital-goods Firms",
                "cadence": "every completed week plus initial",
            },
        }
        for filename, spec in contracts.items():
            fields = next(fields for file_name, fields in self.FILES.values() if file_name == filename)
            for field in fields:
                rows.append({
                    "observability_mode": self.observability_mode,
                    "dataset": filename,
                    "field": field,
                    "semantic_definition": self._field_semantic(filename, field),
                    "entity": spec["entity"],
                    "time_grain": spec["grain"],
                    "stock_flow_event": spec["kind"],
                    "unit": self._field_unit(field),
                    "source_runtime_field": self._field_source(field),
                    "persistence_cadence": spec["cadence"],
                    "allowed_derived_statistics": self._allowed_statistics(filename),
                    "account_scope": spec["scope"],
                })
        fields = list(rows[0])
        path = self.output_dir / "statistical_persistence_contract.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _field_semantic(filename, field):
        explicit = {
            "age_year": "floor(Person.age_years), an exact attained-age bin",
            "sex": "authoritative Person.sex value; no category is inferred",
            "current_week_income": "Household.income_this_step after weekly settlement",
            "current_week_consumption": "Household.consumption_this_step for the completed week",
            "current_week_saving": "Household.saving_this_step for the completed week",
            "opening_cash": "same Household cash at the preceding persisted social-Household snapshot, or zero for a newly formed Household",
            "interval_labor_income": "sum of Household.wage_income_this_step over completed weeks since the preceding snapshot",
            "interval_labor_eligible_person_weeks": "sum of end-of-week labor-eligible Social-Household members over the interval",
            "interval_employed_person_weeks": "sum of end-of-week labor-eligible members with a Firm assignment over the interval",
            "interval_paid_person_weeks": "count of Person-week payroll records with positive authoritative paid wage over the interval",
            "interval_scheduled_wage": "sum of authoritative scheduled payroll over the interval",
            "interval_paid_wage": "sum of authoritative person-level paid payroll over the interval",
            "interval_payroll_shortfall": "scheduled payroll minus paid payroll over the interval",
            "interval_food_paid_wage": "authoritative paid payroll assigned to Food-sector Firms over the interval",
            "interval_capital_goods_paid_wage": "authoritative paid payroll assigned to capital-goods Firms over the interval",
            "interval_dividend_income": "sum of Household.dividend_income_this_step over completed weeks since the preceding snapshot",
            "interval_other_recorded_income": "residual inside Household.income_this_step after authoritative wage and dividend components",
            "interval_total_authoritative_income": "sum of Household.income_this_step over completed weeks since the preceding snapshot",
            "interval_consumption": "sum of Household.consumption_this_step over completed weeks since the preceding snapshot",
            "interval_saving": "sum of Household.saving_this_step over completed weeks since the preceding snapshot",
            "interval_minimum_consumption": "sum of Household.necessary_consumption_this_step over completed weeks since the preceding snapshot",
            "interval_lifecycle_transfer_in": "sum of authoritative lifecycle-transfer event amounts received by the Household",
            "interval_lifecycle_transfer_out": "sum of authoritative lifecycle-transfer event amounts paid by the Household",
            "interval_cash_bridge_gap": "closing cash minus opening cash and all authoritative interval flow components",
            "cash": "Household.wealth legacy field, explicitly interpreted as cash",
            "equity_assets_at_cost": "Household.equity_asset_value at acquisition-cost accounting basis",
            "other_financial_assets": "authoritatively inactive Household asset category, currently zero",
            "liabilities": "authoritatively inactive Household liability category, currently zero",
            "working_age_population_20_60": "alive Persons with 20 <= age_years <= 60",
            "settlement_valid_working_age_20_60": "working-age Persons with a valid settlement Household",
            "labor_age_eligible": "alive Persons satisfying age_productivity(age) > 0",
            "labor_match_eligible": "labor-age eligible Persons with a valid settlement Household",
            "employed_eligible": "labor-match eligible Persons with firm_id assigned",
            "unassigned_eligible": "labor-match eligible Persons with firm_id=None",
            "gross_capital_book_value": "sum of CapitalAsset acquisition_cost before accumulated depreciation",
            "depreciation_flow": "Firm capital_depreciation_expense_this_step, a non-cash weekly flow",
        }
        return explicit.get(field, f"authoritative persisted {field.replace('_', ' ')}")

    @staticmethod
    def _field_unit(field):
        if field in {"global_step"}:
            return "simulation week"
        if "fraction" in field or field.endswith("_gap"):
            return "ratio/count identity"
        if "person_weeks" in field:
            return "person-weeks"
        if any(token in field for token in ("income", "consumption", "saving", "cash", "value", "expenditure", "assets", "liabilities", "depreciation")):
            return "currency or currency/week as named"
        if "service" in field:
            return "capital-service units"
        return "count/id/category as named"

    @staticmethod
    def _field_source(field):
        if field.startswith("current_week_"):
            return "Household.*_this_step"
        if field in {"age_year", "sex", "population_count"}:
            return "Person.age_years; Person.sex; alive population"
        if field.startswith("labor_") or field in {"employed", "employed_eligible", "unassigned_eligible"}:
            return "Person.alive; age_productivity; settlement Household; Person.firm_id"
        if "capital" in field or field in {"acquisition_count", "retirement_count", "depreciation_flow", "replacement_investment", "expansion_investment"}:
            return "Firm.capital_stock and authoritative Firm weekly capital diagnostics"
        return "authoritative runtime state at observation time"

    @staticmethod
    def _allowed_statistics(filename):
        if filename == "social_household_snapshots.csv":
            return "histogram; ECDF; quantiles; Gini; Lorenz; top/bottom shares; snapshot trend"
        if filename == "age_sex_histogram.csv":
            return "age pyramid/profile; age shares; dependency ratios"
        if filename == "labor_events.csv":
            return "event counts; hires; releases; transfers; turnover; net flow"
        if filename == "firm_capital_history.csv":
            return "Firm/sector levels; flows; capital intensity where employment denominator is valid"
        return "counts; rates; reconciliation"

    @staticmethod
    def _person_state(world, person):
        firm_id = getattr(person, "firm_id", None)
        firm = world.get_firm_by_id(firm_id) if firm_id is not None else None
        labor_age = bool(getattr(person, "alive", False) and age_productivity(person.age) > 0.0)
        settlement_valid = bool(labor_age and world.has_valid_settlement_household(person))
        return {
            "labor_age_eligible": labor_age,
            "settlement_valid": settlement_valid,
            "eligible": settlement_valid,
            "firm_id": firm_id,
            "sector": getattr(firm, "sector_id", "") if firm is not None else "",
            "age": float(person.age_years),
        }

    def _people_state(self, world):
        return {
            person.id: self._person_state(world, person)
            for person in getattr(world, "population", [])
            if getattr(person, "alive", False)
        }

    @staticmethod
    def _settlement_state(world):
        owner_by_account = {
            account_id: person_id
            for person_id, account_id in getattr(world, "settlement_household_by_person", {}).items()
        }
        return {
            household.id: {
                "owner_person_id": owner_by_account.get(
                    household.id,
                    (household.parents[0] if getattr(household, "parents", []) else ""),
                ),
                "cash": _number(getattr(household, "wealth", 0.0)),
            }
            for household in getattr(world, "households", [])
            if getattr(household, "settlement_only", False)
        }

    def initialize(self, world):
        """Capture the pre-run state and write authoritative initial snapshots."""
        self._previous_people = self._people_state(world)
        self._previous_settlement = self._settlement_state(world)
        self._write_snapshot(world, -1, "INITIAL")
        self._emit("labor_denominators", [self._labor_denominator_row(world, -1)])
        self._emit("capital", self._capital_rows(world, -1))

    def _age_rows(self, world, step, phase):
        counts = {}
        for person in getattr(world, "population", []):
            if not getattr(person, "alive", False):
                continue
            key = (int(math.floor(float(person.age_years))), str(person.sex))
            counts[key] = counts.get(key, 0) + 1
        return [
            {
                "global_step": step,
                "observation_phase": phase,
                "age_year": age,
                "sex": sex,
                "population_count": count,
            }
            for (age, sex), count in sorted(counts.items())
        ]

    @staticmethod
    def _household_members(world, household):
        member_ids = dict.fromkeys([
            *getattr(household, "parents", []),
            *getattr(household, "children", []),
        ])
        return [
            world.person_dict[person_id]
            for person_id in member_ids
            if person_id in world.person_dict and world.person_dict[person_id].alive
        ]

    def _interval(self, household_id, opening_cash=0.0, created=False):
        return self._household_intervals.setdefault(household_id, {
            "opening_cash": float(opening_cash),
            "weeks": 0,
            "labor_income": 0.0,
            "dividend_income": 0.0,
            "other_recorded_income": 0.0,
            "total_income": 0.0,
            "consumption": 0.0,
            "saving": 0.0,
            "minimum_consumption": 0.0,
            "labor_eligible_person_weeks": 0,
            "employed_person_weeks": 0,
            "paid_person_weeks": 0,
            "scheduled_wage": 0.0,
            "paid_wage": 0.0,
            "payroll_shortfall": 0.0,
            "food_paid_wage": 0.0,
            "capital_goods_paid_wage": 0.0,
            "food_paid_person_weeks": 0,
            "capital_goods_paid_person_weeks": 0,
            "transfer_in": 0.0,
            "transfer_out": 0.0,
            "settlement_origin": False,
            "created": bool(created),
        })

    def _accumulate_household_flows(self, world, step):
        transfer_in = {}
        transfer_out = {}
        settlement_destinations = set()
        for event in getattr(world, "household_lifecycle_transfer_events", []):
            if int(_number(event.get("week"), -1)) != step:
                continue
            amount = _number(event.get("amount"), 0.0)
            destination = event.get("destination_household_id")
            source = event.get("source_household_id")
            if destination is not None:
                transfer_in[destination] = transfer_in.get(destination, 0.0) + amount
                if event.get("event_type") == "settlement_to_social_household":
                    settlement_destinations.add(destination)
            if source is not None:
                transfer_out[source] = transfer_out.get(source, 0.0) + amount
        for household in getattr(world, "households", []):
            if getattr(household, "settlement_only", False):
                continue
            household_id = household.id
            created = household_id not in self._household_intervals
            interval = self._interval(household_id, opening_cash=0.0, created=created)
            income = _number(getattr(household, "income_this_step", 0.0))
            labor = _number(getattr(household, "wage_income_this_step", 0.0))
            dividends = _number(getattr(household, "dividend_income_this_step", 0.0))
            interval["weeks"] += 1
            interval["labor_income"] += labor
            interval["dividend_income"] += dividends
            interval["other_recorded_income"] += income - labor - dividends
            interval["total_income"] += income
            interval["consumption"] += _number(getattr(household, "consumption_this_step", 0.0))
            interval["saving"] += _number(getattr(household, "saving_this_step", 0.0))
            interval["minimum_consumption"] += _number(
                getattr(household, "necessary_consumption_this_step", 0.0)
            )
            members = self._household_members(world, household)
            eligible_members = [
                person for person in members
                if age_productivity(person.age) > 0.0
                and world.has_valid_settlement_household(person)
            ]
            interval["labor_eligible_person_weeks"] += len(eligible_members)
            interval["employed_person_weeks"] += sum(
                getattr(person, "firm_id", None) is not None
                for person in eligible_members
            )
            payroll = getattr(
                world, "_household_payroll_provenance_weekly_state", {}
            ).get(household_id, {})
            interval["paid_person_weeks"] += int(payroll.get("paid_person_count", 0))
            for component in payroll.get("firm_components", []):
                scheduled = _number(component.get("scheduled_wage"), 0.0)
                paid = _number(component.get("paid_wage"), 0.0)
                paid_people = int(_number(component.get("paid_person_count"), 0))
                interval["scheduled_wage"] += scheduled
                interval["paid_wage"] += paid
                interval["payroll_shortfall"] += max(0.0, scheduled - paid)
                if component.get("sector_id") == "capital_goods":
                    interval["capital_goods_paid_wage"] += paid
                    interval["capital_goods_paid_person_weeks"] += paid_people
                else:
                    interval["food_paid_wage"] += paid
                    interval["food_paid_person_weeks"] += paid_people
            interval["transfer_in"] += transfer_in.get(household_id, 0.0)
            interval["transfer_out"] += transfer_out.get(household_id, 0.0)
            interval["settlement_origin"] = (
                interval["settlement_origin"] or household_id in settlement_destinations
            )

    def _reset_household_intervals(self, world):
        current_ids = set()
        for household in getattr(world, "households", []):
            if getattr(household, "settlement_only", False):
                continue
            current_ids.add(household.id)
            self._household_intervals[household.id] = {
                "opening_cash": _number(getattr(household, "wealth", 0.0)),
                "weeks": 0,
                "labor_income": 0.0,
                "dividend_income": 0.0,
                "other_recorded_income": 0.0,
                "total_income": 0.0,
                "consumption": 0.0,
                "saving": 0.0,
                "minimum_consumption": 0.0,
                "labor_eligible_person_weeks": 0,
                "employed_person_weeks": 0,
                "paid_person_weeks": 0,
                "scheduled_wage": 0.0,
                "paid_wage": 0.0,
                "payroll_shortfall": 0.0,
                "food_paid_wage": 0.0,
                "capital_goods_paid_wage": 0.0,
                "food_paid_person_weeks": 0,
                "capital_goods_paid_person_weeks": 0,
                "transfer_in": 0.0,
                "transfer_out": 0.0,
                "settlement_origin": False,
                "created": False,
            }
        self._household_intervals = {
            household_id: interval
            for household_id, interval in self._household_intervals.items()
            if household_id in current_ids
        }

    def _household_rows(self, world, step, phase):
        rows = []
        for household in getattr(world, "households", []):
            if getattr(household, "settlement_only", False):
                continue
            members = self._household_members(world, household)
            cash = _number(getattr(household, "wealth", 0.0))
            equity = _number(getattr(household, "equity_asset_value", 0.0))
            interval = self._interval(household.id, opening_cash=cash)
            eligible = [
                person for person in members
                if age_productivity(person.age) > 0.0
                and world.has_valid_settlement_household(person)
            ]
            bridge = (
                cash - interval["opening_cash"] - interval["total_income"]
                - interval["transfer_in"] + interval["consumption"]
                + interval["transfer_out"]
            )
            row = {
                "global_step": step,
                "observation_phase": phase,
                "household_id": household.id,
                "member_count": len(members),
                "adult_count": sum(float(person.age_years) >= 18.0 for person in members),
                "child_count": sum(float(person.age_years) < 18.0 for person in members),
                "elderly_count": sum(float(person.age_years) >= 65.0 for person in members),
                "employed_member_count": sum(getattr(person, "firm_id", None) is not None for person in members),
                "labor_eligible_member_count": len(eligible),
                "settlement_origin_since_snapshot": bool(interval["settlement_origin"]),
                "current_week_income": _number(getattr(household, "income_this_step", 0.0)),
                "current_week_consumption": _number(getattr(household, "consumption_this_step", 0.0)),
                "current_week_saving": _number(getattr(household, "saving_this_step", 0.0)),
                "cash": cash,
                "equity_assets_at_cost": equity,
                "other_financial_assets": 0.0,
                "liabilities": 0.0,
                "total_financial_assets": cash + equity,
                "financial_net_worth": cash + equity,
                "interval_weeks": interval["weeks"],
                "opening_cash": interval["opening_cash"],
                "interval_labor_income": interval["labor_income"],
                "interval_labor_eligible_person_weeks": interval["labor_eligible_person_weeks"],
                "interval_employed_person_weeks": interval["employed_person_weeks"],
                "interval_paid_person_weeks": interval["paid_person_weeks"],
                "interval_scheduled_wage": interval["scheduled_wage"],
                "interval_paid_wage": interval["paid_wage"],
                "interval_payroll_shortfall": interval["payroll_shortfall"],
                "interval_food_paid_wage": interval["food_paid_wage"],
                "interval_capital_goods_paid_wage": interval["capital_goods_paid_wage"],
                "interval_food_paid_person_weeks": interval["food_paid_person_weeks"],
                "interval_capital_goods_paid_person_weeks": interval["capital_goods_paid_person_weeks"],
                "interval_dividend_income": interval["dividend_income"],
                "interval_other_recorded_income": interval["other_recorded_income"],
                "interval_total_authoritative_income": interval["total_income"],
                "interval_consumption": interval["consumption"],
                "interval_saving": interval["saving"],
                "interval_minimum_consumption": interval["minimum_consumption"],
                "interval_lifecycle_transfer_in": interval["transfer_in"],
                "interval_lifecycle_transfer_out": interval["transfer_out"],
                "interval_other_authoritative_inflow": 0.0,
                "interval_other_authoritative_outflow": 0.0,
                "interval_cash_bridge_gap": bridge,
                "created_since_previous_snapshot": bool(interval["created"]),
            }
            if self.observability_mode == "RESEARCH_FAST":
                # Fast mode intentionally does not scan household flows on
                # non-snapshot weeks. Interval flow fields are unavailable,
                # not zero, and the current-week snapshot fields remain valid.
                for key in row:
                    if key.startswith("interval_"):
                        row[key] = math.nan
                row["opening_cash"] = cash
                row["interval_weeks"] = math.nan
            rows.append(row)
        return rows

    def _settlement_rows(self, world, step, phase):
        current = self._settlement_state(world)
        return [
            {
                "global_step": step,
                "observation_phase": phase,
                "settlement_account_id": account_id,
                "owner_person_id": state["owner_person_id"],
                "cash": state["cash"],
                "active": True,
                "transition_to_social_household": False,
                "destination_social_household_id": "",
            }
            for account_id, state in sorted(current.items(), key=lambda item: str(item[0]))
        ]

    def _settlement_transition_rows(self, world, step, current):
        rows = []
        for account_id in self._previous_settlement.keys() - current.keys():
            previous = self._previous_settlement[account_id]
            person = world.person_dict.get(previous["owner_person_id"])
            destination = getattr(person, "household_id", "") if person is not None else ""
            transitioned = bool(
                person is not None
                and destination not in (None, "")
                and world.get_household(destination) is not None
                and not getattr(world.get_household(destination), "settlement_only", False)
            )
            rows.append({
                "global_step": step,
                "observation_phase": "TRANSITION" if transitioned else "DEACTIVATION",
                "settlement_account_id": account_id,
                "owner_person_id": previous["owner_person_id"],
                "cash": previous["cash"],
                "active": False,
                "transition_to_social_household": transitioned,
                "destination_social_household_id": destination if transitioned else "",
            })
        return rows

    def _write_snapshot(self, world, step, phase):
        self._emit("age", self._age_rows(world, step, phase))
        self._emit("households", self._household_rows(world, step, phase))
        self._emit("settlement", self._settlement_rows(world, step, phase))
        self._reset_household_intervals(world)

    @staticmethod
    def _labor_denominator_row(world, step):
        people = [person for person in getattr(world, "population", []) if person.alive]
        working = [person for person in people if 20.0 <= float(person.age_years) <= 60.0]
        settlement_working = [person for person in working if world.has_valid_settlement_household(person)]
        labor_age = [person for person in people if age_productivity(person.age) > 0.0]
        eligible = [person for person in labor_age if world.has_valid_settlement_household(person)]
        employed_eligible = [person for person in eligible if getattr(person, "firm_id", None) is not None]
        unassigned = [person for person in eligible if getattr(person, "firm_id", None) is None]
        employed = sum(len(getattr(firm, "employee_ids", [])) for firm in world.operating_firms())
        return {
            "global_step": step,
            "population": len(people),
            "working_age_population_20_60": len(working),
            "settlement_valid_working_age_20_60": len(settlement_working),
            "labor_age_eligible": len(labor_age),
            "labor_match_eligible": len(eligible),
            "employed": employed,
            "employed_eligible": len(employed_eligible),
            "unassigned_eligible": len(unassigned),
            "eligible_identity_gap": len(eligible) - len(employed_eligible) - len(unassigned),
        }

    def _labor_event_rows(self, world, step, current):
        rows = []
        death_ids = {
            row.get("person_id")
            for row in getattr(world, "demographic_event_rows", [])
            if _step(row) == step and str(row.get("event_type", "")).lower() == "death"
        }
        all_ids = self._previous_people.keys() | current.keys()
        for person_id in sorted(all_ids, key=str):
            before = self._previous_people.get(person_id)
            after = current.get(person_id)
            if before is not None and before["eligible"] and (after is None or not after["eligible"]):
                reason = "death" if person_id in death_ids else (
                    "settlement_account_unavailable"
                    if after is not None and after["labor_age_eligible"] else
                    "age_productivity_boundary_or_population_exit"
                )
                rows.append(self._labor_event(step, person_id, "eligibility_exit", before, after, reason))
            if after is not None and after["eligible"] and (before is None or not before["eligible"]):
                reason = (
                    "settlement_account_available"
                    if before is not None and before["labor_age_eligible"] else
                    "age_productivity_boundary"
                )
                rows.append(self._labor_event(step, person_id, "eligibility_entry", before, after, reason))
            before_firm = before.get("firm_id") if before else None
            after_firm = after.get("firm_id") if after else None
            if before_firm is None and after_firm is not None:
                rows.append(self._labor_event(step, person_id, "hire", before, after, "employer_assignment_change"))
            elif before_firm is not None and after_firm is None:
                reason = "death" if person_id in death_ids else "employer_assignment_change"
                rows.append(self._labor_event(step, person_id, "release", before, after, reason))
            elif before_firm is not None and after_firm is not None and before_firm != after_firm:
                rows.append(self._labor_event(step, person_id, "sector_transfer", before, after, "employer_assignment_change"))
        return rows

    @staticmethod
    def _labor_event(step, person_id, event_type, before, after, reason):
        return {
            "global_step": step,
            "person_id": person_id,
            "event_type": event_type,
            "from_firm_id": before.get("firm_id", "") if before else "",
            "to_firm_id": after.get("firm_id", "") if after else "",
            "from_sector": before.get("sector", "") if before else "",
            "to_sector": after.get("sector", "") if after else "",
            "reason": reason,
        }

    @staticmethod
    def _asset_week(asset, field):
        value = getattr(asset, "capacity_metadata", {}).get(field)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _capital_rows(self, world, step):
        raw_by_id = _latest_firm_rows(world, step)
        rows = []
        for firm in world.operating_firms():
            raw = raw_by_id.get(str(getattr(firm, "firm_id", "")), {})
            stock = getattr(firm, "capital_stock", None)
            assets = list(getattr(stock, "assets", []))
            acquisitions = [asset for asset in assets if self._asset_week(asset, "acquisition_week") == step]
            retirements = [asset for asset in assets if self._asset_week(asset, "retirement_week") == step]
            retired_service = math.fsum(
                _number(getattr(asset, "capacity_metadata", {}).get("capital_service_capacity_per_unit"), 0.0)
                * _number(getattr(asset, "quantity", 0.0), 0.0)
                for asset in retirements
            )
            service = (
                _number(stock.capital_service_capacity(engineering_capacity_per_unit=1.0), 0.0)
                if stock is not None else 0.0
            )
            rows.append({
                "global_step": step,
                "firm_id": getattr(firm, "firm_id", ""),
                "sector_id": getattr(firm, "sector_id", "UNAVAILABLE"),
                "employment": len(getattr(firm, "employee_ids", [])),
                "active_capital_service": service,
                "capital_asset_count": len(assets),
                "active_capital_asset_count": sum(bool(getattr(asset, "is_active", False)) for asset in assets),
                "gross_capital_book_value": math.fsum(_number(getattr(asset, "acquisition_cost", 0.0), 0.0) for asset in assets),
                "closing_capital_book_value": math.fsum(_number(getattr(asset, "remaining_book_value", 0.0), 0.0) for asset in assets),
                "accumulated_depreciation": math.fsum(_number(getattr(asset, "accumulated_depreciation", 0.0), 0.0) for asset in assets),
                "depreciation_flow": raw.get("capital_depreciation_expense", getattr(firm, "capital_depreciation_expense_this_step", 0.0)),
                "acquisition_count": len(acquisitions),
                "acquisition_expenditure": math.fsum(_number(getattr(asset, "acquisition_cost", 0.0), 0.0) for asset in acquisitions),
                "retirement_count": len(retirements),
                "retired_service_capacity": retired_service,
                "replacement_investment": raw.get("executed_replacement_investment", getattr(firm, "executed_replacement_investment_this_step", 0.0)),
                "expansion_investment": raw.get("executed_expansion_investment", getattr(firm, "executed_expansion_investment_this_step", 0.0)),
            })
        return rows

    def write_step(self, world, step):
        final_step = int(getattr(world, "steps", 0)) - 1
        micro_due = (
            self.observability_mode != "RESEARCH_FAST"
            or step % self.snapshot_cadence == 0
            or step == final_step
        )
        # RESEARCH_FAST keeps weekly aggregate diagnostics in the canonical
        # persistence layer, while micro state is sampled at the declared
        # snapshot cadence. This runs after economic execution.
        current_people = self._people_state(world) if micro_due else None
        current_settlement = (
            self._settlement_state(world) if micro_due else None
        )
        if micro_due:
            self._emit(
                "labor_events",
                self._labor_event_rows(world, step, current_people),
            )
        self._emit(
            "labor_denominators",
            [self._labor_denominator_row(world, step)],
        )
        self._emit("capital", self._capital_rows(world, step))
        if micro_due:
            transitions = self._settlement_transition_rows(
                world, step, current_settlement
            )
            self._emit("settlement", transitions)
            self._accumulate_household_flows(world, step)

        household_due = step % self.snapshot_cadence == 0 or step == final_step
        age_due = step % self.age_cadence == 0 or step == final_step
        phase = "FINAL" if step == final_step else "PERIODIC"
        if age_due and micro_due:
            self._emit("age", self._age_rows(world, step, phase))
        if household_due and micro_due:
            self._emit(
                "households", self._household_rows(world, step, phase)
            )
            self._emit(
                "settlement", self._settlement_rows(world, step, phase)
            )
            self._reset_household_intervals(world)

        if micro_due:
            self._previous_people = current_people
            self._previous_settlement = current_settlement
        return True


__all__ = ["StatisticalObservabilityPersistence"]
