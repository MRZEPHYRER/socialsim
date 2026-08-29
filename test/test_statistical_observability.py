from __future__ import annotations

import csv
import random

import pytest

from world import World


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture()
def observed_world(tmp_path):
    world = World(
        80,
        seed=42,
        diagnostics_mode="full",
        scenario_name="statistical_observability_test",
        scenario_overrides={"ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED": True},
    )
    world.split_firms(2)
    world.ensure_labor_settlement_households()
    world.household_employer_exposure_instrumentation_enabled = True
    world.steps = 2
    world.configure_diagnostic_persistence(
        tmp_path / "canonical",
        cadence=1,
        statistical_observability=True,
        statistical_output_dir=tmp_path / "statistics",
        statistical_snapshot_cadence=1,
        statistical_age_cadence=1,
    )
    world.step()
    world.step()
    return world, tmp_path / "statistics"


def test_exact_age_sex_histogram_sums_to_population(observed_world):
    world, output = observed_world
    rows = read_rows(output / "age_sex_histogram.csv")
    initial = [row for row in rows if int(row["global_step"]) == -1]
    final = [row for row in rows if int(row["global_step"]) == 1]
    assert sum(int(row["population_count"]) for row in initial) == 80
    assert sum(int(row["population_count"]) for row in final) == len(world.population)
    assert {row["sex"] for row in rows}.issubset({"M", "F"})


def test_social_household_snapshots_exclude_settlement_only_accounts(observed_world):
    world, output = observed_world
    rows = read_rows(output / "social_household_snapshots.csv")
    settlement_ids = {
        str(household.id)
        for household in world.households
        if getattr(household, "settlement_only", False)
    }
    assert rows
    assert not ({row["household_id"] for row in rows} & settlement_ids)
    for row in rows:
        assert int(row["member_count"]) == int(row["adult_count"]) + int(row["child_count"])
        assert float(row["total_financial_assets"]) == pytest.approx(
            float(row["cash"]) + float(row["equity_assets_at_cost"])
        )


def test_household_snapshot_interval_cash_bridge_is_authoritative(observed_world):
    _, output = observed_world
    rows = read_rows(output / "social_household_snapshots.csv")
    completed = [row for row in rows if int(row["global_step"]) >= 0]
    assert completed
    assert all("employed_member_count" in row for row in completed)
    assert all("interval_labor_income" in row for row in completed)
    assert all("interval_employed_person_weeks" in row for row in completed)
    assert all("interval_paid_wage" in row for row in completed)
    for row in completed:
        assert int(row["interval_employed_person_weeks"]) <= int(
            row["interval_labor_eligible_person_weeks"]
        )
        assert float(row["interval_labor_income"]) == pytest.approx(
            float(row["interval_paid_wage"])
        )
        bridge = (
            float(row["cash"])
            - float(row["opening_cash"])
            - float(row["interval_total_authoritative_income"])
            - float(row["interval_lifecycle_transfer_in"])
            + float(row["interval_consumption"])
            + float(row["interval_lifecycle_transfer_out"])
        )
        assert float(row["interval_cash_bridge_gap"]) == pytest.approx(bridge)


def test_settlement_accounts_are_persisted_separately(observed_world):
    world, output = observed_world
    rows = read_rows(output / "settlement_account_snapshots.csv")
    active = [row for row in rows if row["active"] == "True"]
    assert active
    assert all(row["owner_person_id"] != "" for row in active)
    assert all(row["settlement_account_id"] != "" for row in active)


def test_labor_denominator_identity_is_exact(observed_world):
    _, output = observed_world
    rows = read_rows(output / "labor_denominators.csv")
    assert {int(row["global_step"]) for row in rows} == {-1, 0, 1}
    for row in rows:
        eligible = int(row["labor_match_eligible"])
        employed = int(row["employed_eligible"])
        unassigned = int(row["unassigned_eligible"])
        assert eligible == employed + unassigned
        assert int(row["eligible_identity_gap"]) == 0


def test_firm_capital_history_aggregates_runtime_stock(observed_world):
    world, output = observed_world
    rows = read_rows(output / "firm_capital_history.csv")
    final = [row for row in rows if int(row["global_step"]) == 1]
    assert len(final) == len(world.operating_firms())
    expected = sum(
        firm.capital_stock.capital_service_capacity(engineering_capacity_per_unit=1.0)
        for firm in world.operating_firms()
    )
    assert sum(float(row["active_capital_service"]) for row in final) == pytest.approx(expected)
    assert all(
        float(row["gross_capital_book_value"])
        >= float(row["closing_capital_book_value"])
        for row in final
    )


def test_observer_contract_is_written(observed_world):
    _, output = observed_world
    rows = read_rows(output / "statistical_persistence_contract.csv")
    assert {row["dataset"] for row in rows} == {
        "age_sex_histogram.csv",
        "social_household_snapshots.csv",
        "settlement_account_snapshots.csv",
        "labor_events.csv",
        "labor_denominators.csv",
        "firm_capital_history.csv",
    }


def test_diagnostics_on_off_are_behaviorally_identical(tmp_path):
    def run(observed):
        world = World(
            120,
            seed=7,
            diagnostics_mode="full",
            scenario_name="observer_noninterference",
            scenario_overrides={"ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED": True},
        )
        world.split_firms(3)
        world.steps = 8
        if observed:
            world.configure_diagnostic_persistence(
                tmp_path / "canonical",
                statistical_observability=True,
                statistical_output_dir=tmp_path / "statistics",
                statistical_snapshot_cadence=2,
                statistical_age_cadence=2,
            )
        for _ in range(world.steps):
            world.step()
        fields = (
            "population", "births", "deaths", "marriages", "labor",
            "total_income", "total_consumption", "total_saving",
            "total_household_wealth", "total_money_stock",
            "monetary_accounting_gap",
        )
        trajectory = [tuple(row.get(field) for field in fields) for row in world.diagnostics_rows]
        firm_cash = [
            (row.get("global_step"), row.get("firm_id"), row.get("cash"))
            for row in world.firm_diagnostics_rows
        ]
        return trajectory, firm_cash, random.getstate()

    control = run(False)
    observed = run(True)
    assert observed == control
