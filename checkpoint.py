import os
import pickle
import random
import sys

from central_bank import config as central_bank_config
from economy import config as economy_config
from fertility import config as fertility_config
from scenarios import git_version_identifier
from time_system import MARRIAGE_MARKET_INTERVAL_WEEKS


CHECKPOINT_VERSION = 1
CHECKPOINT_MAGIC = "socialsim.world.checkpoint"


def _weekly_checkpoint_snapshot(world):
    firms = []
    for firm in getattr(world, "firms", []):
        firms.append(
            {
                "firm_id": getattr(firm, "firm_id", None),
                "price": getattr(firm, "price", 0.0),
                "market_share": getattr(firm, "unit_market_share", 0.0),
                "sales": getattr(firm, "sales", 0.0),
                "production": getattr(firm, "production", 0.0),
                "inventory": getattr(firm, "inventory_units", 0.0),
                "expected_demand": getattr(firm, "expected_demand", 0.0),
                "production_plan": getattr(firm, "production_plan", 0.0),
                "profit": getattr(firm, "profit", 0.0),
                "cash": getattr(firm, "cash", 0.0),
                "loan_balance": getattr(firm, "loan_balance", 0.0),
            }
        )

    household_wealth = sum(
        getattr(household, "wealth", 0.0)
        for household in getattr(world, "households", [])
    )
    total_inventory = sum(
        getattr(firm, "inventory_units", 0.0)
        for firm in getattr(world, "firms", [])
    )
    total_cash = sum(
        getattr(firm, "cash", 0.0)
        for firm in getattr(world, "firms", [])
    )
    total_loans = sum(
        getattr(firm, "loan_balance", 0.0)
        for firm in getattr(world, "firms", [])
    )
    return {
        "population": len(getattr(world, "population", [])),
        "household_count": len(getattr(world, "households", [])),
        "total_consumption": getattr(
            getattr(world, "consumption_system", None),
            "total_consumption",
            0.0,
        ),
        "total_production": getattr(
            getattr(world, "firm_system", None),
            "food_output_units",
            0.0,
        ),
        "total_inventory": total_inventory,
        "total_household_wealth": household_wealth,
        "total_firm_cash": total_cash,
        "total_firm_loan_balance": total_loans,
        "total_public_cash_wealth": (
            getattr(world, "public_wealth", 0.0)
            +
            getattr(
                getattr(getattr(world, "firm_system", None), "central_bank", None),
                "public_income_balance",
                0.0,
            )
        ),
        "legacy_owner_cash": getattr(world, "legacy_owner_cash", 0.0),
        "total_money": getattr(
            getattr(getattr(world, "firm_system", None), "central_bank", None),
            "money_supply",
            0.0,
        ),
        "age_group_shares": world.demographic_structure_diagnostics()
        if hasattr(world, "demographic_structure_diagnostics")
        else {},
        "firms": firms,
    }


def _numpy_module():
    try:
        import numpy as np
    except Exception:
        return None

    return np


def _config_snapshot(module):
    return {
        name: getattr(module, name)
        for name in dir(module)
        if name.isupper()
    }


def current_config_state():
    return {
        "economy.config": _config_snapshot(economy_config),
        "fertility.config": _config_snapshot(fertility_config),
        "central_bank.config": _config_snapshot(central_bank_config),
    }


def restore_config_state(config_state):
    modules = {
        "economy.config": economy_config,
        "fertility.config": fertility_config,
        "central_bank.config": central_bank_config,
    }

    for module_name, values in config_state.items():
        if module_name not in modules:
            raise ValueError(
                f"Unsupported config module in checkpoint: {module_name}"
            )

        module = modules[module_name]

        for name, value in values.items():
            if not hasattr(module, name):
                raise ValueError(
                    f"Config field missing in current code: {module_name}.{name}"
                )

            setattr(module, name, value)


def capture_rng_state():
    np = _numpy_module()
    numpy_state = None

    if np is not None:
        numpy_state = np.random.get_state()

    return {
        "python_random": random.getstate(),
        "numpy_random": numpy_state,
    }


def restore_rng_state(rng_state):
    if "python_random" not in rng_state:
        raise ValueError("Checkpoint missing Python random state")

    random.setstate(rng_state["python_random"])

    numpy_state = rng_state.get("numpy_random")
    if numpy_state is not None:
        np = _numpy_module()

        if np is None:
            raise ValueError(
                "Checkpoint includes NumPy RNG state but NumPy is unavailable"
            )

        np.random.set_state(numpy_state)


def world_global_step(world):
    return len(getattr(world, "population_history", []))


def build_checkpoint_metadata(world, extra_metadata=None):
    global_step = world_global_step(world)
    metadata = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "magic": CHECKPOINT_MAGIC,
        "python_version": sys.version,
        "version_identifier": git_version_identifier(),
        "scenario": getattr(world, "scenario_name", None),
        "scenario_overrides": getattr(world, "scenario_overrides", {}),
        "seed": getattr(world, "seed", None),
        "initial_population": getattr(world, "initial_population", None),
        "global_step": global_step,
        "diagnostics_mode": getattr(world, "diagnostics_mode", None),
        "observability_mode": getattr(
            world, "observability_mode", "FULL_DIAGNOSTIC"
        ),
        "age_schema": "weekly_age_weeks_v1",
        "initial_age_phase_mode": getattr(
            world, "initial_age_phase_mode", "legacy_synchronized"
        ),
        "initial_age_phase_schema": getattr(
            world,
            "initial_age_phase_schema",
            "legacy_integer_year_synchronized",
        ),
        "initial_age_phase_rng_persisted": hasattr(world, "age_phase_rng"),
        "birth_spacing_schema": "last_birth_step_v1",
        "default_bookkeeping_schema": "step13_7c_contractual_cure_v1",
        "default_history_before_checkpoint_known": bool(
            getattr(world, "default_bookkeeping_history_complete", False)
        ),
        "marriage_rng_schema": "dedicated_marriage_rng_v1",
        "marriage_rng_isolated": True,
        "legacy_marriage_rng_migration": bool(
            getattr(
                getattr(world, "marriage_system", None),
                "legacy_marriage_rng_migration",
                False,
            )
        ),
        "marriage_schedule_schema": "annual_equivalent_marriage_schedule_v1",
        "next_marriage_market_step": getattr(
            world,
            "next_marriage_market_step",
            world_global_step(world) + MARRIAGE_MARKET_INTERVAL_WEEKS,
        ),
        "legacy_marriage_schedule_migration": bool(
            getattr(world, "legacy_marriage_schedule_migration", False)
        ),
        "time_convention": "1_step_1_week_52_steps_1_year",
        "time_schema": "weekly",
        "simulation_step_unit": "week",
        "steps_per_year": 52,
        "simulation_week": float(global_step),
        "simulation_year": float(global_step) / 52.0,
        "completed_steps": global_step,
        "next_step_to_execute": global_step,
        "checkpoint_boundary": "after_completed_weekly_steps",
        "project_stage": "Step 12.11",
        "baseline_type": "formal_weekly_baseline",
        "source": "fresh_weekly_simulation",
        "firm_count": len(getattr(world, "firms", [])),
        "multi_firm_bootstrap_complete": bool(
            getattr(world, "multi_firm_bootstrap_complete", False)
        ),
        "multi_firm_bootstrap_execution_count": int(
            getattr(world, "multi_firm_bootstrap_execution_count", 0)
        ),
        "household_planning_price_schema": (
            "lagged_unit_share_x_current_posted_price"
        ),
        "household_reserve_schema": "six_calendar_months_26_weeks",
        "lifecycle_signed_balance_fix": "enabled",
        "historical_checkpoint_compatible_for_regression": False,
        "weekly_schema": {
            "age_schema": "weekly_age_weeks_v1",
            "mortality_semantics": "annual_hazard_converted_to_weekly_probability",
            "fertility_semantics": "final_annual_composite_probability_converted_to_weekly",
            "minimum_birth_interval_weeks": 52,
            "marriage_market_interval_weeks": 52,
            "household_target_wealth_reserve_months": 6,
            "household_target_wealth_reserve_weeks": 26,
            "firm_production_review_interval_weeks": 5,
            "firm_inventory_target_coverage_weeks": 15,
            "firm_price_review_probability_per_week": 0.20,
            "firm_price_evaluation_window_weeks": 12,
            "inventory_spoilage_rate_per_week": 0.01,
            "principal_repayment_target_rate_per_week": 0.35,
            "principal_repayment_cash_constrained": True,
            "dividend_share_of_eligible_profit": 0.15,
            "interest_active": False,
            "credit_limit_active": False,
            "default_active": False,
        "bankruptcy_active": False,
        "credit_capacity_active": bool(
            getattr(
                central_bank_config,
                "CENTRAL_BANK_CREDIT_CAPACITY_ENABLED",
                False,
            )
        ),
        "credit_capacity_schema": "scheduled_pre_financing_wage_bill_trailing_mean_v1",
        "credit_capacity_k_weeks": getattr(
            central_bank_config,
            "CENTRAL_BANK_CREDIT_CAPACITY_K_WEEKS",
            None,
        ),
        "credit_capacity_smoothing_weeks": getattr(
            central_bank_config,
            "CENTRAL_BANK_CREDIT_CAPACITY_SMOOTHING_WEEKS",
            26,
        ),
        },
        "state_snapshot": _weekly_checkpoint_snapshot(world),
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    return metadata


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint: payload is not a dict")

    required = {
        "metadata",
        "world",
        "rng_state",
        "config_state",
    }
    missing = required - set(payload)

    if missing:
        raise ValueError(
            "Invalid checkpoint: missing keys "
            +
            ", ".join(sorted(missing))
        )

    metadata = payload["metadata"]

    if metadata.get("magic") != CHECKPOINT_MAGIC:
        raise ValueError("Invalid checkpoint: wrong magic")

    if metadata.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            "Unsupported checkpoint version: "
            f"{metadata.get('checkpoint_version')}"
        )


def save_world_checkpoint(path, world, extra_metadata=None):
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    payload = {
        "metadata": build_checkpoint_metadata(world, extra_metadata),
        "world": world,
        "rng_state": capture_rng_state(),
        "config_state": current_config_state(),
    }

    with open(path, "wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    return payload["metadata"]


def load_world_checkpoint(path):
    with open(path, "rb") as file:
        payload = pickle.load(file)

    validate_payload(payload)
    restore_config_state(payload["config_state"])
    restore_rng_state(payload["rng_state"])
    world = payload["world"]
    checkpoint_step = payload["metadata"].get("global_step")

    # Older accepted checkpoints predate passive Default bookkeeping.  Add
    # only neutral defaults; historical events before the checkpoint remain
    # explicitly unknown and are never fabricated here.
    if hasattr(world, "ensure_default_bookkeeping_state"):
        world.ensure_default_bookkeeping_state()
    else:
        for firm in getattr(world, "firms", []):
            for name, default in {
                "default_history_count": 0,
                "active_contract_default": False,
                "active_default_episode": False,
                "default_event_this_week": False,
                "consecutive_d3_weeks": 0,
                "consecutive_non_d3_weeks": 0,
                "consecutive_no_breach_weeks": 0,
                "contract_cure_this_week": False,
                "acute_default_phase_exit_this_week": False,
                "default_distress_history": [],
            }.items():
                if not hasattr(firm, name):
                    setattr(firm, name, list(default) if isinstance(default, list) else default)
        world.default_bookkeeping_history_complete = False

    # Step 15B passive capital-stock migration.  Missing state becomes an
    # explicitly empty container; no historical assets or depreciation are
    # fabricated and no production behavior is activated.
    if hasattr(world, "ensure_capital_stock_state"):
        world.ensure_capital_stock_state()

    # Step17.D active research institutions migrate as neutral default-off
    # state when loading older checkpoints.
    if hasattr(world, "ensure_active_social_policy_state"):
        world.ensure_active_social_policy_state()

    # Step17.V founder fields are opt-in and neutral for old checkpoints.
    if not hasattr(world, "person_founder_bootstrap_enabled"):
        world.person_founder_bootstrap_enabled = False
    if not hasattr(world, "new_world_formation_context"):
        world.new_world_formation_context = False
    if not hasattr(world, "founder_assignment_rows"):
        world.founder_assignment_rows = []
    if not hasattr(world, "founder_assigned_firm_ids"):
        world.founder_assigned_firm_ids = set()

    # Step 15E.1 passive ownership migration.  Legacy checkpoints receive a
    # valid 100% LegacyOwnershipPool table; no Person shares are fabricated.
    if hasattr(world, "ensure_ownership_state"):
        world.ensure_ownership_state()

    if hasattr(world, "ensure_adult_settlement_labor_eligibility_state"):
        world.ensure_adult_settlement_labor_eligibility_state()

    if hasattr(world, "rebuild_runtime_id_indexes"):
        world.rebuild_runtime_id_indexes()

    if not hasattr(world, "next_marriage_market_step"):
        world.next_marriage_market_step = (
            checkpoint_step + MARRIAGE_MARKET_INTERVAL_WEEKS
        )
        world.legacy_marriage_schedule_migration = True
    if not hasattr(world, "marriage_schedule_schema"):
        world.marriage_schedule_schema = (
            "annual_equivalent_marriage_schedule_v1"
        )
    if not hasattr(world, "marriage_market_execution_count"):
        world.marriage_market_execution_count = 0
    if not hasattr(world, "demographic_events"):
        world.demographic_events = []
    if not hasattr(world, "marriage_market_diagnostics"):
        world.marriage_market_diagnostics = []
    for name, default in {
        "marriage_market_executed": False,
        "marriage_market_last_result": 0,
        "marriage_market_last_eligible_males": 0,
        "marriage_market_last_eligible_females": 0,
        "marriage_market_last_unmatched_males": 0,
        "marriage_market_last_unmatched_females": 0,
    }.items():
        if not hasattr(world, name):
            setattr(world, name, default)

    # Step 14A identity migration is passive and checkpoint-safe.  Old Food
    # checkpoints receive neutral Food identities; no active registry is
    # created unless the checkpoint explicitly carries the feature flag.
    if hasattr(world, "ensure_multisector_foundation_contracts"):
        world.ensure_multisector_foundation_contracts()

    actual_step = world_global_step(world)

    if checkpoint_step != actual_step:
        raise ValueError(
            "Checkpoint step mismatch: metadata global_step="
            f"{checkpoint_step}, world history length={actual_step}"
        )

    if getattr(world, "ledger", None) is not None:
        world.ledger.world = world
        world.ledger.end_step()

    return world, payload["metadata"]

