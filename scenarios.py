import copy
import json
import os
import subprocess

from economy import config as economy_config
from central_bank import config as central_bank_config
from fertility import config as fertility_config


MODEL_NAME = "single_firm_single_good"


SCENARIOS = {
    "baseline": {},
    "credit_capacity_payroll_1x": {
        "CREDIT_CAPACITY_ENABLED": True,
        "CREDIT_CAPACITY_K_WEEKS": 11.59038588550643,
    },
    "credit_capacity_payroll_2x": {
        "CREDIT_CAPACITY_ENABLED": True,
        "CREDIT_CAPACITY_K_WEEKS": 23.18077177101286,
    },
    "credit_capacity_payroll_4x": {
        "CREDIT_CAPACITY_ENABLED": True,
        "CREDIT_CAPACITY_K_WEEKS": 46.36154354202572,
    },
    "credit_capacity_payroll_control_100x": {
        "CREDIT_CAPACITY_ENABLED": True,
        "CREDIT_CAPACITY_K_WEEKS": 1159.038588550643,
    },
    "interest_behavioral_5pct": {
        "CREDIT_CAPACITY_ENABLED": True,
        "CREDIT_CAPACITY_K_WEEKS": 46.36154354202572,
        "FIRM_LOAN_INTEREST_ANNUAL_RATE": 0.05,
    },
    "interest_behavioral_0pct": {
        "CREDIT_CAPACITY_ENABLED": True,
        "CREDIT_CAPACITY_K_WEEKS": 46.36154354202572,
        "FIRM_LOAN_INTEREST_ANNUAL_RATE": 0.0,
    },
    "low_firm_cash": {
        "INITIAL_FIRM_CASH_MULTIPLIER": 0.25,
    },
    "liquidity_ratio_10": {
        "INITIAL_FIRM_CASH_TO_INITIAL_WAGE_BILL": 10.0,
    },
    "liquidity_ratio_5": {
        "INITIAL_FIRM_CASH_TO_INITIAL_WAGE_BILL": 5.0,
    },
    "liquidity_ratio_2": {
        "INITIAL_FIRM_CASH_TO_INITIAL_WAGE_BILL": 2.0,
    },
    "liquidity_ratio_1": {
        "INITIAL_FIRM_CASH_TO_INITIAL_WAGE_BILL": 1.0,
    },
    "liquidity_ratio_0_5": {
        "INITIAL_FIRM_CASH_TO_INITIAL_WAGE_BILL": 0.5,
    },
    "wage_shock": {
        "WAGE_MULTIPLIER_STEP": 1000,
        "WAGE_MULTIPLIER": 1.30,
    },
    "wage_shock_1_2": {
        "WAGE_MULTIPLIER_STEP": 1000,
        "WAGE_MULTIPLIER": 1.20,
    },
    "wage_shock_1_25": {
        "WAGE_MULTIPLIER_STEP": 1000,
        "WAGE_MULTIPLIER": 1.25,
    },
    "wage_shock_1_5": {
        "WAGE_MULTIPLIER_STEP": 1000,
        "WAGE_MULTIPLIER": 1.50,
    },
    "wage_shock_1_35": {
        "WAGE_MULTIPLIER_STEP": 1000,
        "WAGE_MULTIPLIER": 1.35,
    },
    "wage_shock_1_4": {
        "WAGE_MULTIPLIER_STEP": 1000,
        "WAGE_MULTIPLIER": 1.40,
    },
    "wage_shock_1_45": {
        "WAGE_MULTIPLIER_STEP": 1000,
        "WAGE_MULTIPLIER": 1.45,
    },
    "wage_shock_1_46": {
        "WAGE_MULTIPLIER_STEP": 1000,
        "WAGE_MULTIPLIER": 1.46,
    },
    "wage_shock_1_47": {
        "WAGE_MULTIPLIER_STEP": 1000,
        "WAGE_MULTIPLIER": 1.47,
    },
    "wage_shock_1_48": {
        "WAGE_MULTIPLIER_STEP": 1000,
        "WAGE_MULTIPLIER": 1.48,
    },
    "wage_shock_1_49": {
        "WAGE_MULTIPLIER_STEP": 1000,
        "WAGE_MULTIPLIER": 1.49,
    },
    "demand_shock": {
        "DEMAND_SHOCK_STEP": 1000,
        "DEMAND_SHOCK_MULTIPLIER": 0.75,
    },
    "demand_shock_0_5": {
        "DEMAND_SHOCK_STEP": 1000,
        "DEMAND_SHOCK_MULTIPLIER": 0.50,
    },
    "population_growth": {
        "BASE_BIRTH_RATE_MULTIPLIER": 1.35,
    },
    "population_boom": {
        "BIRTH_RATE_SHOCK_START": 500,
        "BIRTH_RATE_SHOCK_END": 1500,
        "BIRTH_RATE_SHOCK_MULTIPLIER": 1.75,
    },
}


def scenario_names():
    return sorted(SCENARIOS.keys())


def git_version_identifier():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def apply_scenario(name):
    if name not in SCENARIOS:
        valid = ", ".join(scenario_names())
        raise ValueError(f"Unknown scenario '{name}'. Valid scenarios: {valid}")

    overrides = copy.deepcopy(SCENARIOS[name])
    applied = {}

    if "INITIAL_FIRM_CASH_MULTIPLIER" in overrides:
        base_value = economy_config.FIRM_INITIAL_CASH
        multiplier = overrides["INITIAL_FIRM_CASH_MULTIPLIER"]
        effective_value = base_value * multiplier
        economy_config.FIRM_INITIAL_CASH = effective_value
        applied["economy.config.FIRM_INITIAL_CASH"] = {
            "base": base_value,
            "effective": effective_value,
            "source": "INITIAL_FIRM_CASH_MULTIPLIER",
            "multiplier": multiplier,
        }

    if "BASE_BIRTH_RATE_MULTIPLIER" in overrides:
        base_value = fertility_config.BASE_BIRTH_RATE
        multiplier = overrides["BASE_BIRTH_RATE_MULTIPLIER"]
        effective_value = base_value * multiplier
        fertility_config.BASE_BIRTH_RATE = effective_value
        applied["fertility.config.BASE_BIRTH_RATE"] = {
            "base": base_value,
            "effective": effective_value,
            "source": "BASE_BIRTH_RATE_MULTIPLIER",
            "multiplier": multiplier,
        }

    if "FOOD_TARGET_INVENTORY_DAYS" in overrides:
        base_value = economy_config.FOOD_TARGET_INVENTORY_DAYS
        effective_value = overrides["FOOD_TARGET_INVENTORY_DAYS"]
        economy_config.FOOD_TARGET_INVENTORY_DAYS = effective_value
        applied["economy.config.FOOD_TARGET_INVENTORY_DAYS"] = {
            "base": base_value,
            "effective": effective_value,
            "source": "FOOD_TARGET_INVENTORY_DAYS",
        }

    if "PRICE_CHOICE_SENSITIVITY" in overrides:
        base_value = economy_config.PRICE_CHOICE_SENSITIVITY
        effective_value = overrides["PRICE_CHOICE_SENSITIVITY"]
        economy_config.PRICE_CHOICE_SENSITIVITY = effective_value
        applied["economy.config.PRICE_CHOICE_SENSITIVITY"] = {
            "base": base_value,
            "effective": effective_value,
            "source": "PRICE_CHOICE_SENSITIVITY",
        }

    if "INITIAL_FIRM_RELATIVE_PRICE_SPREAD" in overrides:
        base_value = economy_config.INITIAL_FIRM_RELATIVE_PRICE_SPREAD
        effective_value = overrides["INITIAL_FIRM_RELATIVE_PRICE_SPREAD"]
        economy_config.INITIAL_FIRM_RELATIVE_PRICE_SPREAD = effective_value
        applied["economy.config.INITIAL_FIRM_RELATIVE_PRICE_SPREAD"] = {
            "base": base_value,
            "effective": effective_value,
            "source": "INITIAL_FIRM_RELATIVE_PRICE_SPREAD",
        }

    for key in [
        "FIRM_PRICE_REVIEW_PROBABILITY",
        "FIRM_PRICE_TRIAL_STEP_SMALL",
        "FIRM_PRICE_TRIAL_STEP_LARGE",
        "FIRM_PRICE_KEEP_PROBABILITY",
        "FIRM_PRICE_DIRECTION_PERSIST_PROBABILITY",
        "FIRM_PRICE_DIRECTION_REVERSE_PROBABILITY",
        "FIRM_PRICE_INVENTORY_WORSEN_TOLERANCE",
        "FIRM_PRICE_MIN_MARKUP",
        "FIRM_PRICE_EVALUATION_WINDOW",
        "FIRM_PRICE_PROFIT_EMA_ALPHA",
        "FIRM_PRICE_EXPLORATION_PROBABILITY",
    ]:
        if key not in overrides:
            continue

        base_value = getattr(economy_config, key)
        effective_value = overrides[key]
        setattr(economy_config, key, effective_value)
        applied[f"economy.config.{key}"] = {
            "base": base_value,
            "effective": effective_value,
            "source": key,
        }

    if "CREDIT_CAPACITY_ENABLED" in overrides:
        base_value = central_bank_config.CENTRAL_BANK_CREDIT_CAPACITY_ENABLED
        effective_value = bool(overrides["CREDIT_CAPACITY_ENABLED"])
        central_bank_config.CENTRAL_BANK_CREDIT_CAPACITY_ENABLED = effective_value
        applied["central_bank.config.CENTRAL_BANK_CREDIT_CAPACITY_ENABLED"] = {
            "base": base_value,
            "effective": effective_value,
            "source": "CREDIT_CAPACITY_ENABLED",
        }

    if "CREDIT_CAPACITY_K_WEEKS" in overrides:
        base_value = central_bank_config.CENTRAL_BANK_CREDIT_CAPACITY_K_WEEKS
        effective_value = float(overrides["CREDIT_CAPACITY_K_WEEKS"])
        central_bank_config.CENTRAL_BANK_CREDIT_CAPACITY_K_WEEKS = effective_value
        applied["central_bank.config.CENTRAL_BANK_CREDIT_CAPACITY_K_WEEKS"] = {
            "base": base_value,
            "effective": effective_value,
            "source": "CREDIT_CAPACITY_K_WEEKS",
        }

    if "FIRM_LOAN_INTEREST_ANNUAL_RATE" in overrides:
        base_value = central_bank_config.CENTRAL_BANK_FIRM_LOAN_INTEREST_ANNUAL_RATE
        effective_value = float(overrides["FIRM_LOAN_INTEREST_ANNUAL_RATE"])
        central_bank_config.CENTRAL_BANK_FIRM_LOAN_INTEREST_ANNUAL_RATE = effective_value
        applied["central_bank.config.CENTRAL_BANK_FIRM_LOAN_INTEREST_ANNUAL_RATE"] = {
            "base": base_value,
            "effective": effective_value,
            "source": "FIRM_LOAN_INTEREST_ANNUAL_RATE",
        }

    return {
        "name": name,
        "overrides": overrides,
        "applied_parameters": applied,
    }


def apply_runtime_scenario(scenario, world):
    overrides = scenario["overrides"]
    applied = scenario["applied_parameters"]

    if "INITIAL_FIRM_CASH_TO_INITIAL_WAGE_BILL" in overrides:
        ratio = overrides["INITIAL_FIRM_CASH_TO_INITIAL_WAGE_BILL"]
        firm = world.firm_system
        total_labor = firm.total_labor()
        initial_wage_bill = firm.base_wage_bill(total_labor)
        base_value = firm.cash
        effective_value = ratio * initial_wage_bill
        firm.cash = effective_value
        world.initial_private_money_stock = (
            firm.cash
            +
            sum(household.wealth for household in world.households)
            +
            getattr(world, "public_wealth", 0.0)
        )
        applied["firm.cash"] = {
            "base": base_value,
            "effective": effective_value,
            "source": "INITIAL_FIRM_CASH_TO_INITIAL_WAGE_BILL",
            "ratio": ratio,
            "initial_wage_bill_estimate": initial_wage_bill,
            "total_labor_estimate": total_labor,
        }

    return scenario


def build_manifest(
    scenario,
    seed,
    population,
    steps,
    diagnostics_csv=None,
    firm_diagnostics_csv=None,
    ledger_csv=None,
    output_dir=None,
    firm_count=None,
    analysis_start_global_step=0,
    initial_age_phase_mode=None,
):
    timing_contract = {
        "simulation_step_unit": "week",
        "steps_per_year": 52,
        "firm_expected_demand_update": "weekly",
        "firm_expected_demand_alpha": economy_config.FIRM_PRODUCTION_EXPECTED_DEMAND_ALPHA,
        "firm_production_review_interval_weeks": economy_config.FIRM_PRODUCTION_REVIEW_INTERVAL_WEEKS,
        "firm_inventory_target_coverage_weeks": economy_config.FIRM_PRODUCTION_TARGET_INVENTORY_COVERAGE_WEEKS,
        "firm_price_review_probability_per_week": economy_config.FIRM_PRICE_REVIEW_PROBABILITY,
        "firm_price_evaluation_window_weeks": economy_config.FIRM_PRICE_EVALUATION_WINDOW_WEEKS,
        "firm_price_profit_ema_alpha": economy_config.FIRM_PRICE_PROFIT_EMA_ALPHA,
        "firm_price_trial_small": economy_config.FIRM_PRICE_TRIAL_STEP_SMALL,
        "firm_price_trial_large": economy_config.FIRM_PRICE_TRIAL_STEP_LARGE,
        "firm_price_exploration_probability_per_review": economy_config.FIRM_PRICE_EXPLORATION_PROBABILITY,
        "firm_expected_sales_ema_semantics": "legacy_weekly_sales_ema",
        "food_price_cash_window_weeks": economy_config.FOOD_PRICE_CASH_WINDOW,
        "working_capital_check_frequency": "weekly",
        "working_capital_wage_buffer_weeks": 1.0,
        "working_capital_fixed_cash_buffer": 150000.0,
        "principal_repayment_check_frequency": "weekly",
        "principal_repayment_target_rate_per_week": (
            central_bank_config.CENTRAL_BANK_FIRM_WEEKLY_PRINCIPAL_REPAYMENT_TARGET_RATE
        ),
        "principal_repayment_cash_constrained": True,
        "inventory_spoilage_frequency": "weekly",
        "inventory_spoilage_rate_per_week": (
            economy_config.FOOD_INVENTORY_SPOILAGE_RATE_PER_WEEK
        ),
        "dividend_check_frequency": "weekly",
        "dividend_share_of_eligible_profit": economy_config.FIRM_DIVIDEND_SHARE,
        "dividend_profit_basis": "lagged eligible weekly profit",
        "public_support_frequency": "weekly_conditional",
        "estate_transfer_frequency": "lifecycle_event",
        "financial_interest_active": (
            central_bank_config.CENTRAL_BANK_FIRM_LOAN_INTEREST_ANNUAL_RATE > 0
        ),
        "credit_limit_active": bool(
            central_bank_config.CENTRAL_BANK_CREDIT_CAPACITY_ENABLED
        ),
        "default_bookkeeping_active": True,
        "default_consequence": "record_only",
        "temporary_interest_relief_active": bool(
            central_bank_config.CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_ENABLED
        ),
        "snapshot_legacy_arrears_termout_active": bool(
            central_bank_config.CENTRAL_BANK_SNAPSHOT_LEGACY_ARREARS_TERMOUT_ENABLED
        ),
        "bankruptcy_active": False,
    }

    return {
        "model": MODEL_NAME,
        "scenario": scenario["name"],
        "seed": seed,
        "population": population,
        "steps": steps,
        "firm_count": firm_count,
        "overrides": scenario["overrides"],
        "applied_parameters": scenario["applied_parameters"],
        "output_dir": output_dir,
        "diagnostics_csv": diagnostics_csv,
        "firm_diagnostics_csv": firm_diagnostics_csv,
        "ledger_csv": ledger_csv,
        "version_identifier": git_version_identifier(),
        "time_schema": "weekly",
        "steps_per_year": 52,
        "flow_frequency": "weekly",
        "analysis_start_global_step": analysis_start_global_step,
        "analysis_end_global_step": steps,
        "elapsed_weeks": max(0, steps - analysis_start_global_step),
        "initial_age_phase_mode": initial_age_phase_mode,
        "initial_age_phase_schema": "dedicated_initial_age_phase_rng_v1",
        "firm_timing_contract": timing_contract,
    }


def export_manifest(path, manifest):
    if not path:
        return

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
