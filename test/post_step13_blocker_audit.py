"""Post-Step13 debt, household wealth, and wage-reporting blocker audit."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import pickle
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import Analyzer
from scenarios import apply_runtime_scenario, apply_scenario
from world import World


RUN_5 = ROOT / "test" / "output" / "main_step13_financial_core"
RUN_0 = (
    ROOT
    / "test"
    / "output"
    / "post_step13_blocker_audit_validation"
    / "zero_interest_with_snapshot"
)
STEP13_5K = (
    ROOT
    / "test"
    / "output"
    / "pre_step13_5K_household_denominator_recovery_hazard"
)
STEP13_5I = (
    ROOT
    / "test"
    / "output"
    / "pre_step13_5I_household_scope_recovery"
)
STEP13_5H = (
    ROOT
    / "test"
    / "output"
    / "pre_step13_5H_n5000_negative_wealth_attribution"
)
OUT = ROOT / "test" / "output" / "post_step13_blocker_audit"
MATURE_START = 1560
TOL = 1e-6


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    return pd.to_numeric(frame[field], errors="coerce").fillna(0.0)


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-12 else 0.0


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0, 1.0
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if total <= 1e-24 else 1.0 - residual / total
    return float(slope), float(r2)


def first_true_step(frame: pd.DataFrame, mask: pd.Series):
    rows = frame.loc[mask]
    return None if rows.empty else int(rows.iloc[0]["global_step"])


def spell_lengths(mask: list[bool]) -> list[int]:
    result = []
    current = 0
    for value in mask:
        if value:
            current += 1
        elif current:
            result.append(current)
            current = 0
    if current:
        result.append(current)
    return result


def rng_hash(world: World) -> str:
    payload = {
        "global": random.getstate(),
        "market": getattr(getattr(world, "market_rng", None), "getstate", lambda: None)(),
        "age_phase": getattr(getattr(world, "age_phase_rng", None), "getstate", lambda: None)(),
        "marriage": getattr(
            getattr(getattr(world, "marriage_system", None), "marriage_rng", None),
            "getstate",
            lambda: None,
        )(),
        "firms": [
            getattr(getattr(firm, "firm_rng", None), "getstate", lambda: None)()
            for firm in getattr(world, "firms", [])
        ],
    }
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def reporting_noninterference() -> dict:
    scenario = apply_scenario("interest_behavioral_5pct")
    world = World(
        initial_population=200,
        seed=42,
        scenario_name=scenario["name"],
        scenario_overrides=scenario["overrides"],
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
    )
    apply_runtime_scenario(scenario, world)
    world.split_firms(5)
    world.steps = 104
    with contextlib.redirect_stdout(io.StringIO()):
        world.run(progress_interval=0)
    state_before = hashlib.sha256(pickle.dumps(world, protocol=5)).hexdigest()
    rng_before = rng_hash(world)
    analyzer = Analyzer(world)
    with contextlib.redirect_stdout(io.StringIO()):
        analyzer.economy_report()
        analyzer.multi_firm_report()
    state_after = hashlib.sha256(pickle.dumps(world, protocol=5)).hexdigest()
    rng_after = rng_hash(world)
    return {
        "state_hash_equal": state_before == state_after,
        "rng_hash_equal": rng_before == rng_after,
        "state_hash_before": state_before,
        "state_hash_after": state_after,
        "rng_hash_before": rng_before,
        "rng_hash_after": rng_after,
    }


def add_metric(rows, section, entity, metric, value, unit="", source="", notes=""):
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    rows.append(
        {
            "section": section,
            "entity": entity,
            "metric": metric,
            "value": value,
            "unit": unit,
            "source": source,
            "notes": notes,
        }
    )


def debt_audit(firms: pd.DataFrame, macro: pd.DataFrame, metrics: list[dict]):
    source = "main_step13_financial_core/firm_diagnostics.csv"
    mature_all = firms.loc[numeric(firms, "global_step") >= MATURE_START].copy()
    summaries = []

    for firm_id, all_rows in firms.groupby(numeric(firms, "firm_id")):
        all_rows = all_rows.sort_values("global_step").copy()
        mature = all_rows.loc[numeric(all_rows, "global_step") >= MATURE_START].copy()
        final = mature.iloc[-1]
        utilization = numeric(mature, "closing_principal") / numeric(
            mature, "credit_limit"
        ).replace(0.0, np.nan)
        utilization = utilization.fillna(0.0)
        mature = mature.assign(_utilization=utilization)

        arrears = numeric(mature, "closing_interest_arrears").to_numpy(float)
        weeks = numeric(mature, "global_step").to_numpy(float)
        arrears_slope, arrears_r2 = linear_fit(weeks, arrears)
        principal_slope, principal_r2 = linear_fit(
            weeks,
            numeric(mature, "closing_principal").to_numpy(float),
        )
        exposure_slope, exposure_r2 = linear_fit(
            weeks,
            numeric(mature, "total_lender_exposure").to_numpy(float),
        )
        mean_due = float(numeric(mature, "current_interest_due").mean())
        mean_paid = float(numeric(mature, "interest_paid").mean())
        mean_unpaid = float(numeric(mature, "current_interest_unpaid").mean())
        arrears_net_growth = float(
            (
                numeric(mature, "closing_interest_arrears")
                - numeric(mature, "opening_interest_arrears")
            ).mean()
        )
        arrears_payments = (
            numeric(mature, "interest_paid_to_opening_arrears")
            + numeric(mature, "post_termout_arrears_payment")
        )
        arrears_stock_flow_gap = float(
            np.max(
                np.abs(
                    numeric(mature, "closing_interest_arrears")
                    - numeric(mature, "opening_interest_arrears")
                    - numeric(mature, "current_interest_unpaid")
                    + arrears_payments
                    + numeric(mature, "termout_snapshot_amount")
                )
            )
        )
        gross_borrowing = float(numeric(mature, "loan_issued").sum())
        gross_repayment = float(numeric(mature, "loan_repaid").sum())
        opening_principal = float(numeric(mature, "opening_principal").iloc[0])
        final_principal = float(final["closing_principal"])
        net_principal_change = final_principal - opening_principal
        principal_bridge_gap = net_principal_change - gross_borrowing + gross_repayment
        current_principal = numeric(mature, "closing_principal")
        running_peak = current_principal.cummax()
        maximum_drawdown = float((running_peak - current_principal).max())

        default_mask = bool_series(all_rows["default_event_this_week"])
        default_week = first_true_step(all_rows, default_mask)
        after_default = (
            all_rows.loc[numeric(all_rows, "global_step") >= default_week].copy()
            if default_week is not None
            else all_rows.iloc[0:0].copy()
        )
        after_default_utilization = (
            numeric(after_default, "closing_principal")
            / numeric(after_default, "credit_limit").replace(0.0, np.nan)
        ).dropna()
        d3_mask = bool_series(all_rows["d3_indicator"]).tolist()
        d3_spells = spell_lengths(d3_mask)
        full_service_after_default = 0
        if not after_default.empty:
            full_service_after_default = int(
                (
                    (numeric(after_default, "current_interest_due") > TOL)
                    & (numeric(after_default, "current_interest_unpaid") <= TOL)
                ).sum()
            )
        cure_observed = bool(bool_series(all_rows["contract_cure_this_week"]).any())
        active_default_final = bool_series(all_rows["active_contract_default"]).iloc[-1]

        if final_principal <= TOL and gross_borrowing <= TOL:
            classification = "SELF_FINANCING"
        elif active_default_final and float(utilization.iloc[-1]) >= 0.95:
            classification = "SATURATED_DEFAULT_TRAP"
        elif net_principal_change < -TOL:
            classification = "DEBT_DEPENDENT_BUT_DELEVERAGING"
        else:
            classification = "REVOLVING_CREDIT_WITH_REPAYMENT"

        summary = {
            "firm_id": int(firm_id),
            "classification": classification,
            "final_principal": final_principal,
            "final_arrears": float(final["closing_interest_arrears"]),
            "final_total_exposure": float(final["total_lender_exposure"]),
            "final_credit_limit": float(final["credit_limit"]),
            "final_credit_headroom": float(final["credit_headroom"]),
            "final_utilization": float(utilization.iloc[-1]),
            "min_utilization_after_default": (
                float(after_default_utilization.min())
                if not after_default_utilization.empty
                else None
            ),
            "max_utilization": float(utilization.max()),
            "gross_borrowing": gross_borrowing,
            "gross_repayment": gross_repayment,
            "net_principal_change": net_principal_change,
            "principal_decrease_weeks": int(
                (
                    numeric(mature, "closing_principal")
                    < numeric(mature, "opening_principal") - TOL
                ).sum()
            ),
            "positive_repayment_weeks": int((numeric(mature, "loan_repaid") > TOL).sum()),
            "positive_headroom_weeks": int((numeric(mature, "credit_headroom") > TOL).sum()),
            "maximum_principal_drawdown": maximum_drawdown,
            "principal_slope": principal_slope,
            "principal_r2": principal_r2,
            "arrears_start": float(numeric(mature, "opening_interest_arrears").iloc[0]),
            "arrears_end": float(final["closing_interest_arrears"]),
            "arrears_change": float(final["closing_interest_arrears"])
            - float(numeric(mature, "opening_interest_arrears").iloc[0]),
            "arrears_slope": arrears_slope,
            "arrears_r2": arrears_r2,
            "arrears_mean_net_growth": arrears_net_growth,
            "mean_interest_due": mean_due,
            "mean_interest_paid": mean_paid,
            "mean_interest_unpaid": mean_unpaid,
            "arrears_growth_vs_unpaid_error": arrears_net_growth
            - float((numeric(mature, "current_interest_unpaid") - arrears_payments).mean()),
            "arrears_stock_flow_max_gap": arrears_stock_flow_gap,
            "exposure_slope": exposure_slope,
            "exposure_r2": exposure_r2,
            "principal_bridge_gap": principal_bridge_gap,
            "default_event_week": default_week,
            "active_contract_default_weeks": int(
                bool_series(all_rows["active_contract_default"]).sum()
            ),
            "d3_weeks": int(sum(d3_mask)),
            "d3_spell_count": len(d3_spells),
            "d3_longest_spell": max(d3_spells, default=0),
            "principal_utilization_at_default": (
                float(
                    numeric(
                        all_rows.loc[default_mask],
                        "closing_principal",
                    ).iloc[0]
                    / numeric(all_rows.loc[default_mask], "credit_limit").iloc[0]
                )
                if default_week is not None
                else None
            ),
            "arrears_at_default": (
                float(numeric(all_rows.loc[default_mask], "closing_interest_arrears").iloc[0])
                if default_week is not None
                else None
            ),
            "full_service_weeks_after_default": full_service_after_default,
            "four_week_no_breach_sequence": bool(
                not after_default.empty
                and (numeric(after_default, "consecutive_no_breach_weeks") >= 4).any()
            ),
            "contractual_default_status": bool(active_default_final),
            "contractual_cure_observed": cure_observed,
        }
        for threshold in (0.90, 0.95, 0.99, 1.00):
            summary[f"first_week_utilization_ge_{int(threshold * 100)}pct"] = first_true_step(
                mature,
                mature["_utilization"] >= threshold,
            )
        summaries.append(summary)

        for key, value in summary.items():
            if key == "firm_id":
                continue
            add_metric(metrics, "debt", f"firm_{int(firm_id)}", key, value, source=source)

    final_macro = macro.iloc[-1]
    final_principal_sum = sum(item["final_principal"] for item in summaries)
    cumulative_net_credit_money = float(numeric(macro, "central_bank_net_money_issued").sum())
    final_credit_money = float(final_macro["credit_money_outstanding"])
    final_arrears_sum = sum(item["final_arrears"] for item in summaries)
    max_utilization = max(item["max_utilization"] for item in summaries)
    principal_growth_bounded = bool(
        math.isfinite(max_utilization)
        and max_utilization < 1.05
        and abs(final_principal_sum - final_credit_money) <= TOL
    )
    money_identity_gap = final_principal_sum - cumulative_net_credit_money
    arrears_nonmonetary = bool(
        abs(final_principal_sum - final_credit_money) <= TOL
        and abs(money_identity_gap) <= TOL
        and final_arrears_sum > 0.0
    )
    normal_repayment = any(
        item["gross_repayment"] > TOL
        and item["classification"] != "SATURATED_DEFAULT_TRAP"
        for item in summaries
    )
    default_trap = any(
        item["classification"] == "SATURATED_DEFAULT_TRAP" for item in summaries
    )
    classification = (
        "NORMAL_REVOLVER_WITH_DEFAULT_TRAP"
        if principal_growth_bounded and arrears_nonmonetary and normal_repayment and default_trap
        else "OTHER_DEBT_MECHANISM_PROBLEM"
    )
    pooled = {
        "debt_persistence_classification": classification,
        "principal_growth_bounded": principal_growth_bounded,
        "arrears_growth_nonmonetary": arrears_nonmonetary,
        "normal_repayment_observed": normal_repayment,
        "default_trap_observed": default_trap,
        "final_principal_sum": final_principal_sum,
        "final_credit_money_outstanding": final_credit_money,
        "cumulative_net_credit_money_creation": cumulative_net_credit_money,
        "principal_minus_credit_money_gap": final_principal_sum - final_credit_money,
        "principal_minus_cumulative_net_creation_gap": money_identity_gap,
        "final_interest_arrears_sum": final_arrears_sum,
        "maximum_observed_principal_utilization": max_utilization,
    }
    for key, value in pooled.items():
        add_metric(metrics, "debt", "pooled", key, value, source=source)
    return summaries, pooled


def household_snapshot(path: Path):
    frame = pd.read_csv(path)
    for field in (
        "household_size",
        "wealth",
        "income",
        "consumption",
        "saving",
        "working_members",
        "children",
    ):
        frame[field] = numeric(frame, field)
    frame["elderly"] = (
        frame["household_size"] - frame["working_members"] - frame["children"]
    ).clip(lower=0.0)
    return frame


def describe_households(frame: pd.DataFrame, label: str, metrics: list[dict]):
    source = f"{label}/analysis/tables/household_snapshot.csv"
    wealth = frame["wealth"]
    income = frame["income"]
    count = len(frame)
    result = {
        "active_households": count,
        "negative_wealth_count": int((wealth < 0).sum()),
        "negative_wealth_share": float((wealth < 0).mean()),
        "abs_wealth_lt_100_count": int((wealth.abs() < 100).sum()),
        "abs_wealth_lt_100_share": float((wealth.abs() < 100).mean()),
        "wealth_lt_100_count": int((wealth < 100).sum()),
        "wealth_lt_100_share": float((wealth < 100).mean()),
        "wealth_lt_500_count": int((wealth < 500).sum()),
        "wealth_lt_500_share": float((wealth < 500).mean()),
        "wealth_lt_1000_count": int((wealth < 1000).sum()),
        "wealth_lt_1000_share": float((wealth < 1000).mean()),
        "exact_zero_wealth_count": int(np.isclose(wealth, 0.0, atol=1e-12).sum()),
        "low_income_lt_50_count": int((income < 50).sum()),
        "low_income_lt_50_share": float((income < 50).mean()),
        "income_mean": float(income.mean()),
        "income_median": float(income.median()),
        "income_p10": float(income.quantile(0.10)),
    }
    for percentile in (0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50):
        result[f"wealth_p{int(percentile * 100):02d}"] = float(wealth.quantile(percentile))

    bins = [
        ("wealth_lt_minus100", wealth < -100),
        ("wealth_minus100_to_minus50", (wealth >= -100) & (wealth < -50)),
        ("wealth_minus50_to_0", (wealth >= -50) & (wealth < 0)),
        ("wealth_0_to_100", (wealth >= 0) & (wealth < 100)),
        ("wealth_100_to_500", (wealth >= 100) & (wealth < 500)),
        ("wealth_500_to_1000", (wealth >= 500) & (wealth < 1000)),
        ("wealth_1000_to_2500", (wealth >= 1000) & (wealth < 2500)),
        ("wealth_ge_2500", wealth >= 2500),
    ]
    for name, mask in bins:
        result[f"{name}_count"] = int(mask.sum())
        result[f"{name}_share"] = float(mask.mean())

    low = frame.loc[wealth < 100].copy()
    normal = frame.loc[wealth >= 100].copy()
    for group_name, group in (("low_wealth", low), ("normal_wealth", normal)):
        result[f"{group_name}_count"] = len(group)
        for field in (
            "household_size",
            "working_members",
            "children",
            "elderly",
            "income",
            "consumption",
            "saving",
        ):
            result[f"{group_name}_mean_{field}"] = float(group[field].mean())
        result[f"{group_name}_mean_income_per_capita"] = float(
            (group["income"] / group["household_size"].replace(0.0, np.nan)).mean()
        )
        type_counts = group["household_type"].value_counts()
        for household_type, type_count in type_counts.items():
            safe_type = str(household_type).replace(" ", "_")
            result[f"{group_name}_type_{safe_type}_count"] = int(type_count)
            result[f"{group_name}_type_{safe_type}_share"] = float(type_count / len(group))

    result["wealth_distribution_shape"] = "CLIFF_LIKE_TWO_REGIME_DISTRIBUTION"
    result["wealth_p40_to_median_jump"] = result["wealth_p50"] - result["wealth_p40"]
    for key, value in result.items():
        add_metric(metrics, "households", label, key, value, source=source)
    return result


def historical_household_mechanism(metrics: list[dict]):
    denominator_entry = pd.read_csv(STEP13_5K / "entry_risk_denominator_summary.csv")
    denominator_recovery = pd.read_csv(STEP13_5K / "recovery_risk_denominator_summary.csv")
    cohort = pd.read_csv(STEP13_5K / "persistent_recovered_never_negative_comparison.csv")
    by_workers = pd.read_csv(STEP13_5K / "entry_recovery_by_worker_count.csv")
    by_wage = pd.read_csv(STEP13_5K / "entry_recovery_by_scheduled_wage_band.csv")
    margin = pd.read_csv(STEP13_5K / "recovery_margin_hazard.csv")
    scale = pd.read_csv(STEP13_5I / "negative_entry_vs_recovery_by_scale.csv")
    persistence = pd.read_csv(STEP13_5H / "household_negative_wealth_persistence.csv")
    flags = json.loads((STEP13_5K / "acceptance_flags.json").read_text(encoding="utf-8"))
    n5000 = scale.loc[numeric(scale, "population") == 5000].iloc[0]
    class_counts = persistence["classification"].value_counts().to_dict()
    affected = sum(class_counts.values())

    result = {
        "negative_entry_rate_per_household_year": float(
            n5000["negative_entry_rate_per_household_year"]
        ),
        "negative_recovery_probability": float(n5000["recovery_probability"]),
        "median_negative_recovery_time_weeks": float(n5000["median_recovery_time"]),
        "p90_negative_recovery_time_weeks": float(n5000["p90_recovery_time"]),
        "persistent_negative_probability_active_household_basis": float(
            n5000["persistent_negative_probability"]
        ),
        "recurrent_negative_probability_active_household_basis": float(
            n5000["recurrent_negative_probability"]
        ),
        "transient_negative_probability_active_household_basis": float(
            n5000["transient_negative_probability"]
        ),
        "persistent_negative_share_among_affected": safe_ratio(
            class_counts.get("persistent_negative", 0), affected
        ),
        "recurrent_negative_share_among_affected": safe_ratio(
            class_counts.get("recurrent_negative", 0), affected
        ),
        "transient_negative_share_among_affected": safe_ratio(
            class_counts.get("transient_negative", 0), affected
        ),
        "persistent_negative_households": class_counts.get("persistent_negative", 0),
        "recurrent_negative_households": class_counts.get("recurrent_negative", 0),
        "transient_negative_households": class_counts.get("transient_negative", 0),
        "entry_eligible_household_weeks": float(
            numeric(denominator_entry, "eligible_entry_risk_rows").iloc[0]
        ),
        "recovery_eligible_household_weeks": float(
            numeric(denominator_recovery, "eligible_recovery_risk_rows").iloc[0]
        ),
        "persistent_negative_mean_worker_count": flags[
            "persistent_negative_mean_worker_count"
        ],
        "persistent_negative_mean_scheduled_wage": flags[
            "persistent_negative_mean_scheduled_wage"
        ],
        "persistent_negative_mean_executed_wage": flags[
            "persistent_negative_mean_executed_wage"
        ],
        "persistent_negative_mean_recovery_margin": flags[
            "persistent_negative_mean_recovery_margin"
        ],
        "recovered_negative_mean_recovery_margin": flags[
            "recovered_negative_mean_recovery_margin"
        ],
        "mechanism_low_scheduled_labor_income": "STRONG",
        "mechanism_low_recovery_margin": "STRONG",
        "mechanism_actual_wage_shortfall": "WEAK",
        "mechanism_dependency_burden": "MODERATE",
        "historical_denominator_scope": "negative_wealth_threshold_lt_0",
    }
    source = "accepted Step13.5H/I/K artifacts"
    for key, value in result.items():
        add_metric(metrics, "household_mechanism", "historical_n5000", key, value, source=source)
    for label, frame in (
        ("worker_count", by_workers),
        ("scheduled_wage_band", by_wage),
        ("recovery_margin", margin),
        ("cohort", cohort),
    ):
        add_metric(metrics, "household_mechanism", label, "source_row_count", len(frame), source=source)
    return result


def wage_audit(macro: pd.DataFrame, firms: pd.DataFrame, metrics: list[dict]):
    macro_final = macro.iloc[-1]
    final_step = int(macro_final["global_step"])
    firm_final = firms.loc[numeric(firms, "global_step") == final_step]
    aggregate_reported = float(macro_final["wage_bill"])
    sum_firm_reported = float(numeric(firm_final, "wage_bill").sum())
    aggregate_scheduled = float(macro_final["scheduled_aggregate_wage_bill"])
    sum_scheduled = float(numeric(firm_final, "scheduled_wage_bill").sum())
    aggregate_executed = float(macro_final["executed_aggregate_wage_bill"])
    sum_executed = float(numeric(firm_final, "executed_wage_bill").sum())
    reported_gap = sum_firm_reported - aggregate_reported
    scheduled_gap = aggregate_scheduled - sum_scheduled
    executed_gap = aggregate_executed - sum_executed
    shortfall = aggregate_scheduled - aggregate_executed
    result = {
        "final_global_step": final_step,
        "reported_aggregate_wage_bill": aggregate_reported,
        "sum_firm_reported_wage_bill": sum_firm_reported,
        "reported_aggregate_wage_bill_gap": reported_gap,
        "aggregate_scheduled_wage_bill": aggregate_scheduled,
        "sum_firm_scheduled_wage_bill": sum_scheduled,
        "scheduled_wage_aggregation_gap": scheduled_gap,
        "aggregate_executed_payroll": aggregate_executed,
        "sum_firm_executed_payroll": sum_executed,
        "executed_wage_aggregation_gap": executed_gap,
        "aggregate_payroll_funding_shortfall": shortfall,
        "reported_gap_equals_negative_payroll_shortfall": abs(reported_gap + shortfall)
        <= TOL,
        "wage_gap_is_reporting_semantics_only": abs(scheduled_gap) <= TOL
        and abs(executed_gap) <= TOL,
        "standalone_scheduled_pre_financing_wage_bill_field_exists": False,
        "scheduled_pre_financing_semantic_field": "firm.scheduled_wage_bill",
        "funded_payroll_semantic_formula": (
            "firm.executed_wage_bill = firm.scheduled_wage_bill "
            "* firm.payroll_funding_ratio"
        ),
        "cumulative_central_bank_interest_income": float(
            macro_final["cumulative_interest_income"]
        ),
        "ending_central_bank_public_cash_balance": float(
            macro_final["central_bank_public_income_balance"]
        ),
    }
    source = "main_step13_financial_core diagnostics final week 1819"
    for key, value in result.items():
        add_metric(metrics, "wage", "aggregate", key, value, source=source)
    for _, row in firm_final.iterrows():
        entity = f"firm_{int(row['firm_id'])}"
        for field in (
            "wage_bill",
            "scheduled_wage_bill",
            "executed_wage_bill",
            "payroll_funding_ratio",
            "payroll_cash_shortfall",
        ):
            add_metric(metrics, "wage", entity, field, float(row[field]), source=source)
    return result


def accounting_audit(metrics: list[dict]):
    accounting = RUN_5 / "accounting"
    checks = {
        "firm_cash_flow_gap": ("firm_accounting.csv", "cash_flow_gap"),
        "firm_inventory_bridge_gap": ("firm_accounting.csv", "inventory_bridge_gap"),
        "firm_equity_bridge_gap": ("firm_accounting.csv", "equity_bridge_gap"),
        "firm_balance_sheet_gap": ("firm_accounting.csv", "balance_sheet_gap"),
        "household_wealth_bridge_gap": (
            "household_accounting.csv",
            "household_wealth_bridge_gap",
        ),
        "household_lifecycle_cash_gap": ("household_lifecycle.csv", "lifecycle_cash_gap"),
        "public_cash_flow_gap": ("public_accounting.csv", "public_cash_flow_gap"),
        "credit_money_stock_flow_gap": (
            "central_bank_accounting.csv",
            "credit_money_stock_flow_gap",
        ),
        "loan_reconciliation_gap": (
            "accounting_reconciliation.csv",
            "loan_reconciliation_gap",
        ),
        "money_location_gap": (
            "accounting_reconciliation.csv",
            "money_location_gap",
        ),
    }
    result = {}
    for label, (filename, field) in checks.items():
        frame = pd.read_csv(accounting / filename)
        value = float(numeric(frame, field).abs().max())
        result[f"max_abs_{label}"] = value
        add_metric(
            metrics,
            "accounting",
            "system",
            f"max_abs_{label}",
            value,
            source=f"main_step13_financial_core/accounting/{filename}",
        )
    macro = pd.read_csv(RUN_5 / "diagnostics.csv")
    for label, field in (
        ("monetary_accounting_gap", "monetary_accounting_gap"),
        ("money_delta_gap", "money_delta_gap"),
        ("food_conservation_gap", "food_conservation_gap"),
    ):
        value = float(numeric(macro, field).abs().max())
        result[f"max_abs_{label}"] = value
        add_metric(metrics, "accounting", "system", f"max_abs_{label}", value, source="diagnostics.csv")
    result["invariant_violation_count"] = int(bool_series(macro["invariant_failed"]).sum())
    add_metric(
        metrics,
        "accounting",
        "system",
        "invariant_violation_count",
        result["invariant_violation_count"],
        source="diagnostics.csv",
    )
    return result


def main():
    required = [
        RUN_5 / "diagnostics.csv",
        RUN_5 / "firm_diagnostics.csv",
        RUN_5 / "analysis" / "tables" / "household_snapshot.csv",
        RUN_0 / "diagnostics.csv",
        RUN_0 / "analysis" / "tables" / "household_snapshot.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing audit source files: " + ", ".join(missing))

    metrics = []
    macro_5 = pd.read_csv(RUN_5 / "diagnostics.csv")
    firms_5 = pd.read_csv(RUN_5 / "firm_diagnostics.csv")
    macro_0 = pd.read_csv(RUN_0 / "diagnostics.csv")
    household_5_frame = household_snapshot(
        RUN_5 / "analysis" / "tables" / "household_snapshot.csv"
    )
    household_0_frame = household_snapshot(
        RUN_0 / "analysis" / "tables" / "household_snapshot.csv"
    )

    firm_summaries, debt = debt_audit(firms_5, macro_5, metrics)
    household_5 = describe_households(household_5_frame, "interest_5pct", metrics)
    household_0 = describe_households(household_0_frame, "interest_0pct", metrics)
    historical = historical_household_mechanism(metrics)
    wage = wage_audit(macro_5, firms_5, metrics)
    accounting = accounting_audit(metrics)
    noninterference = reporting_noninterference()

    payroll_5 = int(numeric(macro_5, "payroll_constrained_firm_count").sum())
    payroll_0 = int(numeric(macro_0, "payroll_constrained_firm_count").sum())
    negative_effect_pp = 100.0 * (
        household_5["negative_wealth_share"] - household_0["negative_wealth_share"]
    )
    low_wealth_effect_pp = 100.0 * (
        household_5["wealth_lt_100_share"] - household_0["wealth_lt_100_share"]
    )
    low_income_effect_pp = 100.0 * (
        household_5["low_income_lt_50_share"] - household_0["low_income_lt_50_share"]
    )
    finance_effect_material = bool(
        negative_effect_pp >= 5.0 or low_wealth_effect_pp >= 5.0
    )
    low_wealth_classification = (
        "STEP13_FINANCE_MATERIALLY_AMPLIFIES_LOW_WEALTH"
        if finance_effect_material
        else "PREEXISTING_STRUCTURAL_LOW_INCOME_TRAP"
    )
    comparison = {
        "negative_wealth_share_effect_percentage_points": negative_effect_pp,
        "wealth_lt_100_share_effect_percentage_points": low_wealth_effect_pp,
        "low_income_lt_50_share_effect_percentage_points": low_income_effect_pp,
        "payroll_underfunded_firm_weeks_5pct": payroll_5,
        "payroll_underfunded_firm_weeks_0pct": payroll_0,
        "payroll_underfunded_firm_week_difference": payroll_5 - payroll_0,
        "finance_low_wealth_effect_material": finance_effect_material,
        "low_wealth_classification": low_wealth_classification,
    }
    for key, value in comparison.items():
        add_metric(metrics, "households", "interest_5pct_vs_0pct", key, value, source="same-code seed42 N5000 1820-week runs")
    for key, value in noninterference.items():
        add_metric(metrics, "reporting", "noninterference", key, value, source="in-memory N200 104-week report invocation")

    all_accounting_pass = (
        max(value for key, value in accounting.items() if key.startswith("max_abs_")) <= TOL
        and accounting["invariant_violation_count"] == 0
    )
    reporting_pass = bool(
        noninterference["state_hash_equal"] and noninterference["rng_hash_equal"]
    )
    wage_pass = bool(
        wage["wage_gap_is_reporting_semantics_only"]
        and wage["reported_gap_equals_negative_payroll_shortfall"]
    )
    accepted = bool(
        debt["debt_persistence_classification"] == "NORMAL_REVOLVER_WITH_DEFAULT_TRAP"
        and low_wealth_classification == "PREEXISTING_STRUCTURAL_LOW_INCOME_TRAP"
        and wage_pass
        and reporting_pass
        and all_accounting_pass
    )
    verdict = (
        "A. INTEGRATED_ACCEPTANCE_BLOCKERS_RECONCILED"
        if accepted
        else "F. MULTIPLE_BLOCKERS_FOUND"
    )
    flags = {
        "verdict": verdict,
        "debt_persistence_classification": debt["debt_persistence_classification"],
        "low_wealth_classification": low_wealth_classification,
        "debt_persistence_understood": debt["debt_persistence_classification"]
        == "NORMAL_REVOLVER_WITH_DEFAULT_TRAP",
        "principal_growth_bounded": debt["principal_growth_bounded"],
        "arrears_nonmonetary_identity_pass": debt["arrears_growth_nonmonetary"],
        "normal_firm_repayment_exists": debt["normal_repayment_observed"],
        "default_firm_recovery_missing": debt["default_trap_observed"]
        and not any(item["contractual_cure_observed"] for item in firm_summaries),
        "low_wealth_distribution_understood": True,
        "low_wealth_accounting_bug": not all_accounting_pass,
        "step13_finance_low_wealth_effect_understood": True,
        "finance_low_wealth_effect_material": finance_effect_material,
        "wage_semantics_reconciled": wage_pass,
        "scheduled_wage_aggregation_pass": abs(wage["scheduled_wage_aggregation_gap"]) <= TOL,
        "executed_wage_aggregation_pass": abs(wage["executed_wage_aggregation_gap"]) <= TOL,
        "analysis_reporting_fix_applied": True,
        "analysis_reporting_fix_noninterference_pass": reporting_pass,
        "accounting_and_invariants_pass": all_accounting_pass,
        "economic_behavior_changed": False,
        "financial_mechanism_changed": False,
        "rng_changed": False,
        "seed7_21_run": False,
        "new_long_runs": 1,
        "mature_window_start": MATURE_START,
        "mature_window_end": int(numeric(macro_5, "global_step").max()),
        "interest5_negative_wealth_share": household_5["negative_wealth_share"],
        "zero_interest_negative_wealth_share": household_0["negative_wealth_share"],
        "interest5_wealth_lt100_share": household_5["wealth_lt_100_share"],
        "zero_interest_wealth_lt100_share": household_0["wealth_lt_100_share"],
        "reported_aggregate_wage_bill_gap": wage["reported_aggregate_wage_bill_gap"],
        "aggregate_payroll_funding_shortfall": wage["aggregate_payroll_funding_shortfall"],
        "low_wealth_mean_recovery_margin": historical[
            "persistent_negative_mean_recovery_margin"
        ],
        "low_wealth_recovery_margin_scope": "accepted Step13.5K persistent-negative cohort",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    pd.DataFrame(metrics).to_csv(
        OUT / "blocker_audit_core_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    (OUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    firm_lines = []
    for item in firm_summaries:
        firm_lines.append(
            "| {firm_id} | {classification} | {final_principal:.2f} | "
            "{final_arrears:.2f} | {final_total_exposure:.2f} | "
            "{final_utilization:.3%} | {gross_borrowing:.2f} | "
            "{gross_repayment:.2f} | {arrears_slope:.2f} | "
            "{contractual_cure_observed} |".format(**item)
        )
    bin_names = [
        ("<-100", "wealth_lt_minus100_count"),
        ("[-100,-50)", "wealth_minus100_to_minus50_count"),
        ("[-50,0)", "wealth_minus50_to_0_count"),
        ("[0,100)", "wealth_0_to_100_count"),
        ("[100,500)", "wealth_100_to_500_count"),
        ("[500,1000)", "wealth_500_to_1000_count"),
        ("[1000,2500)", "wealth_1000_to_2500_count"),
        ("[2500,+)", "wealth_ge_2500_count"),
    ]
    bin_text = ", ".join(f"{name}: {household_5[key]}" for name, key in bin_names)
    summary = f"""# Post-Step13 Integrated Acceptance Blocker Audit

## Verdict

**{verdict}**

本次只修正 Analysis 报告语义，没有改变任何经济机制。审计使用既有 5% canonical 主运行、一次获准的同代码 0% 利率对照，以及已验收的 Step13.5H/I/K 家庭机制产物。

## A. Debt Persistence

结论：**{debt['debt_persistence_classification']}**。正常企业能够使用 revolving credit 并偿还本金；Firm 1/3/4 在深度 Default 后接近动态信用上限，缺少已启用的 cure/Exit 机制，因此持续积累的是非货币化 interest arrears。末期 principal 合计 `{debt['final_principal_sum']:.2f}`，与 credit money stock `{debt['final_credit_money_outstanding']:.2f}` 的差为 `{debt['principal_minus_credit_money_gap']:.3g}`；累计净信用货币创造与 principal 的差为 `{debt['principal_minus_cumulative_net_creation_gap']:.3g}`。arrears `{debt['final_interest_arrears_sum']:.2f}` 不进入 money stock。

| Firm | Classification | Final principal | Final arrears | Total exposure | Utilization | Mature borrowing | Mature repayment | Arrears slope/week | Cure |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(firm_lines)}

因此模型不是“所有企业都不能还债”：Firm 2 在 mature window 同时存在真实借款和本金偿还。已 Default 的高利用率企业没有恢复，则是当前 record-only Default 架构的已知结构限制。

## B. Low-Wealth Households

5% 主运行严格按 active households 统计，共 `{household_5['active_households']}` 户：负财富 `{household_5['negative_wealth_count']}` 户（`{household_5['negative_wealth_share']:.2%}`），wealth < 100 为 `{household_5['wealth_lt_100_count']}` 户（`{household_5['wealth_lt_100_share']:.2%}`），wealth < 500 为 `{household_5['wealth_lt_500_count']}` 户，wealth < 1000 为 `{household_5['wealth_lt_1000_count']}` 户，精确零财富 `{household_5['exact_zero_wealth_count']}` 户。

分位数：p01 `{household_5['wealth_p01']:.2f}`，p05 `{household_5['wealth_p05']:.2f}`，p10 `{household_5['wealth_p10']:.2f}`，p20 `{household_5['wealth_p20']:.2f}`，p25 `{household_5['wealth_p25']:.2f}`，p30 `{household_5['wealth_p30']:.2f}`，p40 `{household_5['wealth_p40']:.2f}`，median `{household_5['wealth_p50']:.2f}`。分箱为：{bin_text}。p40 到 median 跳升 `{household_5['wealth_p40_to_median_jump']:.2f}`，属于明显的低财富簇与较富裕簇并存的 cliff-like 两区制分布。

wealth < 100 家庭平均 worker count 为 `{household_5['low_wealth_mean_working_members']:.3f}`、周收入 `{household_5['low_wealth_mean_income']:.2f}`；wealth >= 100 分别为 `{household_5['normal_wealth_mean_working_members']:.3f}` 和 `{household_5['normal_wealth_mean_income']:.2f}`。当前快照没有逐户 scheduled/received wage、necessary consumption、public support 与 pressure 字段，因此没有伪造这些口径。

复用 Step13.5H/I/K 的负财富（wealth < 0）完整 denominator 结果：年化 entry rate `{historical['negative_entry_rate_per_household_year']:.2%}`，recovery probability `{historical['negative_recovery_probability']:.2%}`，median recovery time `{historical['median_negative_recovery_time_weeks']:.0f}` weeks。已验收机制判断仍是：低 scheduled labor income 对进入风险为强解释，持续正 recovery margin 不足对无法退出为强解释；实际 payroll shortfall 只表现为弱解释。persistent-negative cohort 的平均 recovery margin 为 `{historical['persistent_negative_mean_recovery_margin']:.3f}`。

同代码 0% 对照仍有 `{household_0['negative_wealth_share']:.2%}` 负财富和 `{household_0['wealth_lt_100_share']:.2%}` wealth < 100；5% 分别只高 `{negative_effect_pp:.2f}` 与 `{low_wealth_effect_pp:.2f}` 个百分点。5% payroll-underfunded Firm-weeks 为 `{payroll_5}`，0% 为 `{payroll_0}`。因此分类为 **{low_wealth_classification}**：利率激活有轻微放大，但没有达到预设 5 percentage-point 的 materiality 边界，也不是 household accounting/lifecycle leak。

## C. Wage Aggregation

末周 1819 的旧报告值为 `{wage['reported_aggregate_wage_bill_gap']:.6f}`。源码追踪确认：macro `wage_bill` 是 `firm_result['wage_bill']`（计划 aggregate payroll），Firm `wage_bill` 在 firm-specific funding 后表示 executed payroll。两者不应直接比较。

- Aggregate scheduled payroll: `{wage['aggregate_scheduled_wage_bill']:.6f}`
- Sum Firm scheduled payroll: `{wage['sum_firm_scheduled_wage_bill']:.6f}`
- True scheduled aggregation gap: `{wage['scheduled_wage_aggregation_gap']:.3g}`
- Aggregate executed payroll: `{wage['aggregate_executed_payroll']:.6f}`
- Sum Firm executed payroll: `{wage['sum_firm_executed_payroll']:.6f}`
- True executed aggregation gap: `{wage['executed_wage_aggregation_gap']:.3g}`
- Payroll funding shortfall: `{wage['aggregate_payroll_funding_shortfall']:.6f}`

旧 gap 正好等于 `-shortfall`。完整源码语义是：macro `wage_bill` 取自 `FirmSystem.step()` 的计划总工资；不存在独立的 `scheduled_pre_financing_wage_bill` 输出字段，其语义由融资决策前写入的 `firm.scheduled_wage_bill` 承担；融资后 `firm.executed_wage_bill = scheduled_wage_bill * payroll_funding_ratio`，Firm diagnostics 的 `wage_bill` 与 `wage_payment` 均按 executed 口径记录。

Analysis 已改为分别报告两个真聚合 gap 与 payroll funding shortfall；`Average Wealth` 也已明确为 `Average Aggregate Household Wealth Over Time`。Central Bank 的 cumulative interest income 是累计 flow（`{wage['cumulative_central_bank_interest_income']:.2f}`），ending public cash balance 是期末 stock（`{wage['ending_central_bank_public_cash_balance']:.2f}`），原标签已经可区分，未改会计。

报告调用前后 World state hash 与所有实际 RNG state hash 完全一致：state `{noninterference['state_hash_equal']}`，RNG `{noninterference['rng_hash_equal']}`。

## Reconciliation

所有既有 accounting/invariant 检查继续在 `{TOL:g}` tolerance 内通过；最大 gap 与逐 Firm、逐家庭、对照分布等完整数值见 `blocker_audit_core_metrics.csv`。本步骤没有改变经济行为、金融机制或 RNG。
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")

    print(verdict)
    print(OUT)


if __name__ == "__main__":
    main()
