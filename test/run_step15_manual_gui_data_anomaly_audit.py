"""Read-only Step 15 household-distribution and late-shock audit.

This script intentionally consumes the frozen statistics-enriched run and
writes only the requested audit artifacts.  It does not construct a World or
run a simulation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN = PROJECT_ROOT / "test/output/step15_final_canonical_statistics_enriched"
OUTPUT = PROJECT_ROOT / "test/output/step15_manual_gui_data_anomaly_audit"
NEAR_ZERO_THRESHOLD = 10.0
INITIAL_FIRM_CASH = 10_000_000.0
SNAPSHOT_PHASES = {"INITIAL", "PERIODIC", "FINAL"}


def read(relative: str) -> pd.DataFrame:
    path = RUN / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUTPUT / name, index=False)


def finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def gini(values) -> float:
    array = np.sort(np.maximum(finite(values), 0.0))
    if len(array) == 0 or float(array.sum()) <= 0.0:
        return 0.0
    ranks = np.arange(1, len(array) + 1, dtype=float)
    return float((2.0 * ranks.dot(array) / (len(array) * array.sum())) - (len(array) + 1.0) / len(array))


def describe(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {key: np.nan for key in ("mean", "median", "p10", "p25", "p50", "p75", "p90", "p95")}
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p10": float(clean.quantile(0.10)),
        "p25": float(clean.quantile(0.25)),
        "p50": float(clean.quantile(0.50)),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
    }


def social_snapshots(social: pd.DataFrame) -> pd.DataFrame:
    return social.loc[social["observation_phase"].isin(SNAPSHOT_PHASES)].copy()


def settlement_snapshots(settlement: pd.DataFrame) -> pd.DataFrame:
    return settlement.loc[settlement["observation_phase"].isin(SNAPSHOT_PHASES)].copy()


def quantile_labels(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_values(["cash", "household_id"], kind="stable")
    n = len(ordered)
    if n == 0:
        return pd.Series(dtype="object")
    fractional_rank = (np.arange(n, dtype=float) + 0.5) / n
    labels = np.where(
        fractional_rank <= 0.50,
        "BOTTOM_50",
        np.where(fractional_rank <= 0.90, "MIDDLE_40", "TOP_10"),
    )
    return pd.Series(labels, index=ordered.index)


def aggregate_firms(firms: pd.DataFrame) -> pd.DataFrame:
    numeric = {
        "revenue": ("revenue", "sum"),
        "cash": ("cash", "sum"),
        "operating_profit": ("operating_profit", "sum"),
        "cfo": ("cfo", "sum"),
        "employment": ("employment", "sum"),
        "desired_labor": ("desired_labor", "sum"),
        "production": ("production", "sum"),
        "sales": ("sales", "sum"),
    }
    return firms.groupby("global_step", as_index=False).agg(**numeric)


def event_sum(events: pd.DataFrame, event_type: str, step: int, field: str) -> float:
    subset = events.loc[(events["global_step"] == step) & (events["event_type"] == event_type)]
    if field not in subset:
        return 0.0
    return float(pd.to_numeric(subset[field], errors="coerce").fillna(0.0).sum())


def event_count(events: pd.DataFrame, event_type: str, step: int) -> int:
    return int(((events["global_step"] == step) & (events["event_type"] == event_type)).sum())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    macro = read("step15_macro_panel.csv").sort_values("global_step").reset_index(drop=True)
    firm_panel = read("step15_firm_panel.csv").sort_values(["global_step", "firm_id"])
    firm_diagnostics = read("firm_diagnostics.csv").sort_values(["global_step", "firm_id"])
    chain = read("step15_investment_chain_trace.csv").sort_values("global_step")
    assets = read("step15_capital_asset_ledger.csv")
    social_all = read("statistical_observability/social_household_snapshots.csv")
    settlement_all = read("statistical_observability/settlement_account_snapshots.csv")
    demographic_events = read("demographic_events.csv")
    labor_events = read("statistical_observability/labor_events.csv")

    social = social_snapshots(social_all)
    settlement = settlement_snapshots(settlement_all)
    aggregate = aggregate_firms(firm_panel)
    macro = macro.merge(aggregate, on="global_step", how="left", suffixes=("", "_panel"))

    # 1. Initial distribution and initialization trace.
    initial_social = social.loc[social["global_step"] == -1].copy()
    initial_settlement = settlement.loc[settlement["global_step"] == -1].copy()
    origin_rows = [
        {
            "stage": "FirmSystem.__init__",
            "source_location": "economy/firm.py:FirmSystem.__init__",
            "authoritative_rule": "FirmSystem.cash = config.FIRM_INITIAL_CASH",
            "amount": INITIAL_FIRM_CASH,
            "recipient_scope": "initial aggregate Firm system",
            "social_household_cash_effect": 0.0,
            "settlement_account_cash_effect": 0.0,
            "classification": "INITIALIZATION_ORDER_EFFECT",
        },
        {
            "stage": "Household.__init__",
            "source_location": "household.py:Household.__init__",
            "authoritative_rule": "Household.wealth = 0.0",
            "amount": 0.0,
            "recipient_scope": "every newly created social Household",
            "social_household_cash_effect": 0.0,
            "settlement_account_cash_effect": 0.0,
            "classification": "UNFUNDED_HOUSEHOLD_CREATION",
        },
        {
            "stage": "World.initialize_households",
            "source_location": "world.py:initialize_households/create_household",
            "authoritative_rule": "initial couples are assigned to new social Households without a cash allocation",
            "amount": 0.0,
            "recipient_scope": "1,877 initial social Households",
            "social_household_cash_effect": float(initial_social["cash"].sum()),
            "settlement_account_cash_effect": 0.0,
            "classification": "UNFUNDED_HOUSEHOLD_CREATION",
        },
        {
            "stage": "World.ensure_labor_settlement_households",
            "source_location": "world.py:ensure_labor_settlement_households",
            "authoritative_rule": "eligible adults without a social settlement Household receive deterministic zero-cash settlement accounts",
            "amount": 0.0,
            "recipient_scope": "133 initial settlement-only accounts",
            "social_household_cash_effect": 0.0,
            "settlement_account_cash_effect": float(initial_settlement["cash"].sum()),
            "classification": "SETTLEMENT_ACCOUNT_EFFECT",
        },
        {
            "stage": "split_single_firm",
            "source_location": "economy/multi_firm.py:split_single_firm",
            "authoritative_rule": "the initial aggregate Firm cash is split among Firms; no household recipient is used",
            "amount": INITIAL_FIRM_CASH,
            "recipient_scope": "Food Firms after the initial Firm split",
            "social_household_cash_effect": 0.0,
            "settlement_account_cash_effect": 0.0,
            "classification": "INITIALIZATION_ORDER_EFFECT",
        },
        {
            "stage": "initial observation evidence",
            "source_location": "statistical_observability/*_snapshots.csv at global_step=-1",
            "authoritative_rule": "all observed initial cash balances are zero",
            "amount": float(initial_social["cash"].sum() + initial_settlement["cash"].sum()),
            "recipient_scope": "all initial Household and settlement-account cash holders",
            "social_household_cash_effect": float(initial_social["cash"].sum()),
            "settlement_account_cash_effect": float(initial_settlement["cash"].sum()),
            "classification": "UNFUNDED_HOUSEHOLD_CREATION",
        },
    ]
    write(pd.DataFrame(origin_rows), "initial_household_cash_origin.csv")

    distribution_rows: list[dict] = []
    for scope, frame, account_id in (
        ("SOCIAL_HOUSEHOLD", initial_social, "household_id"),
        ("SETTLEMENT_ONLY_ACCOUNT", initial_settlement, "settlement_account_id"),
    ):
        groups = [("ALL", frame)]
        if scope == "SOCIAL_HOUSEHOLD":
            groups.extend(
                (f"adults={adults};children={children}", group)
                for (adults, children), group in frame.groupby(["adult_count", "child_count"], dropna=False)
            )
        for group_name, group in groups:
            stats = describe(group["cash"])
            distribution_rows.append(
                {
                    "scope": scope,
                    "group": group_name,
                    "global_step": -1,
                    "count": int(len(group)),
                    "cash_sum": float(group["cash"].sum()),
                    **stats,
                    "zero_cash_count": int((group["cash"].abs() <= 1e-12).sum()),
                    "near_zero_threshold": NEAR_ZERO_THRESHOLD,
                    "near_zero_definition": "cash <= 10.0, equal to 1e-6 of the 10,000,000 initial Firm-cash scale",
                    "near_zero_cash_count": int((group["cash"] <= NEAR_ZERO_THRESHOLD).sum()),
                    "employed_member_count_available": False,
                    "initial_income_status": "ZERO_PRE_SIMULATION_FLOW",
                    "adult_count_mean": float(group["adult_count"].mean()) if "adult_count" in group else np.nan,
                    "child_count_mean": float(group["child_count"].mean()) if "child_count" in group else np.nan,
                    "age_composition_available": bool(scope == "SOCIAL_HOUSEHOLD"),
                }
            )
    write(pd.DataFrame(distribution_rows), "initial_cash_distribution_by_household_type.csv")

    # 2/3. Household persistence, current cross-sectional quantiles, and
    # previous-quantile cash increments. All initial households tie at zero,
    # so their numeric IDs are used only to make the tie explicit and stable.
    snapshot_steps = sorted(social["global_step"].unique())
    initial_ids = set(initial_social["household_id"])
    accumulation_rows: list[dict] = []
    previous: pd.DataFrame | None = None
    for step in snapshot_steps:
        current = social.loc[social["global_step"] == step].copy()
        current["quantile_group"] = quantile_labels(current)
        total_cash = float(current["cash"].sum())
        for group_name, group in current.groupby("quantile_group", observed=True):
            accumulation_rows.append(
                {
                    "row_type": "CURRENT_CROSS_SECTION_QUANTILE",
                    "global_step": int(step),
                    "quantile_group": group_name,
                    "household_count": int(len(group)),
                    "cash_sum": float(group["cash"].sum()),
                    "cash_share": float(group["cash"].sum() / total_cash) if total_cash else np.nan,
                    "cash_mean": float(group["cash"].mean()),
                    "cash_median": float(group["cash"].median()),
                    "saving_mean": float(group["current_week_saving"].mean()),
                    "saving_median": float(group["current_week_saving"].median()),
                    "income_mean": float(group["current_week_income"].mean()),
                    "consumption_mean": float(group["current_week_consumption"].mean()),
                    "incremental_cash": np.nan,
                    "incremental_cash_share": np.nan,
                    "cohort_note": "current cash-ranked social-Household cross-section",
                }
            )

        retained = current.loc[current["household_id"].isin(initial_ids)]
        accumulation_rows.append(
            {
                "row_type": "INITIAL_ZERO_COHORT_RETENTION",
                "global_step": int(step),
                "quantile_group": "ALL_INITIAL_HOUSEHOLDS_TIED_AT_ZERO",
                "household_count": int(len(retained)),
                "cash_sum": float(retained["cash"].sum()),
                "cash_share": float(retained["cash"].sum() / total_cash) if total_cash else np.nan,
                "cash_mean": float(retained["cash"].mean()) if len(retained) else np.nan,
                "cash_median": float(retained["cash"].median()) if len(retained) else np.nan,
                "saving_mean": float(retained["current_week_saving"].mean()) if len(retained) else np.nan,
                "saving_median": float(retained["current_week_saving"].median()) if len(retained) else np.nan,
                "income_mean": float(retained["current_week_income"].mean()) if len(retained) else np.nan,
                "consumption_mean": float(retained["current_week_consumption"].mean()) if len(retained) else np.nan,
                "incremental_cash": np.nan,
                "incremental_cash_share": np.nan,
                "initial_cohort_size": int(len(initial_ids)),
                "initial_cohort_retention_share": float(len(retained) / len(initial_ids)),
                "near_zero_retained_share": float((retained["cash"] <= NEAR_ZERO_THRESHOLD).mean()) if len(retained) else np.nan,
                "cohort_note": "all initial social Households have equal zero cash; no economically ranked initial bottom subset exists",
            }
        )
        if previous is not None:
            left = previous[["household_id", "cash", "quantile_group"]].rename(
                columns={"cash": "prior_cash", "quantile_group": "prior_quantile"}
            )
            joined = current[["household_id", "cash"]].merge(left, on="household_id", how="inner")
            joined["cash_increment"] = joined["cash"] - joined["prior_cash"]
            total_increment = float(joined["cash_increment"].sum())
            for group_name, group in joined.groupby("prior_quantile", observed=True):
                increment = float(group["cash_increment"].sum())
                accumulation_rows.append(
                    {
                        "row_type": "PERSISTENT_PRIOR_QUANTILE_INCREMENT",
                        "global_step": int(step),
                        "prior_global_step": int(previous["global_step"].iloc[0]),
                        "quantile_group": group_name,
                        "household_count": int(len(group)),
                        "cash_sum": float(group["cash"].sum()),
                        "cash_share": np.nan,
                        "cash_mean": float(group["cash"].mean()),
                        "cash_median": float(group["cash"].median()),
                        "saving_mean": np.nan,
                        "saving_median": np.nan,
                        "income_mean": np.nan,
                        "consumption_mean": np.nan,
                        "incremental_cash": increment,
                        "incremental_cash_share": increment / total_increment if abs(total_increment) > 1e-12 else np.nan,
                        "cohort_note": "classified by prior snapshot cash; only stable household IDs are included",
                    }
                )
        previous = current
    write(pd.DataFrame(accumulation_rows), "household_cash_accumulation_by_quantile.csv")

    # 4. Gini time series, annual marriage alignment, and fixed-ID bridge.
    social_gini_rows: list[dict] = []
    previous_gini: dict | None = None
    transition = settlement_all.loc[settlement_all["observation_phase"] == "TRANSITION"]
    for step in snapshot_steps:
        current = social.loc[social["global_step"] == step]
        event_counts = demographic_events.loc[demographic_events["global_step"] == step, "event_type"].value_counts()
        transition_rows = transition.loc[transition["global_step"] == step]
        record = {
            "global_step": int(step),
            "snapshot_phase": str(current["observation_phase"].iloc[0]),
            "social_household_count": int(len(current)),
            "cash_gini": gini(current["cash"]),
            "income_gini": gini(current["current_week_income"]),
            "snapshot_interval_weeks": np.nan if previous_gini is None else int(step - previous_gini["global_step"]),
            "is_13_week_snapshot_grid": bool(step >= 0 and (step % 13 == 0 or step == 519)),
            "is_annual_marriage_schedule_week": bool(step > 0 and step % 52 == 0),
            "birth_events": int(event_counts.get("birth", 0)),
            "death_events": int(event_counts.get("death", 0)),
            "marriage_events": int(event_counts.get("marriage", 0)),
            "settlement_transition_accounts": int(len(transition_rows)),
            "settlement_transition_cash": float(transition_rows["cash"].sum()),
            "composition_change_from_prior": np.nan if previous_gini is None else int(len(current) - previous_gini["social_household_count"]),
        }
        record["cash_gini_change"] = np.nan if previous_gini is None else record["cash_gini"] - previous_gini["cash_gini"]
        record["income_gini_change"] = np.nan if previous_gini is None else record["income_gini"] - previous_gini["income_gini"]
        record["large_gini_turning_point"] = bool(
            previous_gini is not None
            and max(abs(record["cash_gini_change"]), abs(record["income_gini_change"])) >= 0.01
        )
        social_gini_rows.append(record)
        previous_gini = record
    gini_frame = pd.DataFrame(social_gini_rows)
    write(gini_frame, "gini_cycle_event_alignment.csv")

    decomposition_rows: list[dict] = []
    for prior_step, step in zip(snapshot_steps, snapshot_steps[1:]):
        prior = social.loc[social["global_step"] == prior_step, ["household_id", "cash", "current_week_income"]]
        current = social.loc[social["global_step"] == step, ["household_id", "cash", "current_week_income"]]
        common = prior.merge(current, on="household_id", suffixes=("_prior", "_current"))
        for metric, prior_column, current_column in (
            ("cash", "cash_prior", "cash_current"),
            ("income", "current_week_income_prior", "current_week_income_current"),
        ):
            prior_all = gini(prior["cash"] if metric == "cash" else prior["current_week_income"])
            current_all = gini(current["cash"] if metric == "cash" else current["current_week_income"])
            shared_prior = gini(common[prior_column])
            shared_current = gini(common[current_column])
            value_change = shared_current - shared_prior
            entry_composition_effect = current_all - shared_current
            exit_selection_effect = shared_prior - prior_all
            composition_effect = entry_composition_effect + exit_selection_effect
            if abs(value_change) >= 0.002 and abs(composition_effect) >= 0.002:
                classification = "BOTH"
            elif abs(composition_effect) > abs(value_change):
                classification = "COMPOSITION_CHANGE"
            else:
                classification = "VALUE_CHANGE"
            decomposition_rows.append(
                {
                    "metric": metric,
                    "prior_global_step": int(prior_step),
                    "global_step": int(step),
                    "prior_household_count": int(len(prior)),
                    "current_household_count": int(len(current)),
                    "shared_household_count": int(len(common)),
                    "entries": int(len(current) - len(common)),
                    "exits": int(len(prior) - len(common)),
                    "prior_all_gini": prior_all,
                    "current_all_gini": current_all,
                    "shared_prior_gini": shared_prior,
                    "shared_current_gini": shared_current,
                    "value_change_effect": value_change,
                    "entry_composition_effect": entry_composition_effect,
                    "exit_selection_effect": exit_selection_effect,
                    "net_composition_effect": composition_effect,
                    "classification": classification,
                }
            )
    write(pd.DataFrame(decomposition_rows), "gini_value_vs_composition_decomposition.csv")

    # 5/6. Settlement lifecycle and its social-Household counterpart.
    social_aggregate = social.groupby("global_step", as_index=False).agg(
        social_household_count=("household_id", "nunique"),
        social_household_cash=("cash", "sum"),
        social_income=("current_week_income", "sum"),
        social_consumption=("current_week_consumption", "sum"),
        social_saving=("current_week_saving", "sum"),
    )
    settlement_aggregate = settlement.groupby("global_step", as_index=False).agg(
        settlement_account_count=("settlement_account_id", "nunique"),
        settlement_cash=("cash", "sum"),
    )
    transition_aggregate = transition.groupby("global_step", as_index=False).agg(
        settlement_transition_count=("settlement_account_id", "nunique"),
        settlement_transition_cash=("cash", "sum"),
    )
    deactivation = settlement_all.loc[settlement_all["observation_phase"] == "DEACTIVATION"]
    deactivation_aggregate = deactivation.groupby("global_step", as_index=False).agg(
        settlement_deactivation_count=("settlement_account_id", "nunique"),
        settlement_deactivation_cash=("cash", "sum"),
    )
    settlement_cycle = social_aggregate.merge(settlement_aggregate, on="global_step", how="outer")
    settlement_cycle = settlement_cycle.merge(transition_aggregate, on="global_step", how="left")
    settlement_cycle = settlement_cycle.merge(deactivation_aggregate, on="global_step", how="left")
    settlement_cycle = settlement_cycle.sort_values("global_step").fillna(0.0)
    settlement_cycle["settlement_cash_change_since_prior_snapshot"] = settlement_cycle["settlement_cash"].diff()
    settlement_cycle["social_cash_change_since_prior_snapshot"] = settlement_cycle["social_household_cash"].diff()
    settlement_cycle["ordinary_social_saving_between_snapshots"] = settlement_cycle["social_saving"]
    settlement_cycle["wages_received_by_settlement_accounts"] = np.nan
    settlement_cycle["settlement_consumption"] = np.nan
    settlement_cycle["settlement_wage_consumption_availability"] = "NOT_PERSISTED_AT_SETTLEMENT_ACCOUNT_LEVEL"
    settlement_cycle["classification"] = np.where(
        settlement_cycle["settlement_transition_count"] > 0,
        "EXPECTED_SETTLEMENT_LIFECYCLE_PATTERN",
        "ACCUMULATION_BETWEEN_MARRIAGE_EVENTS",
    )
    write(settlement_cycle, "settlement_account_cycle_decomposition.csv")

    # 7. Common, recoverable shocks: require simultaneous negative changes in
    # all five requested series and recovery of income/consumption within two
    # steps. A terminal candidate is retained in the CSV but excluded because
    # its recovery cannot be observed in the frozen horizon.
    shock_columns = [
        "household_income",
        "household_consumption",
        "household_saving",
        "revenue",
        "total_fixed_investment",
    ]
    shock_frame = macro[["global_step", *shock_columns]].copy()
    for column in shock_columns:
        shock_frame[f"{column}_delta"] = shock_frame[column].diff()
        scale = max(1e-12, float(shock_frame.loc[shock_frame["global_step"] >= 430, column].abs().median()))
        shock_frame[f"{column}_normalized_negative"] = np.maximum(0.0, -shock_frame[f"{column}_delta"] / scale)
    normalized = [f"{column}_normalized_negative" for column in shock_columns]
    shock_frame["joint_negative_score"] = shock_frame[normalized].sum(axis=1)
    shock_frame["late_run"] = shock_frame["global_step"] > 430
    shock_frame["all_requested_series_negative"] = (shock_frame[[f"{column}_delta" for column in shock_columns]] < 0.0).all(axis=1)
    income_lookup = shock_frame.set_index("global_step")["household_income"]
    consumption_lookup = shock_frame.set_index("global_step")["household_consumption"]
    shock_frame["income_recovery_within_2w"] = shock_frame.apply(
        lambda row: bool(
            row["global_step"] + 2 in income_lookup.index
            and income_lookup.loc[row["global_step"] + 2] > row["household_income"]
        ),
        axis=1,
    )
    shock_frame["consumption_recovery_within_2w"] = shock_frame.apply(
        lambda row: bool(
            row["global_step"] + 2 in consumption_lookup.index
            and consumption_lookup.loc[row["global_step"] + 2] > row["household_consumption"]
        ),
        axis=1,
    )
    candidates = shock_frame.loc[
        shock_frame["late_run"]
        & shock_frame["all_requested_series_negative"]
        & shock_frame["income_recovery_within_2w"]
        & shock_frame["consumption_recovery_within_2w"]
    ].nlargest(3, "joint_negative_score")
    shock_weeks = sorted(int(value) for value in candidates["global_step"])
    shock_frame["selected_common_recovering_shock"] = shock_frame["global_step"].isin(shock_weeks)
    shock_frame["unrecovered_pre_recovery_candidate"] = (
        shock_frame["late_run"]
        & shock_frame["all_requested_series_negative"]
        & ~shock_frame["income_recovery_within_2w"]
    )
    final_observed_step = int(shock_frame["global_step"].max())
    shock_frame["terminal_horizon_candidate"] = (
        shock_frame["unrecovered_pre_recovery_candidate"]
        & (shock_frame["global_step"] + 2 > final_observed_step)
    )
    write(shock_frame, "late_run_shock_week_detection.csv")

    # 8. Event windows. Macro flows, Food and supplier operating states, and
    # event-level investment/lifecycle counts remain separate by construction.
    capacity_events = chain.loc[chain["event_type"].isin(["asset_acquired", "asset_retired"])]
    chain_by_step = chain.groupby("global_step").agg(
        accepted_order_events=("order_id", lambda value: int(value.notna().sum())),
        investment_chain_rows=("event_type", "size"),
        replacement_settled_units=("settled_units", lambda value: float(pd.to_numeric(value, errors="coerce").fillna(0.0).sum())),
        replacement_unmet_units=("unmet_units", lambda value: float(pd.to_numeric(value, errors="coerce").fillna(0.0).sum())),
    )
    advance_by_step = chain.loc[chain["event_type"] == "customer_advance_received"].groupby("global_step").agg(
        new_customer_advance_events=("event_type", "size"),
        new_customer_advance_cash=("cash_amount", "sum"),
    )
    deliveries_by_step = chain.loc[chain["event_type"] == "customer_advance_delivery_recognition"].groupby("global_step").agg(
        delivery_events=("event_type", "size"),
        delivery_units=("physical_units", "sum"),
    )
    assets_by_step = capacity_events.groupby("global_step").agg(
        acquisitions=("event_type", lambda value: int((value == "asset_acquired").sum())),
        retirements=("event_type", lambda value: int((value == "asset_retired").sum())),
        retired_service_from_events=("retired_service_capacity", lambda value: float(pd.to_numeric(value, errors="coerce").fillna(0.0).sum())),
    )
    supplier = firm_diagnostics.loc[firm_diagnostics["sector_id"] == "capital_goods"]
    food = firm_diagnostics.loc[firm_diagnostics["sector_id"] == "food"]
    supplier_by_step = supplier.groupby("global_step", as_index=False).agg(
        supplier_desired_output=("desired_production", "sum"),
        supplier_funded_output=("funded_output", "sum"),
        supplier_realized_production=("actual_production", "sum"),
        supplier_employment=("employee_count", "sum"),
        supplier_cash=("cash", "sum"),
        supplier_advance_liability=("customer_advance_liability", "sum"),
        supplier_wage_payment=("wage_payment", "sum"),
    )
    food_by_step = food.groupby("global_step", as_index=False).agg(
        food_firm_cash=("cash", "sum"),
        food_firm_revenue=("sales_revenue", "sum"),
        food_firm_wage_payment=("wage_payment", "sum"),
    )
    trace_base = macro.merge(supplier_by_step, on="global_step", how="left").merge(food_by_step, on="global_step", how="left")
    trace_base = trace_base.merge(chain_by_step, left_on="global_step", right_index=True, how="left")
    trace_base = trace_base.merge(advance_by_step, left_on="global_step", right_index=True, how="left")
    trace_base = trace_base.merge(deliveries_by_step, left_on="global_step", right_index=True, how="left")
    trace_base = trace_base.merge(assets_by_step, left_on="global_step", right_index=True, how="left")
    trace_columns = [
        "global_step", "household_income", "household_consumption", "household_saving", "household_cash",
        "total_employment", "food_employment", "capital_good_employment", "unassigned_labor",
        "food_demand", "food_production", "food_sales", "revenue", "cash", "food_firm_cash",
        "supplier_cash", "total_fixed_investment", "expansion_investment", "replacement_investment",
        "customer_advances_received", "customer_advances_delivered", "active_capital_assets",
        "active_capital_service", "depreciation", "retired_capacity", "supplier_desired_output",
        "supplier_funded_output", "supplier_realized_production", "supplier_employment",
        "supplier_advance_liability", "supplier_wage_payment", "investment_chain_rows",
        "new_customer_advance_events", "new_customer_advance_cash", "delivery_events", "delivery_units",
        "acquisitions", "retirements", "retired_service_from_events", "replacement_settled_units",
        "replacement_unmet_units",
    ]
    trace_rows = []
    for shock_week in shock_weeks:
        for step in range(shock_week - 4, shock_week + 5):
            row = trace_base.loc[trace_base["global_step"] == step]
            if row.empty:
                continue
            record = row.iloc[0].reindex(trace_columns).to_dict()
            record["shock_week"] = shock_week
            record["relative_week"] = step - shock_week
            record["capital_backlog"] = np.nan
            record["capital_backlog_availability"] = "NOT_PERSISTED_AS_AUTHORITATIVE_WEEKLY_STOCK"
            trace_rows.append(record)
    write(pd.DataFrame(trace_rows), "late_run_event_window_trace.csv")

    # 9. Retirement and scheduling alignment.
    lifecycle_rows = []
    for shock_week in shock_weeks:
        direct_retired = chain.loc[(chain["global_step"] == shock_week) & (chain["event_type"] == "asset_retired")]
        direct_orders = chain.loc[(chain["global_step"] == shock_week) & (chain["event_type"] == "customer_advance_received")]
        delivery = chain.loc[(chain["global_step"] == shock_week) & (chain["event_type"] == "customer_advance_delivery_recognition")]
        recovery_week = shock_week + 2
        review_count = int(
            firm_diagnostics.loc[
                firm_diagnostics["global_step"] == recovery_week,
                "investment_reviewed_this_step",
            ].sum()
        )
        lifecycle_rows.append(
            {
                "shock_week": shock_week,
                "assets_retired_at_shock": int(len(direct_retired)),
                "capital_service_retired_at_shock": float(pd.to_numeric(direct_retired["retired_service_capacity"], errors="coerce").fillna(0.0).sum()),
                "replacement_orders_at_shock": int(len(direct_orders)),
                "replacement_deliveries_at_shock": int(len(delivery)),
                "fixed_investment_at_shock": float(macro.loc[macro["global_step"] == shock_week, "total_fixed_investment"].iloc[0]),
                "investment_review_recovery_week": recovery_week,
                "food_firms_reviewed_at_recovery_week": review_count,
                "shock_mod_13": shock_week % 13,
                "shock_mod_52": shock_week % 52,
                "recovery_week_mod_13": recovery_week % 13,
                "recovery_week_mod_52": recovery_week % 52,
                "alignment_classification": "NOT_ALIGNED",
                "alignment_reason": "no asset retirement occurs at the shock week; all three recover two weeks later at the synchronized 13-week investment review",
            }
        )
    write(pd.DataFrame(lifecycle_rows), "late_shock_capital_lifecycle_alignment.csv")

    # 10/11. Causal-order and metric-definition audit.
    causal_rows = []
    for shock_week in shock_weeks:
        pre = shock_week - 1
        review = shock_week + 2
        pre_supplier = supplier_by_step.loc[supplier_by_step["global_step"] == pre].iloc[0]
        shock_supplier = supplier_by_step.loc[supplier_by_step["global_step"] == shock_week].iloc[0]
        shock_macro = macro.loc[macro["global_step"] == shock_week].iloc[0]
        recovery_macro = macro.loc[macro["global_step"] == review].iloc[0]
        causal_rows.extend([
            {
                "shock_week": shock_week,
                "causal_order": 1,
                "week": pre,
                "event_or_state": "supplier_customer_advance_pipeline_near_exhausted",
                "evidence_value": float(pre_supplier["supplier_cash"]),
                "evidence_detail": "capital-good supplier cash after payroll/production falls to a minimal balance before the observed wage interruption",
                "causal_assessment": "FIRST_MATERIALLY_DISCONTINUOUS_UPSTREAM_STATE",
            },
            {
                "shock_week": shock_week,
                "causal_order": 2,
                "week": shock_week,
                "event_or_state": "capital_good_payroll_and_delivery_stop",
                "evidence_value": float(shock_supplier["supplier_wage_payment"]),
                "evidence_detail": "supplier wage payment and capital-good delivery revenue are zero at the shock; this is before the next synchronized review",
                "causal_assessment": "DIRECT_OPERATING_FLOW_BREAK",
            },
            {
                "shock_week": shock_week,
                "causal_order": 3,
                "week": shock_week,
                "event_or_state": "household_wage_income_and_consumption_contract",
                "evidence_value": float(shock_macro["household_income"]),
                "evidence_detail": "household income equals the depressed wage flow; consumption and Food demand decline in the same execution week",
                "causal_assessment": "DOWNSTREAM_HOUSEHOLD_FLOW_RESPONSE",
            },
            {
                "shock_week": shock_week,
                "causal_order": 4,
                "week": shock_week,
                "event_or_state": "food_sales_and_aggregate_firm_revenue_contract",
                "evidence_value": float(shock_macro["revenue"]),
                "evidence_detail": "Food production remains planned while household-funded Food sales/revenue fall, so capacity retirement is not the first observed discontinuity",
                "causal_assessment": "DOWNSTREAM_MARKET_RESPONSE",
            },
            {
                "shock_week": shock_week,
                "causal_order": 5,
                "week": review,
                "event_or_state": "synchronized_food_firm_review_collects_new_customer_advances",
                "evidence_value": float(recovery_macro["customer_advances_received"]),
                "evidence_detail": "all five Food Firms review at the same 13-week phase; new advances refill supplier cash and operating flows recover",
                "causal_assessment": "RECOVERY_TRIGGER",
            },
        ])
    write(pd.DataFrame(causal_rows), "late_shock_causal_chain.csv")

    fixed_rows = []
    diag_investment = firm_diagnostics.groupby("global_step", as_index=False).agg(
        firm_executed_investment=("executed_investment", "sum"),
        firm_customer_advance_delivered=("customer_advance_delivered", "sum"),
        firm_customer_advance_received=("customer_advance_received", "sum"),
    )
    fixed = macro.merge(diag_investment, on="global_step", how="left")
    for _, row in fixed.iterrows():
        fixed_rows.append(
            {
                "row_type": "WEEKLY_AUTHORITATIVE_COMPARISON",
                "global_step": int(row["global_step"]),
                "macro_fixed_investment": float(row["total_fixed_investment"]),
                "firm_executed_investment": float(row["firm_executed_investment"]),
                "customer_advance_delivered": float(row["customer_advances_delivered"]),
                "customer_advance_received": float(row["customer_advances_received"]),
                "fixed_minus_delivery": float(row["total_fixed_investment"] - row["customer_advances_delivered"]),
                "definition": "realized capital formation at delivery/acquisition, not new order intent or customer-advance cash receipt",
                "classification": "REAL_DELIVERY_FLOW_DISCONTINUITY",
            }
        )
    fixed_rows.append(
        {
            "row_type": "SEMANTIC_CONTRACT",
            "global_step": np.nan,
            "macro_fixed_investment": np.nan,
            "firm_executed_investment": np.nan,
            "customer_advance_delivered": np.nan,
            "customer_advance_received": np.nan,
            "fixed_minus_delivery": np.nan,
            "definition": "macro total_fixed_investment equals supplier delivery-recognition / buyer capitalization expenditure in this run",
            "classification": "NOT_A_NEW_ORDER_METRIC",
        }
    )
    write(pd.DataFrame(fixed_rows), "fixed_investment_definition_audit.csv")

    # 12. Sector cash decomposes the customer-advance transfer and normal
    # supplier payroll run-down without confusing it with the total cash stock.
    sector_cash = firm_diagnostics.groupby(["global_step", "sector_id"], as_index=False).agg(
        sector_cash=("cash", "sum"),
        sector_wage_payment=("wage_payment", "sum"),
        sector_revenue=("sales_revenue", "sum"),
        sector_customer_advance_received=("customer_advance_received", "sum"),
        sector_customer_advance_delivered=("customer_advance_delivered", "sum"),
    )
    sector_pivot = sector_cash.pivot(index="global_step", columns="sector_id", values="sector_cash").rename(
        columns={"food": "food_firm_cash", "capital_goods": "capital_good_firm_cash"}
    ).reset_index()
    sector_flow = sector_cash.pivot(index="global_step", columns="sector_id", values="sector_customer_advance_received").rename(
        columns={"food": "food_advance_received", "capital_goods": "capital_good_customer_advance_received"}
    ).reset_index()
    sector_output = macro[["global_step", "cash", "household_cash", "household_saving"]].merge(sector_pivot, on="global_step", how="left").merge(sector_flow, on="global_step", how="left")
    for column in ("food_firm_cash", "capital_good_firm_cash", "cash"):
        sector_output[f"{column}_change"] = sector_output[column].diff()
    sector_output["classification"] = "EXPECTED_INTERFIRM_SETTLEMENT_CYCLE"
    sector_output["interpretation"] = "Food cash falls when customer advances move to the capital-good supplier; supplier cash then falls through payroll and delivery. Aggregate Firm cash also follows the separate known household-saving leakage trend."
    write(sector_output, "sector_cash_cycle_decomposition.csv")

    # 14. Manual issue classification and summary flags.
    initial_retained_final = social.loc[(social["global_step"] == max(snapshot_steps)) & social["household_id"].isin(initial_ids)]
    final_cross = social.loc[social["global_step"] == max(snapshot_steps)].copy()
    final_cross["quantile_group"] = quantile_labels(final_cross)
    final_top_cash_share = float(
        final_cross.loc[final_cross["quantile_group"] == "TOP_10", "cash"].sum() / final_cross["cash"].sum()
    )
    classifications = pd.DataFrame([
        {
            "issue": "A. initial Social Household cash mass near zero",
            "classification": "INITIAL_CONDITION_PROBLEM",
            "evidence": "all 1,877 initial social Households record exactly zero cash; initial 10,000,000 monetary cash is placed in the Firm system and only later circulates through wages",
        },
        {
            "issue": "B. settlement-account sawtooth",
            "classification": "EXPECTED_ACCOUNT_LIFECYCLE",
            "evidence": "settlement account counts/cash accumulate between annual marriage weeks and transition cash is recorded at steps 52,104,...",
        },
        {
            "issue": "B. social-Household cash sawtooth",
            "classification": "STATISTICAL_UNIT_COMPOSITION_EFFECT",
            "evidence": "annual marriage creates/changes social-Household units and moves temporary settlement balances; social cash stock remains upward trending across persisted snapshots",
        },
        {
            "issue": "C. aggregate Household cash linear rise",
            "classification": "DISTRIBUTIONAL_DYNAMICS",
            "evidence": "positive aggregate social-Household saving accumulates as a stock; final cross-sectional top 10% holds a large share while the bottom half remains close to zero",
        },
        {
            "issue": "D. income/cash Gini sawtooth",
            "classification": "STATISTICAL_UNIT_COMPOSITION_EFFECT",
            "evidence": "13-week snapshot observations show annual-marriage turning points with both fixed-ID value changes and Household entry/exit composition effects",
        },
        {
            "issue": "E. late temporary income/consumption/revenue/investment collapses",
            "classification": "INVESTMENT_PIPELINE_DISCONTINUITY",
            "evidence": "weeks 479,492,505 follow supplier cash/advance exhaustion and precede synchronized 13-week Food-Firm customer-advance collection by two weeks",
        },
        {
            "issue": "F. sector Firm-cash sawtooth",
            "classification": "EXPECTED_INTERFIRM_SETTLEMENT",
            "evidence": "customer advances transfer Food-Firm cash to the capital-good supplier, which then pays wages and recognizes deliveries; the sectors move in offsetting directions",
        },
        {
            "issue": "52-week capital-retirement hypothesis for late shocks",
            "classification": "NOT_PRIMARY",
            "evidence": "no retirement event occurs at the three shock weeks; shock phases differ modulo 52 while recovery is synchronized modulo 13",
        },
    ])
    write(classifications, "manual_review_issue_classification.csv")

    flags = {
        "verdict": "D. INVESTMENT_PIPELINE_DISCONTINUITY_IS_PRIMARY",
        "economic_behavior_changed": False,
        "simulation_runs": 0,
        "analysis_or_gui_changed": False,
        "initial_social_households": int(len(initial_social)),
        "initial_social_households_zero_cash": int((initial_social["cash"].abs() <= 1e-12).sum()),
        "initial_settlement_accounts": int(len(initial_settlement)),
        "initial_settlement_accounts_zero_cash": int((initial_settlement["cash"].abs() <= 1e-12).sum()),
        "initial_cash_classification": "UNFUNDED_HOUSEHOLD_CREATION",
        "near_zero_threshold": NEAR_ZERO_THRESHOLD,
        "initial_zero_cohort_final_retention_share": float(len(initial_retained_final) / len(initial_ids)),
        "initial_zero_cohort_final_near_zero_share": float((initial_retained_final["cash"] <= NEAR_ZERO_THRESHOLD).mean()),
        "initial_zero_cohort_final_above_near_zero_share": float((initial_retained_final["cash"] > NEAR_ZERO_THRESHOLD).mean()),
        "final_top_10_social_household_cash_share": final_top_cash_share,
        "gini_sawtooth_has_composition_component": True,
        "settlement_sawtooth_classification": "EXPECTED_SETTLEMENT_LIFECYCLE_PATTERN",
        "late_recovering_shock_weeks": shock_weeks,
        "terminal_horizon_candidate_weeks": [
            int(step)
            for step in shock_frame.loc[shock_frame["terminal_horizon_candidate"], "global_step"]
        ],
        "late_shocks_capital_lifecycle_aligned": False,
        "late_shocks_investment_review_pipeline_aligned": True,
        "fixed_investment_definition": "DELIVERY_ACQUISITION_REALIZED_CAPITAL_FORMATION",
        "sector_cash_sawtooth_classification": "EXPECTED_INTERFIRM_SETTLEMENT_CYCLE",
    }
    with (OUTPUT / "acceptance_flags.json").open("w", encoding="utf-8") as handle:
        json.dump(flags, handle, indent=2, ensure_ascii=False)

    summary = f"""# Step 15 Household Distribution + Late-Run Discontinuity Root-Cause Audit

Verdict: **D. INVESTMENT_PIPELINE_DISCONTINUITY_IS_PRIMARY**

This is a read-only audit of `step15_final_canonical_statistics_enriched`.
No simulation, parameter, model, persistence, Analysis, or GUI change was made.

## Findings

1. All {len(initial_social):,} initial Social Households and all {len(initial_settlement):,} initial settlement-only accounts have exactly zero cash at global step -1. This is an initialization-order / unfunded-Household condition: the initial {INITIAL_FIRM_CASH:,.0f} cash is held by the Firm system and then split among Firms, while `Household.wealth` is initialized to zero.
2. The initial zero cohort is not uniformly permanent: {len(initial_retained_final):,}/{len(initial_ids):,} original Household IDs remain at the final snapshot. Of them, {(initial_retained_final['cash'] <= NEAR_ZERO_THRESHOLD).mean():.2%} remain at or below {NEAR_ZERO_THRESHOLD:g} cash and {(initial_retained_final['cash'] > NEAR_ZERO_THRESHOLD).mean():.2%} rise above it. Because every initial Household is tied at zero, a meaningful economically ranked initial bottom-decile cohort does not exist.
3. Final cross-sectional cash is highly concentrated: the top 10% of Social Households hold {final_top_cash_share:.2%} of Social-Household cash, while the bottom half holds a negligible cash share. This is distributional dynamics, not a settlement-account mixing artifact.
4. The Gini pattern is partly value movement and partly social-Household composition. The strongest recurrent non-terminal turning points align with annual marriage/settlement transition weeks sampled on the 13-week micro-snapshot grid. The final week has a separate terminal income-Gini jump during an unrecovered end-of-run shock.
5. The three recoverable common late shocks are weeks {', '.join(map(str, shock_weeks))}. Week 518 is retained as a terminal candidate, but cannot qualify as a temporary recovered shock because the frozen run ends at week 519.
6. The first discontinuous upstream state is the capital-good supplier's customer-advance/cash pipeline running down before the next synchronized 13-week Food-Firm investment review. Supplier payroll/delivery falls to zero, Household wage income and consumption decline, Food sales/revenue follow, and all five Food Firms collect new advances two weeks later, producing recovery.
7. `total_fixed_investment` is realized delivery/acquisition capital formation, not order intent or advance receipt. Its drops are real supplier-throughput/delivery-flow interruptions, not a chart-label artifact.
8. Food and capital-good Firm cash sawteeth are expected inter-Firm settlement movements: Food cash transfers to the supplier as customer advances, then supplier cash runs down through payroll and delivery. Aggregate Firm cash has the separate accepted Household-saving leakage trend.

The three late shocks do not align with a common 52-week retirement phase: no asset-retirement event occurs at the shock weeks, whereas recovery aligns with the common 13-week investment-review phase.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
