"""Read-only post-freeze Step 15 manual-review audit.

The audit consumes the frozen canonical CSVs and writes only the requested
diagnostic artifacts.  It never constructs a World or advances simulation
state.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN = PROJECT_ROOT / "test/output/step15_final_canonical_integrated_v2"
OUTPUT = PROJECT_ROOT / "test/output/step15_post_freeze_manual_review_audit"
STAT = RUN / "statistical_observability"
TOLERANCE = 1e-6
NEAR_ZERO_CASH = 10.0
SNAPSHOT_PHASES = {"INITIAL", "PERIODIC", "FINAL"}


INPUTS = {
    "macro": RUN / "step15_macro_panel.csv",
    "firms": RUN / "step15_firm_panel.csv",
    "accounting": RUN / "step15_accounting_reconciliation.csv",
    "raw_firms": RUN / "firm_diagnostics.csv",
    "chain": RUN / "step15_investment_chain_trace.csv",
    "assets": RUN / "step15_capital_asset_ledger.csv",
    "demographic_events": RUN / "demographic_events.csv",
    "social_households": STAT / "social_household_snapshots.csv",
    "settlement_accounts": STAT / "settlement_account_snapshots.csv",
    "labor_events": STAT / "labor_events.csv",
    "labor_denominators": STAT / "labor_denominators.csv",
    "firm_capital": STAT / "firm_capital_history.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(name: str) -> pd.DataFrame:
    path = INPUTS[name]
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def write(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUTPUT / filename, index=False)


def number(value, default=math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_ratio(numerator, denominator) -> float:
    numerator = number(numerator)
    denominator = number(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-15:
        return math.nan
    return numerator / denominator


def population_cv(values) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty or abs(float(series.mean())) <= 1e-15:
        return math.nan
    return float(series.std(ddof=0) / abs(series.mean()))


def gini(values) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    array = np.maximum(array, 0.0)
    if len(array) == 0 or float(array.sum()) <= 1e-15:
        return 0.0
    array.sort()
    ranks = np.arange(1, len(array) + 1, dtype=float)
    return float(
        2.0 * ranks.dot(array) / (len(array) * array.sum())
        - (len(array) + 1.0) / len(array)
    )


def bottom_half_ids(frame: pd.DataFrame) -> set:
    ordered = frame.sort_values(["cash", "household_id"], kind="stable")
    return set(ordered.iloc[: math.ceil(len(ordered) / 2)]["household_id"])


def accounting_firms(accounting: pd.DataFrame) -> pd.DataFrame:
    result = accounting.loc[accounting["record_type"] == "firm"].copy()
    result["firm_id_numeric"] = pd.to_numeric(result["firm_id"], errors="coerce")
    return result


def build_household_cycle(
    macro: pd.DataFrame,
    settlement_all: pd.DataFrame,
    demographic_events: pd.DataFrame,
) -> pd.DataFrame:
    transitions = settlement_all.loc[
        settlement_all["observation_phase"].eq("TRANSITION")
    ].copy()
    deactivations = settlement_all.loc[
        settlement_all["observation_phase"].eq("DEACTIVATION")
    ].copy()
    transition_by_step = transitions.groupby("global_step").agg(
        transition_account_count=("settlement_account_id", "size"),
        settlement_to_social_cash=("cash", "sum"),
    )
    deactivation_by_step = deactivations.groupby("global_step").size()
    marriages = (
        demographic_events.loc[demographic_events["event_type"].eq("marriage")]
        .groupby("global_step")
        .size()
    )

    frame = macro.sort_values("global_step").copy()
    frame["social_household_cash_change"] = frame["social_household_cash"].diff()
    frame["settlement_only_cash_change"] = frame["settlement_only_cash"].diff()
    frame["combined_household_cash"] = (
        frame["social_household_cash"] + frame["settlement_only_cash"]
    )
    frame["combined_household_cash_change"] = frame["combined_household_cash"].diff()
    frame["pending_formation_cash_change"] = frame[
        "pending_household_formation_wealth"
    ].diff()
    frame["settlement_account_count_change"] = frame[
        "settlement_only_account_count"
    ].diff()
    frame["marriage_events"] = frame["global_step"].map(marriages).fillna(0).astype(int)
    frame["settlement_transition_count"] = (
        frame["global_step"]
        .map(transition_by_step["transition_account_count"])
        .fillna(0)
        .astype(int)
    )
    frame["settlement_deactivation_count"] = (
        frame["global_step"].map(deactivation_by_step).fillna(0).astype(int)
    )
    frame["settlement_to_social_cash"] = (
        frame["global_step"]
        .map(transition_by_step["settlement_to_social_cash"])
        .fillna(0.0)
    )
    closures = frame["settlement_transition_count"] + frame["settlement_deactivation_count"]
    frame["settlement_account_creations_inferred"] = (
        frame["settlement_account_count_change"].fillna(0) + closures
    ).astype(int)

    # A pure account reclassification contributes +T to Social cash and -T to
    # settlement cash, hence zero to their combined cash.  The frozen run also
    # posts T through pending_household_formation_wealth a second time.
    transfer = frame["settlement_to_social_cash"]
    frame["pure_reclassification_social_effect"] = transfer
    frame["pure_reclassification_settlement_effect"] = -transfer
    frame["pure_reclassification_combined_effect"] = 0.0
    frame["social_cash_change_excluding_reclassification"] = (
        frame["social_household_cash_change"] - transfer
    )
    frame["settlement_cash_change_excluding_reclassification"] = (
        frame["settlement_only_cash_change"] + transfer
    )
    frame["combined_cash_change_less_weekly_saving"] = (
        frame["combined_household_cash_change"] - frame["household_saving"]
    )
    frame["pending_offset_identity_gap"] = (
        frame["combined_cash_change_less_weekly_saving"]
        + frame["pending_formation_cash_change"]
    )
    frame["transition_double_posting_gap"] = (
        frame["combined_cash_change_less_weekly_saving"] - transfer
    )
    frame["reclassification_share_of_social_cash_change"] = [
        safe_ratio(abs(t), abs(d))
        for t, d in zip(transfer, frame["social_household_cash_change"])
    ]
    frame["new_money_created_by_reclassification"] = 0.0
    frame["cycle_phase"] = np.where(
        frame["settlement_transition_count"] > 0,
        "ANNUAL_MARRIAGE_TRANSFER_WEEK",
        np.where(
            frame["settlement_account_creations_inferred"] > 0,
            "SETTLEMENT_ACCOUNT_ACCUMULATION_WEEK",
            "ORDINARY_SETTLEMENT_WEEK",
        ),
    )
    frame["audit_interpretation"] = np.where(
        frame["settlement_transition_count"] > 0,
        "Settlement cash is reclassified at marriage, but the same balance is also posted through the pending-formation transfer; combined Household cash rises by T and pending formation falls by T, so global money is unchanged.",
        "Weekly payroll less consumption drives economic-account cash; account creations are inferred from the authoritative weekly count bridge.",
    )
    columns = [
        "global_step",
        "cycle_phase",
        "marriage_events",
        "social_household_count",
        "settlement_only_account_count",
        "settlement_account_count_change",
        "settlement_account_creations_inferred",
        "settlement_transition_count",
        "settlement_deactivation_count",
        "settlement_to_social_cash",
        "social_household_cash",
        "settlement_only_cash",
        "combined_household_cash",
        "social_household_cash_change",
        "settlement_only_cash_change",
        "combined_household_cash_change",
        "household_wage_income",
        "household_income",
        "household_consumption",
        "household_saving",
        "pure_reclassification_social_effect",
        "pure_reclassification_settlement_effect",
        "pure_reclassification_combined_effect",
        "social_cash_change_excluding_reclassification",
        "settlement_cash_change_excluding_reclassification",
        "combined_cash_change_less_weekly_saving",
        "pending_household_formation_wealth",
        "pending_formation_cash_change",
        "pending_offset_identity_gap",
        "transition_double_posting_gap",
        "reclassification_share_of_social_cash_change",
        "new_money_created_by_reclassification",
        "money_stock",
        "full_money_location_gap",
        "audit_interpretation",
    ]
    return frame[columns]


def build_household_grain_registry(
    macro: pd.DataFrame,
    social: pd.DataFrame,
    settlement: pd.DataFrame,
) -> pd.DataFrame:
    macro_steps = sorted(macro["global_step"].unique())
    snapshot_steps = sorted(social["global_step"].unique())
    settlement_steps = sorted(settlement["global_step"].unique())
    macro_coverage = f"{macro_steps[0]}..{macro_steps[-1]} ({len(macro_steps)} weekly observations)"
    social_coverage = (
        f"{snapshot_steps[0]}..{snapshot_steps[-1]} ({len(snapshot_steps)} snapshots; "
        "initial plus 13-week grid; no step-519 micro snapshot)"
    )
    settlement_coverage = (
        f"{settlement_steps[0]}..{settlement_steps[-1]} ({len(settlement_steps)} snapshots; transitions persisted weekly)"
    )
    rows = [
        ("Household cash, all economic accounts", "household_cash", "WEEKLY_AGGREGATE_PERSISTED", "step15_macro_panel.csv", macro_coverage, "Weekly line valid; includes Social and settlement-only accounts."),
        ("Social Household cash aggregate", "social_household_cash", "WEEKLY_AGGREGATE_PERSISTED", "step15_macro_panel.csv", macro_coverage, "Weekly line valid; excludes settlement-only accounts."),
        ("Settlement-only cash aggregate", "settlement_only_cash", "WEEKLY_AGGREGATE_PERSISTED", "step15_macro_panel.csv", macro_coverage, "Weekly line valid; temporary economic-account stock."),
        ("Household income aggregate", "household_income", "WEEKLY_AGGREGATE_PERSISTED", "step15_macro_panel.csv", macro_coverage, "Weekly line valid."),
        ("Household consumption aggregate", "household_consumption", "WEEKLY_AGGREGATE_PERSISTED", "step15_macro_panel.csv", macro_coverage, "Weekly line valid."),
        ("Household saving aggregate", "household_saving", "WEEKLY_AGGREGATE_PERSISTED", "step15_macro_panel.csv", macro_coverage, "Weekly line valid."),
        ("Social Household cash micro", "cash", "PERIODIC_MICRO_SNAPSHOT", "social_household_snapshots.csv", social_coverage, "Markers/snapshot table only; never interpolate as weekly history."),
        ("Social Household income micro", "current_week_income", "PERIODIC_MICRO_SNAPSHOT", "social_household_snapshots.csv", social_coverage, "One observed week's flow at each snapshot, not a 13-week accumulated flow."),
        ("Social Household consumption micro", "current_week_consumption", "PERIODIC_MICRO_SNAPSHOT", "social_household_snapshots.csv", social_coverage, "One observed week's flow at each snapshot."),
        ("Social Household saving micro", "current_week_saving", "PERIODIC_MICRO_SNAPSHOT", "social_household_snapshots.csv", social_coverage, "One observed week's flow at each snapshot."),
        ("Settlement-account cash micro", "cash", "PERIODIC_MICRO_SNAPSHOT", "settlement_account_snapshots.csv", settlement_coverage, "Periodic active-account snapshots plus exact transition rows."),
        ("Cash Gini", "cash_gini", "DERIVED_FROM_SNAPSHOT", "social_household_snapshots.csv:cash", social_coverage, "Plot discrete snapshot markers; connecting lines imply unavailable weekly values."),
        ("Income Gini", "income_gini", "DERIVED_FROM_SNAPSHOT", "social_household_snapshots.csv:current_week_income", social_coverage, "Plot discrete snapshot markers."),
        ("Cash Lorenz curve", "cash_lorenz", "DERIVED_FROM_SNAPSHOT", "social_household_snapshots.csv:cash", social_coverage, "Cross-section at the selected authoritative snapshot only."),
        ("Income Lorenz curve", "income_lorenz", "DERIVED_FROM_SNAPSHOT", "social_household_snapshots.csv:current_week_income", social_coverage, "Cross-section at the selected authoritative snapshot only."),
        ("Household cash quantiles", "cash_quantiles", "DERIVED_FROM_SNAPSHOT", "social_household_snapshots.csv:cash", social_coverage, "Snapshot band/markers only."),
        ("Household income quantiles", "income_quantiles", "DERIVED_FROM_SNAPSHOT", "social_household_snapshots.csv:current_week_income", social_coverage, "Snapshot band/markers only."),
        ("Household cash histogram", "cash_histogram", "DERIVED_FROM_SNAPSHOT", "social_household_snapshots.csv:cash", social_coverage, "Selected-snapshot histogram, not a time series."),
    ]
    result = pd.DataFrame(
        rows,
        columns=[
            "gui_metric",
            "authoritative_field",
            "temporal_grain",
            "source",
            "actual_coverage",
            "rendering_contract",
        ],
    )
    result["micro_snapshot_cadence_weeks"] = np.where(
        result["temporal_grain"].isin({"PERIODIC_MICRO_SNAPSHOT", "DERIVED_FROM_SNAPSHOT"}),
        13,
        1,
    )
    result["weekly_micro_values_available"] = result["temporal_grain"].isin(
        {"WEEKLY_RUNTIME", "WEEKLY_AGGREGATE_PERSISTED"}
    )
    result["visual_interpolation_allowed"] = result["temporal_grain"].eq(
        "WEEKLY_AGGREGATE_PERSISTED"
    )
    result["minimum_weekly_distribution_extension"] = (
        "At each weekly diagnostic step compute Social-Household count, Gini, mean, median, P10/P25/P75/P90/P95, top-10 cash share, bottom-50 cash share and near-zero share; persist one aggregate row while retaining 13-week micro snapshots. No Person dump and no interpolation."
    )
    result["extension_status"] = "DESIGNED_NOT_IMPLEMENTED"
    return result


def build_near_zero_diagnosis(
    social: pd.DataFrame,
    settlement_all: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    transitions = settlement_all.loc[
        settlement_all["observation_phase"].eq("TRANSITION")
    ].copy()
    transitions["destination_social_household_id"] = pd.to_numeric(
        transitions["destination_social_household_id"], errors="coerce"
    )
    macro_by_step = macro.set_index("global_step")
    steps = sorted(social["global_step"].unique())
    rows: list[dict] = []
    previous = None

    for step in steps:
        current = social.loc[social["global_step"].eq(step)].copy()
        current["near_zero"] = current["cash"] <= NEAR_ZERO_CASH
        current_median_income = float(current["current_week_income"].median())
        low_income_cutoff = 0.5 * current_median_income
        transition_destinations = set(
            transitions.loc[
                transitions["global_step"].le(step),
                "destination_social_household_id",
            ].dropna()
        )
        current["settlement_origin"] = current["household_id"].isin(
            transition_destinations
        )
        for label, selected in (
            ("NEAR_ZERO", current.loc[current["near_zero"]]),
            ("ABOVE_NEAR_ZERO", current.loc[~current["near_zero"]]),
        ):
            count = len(selected)
            rows.append(
                {
                    "row_type": "SNAPSHOT_GROUP",
                    "global_step": int(step),
                    "prior_global_step": math.nan,
                    "cash_group": label,
                    "near_zero_threshold": NEAR_ZERO_CASH,
                    "household_count": count,
                    "household_share": safe_ratio(count, len(current)),
                    "cash_sum": float(selected["cash"].sum()),
                    "cash_mean": float(selected["cash"].mean()) if count else math.nan,
                    "cash_median": float(selected["cash"].median()) if count else math.nan,
                    "member_count_mean": float(selected["member_count"].mean()) if count else math.nan,
                    "adult_count_mean": float(selected["adult_count"].mean()) if count else math.nan,
                    "child_count_mean": float(selected["child_count"].mean()) if count else math.nan,
                    "households_with_children_share": float((selected["child_count"] > 0).mean()) if count else math.nan,
                    "current_week_income_mean": float(selected["current_week_income"].mean()) if count else math.nan,
                    "current_week_consumption_mean": float(selected["current_week_consumption"].mean()) if count else math.nan,
                    "current_week_saving_mean": float(selected["current_week_saving"].mean()) if count else math.nan,
                    "zero_income_share": float((selected["current_week_income"].abs() <= 1e-12).mean()) if count else math.nan,
                    "low_income_cutoff": low_income_cutoff,
                    "low_income_share": float((selected["current_week_income"] < low_income_cutoff).mean()) if count else math.nan,
                    "negative_saving_share": float((selected["current_week_saving"] < -1e-12).mean()) if count else math.nan,
                    "settlement_origin_share": float(selected["settlement_origin"].mean()) if count else math.nan,
                    "employed_member_count": math.nan,
                    "employed_member_count_status": "UNAVAILABLE_NOT_PERSISTED",
                    "exact_member_age_structure_status": "UNAVAILABLE; adult/child counts only",
                    "interval_income": math.nan,
                    "interval_transfers_dividends_inheritance": math.nan,
                    "interval_consumption": math.nan,
                    "interval_bridge_status": "UNAVAILABLE; micro flows are one-week snapshots, not interval totals",
                    "bottom50_opening_cash": math.nan,
                    "bottom50_closing_cash": math.nan,
                    "bottom50_cash_change": math.nan,
                    "diagnosis": (
                        "LOW_WAGE_INCOME_PLUS_CONSUMPTION_FLOOR"
                        if label == "NEAR_ZERO" and step >= 0
                        else "COMPARISON_GROUP"
                    ),
                }
            )

        if previous is not None:
            prior_step = int(previous["global_step"].iloc[0])
            ids = bottom_half_ids(previous)
            opening = previous.loc[previous["household_id"].isin(ids), ["household_id", "cash"]]
            closing = current.loc[current["household_id"].isin(ids), ["household_id", "cash"]]
            shared = opening.merge(closing, on="household_id", suffixes=("_opening", "_closing"))
            rows.append(
                {
                    "row_type": "BOTTOM_50_COHORT_BRIDGE",
                    "global_step": int(step),
                    "prior_global_step": prior_step,
                    "cash_group": "PRIOR_SNAPSHOT_BOTTOM_50",
                    "near_zero_threshold": NEAR_ZERO_CASH,
                    "household_count": len(shared),
                    "household_share": safe_ratio(len(shared), len(current)),
                    "cash_sum": float(shared["cash_closing"].sum()),
                    "cash_mean": float(shared["cash_closing"].mean()) if len(shared) else math.nan,
                    "cash_median": float(shared["cash_closing"].median()) if len(shared) else math.nan,
                    "bottom50_opening_cash": float(shared["cash_opening"].sum()),
                    "bottom50_closing_cash": float(shared["cash_closing"].sum()),
                    "bottom50_cash_change": float((shared["cash_closing"] - shared["cash_opening"]).sum()),
                    "interval_income": math.nan,
                    "interval_transfers_dividends_inheritance": math.nan,
                    "interval_consumption": math.nan,
                    "interval_bridge_status": "STOCK ENDPOINTS AUTHORITATIVE; INTERVAL MICRO FLOWS NOT PERSISTED",
                    "employed_member_count": math.nan,
                    "employed_member_count_status": "UNAVAILABLE_NOT_PERSISTED",
                    "exact_member_age_structure_status": "UNAVAILABLE; adult/child counts only",
                    "diagnosis": "SNAPSHOT_SAMPLING_LIMITATION",
                }
            )
        previous = current

    # The macro identity confirms current Household income is wage income, but
    # it cannot identify the number of employed members in each Household.
    result = pd.DataFrame(rows)
    result["macro_income_equals_wage_income_max_gap"] = float(
        (macro["household_income"] - macro["household_wage_income"]).abs().max()
    )
    result["initial_buffer_fixed"] = True
    result["opening_near_zero_share"] = float(
        (social.loc[social["global_step"].eq(-1), "cash"] <= NEAR_ZERO_CASH).mean()
    )
    return result


def build_loan_advance_mapping(
    raw_firms: pd.DataFrame,
    accounting: pd.DataFrame,
    firm_panel: pd.DataFrame,
) -> pd.DataFrame:
    raw = raw_firms.loc[raw_firms["sector_id"].eq("capital_goods")].copy()
    raw = raw.sort_values("global_step")
    acc = accounting_firms(accounting)
    acc = acc.loc[acc["firm_id_numeric"].eq(100000)].sort_values("global_step")
    panel = firm_panel.loc[firm_panel["sector"].eq("capital_goods")].copy()
    panel = panel.sort_values("global_step")
    values = raw[[
        "global_step", "loan_balance", "loan_issued", "loan_repaid",
        "opening_principal", "closing_principal", "customer_advance_liability",
    ]].rename(columns={
        "loan_balance": "runtime_true_loan_principal",
        "loan_issued": "runtime_loan_issued",
        "loan_repaid": "runtime_loan_repaid",
        "opening_principal": "runtime_opening_principal",
        "closing_principal": "runtime_closing_principal",
        "customer_advance_liability": "runtime_customer_advance_liability",
    })
    values = values.merge(
        acc[[
            "global_step", "loan_balance", "total_liabilities",
            "revolving_principal_claim", "customer_advance_liability",
        ]].rename(columns={
            "loan_balance": "accounting_export_loan_balance",
            "total_liabilities": "accounting_total_liabilities",
            "revolving_principal_claim": "accounting_revolving_principal_claim",
            "customer_advance_liability": "accounting_customer_advance_liability",
        }),
        on="global_step",
        how="inner",
    )
    values = values.merge(
        panel[["global_step", "principal", "customer_advance_liability"]].rename(
            columns={
                "principal": "firm_panel_principal",
                "customer_advance_liability": "firm_panel_customer_advance_liability",
            }
        ),
        on="global_step",
        how="inner",
    )
    values["analysis_query_loan_principal"] = values["firm_panel_principal"]
    values["gui_displayed_loan_principal"] = values["analysis_query_loan_principal"]
    values["runtime_principal_plus_advance"] = (
        values["runtime_true_loan_principal"]
        + values["runtime_customer_advance_liability"]
    )
    values["accounting_export_semantic_gap"] = (
        values["accounting_export_loan_balance"]
        - values["runtime_true_loan_principal"]
    )
    values["accounting_total_liability_bridge_gap"] = (
        values["accounting_total_liabilities"]
        - values["runtime_principal_plus_advance"]
    )
    values["panel_propagation_gap"] = (
        values["firm_panel_principal"] - values["accounting_export_loan_balance"]
    )
    values["displayed_principal_equals_advance"] = np.isclose(
        values["gui_displayed_loan_principal"],
        values["runtime_customer_advance_liability"],
        atol=1e-9,
        rtol=0.0,
    )
    values["actual_loan_exists"] = values["runtime_true_loan_principal"].abs() > 1e-9
    values["row_type"] = "WEEK_VALUE_TRACE"
    values["pipeline_stage"] = "all numeric stages"
    values["stage_status"] = "CONTAMINATED_AFTER_ACCOUNTING_EXPORT"
    values["source_or_binding"] = "see PIPELINE_STAGE rows"
    values["first_authoritative_divergence"] = "economy/accounting.py:355"
    values["diagnosis"] = "CUSTOMER_ADVANCE_MISLABELED_AS_LOAN_PRINCIPAL"

    selected = values.set_index("global_step")
    stage_specs = [
        ("runtime Firm", "firm_diagnostics.csv:loan_balance", "PASS_TRUE_PRINCIPAL", "0 at weeks 480/481"),
        ("runtime advance", "firm_diagnostics.csv:customer_advance_liability", "PASS_SEPARATE_LIABILITY", "467000.161057 / 648144.841014"),
        ("AccountingLayer export", "economy/accounting.py:245-248,355,357", "FAIL_FIRST_DIVERGENCE", "total liabilities are written into loan_balance and revolving_principal_claim"),
        ("Firm panel materialization", "test/step15_final_canonical_consolidation.py:227-231", "PROPAGATES_UPSTREAM_ERROR", "principal prefers AccountingLayer loan_balance"),
        ("Analysis V2 query", "analysis/gui_v2/query.py:381,445", "PROPAGATES_PANEL_FIELD", "principal and customer advance remain separate names but principal value is already contaminated"),
        ("Metric registry", "analysis/gui_v2/registry.py:90-92", "SEMANTICS_CORRECT_SOURCE_CONTAMINATED", "definitions correctly distinguish loan principal from customer advances"),
        ("GUI binding", "analysis/gui_v2/pages.py:418-425", "DISPLAYS_CONTAMINATED_PANEL_VALUE", "two separate series are drawn, but both values coincide for the supplier"),
    ]
    stages = []
    for stage, source, status, note in stage_specs:
        stages.append({
            "row_type": "PIPELINE_STAGE",
            "global_step": math.nan,
            "pipeline_stage": stage,
            "stage_status": status,
            "source_or_binding": source,
            "stage_note": note,
            "runtime_true_loan_principal_week_480": selected.loc[480, "runtime_true_loan_principal"],
            "runtime_customer_advance_week_480": selected.loc[480, "runtime_customer_advance_liability"],
            "displayed_loan_principal_week_480": selected.loc[480, "gui_displayed_loan_principal"],
            "runtime_true_loan_principal_week_481": selected.loc[481, "runtime_true_loan_principal"],
            "runtime_customer_advance_week_481": selected.loc[481, "runtime_customer_advance_liability"],
            "displayed_loan_principal_week_481": selected.loc[481, "gui_displayed_loan_principal"],
            "first_authoritative_divergence": "economy/accounting.py:355" if stage != "runtime Firm" else "none",
            "diagnosis": "CUSTOMER_ADVANCE_MISLABELED_AS_LOAN_PRINCIPAL",
        })
    return pd.concat([values, pd.DataFrame(stages)], ignore_index=True, sort=False)


def infer_order_sources(chain: pd.DataFrame) -> pd.Series:
    settlements = chain.loc[
        chain["event_type"].eq("investment_order_settled")
        & chain["order_id"].notna()
    ].sort_values("global_step")
    return settlements.groupby("order_id")["investment_source"].first()


def build_advance_cycle(
    macro: pd.DataFrame,
    raw_firms: pd.DataFrame,
    accounting: pd.DataFrame,
    chain: pd.DataFrame,
) -> pd.DataFrame:
    cap_raw = raw_firms.loc[raw_firms["sector_id"].eq("capital_goods")].copy()
    cap_raw = cap_raw.sort_values("global_step")
    cap_acc = accounting_firms(accounting)
    cap_acc = cap_acc.loc[cap_acc["firm_id_numeric"].eq(100000)].sort_values("global_step")
    sources = infer_order_sources(chain)
    advance_events = chain.loc[chain["event_type"].eq("customer_advance_received")].copy()
    advance_events["inferred_source"] = advance_events["order_id"].map(sources)
    delivery_events = chain.loc[
        chain["event_type"].eq("customer_advance_delivery_recognition")
    ].copy()
    delivery_events["inferred_source"] = delivery_events["order_id"].map(sources)

    def source_flow(events, value_field, prefix):
        grouped = (
            events.groupby(["global_step", "inferred_source"])[value_field]
            .sum()
            .unstack(fill_value=0.0)
        )
        return grouped.rename(columns=lambda value: f"{prefix}_{str(value).lower()}")

    advances = source_flow(advance_events, "cash_amount", "new_advance")
    deliveries = source_flow(delivery_events, "revenue_recognized", "delivery_revenue")
    delivery_units = source_flow(delivery_events, "physical_units", "delivery_units")

    frame = macro[[
        "global_step", "customer_advances_received", "customer_advances_delivered",
        "replacement_backlog", "expansion_backlog", "total_backlog",
        "active_capital_assets", "active_capital_service",
    ]].merge(
        cap_raw[[
            "global_step", "cash", "employee_count", "desired_labor",
            "capital_good_production_units", "capital_good_sales_units",
            "customer_advance_liability", "customer_advance_received",
            "customer_advance_delivered", "wage_payment",
        ]],
        on="global_step",
        how="left",
    ).merge(
        cap_acc[["global_step", "cash_start", "cash_end", "customer_advance_liability_bridge_gap"]],
        on="global_step",
        how="left",
    )
    frame = frame.join(advances, on="global_step").join(deliveries, on="global_step").join(delivery_units, on="global_step")
    for column in (
        "new_advance_expansion", "new_advance_replacement",
        "delivery_revenue_expansion", "delivery_revenue_replacement",
        "delivery_units_expansion", "delivery_units_replacement",
    ):
        if column not in frame:
            frame[column] = 0.0
        frame[column] = frame[column].fillna(0.0)
    frame["opening_customer_advance_liability"] = frame[
        "customer_advance_liability"
    ].shift(fill_value=0.0)
    frame["new_customer_advances"] = frame["customer_advance_received"]
    frame["delivered_recognized_advances"] = frame["customer_advance_delivered"]
    frame["closing_customer_advance_liability"] = frame[
        "customer_advance_liability"
    ]
    frame["recomputed_advance_bridge_gap"] = (
        frame["closing_customer_advance_liability"]
        - frame["opening_customer_advance_liability"]
        - frame["new_customer_advances"]
        + frame["delivered_recognized_advances"]
    )

    review_steps = frame.loc[frame["new_customer_advances"] > 1e-9, "global_step"].astype(int).tolist()
    frame["cycle_start_step"] = math.nan
    frame["cycle_end_step"] = math.nan
    frame["cycle_liability_peak"] = math.nan
    frame["cycle_liability_trough"] = math.nan
    frame["cycle_liability_amplitude"] = math.nan
    frame["cycle_cash_amplitude"] = math.nan
    frame["cycle_delivery_total"] = math.nan
    for index, start in enumerate(review_steps):
        end = review_steps[index + 1] - 1 if index + 1 < len(review_steps) else int(frame["global_step"].max())
        mask = frame["global_step"].between(start, end)
        cycle = frame.loc[mask]
        peak = float(cycle["closing_customer_advance_liability"].max())
        trough = float(cycle["closing_customer_advance_liability"].min())
        frame.loc[mask, "cycle_start_step"] = start
        frame.loc[mask, "cycle_end_step"] = end
        frame.loc[mask, "cycle_liability_peak"] = peak
        frame.loc[mask, "cycle_liability_trough"] = trough
        frame.loc[mask, "cycle_liability_amplitude"] = peak - trough
        frame.loc[mask, "cycle_cash_amplitude"] = float(cycle["cash"].max() - cycle["cash"].min())
        frame.loc[mask, "cycle_delivery_total"] = float(cycle["delivered_recognized_advances"].sum())
    frame["cycle_interpretation"] = (
        "Advances arrive mainly on the synchronized 13-week review clock; payroll funds weekly production and deliveries release the liability. After week 260, new advances are replacement orders, and larger throughput/replacement cohorts increase within-cycle swings while the liability stock trends down rather than exploding."
    )
    return frame


def build_capital_good_profit_bridge(
    raw_firms: pd.DataFrame,
    accounting: pd.DataFrame,
) -> pd.DataFrame:
    raw = raw_firms.loc[raw_firms["sector_id"].eq("capital_goods")].copy()
    raw = raw[[
        "global_step", "price",
        "capital_good_production_units", "capital_good_sales_units",
        "wage_payment", "customer_advance_received", "customer_advance_delivered",
    ]]
    # The frozen panel persists the cost-anchored offer price, but not a separate
    # unit-cost column. Keep the derivation source explicit instead of presenting
    # a reconstructed value as persisted runtime history.
    raw["capital_good_unit_production_cost"] = 41.0 / 3.0
    raw["unit_cost_source"] = "ACCEPTED_CONTRACT_DERIVATION_41_DIV_3"
    raw["price_cost_anchor_gap"] = raw["price"] - raw["capital_good_unit_production_cost"]
    acc = accounting_firms(accounting)
    acc = acc.loc[acc["firm_id_numeric"].eq(100000)].copy()
    fields = [
        "global_step", "sales_revenue", "other_existing_operating_revenue",
        "accounting_revenue", "production_wage_cost", "manufacturing_cost",
        "wage_expense", "inventory_cost_or_cogs", "spoilage_or_inventory_loss",
        "capital_depreciation_expense", "accounting_operating_profit",
        "accounting_net_income", "interest_paid", "customer_advance_cash_inflow",
        "sales_collections", "cfo", "cash_start", "cash_end",
    ]
    frame = raw.merge(acc[fields], on="global_step", how="inner")
    frame["row_type"] = "WEEKLY"
    frame["offer_price_contract"] = "authoritative_unit_cost = 41 / 3"
    frame["cost_anchor_price"] = 41.0 / 3.0
    frame["production_payroll_capitalized_to_inventory"] = frame["manufacturing_cost"]
    frame["period_payroll_expense"] = frame["wage_expense"]
    frame["sales_gross_margin"] = frame["sales_revenue"] - frame["inventory_cost_or_cogs"]
    frame["operating_profit_recomputed"] = (
        frame["accounting_revenue"]
        - frame["inventory_cost_or_cogs"]
        - frame["spoilage_or_inventory_loss"]
        - frame["period_payroll_expense"]
        - frame["capital_depreciation_expense"]
    )
    frame["profit_bridge_gap"] = (
        frame["accounting_operating_profit"] - frame["operating_profit_recomputed"]
    )
    frame["zero_margin_sales_gap"] = frame["sales_revenue"] - frame["inventory_cost_or_cogs"]
    frame["classification"] = np.where(
        frame["global_step"].eq(0) & (frame["other_existing_operating_revenue"] > 1e-9),
        "STARTUP_CAPITALIZATION_ACCOUNTING_CLASSIFICATION_EXCEPTION",
        "INTENTIONAL_ZERO_MARGIN_ENGINEERING_CONTRACT",
    )
    frame["interpretation"] = (
        "Production payroll is inventoried and expensed as COGS at delivery; subtracting both payroll and COGS would double count. Cost-anchored price makes delivered revenue equal historical COGS. Customer advances are cash/liability movements, not revenue."
    )

    cumulative_fields = [
        "sales_revenue", "other_existing_operating_revenue", "accounting_revenue",
        "production_wage_cost", "manufacturing_cost", "wage_expense",
        "inventory_cost_or_cogs", "spoilage_or_inventory_loss",
        "capital_depreciation_expense", "accounting_operating_profit",
        "accounting_net_income", "interest_paid", "customer_advance_cash_inflow",
        "sales_collections", "cfo", "capital_good_production_units",
        "capital_good_sales_units", "production_payroll_capitalized_to_inventory",
        "period_payroll_expense", "sales_gross_margin", "operating_profit_recomputed",
        "profit_bridge_gap", "zero_margin_sales_gap",
    ]
    cumulative = {column: float(frame[column].sum()) for column in cumulative_fields}
    cumulative.update({
        "row_type": "CUMULATIVE",
        "global_step": int(frame["global_step"].max()),
        "offer_price_contract": "authoritative_unit_cost = 41 / 3",
        "cost_anchor_price": 41.0 / 3.0,
        "classification": "INTENTIONAL_ZERO_MARGIN_ENGINEERING_CONTRACT_WITH_STARTUP_CLASSIFICATION_EXCEPTION",
        "interpretation": "All sale revenue equals COGS within floating tolerance; the cumulative positive operating profit is the one-time startup-capitalization amount classified as other operating revenue at week 0.",
    })
    return pd.concat([frame, pd.DataFrame([cumulative])], ignore_index=True, sort=False)


def build_sector_labor_reallocation(
    macro: pd.DataFrame,
    raw_firms: pd.DataFrame,
    labor_events: pd.DataFrame,
) -> pd.DataFrame:
    food = raw_firms.loc[raw_firms["sector_id"].eq("food")]
    capital = raw_firms.loc[raw_firms["sector_id"].eq("capital_goods")]
    food_week = food.groupby("global_step", as_index=False).agg(
        desired_labor=("desired_labor", "sum"),
        desired_investment=("desired_investment_expenditure", "sum"),
        executed_investment=("executed_investment", "sum"),
    )
    capital_week = capital.groupby("global_step", as_index=False).agg(
        desired_labor=("desired_labor", "sum"),
        desired_investment=("desired_investment_expenditure", "sum"),
        executed_investment=("executed_investment", "sum"),
    )
    event_names = ["hire", "release", "sector_transfer"]
    event_rows = []
    for step, group in labor_events.groupby("global_step"):
        for sector in ("food", "capital_goods"):
            event_rows.append({
                "global_step": step,
                "sector": sector,
                "hires_into_sector": int(((group["event_type"] == "hire") & (group["to_sector"] == sector)).sum()),
                "releases_from_sector": int(((group["event_type"] == "release") & (group["from_sector"] == sector)).sum()),
                "transfers_into_sector": int(((group["event_type"] == "sector_transfer") & (group["to_sector"] == sector)).sum()),
                "transfers_out_of_sector": int(((group["event_type"] == "sector_transfer") & (group["from_sector"] == sector)).sum()),
            })
    event_frame = pd.DataFrame(event_rows)

    rows = []
    for sector, employment_field, production_field, weekly in (
        ("food", "food_employment", "food_production", food_week),
        ("capital_goods", "capital_good_employment", "capital_good_production", capital_week),
    ):
        frame = macro[[
            "global_step", employment_field, production_field,
            "total_employment", "unassigned_labor", "active_capital_service",
            "active_capital_assets", "total_backlog", "replacement_backlog",
            "expansion_backlog", "total_fixed_investment",
        ]].rename(columns={employment_field: "employment", production_field: "production"})
        frame = frame.merge(weekly, on="global_step", how="left")
        frame["sector"] = sector
        frame["output_per_worker"] = frame["production"].div(
            frame["employment"].replace(0, np.nan)
        )
        frame["sector_employment_share"] = frame["employment"].div(
            frame["total_employment"].replace(0, np.nan)
        )
        frame["capital_service"] = (
            frame["active_capital_service"] if sector == "food" else 0.0
        )
        frame["capital_intensity_per_worker"] = frame["capital_service"].div(
            frame["employment"].replace(0, np.nan)
        )
        frame = frame.merge(event_frame, on=["global_step", "sector"], how="left")
        for column in (
            "hires_into_sector", "releases_from_sector",
            "transfers_into_sector", "transfers_out_of_sector",
        ):
            frame[column] = frame[column].fillna(0).astype(int)
        frame["window"] = pd.cut(
            frame["global_step"],
            bins=[-1, 51, 259, 519],
            labels=["EARLY_0_51", "MIDDLE_52_259", "LATE_260_519"],
        ).astype(str)
        frame["classification"] = "VALID_CAPITAL_DEEPENING_WITH_ENGINEERING_LIFECYCLE_CAVEAT"
        frame["interpretation"] = (
            "Capital-goods hiring is backed by replacement/expansion backlog and real production; Food labor falls while capital service rises. Assignment gaps remain zero and late backlog declines. The 52-week useful life and P3 productivity remain engineering screens, so magnitude is not calibrated."
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True, sort=False).sort_values(["global_step", "sector"])


def structural_profit_reason(row) -> str:
    gross = row["accounting_revenue"] - row["inventory_cost_or_cogs"] - row["wage_expense"]
    after_spoilage = gross - row["spoilage_or_inventory_loss"]
    after_depreciation = after_spoilage - row["capital_depreciation_expense"]
    if after_depreciation >= -1e-9:
        return "NONNEGATIVE_OPERATING_PROFIT"
    if gross < -1e-9:
        return "COGS_EXCEEDS_REVENUE"
    if after_spoilage < -1e-9:
        return "SPOILAGE_ERODES_GROSS_MARGIN"
    return "DEPRECIATION_EXCEEDS_POST_SPOILAGE_SURPLUS"


def build_food_profitability(
    raw_firms: pd.DataFrame,
    accounting: pd.DataFrame,
) -> pd.DataFrame:
    raw = raw_firms.loc[raw_firms["sector_id"].eq("food")].copy()
    raw["firm_id_numeric"] = pd.to_numeric(raw["firm_id"], errors="coerce")
    raw_fields = [
        "global_step", "firm_id_numeric", "price", "unit_labor_cost",
        "realized_ulc", "margin", "sales_units", "sales_revenue",
        "actual_production", "employee_count", "inventory_units", "cash",
        "capital_capacity", "actual_market_share",
    ]
    acc = accounting_firms(accounting)
    acc = acc.loc[acc["firm_id_numeric"].between(0, 4)].copy()
    acc_fields = [
        "global_step", "firm_id_numeric", "sales_revenue",
        "other_existing_operating_revenue", "accounting_revenue",
        "production_wage_cost", "manufacturing_cost", "wage_expense",
        "inventory_cost_or_cogs", "spoilage_or_inventory_loss",
        "capital_depreciation_expense", "accounting_operating_profit",
        "accounting_net_income", "interest_paid", "dividends", "loan_issued",
        "principal_repaid", "cfo", "cfi", "cff", "inventory_book_value",
        "unit_production_cost",
    ]
    frame = acc[acc_fields].merge(raw[raw_fields], on=["global_step", "firm_id_numeric"], how="inner", suffixes=("_accounting", "_runtime"))
    frame = frame.rename(columns={"firm_id_numeric": "firm_id"})
    frame["row_type"] = "WEEKLY_FIRM"
    frame["production_payroll_capitalized_to_inventory"] = frame["manufacturing_cost"]
    frame["period_payroll_expense"] = frame["wage_expense"]
    frame["gross_profit_before_spoilage"] = (
        frame["accounting_revenue"]
        - frame["inventory_cost_or_cogs"]
        - frame["period_payroll_expense"]
    )
    frame["profit_after_spoilage_before_depreciation"] = (
        frame["gross_profit_before_spoilage"] - frame["spoilage_or_inventory_loss"]
    )
    frame["operating_profit_recomputed"] = (
        frame["profit_after_spoilage_before_depreciation"]
        - frame["capital_depreciation_expense"]
    )
    frame["profit_bridge_gap"] = (
        frame["accounting_operating_profit"] - frame["operating_profit_recomputed"]
    )
    frame["price_cost_margin_runtime"] = frame["margin"]
    frame["accounting_gross_margin_rate"] = frame["gross_profit_before_spoilage"].div(
        frame["accounting_revenue"].replace(0, np.nan)
    )
    frame["inventory_units_change"] = frame.groupby("firm_id")["inventory_units"].diff()
    frame["structural_profit_reason"] = frame.apply(structural_profit_reason, axis=1)
    frame["payroll_double_count_warning"] = (
        "Production payroll is inventory cost and reaches profit through COGS; only wage_expense is an additional current-period payroll expense."
    )

    cumulative_rows = []
    flow_fields = [
        "sales_revenue_accounting", "other_existing_operating_revenue",
        "accounting_revenue", "production_wage_cost", "manufacturing_cost",
        "wage_expense", "inventory_cost_or_cogs", "spoilage_or_inventory_loss",
        "capital_depreciation_expense", "accounting_operating_profit",
        "accounting_net_income", "interest_paid", "dividends", "loan_issued",
        "principal_repaid", "cfo", "cfi", "cff",
    ]
    for firm_id, group in frame.groupby("firm_id"):
        row = {column: float(group[column].sum()) for column in flow_fields}
        row.update({
            "row_type": "CUMULATIVE_FIRM",
            "global_step": int(group["global_step"].max()),
            "firm_id": int(firm_id),
            "production_payroll_capitalized_to_inventory": float(group["manufacturing_cost"].sum()),
            "period_payroll_expense": float(group["wage_expense"].sum()),
            "gross_profit_before_spoilage": float(group["accounting_revenue"].sum() - group["inventory_cost_or_cogs"].sum() - group["wage_expense"].sum()),
            "profit_after_spoilage_before_depreciation": float(group["accounting_revenue"].sum() - group["inventory_cost_or_cogs"].sum() - group["wage_expense"].sum() - group["spoilage_or_inventory_loss"].sum()),
            "operating_profit_recomputed": float(group["accounting_revenue"].sum() - group["inventory_cost_or_cogs"].sum() - group["wage_expense"].sum() - group["spoilage_or_inventory_loss"].sum() - group["capital_depreciation_expense"].sum()),
            "profit_bridge_gap": float(group["profit_bridge_gap"].sum()),
            "structural_profit_reason": "CUMULATIVE_DEPRECIATION_EXCEEDS_POST_SPOILAGE_SURPLUS",
            "payroll_double_count_warning": "Production payroll is already capitalized into inventory/COGS.",
        })
        cumulative_rows.append(row)

    sector = frame.groupby("global_step", as_index=False)[[
        "accounting_revenue", "inventory_cost_or_cogs", "wage_expense",
        "spoilage_or_inventory_loss", "capital_depreciation_expense",
        "accounting_operating_profit",
    ]].sum()
    first_negative = int(sector.loc[sector["accounting_operating_profit"] < -1e-9, "global_step"].iloc[0])
    sustained_candidates = []
    negative = sector["accounting_operating_profit"] < -1e-9
    for step in sector.loc[negative, "global_step"]:
        tail = negative.loc[sector["global_step"].between(step, min(step + 25, 519))]
        if len(tail) >= 13 and bool(tail.all()):
            sustained_candidates.append(int(step))
            break
    sector_row = {
        "row_type": "CUMULATIVE_SECTOR",
        "global_step": int(frame["global_step"].max()),
        "firm_id": "ALL_FOOD",
        "sales_revenue_accounting": float(frame["sales_revenue_accounting"].sum()),
        "accounting_revenue": float(frame["accounting_revenue"].sum()),
        "production_wage_cost": float(frame["production_wage_cost"].sum()),
        "manufacturing_cost": float(frame["manufacturing_cost"].sum()),
        "wage_expense": float(frame["wage_expense"].sum()),
        "inventory_cost_or_cogs": float(frame["inventory_cost_or_cogs"].sum()),
        "spoilage_or_inventory_loss": float(frame["spoilage_or_inventory_loss"].sum()),
        "capital_depreciation_expense": float(frame["capital_depreciation_expense"].sum()),
        "accounting_operating_profit": float(frame["accounting_operating_profit"].sum()),
        "accounting_net_income": float(frame["accounting_net_income"].sum()),
        "interest_paid": float(frame["interest_paid"].sum()),
        "dividends": float(frame["dividends"].sum()),
        "loan_issued": float(frame["loan_issued"].sum()),
        "principal_repaid": float(frame["principal_repaid"].sum()),
        "production_payroll_capitalized_to_inventory": float(frame["manufacturing_cost"].sum()),
        "period_payroll_expense": float(frame["wage_expense"].sum()),
        "gross_profit_before_spoilage": float(frame["accounting_revenue"].sum() - frame["inventory_cost_or_cogs"].sum() - frame["wage_expense"].sum()),
        "profit_after_spoilage_before_depreciation": float(frame["accounting_revenue"].sum() - frame["inventory_cost_or_cogs"].sum() - frame["wage_expense"].sum() - frame["spoilage_or_inventory_loss"].sum()),
        "operating_profit_recomputed": float(frame["accounting_revenue"].sum() - frame["inventory_cost_or_cogs"].sum() - frame["wage_expense"].sum() - frame["spoilage_or_inventory_loss"].sum() - frame["capital_depreciation_expense"].sum()),
        "profit_bridge_gap": float(frame["profit_bridge_gap"].sum()),
        "first_negative_sector_week": first_negative,
        "first_sustained_negative_sector_week": sustained_candidates[0] if sustained_candidates else math.nan,
        "structural_profit_reason": "SPOILAGE_ERODES_MIDRUN_MARGIN; DEPRECIATION_DOMINATES_FINAL_CUMULATIVE_LOSS",
        "payroll_double_count_warning": "Production payroll is already capitalized into inventory/COGS.",
    }
    return pd.concat([frame, pd.DataFrame(cumulative_rows + [sector_row])], ignore_index=True, sort=False)


def build_competition_audit(
    raw_firms: pd.DataFrame,
    firm_capital: pd.DataFrame,
) -> pd.DataFrame:
    food = raw_firms.loc[raw_firms["sector_id"].eq("food")].copy()
    food["firm_id_key"] = food["firm_id"].astype(str)
    capital = firm_capital.loc[firm_capital["sector_id"].eq("food"), [
        "global_step", "firm_id", "active_capital_service",
    ]].copy()
    capital["firm_id_key"] = capital["firm_id"].astype(str)
    food = food.merge(
        capital[["global_step", "firm_id_key", "active_capital_service"]],
        on=["global_step", "firm_id_key"],
        how="left",
    )
    fields = {
        "price": "price",
        "unit_cost": "unit_labor_cost",
        "margin": "margin",
        "sales": "sales_units",
        "revenue": "sales_revenue",
        "production": "actual_production",
        "employment": "employee_count",
        "inventory": "inventory_units",
        "cash": "cash",
        "capital_service": "active_capital_service",
    }
    rows = []
    for step, group in food.groupby("global_step", sort=True):
        sales = pd.to_numeric(group["sales_units"], errors="coerce").fillna(0.0)
        shares = sales / sales.sum() if float(sales.sum()) > 1e-15 else pd.Series(np.repeat(0.2, len(group)), index=group.index)
        row = {
            "row_type": "WEEKLY",
            "global_step": int(step),
            "firm_count": int(group["firm_id"].nunique()),
            "sales_hhi": float((shares ** 2).sum()),
            "largest_sales_share": float(shares.max()),
            "smallest_sales_share": float(shares.min()),
            "choice_probability_cv": population_cv(group["choice_probability"]),
            "actual_share_cv": population_cv(shares),
            "mean_abs_choice_probability_minus_realized_share": float((pd.to_numeric(group["choice_probability"], errors="coerce") - shares).abs().mean()),
            "price_share_cross_section_correlation": float(pd.to_numeric(group["price"], errors="coerce").corr(shares)),
            "production_review_phases_synchronized": int(group["production_review_phase_offset"].nunique()) == 1,
            "investment_review_phases_synchronized": int(group["investment_review_phase_offset"].nunique()) == 1,
            "classification": "PARAMETER_SYMMETRY",
            "demand_allocation_contract": "Household-by-Household weighted stochastic choice with weights price^(-2); not a fixed 20% split.",
            "interpretation": "HHI near 0.2 is the minimum for five similarly parameterized active Firms. Price differences affect shares, but common technology, labor/wage rules and synchronized reviews keep differentiation modest.",
        }
        for label, field in fields.items():
            values = pd.to_numeric(group[field], errors="coerce")
            row[f"{label}_mean"] = float(values.mean())
            row[f"{label}_minimum"] = float(values.min())
            row[f"{label}_maximum"] = float(values.max())
            row[f"{label}_range"] = float(values.max() - values.min())
            row[f"{label}_cv"] = population_cv(values)
        rows.append(row)
    result = pd.DataFrame(rows)
    summary = {
        "row_type": "RUN_SUMMARY",
        "global_step": int(result["global_step"].max()),
        "firm_count": 5,
        "sales_hhi": float(result["sales_hhi"].mean()),
        "sales_hhi_minimum": float(result["sales_hhi"].min()),
        "sales_hhi_maximum": float(result["sales_hhi"].max()),
        "price_share_cross_section_correlation": float(result["price_share_cross_section_correlation"].mean()),
        "production_review_phases_synchronized": True,
        "investment_review_phases_synchronized": True,
        "classification": "PARAMETER_SYMMETRY",
        "demand_allocation_contract": "Active price-sensitive consumer choice; no structural equal-share allocator.",
        "interpretation": "Competition is active but the experiment supplies little structural heterogeneity and no entry/exit. Near-minimum HHI alone is not evidence of a mapping bug or exact equal allocation.",
    }
    for label in fields:
        summary[f"{label}_cv"] = float(result[f"{label}_cv"].mean())
    return pd.concat([result, pd.DataFrame([summary])], ignore_index=True, sort=False)


def build_gui_coverage() -> pd.DataFrame:
    rows = [
        ("price", "firm_diagnostics.csv:price", True, False, "none", "AVAILABLE_BUT_NOT_LOADED", "Firm small multiples + cross-section ranked chart", "price vs market share"),
        ("unit cost", "firm_diagnostics.csv:unit_labor_cost; accounting:unit_production_cost", True, True, "accounting table only", "AVAILABLE_BUT_NOT_IN_FIRM_COMPARISON", "Firm small multiples", "price-cost margin vs market share"),
        ("price-cost margin", "firm_diagnostics.csv:margin", True, False, "none", "AVAILABLE_BUT_NOT_LOADED", "Firm small multiples + ranked chart", "margin vs market share"),
        ("profit margin", "derived:operating_profit/revenue", True, True, "Firm cross-section table", "EXPOSED_TABLE_ONLY", "Comparative time series/ranked chart", "profit margin vs market share"),
        ("sales", "step15_firm_panel.csv:sales", True, True, "indexed sales, ranked sales, market-share area", "EXPOSED", "keep", "sales vs price"),
        ("revenue", "step15_firm_panel.csv:revenue", True, True, "single-Firm line + cross-section table", "PARTIALLY_EXPOSED", "multi-Firm comparison", "revenue vs market share"),
        ("market share", "derived:sales/sector sales", True, True, "share area + HHI", "EXPOSED", "keep", "price and margin vs market share"),
        ("inventory", "step15_firm_panel.csv:inventory", True, True, "cross-section table", "EXPOSED_TABLE_ONLY", "Firm small multiples", "inventory vs price"),
        ("employment", "step15_firm_panel.csv:employment", True, True, "cross-section table", "EXPOSED_TABLE_ONLY", "Firm small multiples", "employment vs output"),
        ("output per worker", "derived:production/employment", True, True, "none", "DERIVABLE_BUT_HIDDEN", "Firm small multiples/ranked chart", "capital intensity vs productivity"),
        ("capital service", "firm_capital_history.csv:active_capital_service", True, True, "Firm capital-service small multiples + table", "EXPOSED", "keep", "capital service vs output"),
        ("capital intensity", "derived:active_capital_service/employment", True, True, "none", "DERIVABLE_BUT_HIDDEN", "Firm small multiples/ranked chart", "capital intensity vs productivity"),
        ("cash", "step15_firm_panel.csv:cash", True, True, "single-Firm line + cross-section table", "PARTIALLY_EXPOSED", "multi-Firm comparison", "cash vs profitability"),
        ("true loan principal", "firm_diagnostics.csv:loan_balance", True, False, "GUI shows contaminated panel principal", "BLOCKED_BY_ACCOUNTING_EXPORT_MAPPING_BUG", "Restore after source correction", "true principal vs advance liability"),
        ("customer advance liability", "step15_firm_panel.csv:customer_advance_liability", True, True, "single-Firm and sector balance charts", "EXPOSED_CORRECTLY", "keep separate from loan", "advance liability vs supplier cash"),
    ]
    frame = pd.DataFrame(rows, columns=[
        "metric", "authoritative_source", "authoritative_available",
        "loaded_or_derivable_by_analysis_v2", "current_gui_exposure", "coverage_status",
        "recommended_comparison_view", "recommended_relational_view",
    ])
    frame["semantic_validation"] = np.where(
        frame["metric"].eq("true loan principal"),
        "FAILED_UPSTREAM_SOURCE_MAPPING",
        "PASSED",
    )
    frame["implementation_status"] = "RECOMMENDATION_ONLY_NO_GUI_CHANGE"
    return frame


def build_classification() -> pd.DataFrame:
    rows = [
        ("settlement-only account meaning", "EXPECTED_ACCOUNT_LIFECYCLE", "Temporary economic settlement account, separate from social Household identity.", False),
        ("annual settlement/Social cash synchronization", "ACCOUNTING_DEFINITION_ISSUE", "Marriage closes settlement accounts, but transferred settlement cash is also included in the pending-formation transfer; pending cash becomes equally negative.", True),
        ("Household Gini/quantile line appearance", "SNAPSHOT_SAMPLING_LIMITATION", "Micro observations exist only on the 13-week grid through week 507.", False),
        ("near-zero Social Household cash", "ENDOGENOUS_DISTRIBUTIONAL_RESULT", "Near-zero households have much lower wage income and frequently non-positive saving; dependency and settlement origin are not primary in available data.", True),
        ("capital-good loan principal equals advance", "GUI_METRIC_MAPPING_BUG", "Runtime loan is zero; AccountingLayer writes total liabilities into loan_balance and the panel/GUI propagate it.", True),
        ("customer-advance sawtooth", "EXPECTED_ACCOUNT_LIFECYCLE", "13-week order advances fund weekly supplier payroll/production and are released on delivery; late advances are replacement orders.", False),
        ("capital-good operating profit near zero", "ZERO_MARGIN_ENGINEERING_SIMPLIFICATION", "Cost-anchored price equals historical unit cost; sales revenue equals COGS. Week-0 startup inflow is a classification exception.", False),
        ("capital-good employment rises while Food falls", "VALID_CAPITAL_DEEPENING", "Real backlog, production and delivery support hiring; assignment closes and late backlog declines, subject to non-calibrated lifecycle/productivity scale.", False),
        ("Food revenue rises while operating profit is negative", "FIRM_PROFITABILITY_PROBLEM", "Spoilage consumes most gross margin and depreciation exceeds the remaining cumulative operating surplus.", True),
        ("Food HHI near 0.2", "INSUFFICIENT_COMPETITION", "Choice is price-sensitive rather than fixed-equal, but common technology/strategy and synchronized review phases limit differentiation.", True),
        ("Firm-comparison metrics hidden", "OTHER", "Price, unit cost, margins, output/worker and capital intensity are persisted or derivable but not all exposed in GUI V2.", False),
    ]
    return pd.DataFrame(rows, columns=[
        "manual_review_issue", "classification", "evidence", "independent_model_boundary",
    ])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hashes_before = {name: sha256(path) for name, path in INPUTS.items()}

    macro = read("macro").sort_values("global_step").reset_index(drop=True)
    firm_panel = read("firms").sort_values(["global_step", "firm_id"]).reset_index(drop=True)
    accounting = read("accounting").sort_values(["global_step", "record_type", "firm_id"], na_position="last").reset_index(drop=True)
    raw_firms = read("raw_firms").sort_values(["global_step", "firm_id"]).reset_index(drop=True)
    chain = read("chain").sort_values("global_step").reset_index(drop=True)
    demographic_events = read("demographic_events")
    social_all = read("social_households")
    settlement_all = read("settlement_accounts")
    labor_events = read("labor_events")
    firm_capital = read("firm_capital")

    social = social_all.loc[social_all["observation_phase"].isin(SNAPSHOT_PHASES)].copy()
    settlement = settlement_all.loc[settlement_all["observation_phase"].isin(SNAPSHOT_PHASES)].copy()

    cycle = build_household_cycle(macro, settlement_all, demographic_events)
    grain = build_household_grain_registry(macro, social, settlement)
    near_zero = build_near_zero_diagnosis(social, settlement_all, macro)
    loan_mapping = build_loan_advance_mapping(raw_firms, accounting, firm_panel)
    advance = build_advance_cycle(macro, raw_firms, accounting, chain)
    capital_profit = build_capital_good_profit_bridge(raw_firms, accounting)
    sector_labor = build_sector_labor_reallocation(macro, raw_firms, labor_events)
    food_profit = build_food_profitability(raw_firms, accounting)
    competition = build_competition_audit(raw_firms, firm_capital)
    gui = build_gui_coverage()
    classifications = build_classification()

    write(cycle, "household_settlement_cycle_attribution.csv")
    write(grain, "household_metric_temporal_grain_registry.csv")
    write(near_zero, "near_zero_household_cash_diagnosis.csv")
    write(loan_mapping, "capital_good_loan_advance_mapping_audit.csv")
    write(advance, "capital_good_advance_cycle_decomposition.csv")
    write(capital_profit, "capital_good_profit_bridge.csv")
    write(sector_labor, "sector_labor_productivity_reallocation.csv")
    write(food_profit, "food_firm_profitability_bridge.csv")
    write(competition, "food_competition_symmetry_audit.csv")
    write(gui, "firm_gui_comparison_coverage.csv")
    write(classifications, "post_freeze_manual_review_classification.csv")

    hashes_after = {name: sha256(path) for name, path in INPUTS.items()}
    inputs_unchanged = hashes_before == hashes_after

    mapping_values = loan_mapping.loc[loan_mapping["row_type"].eq("WEEK_VALUE_TRACE")]
    marriage_cycles = cycle.loc[cycle["settlement_transition_count"] > 0]
    snapshot_steps = sorted(social["global_step"].unique())
    latest_step = snapshot_steps[-1]
    latest_near = near_zero.loc[
        near_zero["row_type"].eq("SNAPSHOT_GROUP")
        & near_zero["global_step"].eq(latest_step)
        & near_zero["cash_group"].eq("NEAR_ZERO")
    ].iloc[0]
    latest_other = near_zero.loc[
        near_zero["row_type"].eq("SNAPSHOT_GROUP")
        & near_zero["global_step"].eq(latest_step)
        & near_zero["cash_group"].eq("ABOVE_NEAR_ZERO")
    ].iloc[0]
    cap_cumulative = capital_profit.loc[capital_profit["row_type"].eq("CUMULATIVE")].iloc[0]
    food_sector = food_profit.loc[food_profit["row_type"].eq("CUMULATIVE_SECTOR")].iloc[0]
    competition_summary = competition.loc[competition["row_type"].eq("RUN_SUMMARY")].iloc[0]
    final_macro = macro.iloc[-1]
    early_macro = macro.iloc[0]
    late_advance_cycles = advance.loc[
        advance["global_step"].isin(
            advance.loc[advance["new_customer_advances"] > 1e-9, "global_step"].tail(10)
        )
    ]

    flags = {
        "verdict": "F. MULTIPLE_INDEPENDENT_MODEL_BOUNDARIES",
        "canonical_reference": str(RUN.relative_to(PROJECT_ROOT)),
        "simulation_runs": 0,
        "economic_behavior_changed": False,
        "gui_changed": False,
        "new_rng_draws": 0,
        "canonical_inputs_unchanged": inputs_unchanged,
        "settlement_only_is_social_household": False,
        "settlement_only_overwrites_person_household_id": False,
        "recommended_settlement_gui_label_zh": "临时个人结算账户",
        "micro_snapshot_cadence_weeks": 13,
        "micro_snapshot_first_step": snapshot_steps[0],
        "micro_snapshot_last_step": snapshot_steps[-1],
        "micro_snapshot_final_week_519_present": 519 in snapshot_steps,
        "weekly_distribution_extension_implemented": False,
        "settlement_transition_cash_total": float(marriage_cycles["settlement_to_social_cash"].sum()),
        "final_pending_household_formation_wealth": float(final_macro["pending_household_formation_wealth"]),
        "settlement_pending_double_posting_identity_max_gap": float(marriage_cycles["pending_offset_identity_gap"].abs().max()),
        "near_zero_threshold": NEAR_ZERO_CASH,
        "latest_authoritative_micro_snapshot": latest_step,
        "latest_near_zero_household_share": float(latest_near["household_share"]),
        "latest_near_zero_mean_income": float(latest_near["current_week_income_mean"]),
        "latest_other_mean_income": float(latest_other["current_week_income_mean"]),
        "employed_members_by_household_available": False,
        "interval_household_cash_bridge_available": False,
        "capital_good_runtime_true_loan_principal_max": float(mapping_values["runtime_true_loan_principal"].abs().max()),
        "capital_good_gui_principal_equals_advance_all_weeks": bool(mapping_values["displayed_principal_equals_advance"].all()),
        "loan_advance_first_divergence": "economy/accounting.py:355",
        "analysis_query_mapping_itself_distinguishes_fields": True,
        "advance_liability_bridge_max_gap": float(advance["recomputed_advance_bridge_gap"].abs().max()),
        "late_advance_source": "REPLACEMENT",
        "late_cycle_liability_amplitude_mean": float(late_advance_cycles["cycle_liability_amplitude"].mean()),
        "capital_good_sales_margin_gap": float(cap_cumulative["zero_margin_sales_gap"]),
        "capital_good_profit_classification": cap_cumulative["classification"],
        "food_employment_change": int(final_macro["food_employment"] - early_macro["food_employment"]),
        "capital_good_employment_change": int(final_macro["capital_good_employment"] - early_macro["capital_good_employment"]),
        "active_capital_service_change": float(final_macro["active_capital_service"] - early_macro["active_capital_service"]),
        "late_backlog_declining": bool(np.polyfit(macro["global_step"].tail(52), macro["total_backlog"].tail(52), 1)[0] < 0),
        "food_cumulative_operating_profit": float(food_sector["accounting_operating_profit"]),
        "food_cumulative_gross_profit_before_spoilage": float(food_sector["gross_profit_before_spoilage"]),
        "food_cumulative_spoilage": float(food_sector["spoilage_or_inventory_loss"]),
        "food_cumulative_depreciation": float(food_sector["capital_depreciation_expense"]),
        "food_profitability_problem_confirmed": True,
        "food_hhi_mean": float(competition_summary["sales_hhi"]),
        "food_hhi_min": float(competition_summary["sales_hhi_minimum"]),
        "food_hhi_max": float(competition_summary["sales_hhi_maximum"]),
        "food_competition_classification": competition_summary["classification"],
        "demand_allocation_is_fixed_equal_split": False,
        "available_but_not_fully_exposed_firm_metrics": gui.loc[
            ~gui["coverage_status"].isin(["EXPOSED", "EXPOSED_CORRECTLY"]), "metric"
        ].tolist(),
        "accounting_gap_max": float(macro["accounting_reconciliation_gap"].abs().max()),
        "money_location_gap_max": float(macro["full_money_location_gap"].abs().max()),
        "goods_gap_max": float(macro["goods_gap"].abs().max()),
    }
    def json_scalar(value: object) -> object:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        raise TypeError(f"Unsupported JSON value: {type(value).__name__}")

    with (OUTPUT / "acceptance_flags.json").open("w", encoding="utf-8") as handle:
        json.dump(flags, handle, ensure_ascii=False, indent=2, default=json_scalar)
        handle.write("\n")

    summary = f"""# Step 15 冻结后人工复核审计

**Verdict: F. MULTIPLE_INDEPENDENT_MODEL_BOUNDARIES**

本审计只读取冻结基线 `{RUN.relative_to(PROJECT_ROOT)}`，未构造 World、未运行模拟、未修改经济行为或 GUI；输入哈希复核：`{inputs_unchanged}`。

## 1. settlement_only 的经济含义

`settlement_only` 是社会关系之外的临时经济结算账户：它为尚未归属有效社会家庭的劳动年龄 Person 接收工资、持有现金并执行消费。它不是 Social Household，也不覆盖 `Person.household_id`；权威社会身份仍由后者决定。建议 GUI 标签使用 **“临时个人结算账户”**，tooltip 使用：“为暂未归属社会家庭的成年人提供工资、现金与消费结算；不代表婚姻、亲子或共同居住关系。”

生命周期为：`ensure_labor_settlement_households()` 确定性创建 -> `settlement_household_for_person()` 接收 Food/资本品工资 -> 正常 Household 消费路径 -> 年度婚配形成 Social Household -> `merge_settlement_household_into_social_household()` 转移现金并删除临时账户。

## 2. Social / settlement 现金周期

年度婚配周 52、104、...、468 与临时账户集中关闭完全对齐。共转移 `{marriage_cycles['settlement_to_social_cash'].sum():.6f}` 现金；纯重分类本应对 Social 为 `+T`、对 settlement 为 `-T`、合计为 0。

冻结轨迹还存在一个独立生命周期记账边界：`marriage.py:163-183` 已把 settlement 现金直接转入新 Social Household，却又把同一金额加入从 `pending_household_formation_wealth` 转出的金额。于是婚配周 `(Social+settlement cash change) - saving = T`，同时 pending balance 下降 `-T`；最终 pending 为 `{final_macro['pending_household_formation_wealth']:.6f}`，恰为累计转移的相反数。全局货币没有新增，但这不只是统计重分类，而是一次由负 pending 余额抵消的重复过账。最大闭合残差仅 `{marriage_cycles['pending_offset_identity_gap'].abs().max():.3e}`。

## 3. Household 指标时间粒度

宏观 Household cash / income / consumption / saving 为 520 个逐周权威聚合值。Social Household 微观数据只在 `-1, 0, 13, ..., 507` 共 `{len(snapshot_steps)}` 个截面存在；本次手工 520 周运行没有 step 519 的微观终值。Gini、Lorenz、分位数和直方图均由这些截面派生，不得绘成连续周度历史。建议后续仅新增逐周分布聚合统计，同时保留 13 周微观快照；本阶段未实现。

## 4. 近零 Social Household 现金

初始化缓冲后 opening near-zero share 为 0。最新可用微观截面 week {latest_step} 中，现金 `<= {NEAR_ZERO_CASH:g}` 的家庭占 `{latest_near['household_share']:.2%}`。其周均收入 `{latest_near['current_week_income_mean']:.3f}`，高于阈值组为 `{latest_other['current_week_income_mean']:.3f}`；近零组负储蓄比例 `{latest_near['negative_saving_share']:.2%}`，有子女比例仅 `{latest_near['households_with_children_share']:.2%}`，settlement-origin 比例 `{latest_near['settlement_origin_share']:.2%}`。因此现有证据首先指向较低工资收入与最低消费附近的现金约束，而不是高依赖负担或账户转入。逐户就业成员数及 13 周区间收入/转移/消费未持久化，审计明确标记为 unavailable，未做反推。

## 5. 资本品“贷款本金”

它不是实际贷款。资本品 Firm 全程 runtime `loan_balance` 最大值为 `{mapping_values['runtime_true_loan_principal'].abs().max():.1f}`，贷款发行与偿还也均为 0。第一处分歧在 `economy/accounting.py:355`：`liabilities = true loan + customer advance` 被写入名为 `loan_balance` 的列；Firm panel 优先读取该列，Analysis Query 与 GUI 再原样传播。week 480/481 的 `{mapping_values.set_index('global_step').loc[480, 'gui_displayed_loan_principal']:.6f}` / `{mapping_values.set_index('global_step').loc[481, 'gui_displayed_loan_principal']:.6f}` 实为客户预付款负债。Registry 和 GUI 本身已经使用两个不同指标名，因此本阶段没有做 GUI-only 补丁；正确修复点应在 AccountingLayer/Panel 权威源。

## 6. 客户预付款与供应商现金锯齿

负债桥 `opening + advances - delivered = closing` 的最大残差为 `{advance['recomputed_advance_bridge_gap'].abs().max():.3e}`。预付款在同步的 13 周投资复核点集中流入，随后用于逐周工资、资本品生产和交付；交付确认收入并按比例释放负债，但不重复收现。后期新预付款全部可链接到 replacement orders。随着活跃资本存量和更新批次扩大，复核点补充额、周生产/交付额以及周期内 peak-to-trough 振幅增大；但负债从早期约 111 万下降到期末 `{advance.iloc[-1]['closing_customer_advance_liability']:.3f}`，不是无界预付款堆积。

## 7. 资本品利润

成本锚定价为 `41 / 3 = {41/3:.6f}`。生产工资先资本化为库存成本，交付时进入 COGS，所以不能再把当周 payroll 与 COGS 同时扣除。累计交付收入与 COGS 差为 `{cap_cumulative['zero_margin_sales_gap']:.3e}`，持续经营利润约为 0，分类为 `INTENTIONAL_ZERO_MARGIN_ENGINEERING_CONTRACT`。唯一例外是 week 0 的 `{cap_cumulative['other_existing_operating_revenue']:.2f}` startup capitalization 被 AccountingLayer 归入 other operating revenue，使累计报表利润同额为正；这是展示/定义例外，不是资本品毛利。

## 8. 部门劳动重配

Food 就业从 `{int(early_macro['food_employment'])}` 降至 `{int(final_macro['food_employment'])}`，资本品就业从 `{int(early_macro['capital_good_employment'])}` 升至 `{int(final_macro['capital_good_employment'])}`，总就业反而增加 `{int(final_macro['total_employment'] - early_macro['total_employment'])}`；活跃资本服务增加 `{final_macro['active_capital_service'] - early_macro['active_capital_service']:.3f}`，最后 52 周 backlog 斜率为 `{np.polyfit(macro['global_step'].tail(52), macro['total_backlog'].tail(52), 1)[0]:.3f}/周`。因此轨迹内是有真实订单、生产和交付支持的资本深化/劳动重配，而非空转就业；但 52 周寿命和 P3 生产率仍是非校准工程尺度，幅度不可解释为经验结果。

## 9. Food 收入上升但利润为负

Food 累计 accounting revenue `{food_sector['accounting_revenue']:.2f}`，扣 COGS 后毛利 `{food_sector['gross_profit_before_spoilage']:.2f}`；spoilage `{food_sector['spoilage_or_inventory_loss']:.2f}` 后仅余 `{food_sector['profit_after_spoilage_before_depreciation']:.2f}`，再扣 depreciation `{food_sector['capital_depreciation_expense']:.2f}`，累计 operating profit 为 `{food_sector['accounting_operating_profit']:.2f}`。中段首先是 spoilage 吞噬毛利，后段则是 depreciation 超过 spoilage 后剩余经营盈余。利息为 0，融资流不应混入经营亏损；生产工资已通过库存 COGS 费用化，不能重复扣除。

## 10. Food HHI 与竞争

销售 HHI 均值 `{competition_summary['sales_hhi']:.6f}`，范围 `{competition_summary['sales_hhi_minimum']:.6f}`–`{competition_summary['sales_hhi_maximum']:.6f}`。需求不是固定 20% 分配：每个 Household 按 `price^-2` 权重进行选择，价格与份额呈负向关系。接近 0.2 主要来自五家 Firm 的共同生产率、工资/策略参数、平衡初始化及同步复核，加上大量 Household 选择的聚合平均。分类为 `PARAMETER_SYMMETRY`：竞争通道真实存在，但结构异质性、进入退出和策略差异不足；HHI 本身不能证明硬编码平均分配。

## 11. GUI Firm 比较建议

优先恢复：price、unit cost、price-cost margin、profit margin、revenue、inventory、employment、output per worker、capital intensity、cash 的跨 Firm 图；保留现有 sales/share/HHI/capital-service 图。增加关系图：price vs market share、margin vs market share、capital intensity vs productivity、inventory vs price。true loan principal 必须先修正 AccountingLayer/Panel 源字段，不能仅在 GUI 重命名；customer advance liability 应继续独立展示。

## 停止点

本审计确认至少四个相互独立的边界：settlement/pending 重复过账、资本品贷款字段污染、Food 盈利结构、以及有限的 Firm 结构异质性。因此选择 **F**，而不是把所有现象归为 GUI 语义问题。未自动修复任何一项。
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")

    required = {
        "acceptance_summary.md",
        "acceptance_flags.json",
        "household_settlement_cycle_attribution.csv",
        "household_metric_temporal_grain_registry.csv",
        "near_zero_household_cash_diagnosis.csv",
        "capital_good_loan_advance_mapping_audit.csv",
        "capital_good_advance_cycle_decomposition.csv",
        "capital_good_profit_bridge.csv",
        "sector_labor_productivity_reallocation.csv",
        "food_firm_profitability_bridge.csv",
        "food_competition_symmetry_audit.csv",
        "firm_gui_comparison_coverage.csv",
        "post_freeze_manual_review_classification.csv",
    }
    missing = sorted(name for name in required if not (OUTPUT / name).is_file())
    if missing:
        raise AssertionError(f"Missing required outputs: {missing}")
    if not inputs_unchanged:
        raise AssertionError("Frozen canonical inputs changed during read-only audit")
    if len(macro) != 520:
        raise AssertionError(f"Expected 520 macro weeks, found {len(macro)}")
    if float(advance["recomputed_advance_bridge_gap"].abs().max()) > TOLERANCE:
        raise AssertionError("Customer-advance bridge failed")
    if float(mapping_values["runtime_true_loan_principal"].abs().max()) > TOLERANCE:
        raise AssertionError("Capital-good runtime unexpectedly has loan principal")


if __name__ == "__main__":
    main()
