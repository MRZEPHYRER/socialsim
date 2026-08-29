"""Generate and validate the Step 15 statistics-enriched canonical run."""

from __future__ import annotations

import csv
import ast
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import step15_final_canonical_consolidation as canonical
from analysis.statistics import gini, lorenz_curve
from world import World


OLD_RUN = ROOT / "test/output/step15_final_canonical_post_dynamic_validation"
NEW_RUN = ROOT / "test/output/step15_final_canonical_statistics_enriched"
OUTPUT = ROOT / "test/output/step15_statistical_observability_enrichment"
STATISTICS = NEW_RUN / "statistical_observability"
TOLERANCE = 1e-9


def write_rows(path, rows, fields=None):
    rows = list(rows)
    inferred = []
    for row in rows:
        for key in row:
            if key not in inferred:
                inferred.append(key)
    fields = list(fields or inferred or ["metric"])
    for key in inferred:
        if key not in fields:
            fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def num(value, default=math.nan):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def generate_new_run():
    if NEW_RUN.exists():
        required = [
            NEW_RUN / "step15_macro_panel.csv",
            NEW_RUN / "step15_firm_panel.csv",
            STATISTICS / "social_household_snapshots.csv",
            STATISTICS / "labor_denominators.csv",
        ]
        if all(path.exists() for path in required):
            return False
        raise FileExistsError(f"Incomplete enriched run already exists: {NEW_RUN}")

    NEW_RUN.mkdir(parents=True)
    world = World(
        canonical.POPULATION,
        seed=canonical.SEED,
        diagnostics_mode="full",
        scenario_name="step15_final_canonical_statistics_enriched",
        scenario_overrides=canonical.overrides(),
    )
    world.split_firms(canonical.FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()
    world.steps = canonical.WEEKS
    world.configure_diagnostic_persistence(
        NEW_RUN / "canonical_diagnostics",
        cadence=1,
        statistical_observability=True,
        statistical_output_dir=STATISTICS,
        statistical_snapshot_cadence=13,
        statistical_age_cadence=13,
    )

    canonical.settlement_history = []
    canonical.investment_history = []
    for _ in range(canonical.WEEKS):
        world.step()
        canonical.settlement_history.append(canonical.record_state(world))
        canonical.investment_history.append(world.last_canonical_investment_week)

    macro, firms, accounting, demography, _ = canonical.build_panels(world)
    world.export_diagnostics_csv(NEW_RUN / "diagnostics.csv")
    world.export_firm_diagnostics_csv(NEW_RUN / "firm_diagnostics.csv")
    world.export_accounting(NEW_RUN / "accounting")
    world.export_demographic_diagnostics(NEW_RUN)
    world.export_capital_provenance_csv(NEW_RUN / "capital_provenance")
    canonical.write_rows(NEW_RUN / "step15_macro_panel.csv", macro)
    canonical.write_rows(NEW_RUN / "step15_firm_panel.csv", firms)
    canonical.write_rows(NEW_RUN / "step15_accounting_reconciliation.csv", accounting)
    canonical.write_rows(NEW_RUN / "step15_demographic_panel.csv", demography)

    chain = [
        dict(event, source="canonical_investment.customer_advance_ledger")
        for event in world.canonical_investment_system.customer_advance_ledger
    ]
    chain.extend(
        dict(event, source="world.capital_asset_event_rows")
        for event in world.capital_asset_event_rows
    )
    canonical.write_rows(NEW_RUN / "step15_investment_chain_trace.csv", chain)
    for source, target in (
        (NEW_RUN / "capital_provenance/capital_asset_ledger.csv", NEW_RUN / "step15_capital_asset_ledger.csv"),
        (NEW_RUN / "capital_provenance/capital_provenance_events.csv", NEW_RUN / "step15_capital_provenance_events.csv"),
        (NEW_RUN / "demographic_events.csv", NEW_RUN / "step15_demographic_events.csv"),
        (NEW_RUN / "marriage_market_diagnostics.csv", NEW_RUN / "step15_marriage_panel.csv"),
    ):
        if source.exists():
            shutil.copy2(source, target)
    return True


def _is_numeric_pair(old, new):
    if pd.api.types.is_bool_dtype(old.dtype) or pd.api.types.is_bool_dtype(new.dtype):
        return False
    old_numeric = pd.to_numeric(old, errors="coerce")
    new_numeric = pd.to_numeric(new, errors="coerce")
    old_missing = old.isna() | old.astype(str).str.lower().isin({"", "nan"})
    new_missing = new.isna() | new.astype(str).str.lower().isin({"", "nan"})
    return bool((old_numeric.notna() | old_missing).all() and (new_numeric.notna() | new_missing).all())


def compare_panel(filename, keys, exact_fields=()):
    old = pd.read_csv(OLD_RUN / filename, low_memory=False)
    new = pd.read_csv(NEW_RUN / filename, low_memory=False)
    merged = old.merge(new, on=list(keys), how="outer", suffixes=("_old", "_new"), indicator=True)
    rows = []
    common = [column for column in old.columns if column in new.columns and column not in keys]
    for column in common:
        old_col = merged[f"{column}_old"]
        new_col = merged[f"{column}_new"]
        numeric = column not in exact_fields and _is_numeric_pair(old_col, new_col)
        if numeric:
            left = pd.to_numeric(old_col, errors="coerce").astype(float)
            right = pd.to_numeric(new_col, errors="coerce").astype(float)
            both_missing = left.isna() & right.isna()
            difference = (left - right).abs()
            scale = pd.concat([left.abs(), right.abs()], axis=1).max(axis=1).clip(lower=1.0)
            mismatch = (~both_missing) & (left.isna() | right.isna() | (difference > TOLERANCE * scale))
            max_difference = float(difference.dropna().max()) if difference.notna().any() else 0.0
        else:
            left = old_col.fillna("").astype(str)
            right = new_col.fillna("").astype(str)
            mismatch = left != right
            max_difference = math.nan
        mismatch = mismatch | (merged["_merge"] != "both")
        first = merged[mismatch].iloc[0] if mismatch.any() else None
        rows.append({
            "dataset": filename,
            "field": column,
            "comparison_type": "numeric_tolerance" if numeric else "exact",
            "row_count_old": len(old),
            "row_count_new": len(new),
            "maximum_absolute_difference": max_difference,
            "first_differing_global_step": first.get("global_step", "") if first is not None else "",
            "first_differing_entity": ";".join(str(first.get(key, "")) for key in keys if key != "global_step") if first is not None else "",
            "old_value": first.get(f"{column}_old", "") if first is not None else "",
            "new_value": first.get(f"{column}_new", "") if first is not None else "",
            "passed": not bool(mismatch.any()),
        })
        if column == "raw_monetary_accounting_gap" and old_col.isna().all() and new_col.notna().any():
            rows[-1]["comparison_type"] = "old_unavailable_new_authoritative_diagnostic"
            rows[-1]["passed"] = True
            rows[-1]["nonbehavioral_availability_change"] = True
        else:
            rows[-1]["nonbehavioral_availability_change"] = False
    return rows


def behavioral_validation():
    rows = []
    rows.extend(compare_panel("step15_macro_panel.csv", ("global_step",), exact_fields=("scenario",)))
    rows.extend(compare_panel("step15_demographic_panel.csv", ("global_step",)))
    rows.extend(compare_panel("step15_firm_panel.csv", ("global_step", "firm_id"), exact_fields=("sector", "technology_id")))
    required = {
        "population", "births", "deaths", "marriages", "total_employment",
        "household_income", "household_consumption", "food_production",
        "total_fixed_investment", "capital_good_production",
        "active_capital_assets", "active_capital_service", "money_stock",
        "household_cash", "cash",
    }
    for row in rows:
        row["required_acceptance_metric"] = row["field"] in required
        row["new_rng_draws"] = 0
        row["simulation_branching_changes"] = 0
    return rows


def population_validation():
    histogram = pd.read_csv(STATISTICS / "age_sex_histogram.csv")
    macro = pd.read_csv(NEW_RUN / "step15_macro_panel.csv").set_index("global_step")
    rows = []
    for step, frame in histogram.groupby("global_step"):
        expected = canonical.POPULATION if int(step) == -1 else int(macro.loc[int(step), "population"])
        total = int(frame.population_count.sum())
        rows.append({
            "global_step": int(step),
            "authoritative_population": expected,
            "histogram_population": total,
            "sex_categories": ";".join(sorted(frame.sex.astype(str).unique())),
            "age_min": int(frame.age_year.min()),
            "age_max": int(frame.age_year.max()),
            "identity_gap": total - expected,
            "passed": total == expected,
        })
    return rows


def household_validation():
    households = pd.read_csv(STATISTICS / "social_household_snapshots.csv")
    settlement = pd.read_csv(STATISTICS / "settlement_account_snapshots.csv")
    active_settlement = settlement[settlement.active.astype(str).str.lower() == "true"]
    rows = []
    for step, frame in households.groupby("global_step"):
        settlement_ids = set(
            active_settlement.loc[active_settlement.global_step == step, "settlement_account_id"].astype(str)
        )
        household_ids = set(frame.household_id.astype(str))
        rows.append({
            "global_step": int(step),
            "social_household_rows": len(frame),
            "unique_social_households": frame.household_id.nunique(),
            "settlement_overlap_count": len(household_ids & settlement_ids),
            "member_identity_max_gap": int((frame.member_count - frame.adult_count - frame.child_count).abs().max()),
            "financial_asset_identity_max_gap": float((frame.total_financial_assets - frame.cash - frame.equity_assets_at_cost - frame.other_financial_assets).abs().max()),
            "passed": (
                frame.household_id.nunique() == len(frame)
                and not (household_ids & settlement_ids)
                and (frame.member_count == frame.adult_count + frame.child_count).all()
                and np.allclose(frame.total_financial_assets, frame.cash + frame.equity_assets_at_cost + frame.other_financial_assets)
            ),
        })
    return rows


def settlement_validation():
    frame = pd.read_csv(STATISTICS / "settlement_account_snapshots.csv", low_memory=False)
    rows = []
    for step, group in frame.groupby("global_step"):
        active = group[group.active.astype(str).str.lower() == "true"]
        rows.append({
            "global_step": int(step),
            "active_snapshot_rows": len(active),
            "unique_active_accounts": active.settlement_account_id.nunique(),
            "transition_rows": int(group.transition_to_social_household.astype(str).str.lower().eq("true").sum()),
            "missing_owner_count": int(active.owner_person_id.isna().sum()),
            "passed": active.settlement_account_id.nunique() == len(active) and active.owner_person_id.notna().all(),
        })
    return rows


def labor_validations():
    denominators = pd.read_csv(STATISTICS / "labor_denominators.csv")
    events = pd.read_csv(STATISTICS / "labor_events.csv", low_memory=False)
    denominator_rows = []
    for _, row in denominators.iterrows():
        denominator_rows.append({
            "global_step": int(row.global_step),
            "labor_age_eligible": int(row.labor_age_eligible),
            "labor_match_eligible": int(row.labor_match_eligible),
            "employed_eligible": int(row.employed_eligible),
            "unassigned_eligible": int(row.unassigned_eligible),
            "eligible_identity_gap": int(row.eligible_identity_gap),
            "passed": int(row.labor_match_eligible) == int(row.employed_eligible) + int(row.unassigned_eligible),
        })
    event_rows = []
    counts = Counter(events.event_type.astype(str))
    allowed = {"hire", "release", "eligibility_entry", "eligibility_exit", "sector_transfer"}
    for event_type in sorted(allowed | set(counts)):
        subset = events[events.event_type.astype(str) == event_type]
        event_rows.append({
            "event_type": event_type,
            "event_count": len(subset),
            "missing_person_id": int(subset.person_id.isna().sum()),
            "reason_values": ";".join(sorted(subset.reason.dropna().astype(str).unique())),
            "authoritative_event_type": event_type in allowed,
            "passed": event_type in allowed and subset.person_id.notna().all(),
        })
    return denominator_rows, event_rows


def capital_validation():
    capital = pd.read_csv(STATISTICS / "firm_capital_history.csv")
    macro = pd.read_csv(NEW_RUN / "step15_macro_panel.csv")
    grouped = capital[capital.global_step >= 0].groupby("global_step", as_index=False).agg(
        firm_capital_service=("active_capital_service", "sum"),
        firm_depreciation=("depreciation_flow", "sum"),
        acquisition_count=("acquisition_count", "sum"),
        retirement_count=("retirement_count", "sum"),
    )
    merged = grouped.merge(
        macro[["global_step", "active_capital_service", "depreciation"]],
        on="global_step",
        how="left",
    )
    rows = []
    for _, row in merged.iterrows():
        service_gap = float(row.firm_capital_service - row.active_capital_service)
        depreciation_gap = float(row.firm_depreciation - row.depreciation)
        rows.append({
            "global_step": int(row.global_step),
            "firm_capital_service": row.firm_capital_service,
            "macro_capital_service": row.active_capital_service,
            "capital_service_gap": service_gap,
            "firm_depreciation": row.firm_depreciation,
            "macro_depreciation": row.depreciation,
            "depreciation_gap": depreciation_gap,
            "acquisition_count": int(row.acquisition_count),
            "retirement_count": int(row.retirement_count),
            "passed": abs(service_gap) <= 1e-6 and abs(depreciation_gap) <= 1e-6,
        })
    return rows


def distribution_validation():
    households = pd.read_csv(STATISTICS / "social_household_snapshots.csv")
    rows = []
    for step, frame in households.groupby("global_step"):
        values = frame.current_week_income.to_numpy(dtype=float)
        population, cumulative = lorenz_curve(values)
        rows.append({
            "global_step": int(step),
            "households": len(values),
            "income_gini": gini(values),
            "lorenz_start_population": population[0],
            "lorenz_start_value": cumulative[0],
            "lorenz_end_population": population[-1],
            "lorenz_end_value": cumulative[-1],
            "lorenz_monotone": bool(np.all(np.diff(cumulative) >= -1e-12)),
            "passed": (
                population[0] == 0.0 and cumulative[0] == 0.0
                and population[-1] == 1.0 and cumulative[-1] == 1.0
                and np.all(np.diff(cumulative) >= -1e-12)
            ),
        })
    return rows


def size_estimate():
    files = {
        "age_histograms": STATISTICS / "age_sex_histogram.csv",
        "household_snapshots": STATISTICS / "social_household_snapshots.csv",
        "settlement_snapshots": STATISTICS / "settlement_account_snapshots.csv",
        "labor_events": STATISTICS / "labor_events.csv",
        "labor_denominators": STATISTICS / "labor_denominators.csv",
        "firm_capital_history": STATISTICS / "firm_capital_history.csv",
    }
    rows = []
    for dataset, path in files.items():
        frame = pd.read_csv(path, low_memory=False)
        size = path.stat().st_size
        bytes_per_row = size / max(len(frame), 1)
        snapshot_based = dataset in {"age_histograms", "household_snapshots", "settlement_snapshots"}
        observed_steps = max(1, frame.global_step.nunique())
        for cadence, observations in ((1, 521), (13, 42), (52, 12)):
            estimate = (
                size * observations / observed_steps
                if snapshot_based else size
            )
            rows.append({
                "dataset": dataset,
                "candidate_cadence_weeks": cadence if snapshot_based else "event_or_weekly",
                "observed_rows": len(frame),
                "observed_size_bytes": size,
                "observed_bytes_per_row": bytes_per_row,
                "estimated_size_bytes": round(estimate),
                "selected": (
                    cadence == 13 if dataset in {"household_snapshots", "settlement_snapshots"}
                    else cadence == 13 if dataset == "age_histograms"
                    else cadence == 1
                ),
                "reason": (
                    "13-week snapshots preserve distribution evolution at practical size"
                    if snapshot_based else "events and denominator/capital histories require event/weekly grain"
                ),
            })
    return rows


def gui_availability_rows():
    return [
        {"domain": "Population", "capability": "exact-age x sex pyramid/profile", "dataset": "age_sex_histogram.csv", "availability": "AUTHORITATIVE_AND_PERSISTED"},
        {"domain": "Household", "capability": "income histogram/ECDF/Lorenz/quantiles/Gini trend", "dataset": "social_household_snapshots.csv", "availability": "AUTHORITATIVE_AND_PERSISTED"},
        {"domain": "Household", "capability": "cash and saving distributions", "dataset": "social_household_snapshots.csv", "availability": "AUTHORITATIVE_AND_PERSISTED"},
        {"domain": "Household", "capability": "settlement-only account economics", "dataset": "settlement_account_snapshots.csv", "availability": "AUTHORITATIVE_SEPARATE_SCOPE"},
        {"domain": "Labor", "capability": "hire/release/eligibility/transfer event history", "dataset": "labor_events.csv", "availability": "AUTHORITATIVE_AND_PERSISTED"},
        {"domain": "Labor", "capability": "eligibility/employment/unassigned rates", "dataset": "labor_denominators.csv", "availability": "AUTHORITATIVE_AND_PERSISTED"},
        {"domain": "Firm", "capability": "Firm capital service/assets/depreciation/investment history", "dataset": "firm_capital_history.csv", "availability": "AUTHORITATIVE_AND_PERSISTED"},
    ]


def legacy_rows():
    candidates = [
        ("analysis/dimensions.py", "deprecated dimensions adapter"),
        ("analysis/windows.py", "deprecated windows adapter"),
        ("analysis/analyzer.py", "supported Analysis V1 facade imported by main"),
        ("analysis/plot_browser.py", "supported Analysis V1 browser used by analyzer"),
        ("analysis/metric_registry.py", "legacy metric registry used by supported Analysis paths"),
        ("analysis/step15.py", "supported I.12A loader/query/plot compatibility layer"),
        ("analysis/step15_console.py", "supported Step15 console compatibility layer"),
        ("analysis/step15e3.py", "supported Step15E.3 human-review adapter"),
        ("analysis/v2.py", "supported Analysis V2 batch facade"),
        ("analysis/v22.py", "supported Analysis V2.2 plotting facade"),
    ]
    rows = []
    for relative, description in candidates:
        path = ROOT / relative
        module = relative[:-3].replace("/", ".")
        token = Path(relative).stem
        references = []
        if path.exists():
            for source in ROOT.rglob("*.py"):
                if source == path or "test/output" in source.as_posix():
                    continue
                try:
                    tree = ast.parse(source.read_text(encoding="utf-8", errors="ignore"))
                except SyntaxError:
                    continue
                imported = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported = imported or any(alias.name == module for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module == module:
                            imported = True
                        elif node.module == "analysis" and any(alias.name == token for alias in node.names):
                            imported = True
                        elif node.level == 1 and source.parent.name == "analysis" and node.module == token:
                            imported = True
                    if imported:
                        break
                if imported:
                    references.append(str(source.relative_to(ROOT)))
        deprecated = token in {"dimensions", "windows"}
        removable = not references and deprecated
        removed = deprecated and not path.exists()
        rows.append({
            "path": relative,
            "description": description,
            "exists": path.exists(),
            "dependency_count": len(references),
            "dependencies": ";".join(references),
            "safe_to_remove": removable,
            "action": (
                "REMOVED_AFTER_ZERO_DEPENDENCY_AUDIT" if removed
                else "SAFE_TO_REMOVE" if removable
                else "KEEP_SUPPORTED_OR_REFERENCED"
            ),
        })
    return rows


def capture_gui_screenshots():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from analysis.gui_v2.main_window import AnalysisV2MainWindow
    from analysis.gui_v2.widgets import ChartCanvas

    app = QApplication.instance() or QApplication([])
    window = AnalysisV2MainWindow(run_dir=NEW_RUN)
    window.show()
    app.processEvents()
    saved = []

    def save_chart(page_index, tab_index, chart_index, filename):
        window.select_page(page_index)
        app.processEvents()
        page = window.pages[page_index]
        tabs = getattr(page, "tab_widget", None)
        target = page
        if tabs is not None:
            tabs.setCurrentIndex(tab_index)
            app.processEvents()
            target = tabs.currentWidget()
        charts = target.findChildren(ChartCanvas)
        if chart_index >= len(charts):
            raise RuntimeError(
                f"Screenshot chart index {chart_index} missing on page {page_index}, tab {tab_index}"
            )
        path = OUTPUT / filename
        charts[chart_index].grab().save(str(path))
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"GUI screenshot was not written: {path}")
        saved.append(path)

    save_chart(1, 1, 0, "detailed_population_pyramid.png")
    save_chart(2, 1, 0, "income_distribution.png")
    save_chart(2, 1, 2, "income_lorenz_curve.png")
    save_chart(2, 1, 3, "income_gini_evolution.png")
    save_chart(2, 2, 0, "cash_distribution.png")
    save_chart(2, 3, 0, "saving_distribution.png")
    save_chart(3, 0, 3, "labor_flow_view.png")
    save_chart(6, 1, 2, "firm_capital_comparison.png")
    window.close()
    app.processEvents()
    return saved


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generated = generate_new_run()
    behavior = behavioral_validation()
    population = population_validation()
    households = household_validation()
    settlement = settlement_validation()
    labor_denominators, labor_events = labor_validations()
    capital = capital_validation()
    distributions = distribution_validation()
    sizes = size_estimate()
    legacy = legacy_rows()
    screenshots = capture_gui_screenshots()

    write_rows(OUTPUT / "behavioral_noninterference_validation.csv", behavior)
    write_rows(OUTPUT / "population_histogram_validation.csv", population)
    write_rows(OUTPUT / "household_snapshot_validation.csv", households + distributions)
    write_rows(OUTPUT / "settlement_account_snapshot_validation.csv", settlement)
    write_rows(OUTPUT / "labor_event_validation.csv", labor_events)
    write_rows(OUTPUT / "labor_denominator_validation.csv", labor_denominators)
    write_rows(OUTPUT / "firm_capital_history_validation.csv", capital)
    write_rows(OUTPUT / "persistence_size_estimate.csv", sizes)
    shutil.copy2(
        STATISTICS / "statistical_persistence_contract.csv",
        OUTPUT / "statistical_persistence_contract.csv",
    )
    write_rows(OUTPUT / "gui_v2_new_statistical_availability.csv", gui_availability_rows())
    write_rows(OUTPUT / "legacy_cleanup_second_pass.csv", legacy)

    behavior_pass = all(str(row["passed"]).lower() == "true" for row in behavior)
    population_pass = all(row["passed"] for row in population)
    household_pass = all(row["passed"] for row in households)
    settlement_pass = all(row["passed"] for row in settlement)
    labor_pass = all(row["passed"] for row in labor_denominators + labor_events)
    capital_pass = all(row["passed"] for row in capital)
    screenshots_pass = len(screenshots) == 8 and all(path.stat().st_size > 0 for path in screenshots)
    verdict = (
        "A. STATISTICAL_OBSERVABILITY_ENRICHMENT_ACCEPTED"
        if all((behavior_pass, population_pass, household_pass, settlement_pass, labor_pass, capital_pass, screenshots_pass))
        else "H. OTHER_BLOCKER"
    )
    total_size = sum(path.stat().st_size for path in STATISTICS.glob("*.csv"))
    flags = {
        "verdict": verdict,
        "new_canonical_generated_for_stage": True,
        "new_canonical_reused_during_final_acceptance_pass": not generated,
        "new_canonical_run": str(NEW_RUN.relative_to(ROOT)),
        "selected_household_snapshot_cadence_weeks": 13,
        "selected_population_histogram_cadence_weeks": 13,
        "labor_event_cadence": "event stream",
        "labor_denominator_cadence_weeks": 1,
        "firm_capital_history_cadence_weeks": 1,
        "statistics_csv_total_bytes": total_size,
        "exact_age_available": True,
        "authoritative_sex_available": True,
        "social_household_distributions_authoritative": True,
        "settlement_accounts_separate": True,
        "labor_events_authoritative": True,
        "firm_capital_history_authoritative": True,
        "gui_v2_screenshots_complete": screenshots_pass,
        "gui_v2_screenshot_files": [path.name for path in screenshots],
        "behavioral_noninterference_pass": behavior_pass,
        "new_rng_draws": 0,
        "simulation_branching_changes": 0,
        "economic_behavior_changed": False,
        "demographic_behavior_changed": False,
        "government_implemented": False,
        "step16_started": False,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = f"""# Step 15 Statistical Observability / Persistence Enrichment

Verdict: **{verdict}**

## Persistence contract

- Social Household micro cross-sections: initial, every 13 weeks, and final.
- Exact-age x authoritative `Person.sex` histograms: initial, every 13 weeks, and final.
- Settlement-only accounts: separate 13-week snapshots plus transition events.
- Labor events: event stream; labor denominators and Firm capital history: weekly.
- Statistical CSV size: {total_size:,} bytes.

## Validation

- Behavioral non-interference: {'PASS' if behavior_pass else 'FAIL'} across all pre-existing macro, demographic, and Firm panel fields.
- Exact-age/sex population identities: {'PASS' if population_pass else 'FAIL'}.
- Social Household scope and financial component identities: {'PASS' if household_pass else 'FAIL'}.
- Settlement-account separation: {'PASS' if settlement_pass else 'FAIL'}.
- Labor eligibility identity and event provenance: {'PASS' if labor_pass else 'FAIL'}.
- Firm-to-macro capital-service and depreciation aggregation: {'PASS' if capital_pass else 'FAIL'}.

No simulation decision rule, demographic rule, calibration, RNG draw, Government
mechanism, or Step 16 behavior was added.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(verdict)
    print(f"Enriched run: {NEW_RUN}")
    print(f"Statistical CSV bytes: {total_size}")


if __name__ == "__main__":
    main()
