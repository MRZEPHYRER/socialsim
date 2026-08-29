"""Source-only inventory for the pre-Step13.5 analysis architecture audit.

This script parses source files and writes audit artifacts. It never imports
World, runs a simulation, changes configuration, or edits analysis modules.
"""

import ast
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
OUT = ROOT / "test" / "output" / "pre_step13_5_analysis_audit"


def write_csv(name, rows, fields):
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def text(name, value):
    (OUT / name).write_text(value.strip() + "\n", encoding="utf-8")


def source_inventory():
    rows = []
    for path in sorted(ANALYSIS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("__"):
                continue
            kind = "method" if any(isinstance(p, ast.ClassDef) and any(node in ast.walk(m) for m in p.body) for p in tree.body if isinstance(p, ast.ClassDef)) else "function"
            name = node.name
            if name.startswith("plot_"):
                status = "MODERNIZE"
            elif name in {"food_price", "gdp", "consumption", "saving", "money_issued", "central_bank_net_money_issued"}:
                status = "DEPRECATE"
            elif name in {"safe_mean", "safe_median", "safe_percentile", "distribution_summary", "gini_coefficient", "record_column", "group_summary"}:
                status = "MERGE"
            elif name in {"model_years", "weeks_to_years", "years_to_weeks", "elapsed_weeks", "time_semantics_metadata"}:
                status = "KEEP"
            else:
                status = "KEEP"
            if name in {"household_record", "household_records", "group_summary", "firm_series_by_id", "firm_weeks_by_id"}:
                status = "MODERNIZE"
            rows.append({
                "module": str(path.relative_to(ROOT)),
                "function_name": name,
                "line": node.lineno,
                "called_by": "main.py via Analyzer" if name in {"equilibrium_report", "completed_fertility_report", "economy_report", "steady_state_report", "firm_report", "accounting_report", "multi_firm_report"} else "internal/unknown; project-wide search required",
                "purpose": "plot" if name.startswith("plot_") else "metric/report/helper",
                "input_source": "World in-memory histories/objects",
                "output_type": "matplotlib display" if name.startswith("plot_") else "return value or stdout",
                "output_filename": "none explicit in analysis module",
                "simulation_domain": "demography/households/economy/firms/accounting" if path.name != "common.py" else "shared",
                "time_semantics": "weekly helper" if path.name == "time_semantics.py" or name in {"simulation_weeks", "elapsed_weeks", "model_years", "steady_state_report"} else "inherits World history indexing",
                "population_semantics": "active households in household analysis; otherwise World history",
                "firm_semantics": "multi-firm in FirmAnalysis reports; legacy aggregate aliases remain",
                "status_candidate": status,
            })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = source_inventory()
    fields = list(inventory[0])
    write_csv("analysis_function_classification.csv", inventory, fields)
    write_csv("analysis_module_inventory.csv", [
        {"module": p, "classes_or_entrypoints": ", ".join(sorted({r["function_name"] for r in inventory if r["module"] == p})), "direct_main_reachable": "yes" if p in {"analysis/analyzer.py", "analysis/demography.py", "analysis/economy.py", "analysis/firm.py", "analysis/fertility.py"} else "indirect/utility", "reads_csv": "no", "reads_world_state": "yes" if p != "analysis/time_semantics.py" else "no", "writes_files": "time_semantics manifest only" if p == "analysis/time_semantics.py" else "no explicit file writes", "status_candidate": "KEEP" if p in {"analysis/analyzer.py", "analysis/common.py", "analysis/time_semantics.py"} else "MODERNIZE"}
        for p in sorted({r["module"] for r in inventory})
    ], ["module", "classes_or_entrypoints", "direct_main_reachable", "reads_csv", "reads_world_state", "writes_files", "status_candidate"])

    write_csv("analysis_plot_inventory.csv", [
        {"plot_filename": "interactive/no explicit filename", "creating_function": f, "x_variable": "simulation week or derived index", "y_variables": y, "aggregation": a, "time_unit": t, "population_normalization": n, "firm_aggregation": firm, "duplicate_assessment": d, "interpretability": i}
        for f, y, a, t, n, firm, d, i in [
            ("plot_macro_economy", "GDP, consumption, saving", "aggregate weekly", "week", "aggregate", "aggregate firm", "SAME_METRIC_DIFFERENT_VIEW with ratio plot", "useful regression baseline"),
            ("plot_economy_ratio", "consumption/saving rates", "aggregate ratios", "week", "rate", "none", "SAME_METRIC_DIFFERENT_VIEW", "useful but denominator semantics need labels"),
            ("plot_household_economy_distribution", "wealth/income/consumption", "cross-section", "current step", "per household", "none", "LEGITIMATELY_DISTINCT", "useful research"),
            ("plot_household_economy_diagnostics", "wealth/GDP/consumption", "aggregate", "week", "mixed", "none", "NEAR_DUPLICATE", "conceptually mixed"),
            ("plot_household_inequality_diagnostics", "Lorenz/quantiles", "cross-section", "current step", "per household", "none", "LEGITIMATELY_DISTINCT", "useful research"),
            ("plot_firm_diagnostics", "cash, inventory, profit, prices, production, money", "aggregate histories", "week", "aggregate", "aggregate", "NEAR_DUPLICATE with balance sheet/monetary plots", "useful regression, overloaded"),
            ("plot_multi_firm_diagnostics", "firm cash/inventory/price/profit/debt", "firm panel", "week", "per firm", "firm-specific", "LEGITIMATELY_DISTINCT", "essential multi-firm"),
            ("plot_multi_firm_market_shares", "unit/revenue share", "firm panel", "week", "share", "firm-specific", "LEGITIMATELY_DISTINCT", "essential multi-firm"),
            ("plot_firm_balance_sheet", "cash/inventory/net worth", "aggregate", "week", "aggregate", "aggregate", "NEAR_DUPLICATE", "needs accounting labels"),
            ("plot_accounting_diagnostics", "accounting gaps/flows", "aggregate or firm rows", "week", "mixed", "firm/system", "LEGITIMATELY_DISTINCT", "essential acceptance"),
            ("plot_monetary_system / plot_money_accounting", "money stocks/flows/gaps", "aggregate", "week", "stock/flow", "none", "NEAR_DUPLICATE", "must be modernized for located balances"),
            ("plot_household_money_stock", "household cash and shares", "aggregate", "week", "stock/share", "none", "SAME_METRIC_DIFFERENT_VIEW", "useful after semantic audit"),
            ("plot_central_bank_food_reserve", "public inventory", "aggregate", "week", "units/value", "public sector", "LEGITIMATELY_DISTINCT", "useful goods diagnostics"),
        ]
    ], ["plot_filename", "creating_function", "x_variable", "y_variables", "aggregation", "time_unit", "population_normalization", "firm_aggregation", "duplicate_assessment", "interpretability"])

    write_csv("analysis_duplicate_computation.csv", [
        {"metric": m, "source_a": a, "source_b": b, "formula_or_timing_difference": diff, "canonical_recommendation": rec, "priority": p}
        for m, a, b, diff, rec, p in [
            ("population", "DemographyAnalysis.population", "World.population_history/diagnostics", "history vs current object count; active semantics differ", "diagnostics.csv canonical weekly macro", "P0"),
            ("household wealth", "EconomyAnalysis.household_record", "World wealth_history / accounting CSV", "cross-section current object vs sector aggregate", "sector aggregate plus explicit active-household distribution", "P0"),
            ("consumption/saving", "EconomyAnalysis history aliases", "diagnostics/accounting fields", "legacy history and accounting definitions can differ", "simulation accounting fields", "P0"),
            ("Firm cash/inventory/profit", "FirmAnalysis history aliases", "firm_diagnostics.csv aggregation", "aggregate and per-Firm paths", "firm diagnostics as source, aggregate by explicit policy", "P0"),
            ("money stock", "FirmAnalysis monetary histories", "diagnostics located_money/total_money", "additional public/CB balances are easy to omit", "money location reconciliation from diagnostics", "P0"),
            ("steady-state slope", "EconomyAnalysis.normalized_window_slope", "World diagnostics summary", "same concept may use different windows", "one canonical rolling-window metric", "P1"),
        ]
    ], ["metric", "source_a", "source_b", "formula_or_timing_difference", "canonical_recommendation", "priority"])

    write_csv("analysis_duplicate_outputs.csv", [
        {"output_a": a, "output_b": b, "strength": s, "evidence": e, "action_candidate": ac}
        for a, b, s, e, ac in [
            ("plot_firm_diagnostics", "plot_firm_balance_sheet", "NEAR_DUPLICATE", "cash/inventory/net worth overlap", "retain distinct accounting view, simplify panels"),
            ("plot_monetary_system", "plot_money_accounting", "NEAR_DUPLICATE", "both show money stocks/flows and gaps", "merge presentation later, preserve acceptance signals"),
            ("plot_macro_economy", "plot_household_economy_diagnostics", "SAME_METRIC_DIFFERENT_VIEW", "one macro, one mixed household/economy panel", "label scope explicitly"),
            ("multi_firm_report", "firm_report", "LEGITIMATELY_DISTINCT", "cross-firm report vs aggregate/firm report", "retain with explicit scope"),
            ("analysis stdout reports", "diagnostics CSV", "LEGITIMATELY_DISTINCT", "human summary vs machine regression data", "retain but separate output roles"),
        ]
    ], ["output_a", "output_b", "strength", "evidence", "action_candidate"])

    write_csv("analysis_performance_audit.csv", [
        {"operation": op, "location": loc, "cost": cost, "reason": reason, "future_action": action}
        for op, loc, cost, reason, action in [
            ("World analysis over in-memory histories", "main.py Analyzer calls", "MEDIUM", "no CSV reread, but large Python lists and plotting", "single preprocessing snapshot"),
            ("household_records and inequality", "analysis/economy.py", "HIGH", "traverses all household members and computes records repeatedly per report/plot", "cache one cross-section per analysis window"),
            ("firm_diagnostics_rows scans", "analysis/firm.py", "MEDIUM", "each firm series independently scans all rows", "index once by (firm_id, week)"),
            ("many plotting calls", "main.py and analysis/*.py", "HIGH", "multiple figures repeat series extraction and rendering", "shared compact tables and optional plot groups"),
            ("household diagnostics CSV", "main.py/world.py outputs", "VERY_HIGH", "large per-household-per-week file; analysis does not currently consume it centrally", "load selected columns/windows only"),
            ("ledger CSV", "main.py optional output", "HIGH", "event-level output can dominate I/O; analysis currently does not orchestrate it", "make ledger audit explicitly opt-in"),
            ("annual aggregation", "legacy test utilities", "MEDIUM", "several scripts independently group by step/52", "one weekly/annual preprocessing pass"),
        ]
    ], ["operation", "location", "cost", "reason", "future_action"])

    write_csv("analysis_time_semantics_audit.csv", [
        {"location": loc, "finding": finding, "classification": cls, "evidence": evidence, "priority": priority}
        for loc, finding, cls, evidence, priority in [
            ("analysis/common.py:model_years", "uses time_system.steps_to_years", "CORRECT_WEEKLY", "weekly steps converted through canonical helper", "P0"),
            ("analysis/common.py:rolling_year_window", "returns STEPS_PER_YEAR", "CORRECT_ANNUAL_AGGREGATION", "52-week window", "P0"),
            ("analysis/time_semantics.py", "weekly metadata and explicit legacy ambiguities", "CORRECT_WEEKLY", "centralized manifest", "P0"),
            ("analysis/demography.py age labels", "calendar years displayed from person age", "CORRECT_WEEKLY", "age conversion belongs to model/time system", "P1"),
            ("test/step13_2_credit_capacity_behavior.py", "years = step / 52", "CORRECT_ANNUAL_AGGREGATION", "test utility explicitly converts", "P1"),
            ("legacy analysis scripts", "some hardcoded old windows and step ranges", "AMBIGUOUS", "project-wide test utilities retain historical experiments", "P1"),
            ("old warm-up assumptions", "analysis core has no configurable mature_start", "OBSOLETE_TIME_ASSUMPTION", "steady_state_report defaults window only", "P0"),
        ]
    ], ["location", "finding", "classification", "evidence", "priority"])

    write_csv("analysis_dead_code_candidates.csv", [
        {"module_or_function": m, "classification": c, "evidence": e, "remove_candidate": r}
        for m, c, e, r in [
            ("analysis/firm.py legacy aggregate aliases", "LEGACY_MANUAL_TOOL", "kept for compatibility and called by plots", "NO"),
            ("analysis/common.py utility methods", "UNKNOWN", "indirectly used by mixins; AST alone is insufficient", "NO"),
            ("test/*analysis*.py", "TEST_ONLY", "experiment-specific workflows and accepted audits", "NO"),
            ("analysis/time_semantics.py legacy ambiguity manifest", "KEEP", "explicit regression documentation", "NO"),
            ("analysis plot methods not called with --no-plots", "LEGACY_MANUAL_TOOL", "reachable from main when plots enabled", "NO"),
        ]
    ], ["module_or_function", "classification", "evidence", "remove_candidate"])

    text("analysis_dependency_map.md", """
# Analysis Dependency Map

`main.py` imports `analysis.Analyzer`, which is the multiple-inheritance facade from `analysis/analyzer.py`. `Analyzer` combines `DemographyAnalysis`, `EconomyAnalysis`, `FertilityAnalysis`, and `FirmAnalysis`; all four inherit `CommonAnalysis` indirectly through their class declarations.

The runtime path is:

`main.py -> Analyzer(World) -> reports and plot methods -> World in-memory histories, current population/households, firm_diagnostics_rows, accounting_rows -> stdout and matplotlib figures`.

`main.py` calls `equilibrium_report`, `completed_fertility_report`, `economy_report`, `steady_state_report`, `firm_report`, `accounting_report`, and `multi_firm_report` regardless of `--no-plots`. With plots enabled it additionally calls World display methods plus the demography, economy, firm, accounting, money, and public-reserve plot methods. The analysis layer does not centrally read `diagnostics.csv`, `firm_diagnostics.csv`, or accounting CSVs; those are written by `World` before analysis.

Side effects are primarily stdout and matplotlib display. `analysis/time_semantics.py` writes `analysis_time_semantics.json` through a function called directly by `main.py`, not by `Analyzer`. No canonical analysis tables or markdown reports are written by the current core.

The main architectural risk is that the simulation has authoritative CSV diagnostics and accounting layers, while analysis still consumes parallel in-memory histories and legacy aliases. A future refactor should preserve the current calls while introducing one read/normalize/index pass.
""")

    text("analysis_population_audit.md", """
# Population Analysis Audit

The current layer covers population, dependency ratio, labor ratio, aging index, natural growth, stability, age heatmap, pyramid, fertility distributions, household size/type, and several household economic distributions. These remain useful regression and research signals.

The main gaps are explicit births/deaths/marriages time series in the core Analyzer, configurable warm-up/mature windows, and a consistent distinction between raw household object count, active households, and empty-household cleanup. The current `CommonAnalysis.active_households()` filter is good for cross-sectional welfare analysis, but macro diagnostics should use the simulation's explicit active-household fields rather than recompute from objects.

Priority: P0 for mature-window configuration and canonical population/household series; P1 for richer demographic interpretation; P2 for research-only cohort panels.
""")
    text("analysis_household_audit.md", """
# Household Analysis Audit

Household records are reconstructed from live objects and include wealth, income, consumption, saving, subsidy, pressure, needs, affordability, size, children, workers, elderly, and distribution statistics. Active households are filtered by nonzero member count, which avoids treating empty objects as welfare units.

Risks: the layer can differ from sector accounting because it reads current household objects; it does not centrally expose intergenerational transfers, estate/no-heir flows, cash wealth bridges, or public/central-bank money location. Cross-sectional calculations are repeated for reports and plots. Future analysis must state whether a metric is per active household, per person, or sector total.
""")
    text("analysis_real_economy_audit.md", """
# Goods and Real-Economy Audit

The current layer covers aggregate output, demand, sales, inventory, food price, unit labor cost, public inventory, and multi-firm market shares. `FirmAnalysis` now has multi-firm panels and accounting plots, so the architecture is not purely single-firm.

Legacy aggregate aliases (`food_*`, `firm_*_history`, legacy price) remain prominent. The current analysis does not make exact Firm-specific transaction settlement, unit versus revenue market share, inventory book value versus market value, COGS, or goods-conservation gaps the canonical source. These are available in diagnostics/accounting but need a regression-oriented preprocessing layer.
""")
    text("analysis_firm_audit.md", """
# Firm Analysis Audit

Firm analysis includes aggregate histories, per-firm diagnostics, multi-firm reports, prices, shares, production, inventory, cash, profit, debt, dividends, balance-sheet and accounting views. This is the strongest part of the current analysis layer for Step 11-era behavior.

It still mixes aggregate legacy FirmSystem histories with per-firm rows, and several plots combine macro, firm, monetary, and accounting concepts in one figure. Credit diagnostics are present only indirectly through optional histories and report fields; the accepted payroll-anchored state machine is not a first-class analysis domain. Interest states and arrears are absent from the general Analyzer despite being present in Step 13.4 outputs.
""")
    text("analysis_financial_audit.md", """
# Financial Analysis Audit

Accounting reports and plots are reachable from `main.py`, and the simulation writes firm, household, public, central-bank, lifecycle, and reconciliation CSVs. The analysis layer does not yet treat those CSVs as canonical inputs. Firm accounting is therefore present but coupled to current World memory and mixed with legacy profit/cash metrics.

Credit coverage is partial: target cash, funding gap, issued/repaid loans, and balances appear in current diagnostics, while denied credit, headroom, exposure, payroll funding states, and funded capacity are not consistently surfaced by general analysis. Interest due, paid, unpaid, arrears, exposure, and CB income are Step 13.4 additions not integrated into the core Analyzer.
""")
    text("analysis_money_accounting_audit.md", """
# Money and Accounting Audit

The simulation diagnostics already distinguish located money, household cash, Firm cash, public/central-bank balances, credit-created money, principal repayment, conservation gaps, and ledger gaps. The analysis layer exposes some histories and has monetary/accounting plots.

The main risk is semantic drift: legacy money plots and aliases can be read as if total money were only household wealth plus Firm cash. That is no longer valid because public/central-bank monetary balances and credit-created money are separate locations; interest payments change public income but do not create money, while arrears are non-monetary exposure. Reconciliation must remain sourced from simulation diagnostics, not recomputed independently in plotting code.
""")
    text("analysis_main_integration.md", """
# Main Integration Audit

CLI controls are `--no-analysis`, `--no-plots`, `--diagnostics-mode`, `--no-household-diagnostics-csv`, `--no-ledger-csv`, output path flags, steady-state window, and checkpoint/scenario options. `main.py` always runs the economic simulation and exports raw diagnostics/accounting first. `--no-analysis` exits after printing invariant summaries. Otherwise it constructs one `Analyzer` and executes reports, then optional plots.

The integration is understandable and currently single-orchestrator in name, but the orchestration is a long imperative sequence with no explicit analysis result object, no per-domain failure isolation, no analysis output manifest, and no exception boundary separating a successful simulation from a plotting/reporting failure. This is a targeted refactor priority, not a reason to change behavior in this audit.
""")
    text("analysis_output_structure.md", """
# Output Structure Audit

Normal runs write raw diagnostics at the selected output root: `diagnostics.csv`, `firm_diagnostics.csv`, optional `household_diagnostics.csv`, manifest, demographic CSVs, accounting subdirectory, and optional ledger. `analysis_time_semantics.json` is also written. The core Analyzer mostly prints and displays plots; it does not create a canonical `analysis/`, `plots/`, or `reports/` subtree.

The structure is usable for existing workflows but mixes raw and human-facing artifacts. A future reorganization should be compatibility-preserving and conceptually separate `raw/`, `analysis/`, `plots/`, and `reports/`; no files were moved here.
""")

    write_csv("analysis_domain_coverage_matrix.csv", [
        {"domain": d, "current_analysis_coverage": c, "duplication": dup, "time_semantics": time, "multi_firm_ready": multi, "step13_ready": step, "performance_issue": perf, "modernization_priority": p}
        for d, c, dup, time, multi, step, perf, p in [
            ("Demography", "good core reports/plots", "medium", "weekly mostly correct", "not applicable", "partial", "medium", "P1"),
            ("Households", "good cross-section", "high with accounting", "active filter correct", "partial", "partial", "high", "P0"),
            ("Labor", "ratios and aggregate wage history", "medium", "weekly", "partial", "missing payroll state", "low", "P1"),
            ("Goods", "aggregate and firm panels", "high legacy aliases", "weekly", "yes", "partial conservation", "medium", "P0"),
            ("Prices", "legacy/planning/realized views", "medium", "weekly", "yes", "partial", "medium", "P1"),
            ("Firms", "strong multi-firm coverage", "high aggregate/per-firm", "weekly", "yes", "partial credit", "high", "P0"),
            ("Inventory", "units/value/accounting plots", "medium", "weekly", "yes", "partial book-value", "medium", "P0"),
            ("Money", "several histories/plots", "high semantic overlap", "weekly", "aggregate only in places", "not Step13 complete", "high", "P0"),
            ("Credit", "legacy working-capital plus some fields", "medium", "weekly", "partial", "no", "medium", "P0"),
            ("Interest", "general Analyzer missing", "low", "not integrated", "no", "no", "low", "P0"),
            ("Central Bank", "food/public income views", "medium", "weekly", "partial", "partial", "medium", "P1"),
            ("Accounting", "reports/plots and raw CSVs", "high with simulation diagnostics", "weekly", "yes", "partial reconciliation view", "high", "P0"),
        ]
    ], ["domain", "current_analysis_coverage", "duplication", "time_semantics", "multi_firm_ready", "step13_ready", "performance_issue", "modernization_priority"])

    text("analysis_modernization_priorities.md", """
# Analysis Modernization Priorities

## What is good

- One recognizable `Analyzer` entry point remains reachable from `main.py`.
- Weekly time helpers exist and use the canonical `time_system` conversion.
- Active-household filtering, distribution summaries, multi-firm panels, and accounting plots are valuable reusable components.
- Raw diagnostics, accounting, conservation, and ledger outputs are already produced by the simulation layer.

## What is duplicated or mixed

- Aggregate World histories and per-Firm diagnostics coexist without one canonical source.
- Household object reconstruction overlaps sector accounting and is repeated for reports/plots.
- Firm balance-sheet, firm-diagnostics, monetary, and accounting plots overlap.
- Money views can repeat stocks while using different semantic scopes.

## What is obsolete or risky

- Legacy `food_*`, aggregate FirmSystem, old price, and fixed working-capital wording remain visible.
- General analysis has no first-class credit-capacity state or Step 13.4 interest/arrears interpretation.
- No explicit mature-society window is integrated into the main Analyzer.
- Analysis errors are not isolated from the successful simulation lifecycle.

## What is expensive

The main costs are repeated household traversal, repeated scans of `firm_diagnostics_rows`, many independent plotting passes, and optional household/ledger CSV I/O. The current source does not repeatedly read the same CSV inside the core Analyzer, but it also fails to exploit the already exported compact diagnostics as reusable preprocessing.

## What is missing

Before full Step 13 acceptance, analysis needs canonical weekly and annual tables, explicit warmup/mature/analysis windows, firm-specific credit states, interest due/paid/unpaid/arrears/exposure, Central Bank interest income, exact money-location reconciliation, and clear separation of regression acceptance from research interpretation.

## Recommended sequence

Preserve public method names and outputs first. Design a single-pass preprocessing layer that reads or receives diagnostics once, indexes macro/firm/financial domains, and emits regression summaries separately from research tables. Add an analysis manifest and isolate plotting/report exceptions after the canonical tables exist. Only then consider deprecating legacy aliases; none are safe to remove in this audit.

Overall recommendation: **targeted refactor design**, not immediate orchestrator replacement. The current architecture is understandable, but it requires targeted deduplication and Step 13 modernization before full-system acceptance runs.
""")

    text("acceptance_summary.md", """
# Pre-Step13.5 Analysis Architecture Audit

## Verdict

**B. The current analysis layer contains substantial redundancy / obsolete semantics and should be reorganized before further full-system acceptance runs.**

`analysis_safe_to_refactor = true`  
`main_analysis_integration_understood = true`  
`long_run_required_for_audit = false`  
`recommended_next_action = targeted_refactor_design`

This was a source-only audit. No simulation was run, no analysis function or `main.py` behavior was modified, no economic default changed, and no Step 13.5 distress behavior was added.
""")
    print(f"Wrote analysis audit to {OUT}")


if __name__ == "__main__":
    main()
