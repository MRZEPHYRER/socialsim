"""Analysis v2 orchestration, kept separate from the legacy Analyzer calls."""

import csv
import json
import os
import time
import traceback
from pathlib import Path

from analysis.acceptance import build_acceptance, write_acceptance
from analysis.context import AnalysisContext
from analysis.financial_core import (
    generate_standard_financial_core_plots,
    write_financial_core_analysis,
)
from analysis.research import build_research_summary, write_research


def _write_csv(path, rows):
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


def _metric_manifest(context):
    rows = []
    for table_name, table in context.tables.items():
        if not table:
            continue
        for field in table[0]:
            rows.append({
                "metric": field,
                "source_field": field,
                "source_table": table_name,
                "aggregation": "reported canonical field; annual tables use explicit sum/year_end names",
                "unit": "model-native; see source diagnostics",
                "normalization": "none unless field name says per_capita/share/ratio",
                "time_semantics": "weekly rows; annual derived from 52-week groups",
            })
    return rows


def _plots(context):
    import matplotlib.pyplot as plt

    directory = context.analysis_dir / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    macro = context.tables["weekly_macro"]
    firms = context.tables["weekly_firm"]
    x = [float(row.get("simulation_week", row.get("step", 0))) for row in macro]

    def value(row, field):
        try:
            return float(row.get(field, 0.0))
        except (TypeError, ValueError):
            return float("nan")

    dashboards = [
        ("01_demography.png", "Demography", [("population", "population"), ("births", "births"), ("deaths", "deaths")]),
        ("02_age_structure.png", "Age structure", [("working_age_population", "working age"), ("dependency_ratio", "dependency ratio")]),
        ("03_households.png", "Households", [("household_wealth", "wealth")]),
        ("04_real_economy.png", "Real economy", [("production", "production"), ("sales", "sales"), ("consumption", "consumption"), ("inventory", "inventory")]),
        ("05_prices.png", "Prices", [("planning_price", "planning"), ("realized_transaction_price", "realized")]),
        ("06_firm_competition.png", "Firm competition", []),
        ("07_firm_finance.png", "Firm finance", []),
        ("08_money_location.png", "Money location", [("total_money", "total money"), ("household_money", "household"), ("firm_money", "Firm cash")]),
        ("13_accounting.png", "Accounting stock identities", [("money_location_gap", "authoritative money-location gap"), ("goods_conservation_gap", "goods gap")]),
    ]
    paths = []
    entries = []
    for filename, title, series in dashboards:
        fig, axis = plt.subplots(figsize=(9, 4.5))
        if filename == "06_firm_competition.png":
            for firm_id in sorted({row.get("firm_id") for row in firms}):
                values = [value(row, "market_share") for row in firms if row.get("firm_id") == firm_id]
                axis.plot(values, label=f"Firm {firm_id}")
            axis.set_ylabel("unit market share")
            axis.legend()
        elif filename == "07_firm_finance.png":
            for firm_id in sorted({row.get("firm_id") for row in firms}):
                values = [value(row, "cash") for row in firms if row.get("firm_id") == firm_id]
                axis.plot(values, label=f"Firm {firm_id}")
            axis.set_ylabel("cash")
            axis.legend()
        else:
            for field, label in series:
                axis.plot(x, [value(row, field) for row in macro], label=label)
            axis.legend()
        axis.set_title(title)
        axis.set_xlabel("simulation week")
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=120)
        plt.close(fig)
        path = directory / filename
        paths.append(path)
        category = (
            "Demography" if filename.startswith(("01_", "02_")) else
            "Households" if filename.startswith("03_") else
            "Real Economy" if filename.startswith(("04_", "05_")) else
            "Firm Competition" if filename.startswith("06_") else
            "Firm Operations" if filename.startswith("07_") else
            "Money" if filename.startswith("08_") else
            "Accounting"
        )
        entries.append({
            "category": category,
            "plot_id": path.stem,
            "plot_label": title,
            "filename": str(path.relative_to(context.analysis_dir)),
            "source_function": "analysis.v2._plots",
            "availability": True,
            "required_fields": [field for field, _ in series],
            "default_visibility": True,
            "plot_role": "overview",
        })
    financial_paths, financial_entries = generate_standard_financial_core_plots(
        context
    )
    paths.extend(financial_paths)
    entries.extend(financial_entries)
    return paths, entries


def run_analysis_v2(world, output_dir, profile="standard", plot_mode="browser", no_plots=False):
    started = time.perf_counter()
    context = AnalysisContext(world, output_dir, profile=profile)
    context.write()
    write_financial_core_analysis(context)
    _write_csv(context.analysis_dir / "analysis_metric_manifest.csv", _metric_manifest(context))
    acceptance = build_acceptance(context)
    write_acceptance(context, acceptance)
    research = build_research_summary(context)
    write_research(context, research)
    plot_paths = []
    browser_status = {"opened": False, "mode": "disabled"}
    if not no_plots:
        plot_paths, plot_manifest = _plots(context)
        browser_items = [
            dict(entry, path=context.analysis_dir / entry["filename"])
            for entry in plot_manifest
        ]
        if profile == "full":
            from analysis.v22 import generate_detailed_plots
            plot_paths, plot_manifest = generate_detailed_plots(context)
            browser_items = [dict(entry, path=context.analysis_dir / entry["filename"]) for entry in plot_manifest]
        (context.analysis_dir / "browser_category_manifest.json").write_text(
            json.dumps(plot_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if plot_mode == "browser":
            try:
                from analysis.plot_browser import open_html_plot_browser
                print(
                    f"Analysis plots ready ({len(browser_items)}); opening local dashboard...",
                    flush=True,
                )
                browser_status = open_html_plot_browser(browser_items, metadata={
                    "seed": getattr(world, "seed", "NA"),
                    "profile": profile,
                    "years": f"{len(context.tables['weekly_macro']) / 52:.2f}",
                }, output_path=context.analysis_dir / "analysis_dashboard.html")
                print(
                    "Analysis dashboard:",
                    browser_status["dashboard_path"],
                    flush=True,
                )
            except Exception as exc:
                browser_status = {"opened": False, "mode": "save-only", "error": type(exc).__name__, "message": str(exc)}
                print("Interactive dashboard browser unavailable; plots were saved to", context.analysis_dir / "plots")
        elif plot_mode == "save-only":
            browser_status = {"opened": False, "mode": "save-only", "dashboard_count": len(plot_paths)}
        else:
            browser_status = {"opened": False, "mode": "individual", "dashboard_count": len(plot_paths)}
    (context.analysis_dir / "plot_browser_status.json").write_text(json.dumps(browser_status, indent=2), encoding="utf-8")
    manifest = context.manifest()
    manifest.update({
        "hard_acceptance_status": acceptance["hard_acceptance_status"],
        "analysis_runtime_seconds": time.perf_counter() - started,
        "analysis_completed": True,
    })
    (context.analysis_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (context.analysis_dir / "analysis_deprecation_manifest.json").write_text(json.dumps([
        {"legacy_function": name, "canonical_replacement": replacement, "known_callers": "legacy Analyzer/main.py", "safe_to_remove_later": False}
        for name, replacement in (("gdp", "weekly_macro.production"), ("consumption", "weekly_macro.consumption"), ("saving", "weekly_macro.household saving"), ("food_price", "planning_price/realized_transaction_price"), ("money_issued", "weekly_macro.credit_created_money"))
    ], indent=2), encoding="utf-8")
    return context, acceptance


def write_analysis_failure(output_dir, exception, stage):
    path = Path(output_dir) / "analysis" / "analysis_failure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"simulation_status": "completed", "analysis_status": "failed", "exception_type": type(exception).__name__, "message": str(exception), "stage": stage, "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
    return path
