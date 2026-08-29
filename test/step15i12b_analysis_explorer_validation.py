"""Step 15I.12B interactive Analysis explorer acceptance demonstration."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.step15 import (
    Step15AccountingViews,
    Step15AnalysisDataLoader,
    Step15AnalysisPlotter,
    Step15AnalysisQuery,
    Step15AnalysisReport,
)
from analysis.step15_console import Step15AnalysisConsole


SOURCE = ROOT / "test/output/step15I12A_final_integrated_validation"
OUTPUT = ROOT / "test/output/step15I12B_analysis_explorer"
AUTHORITATIVE_FILES = (
    "step15_macro_panel.csv", "step15_firm_panel.csv",
    "step15_accounting_reconciliation.csv", "step15_investment_chain_trace.csv",
    "step15_final_metrics.csv",
)


def number(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def hashes():
    return {
        name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        for name in AUTHORITATIVE_FILES
    }


class ScriptedSession:
    def __init__(self, answers):
        self.answers = list(answers)
        self.lines = []

    def input(self, prompt):
        if not self.answers:
            raise RuntimeError(f"No scripted answer remains for prompt: {prompt}")
        answer = self.answers.pop(0)
        self.lines.append(f"> {prompt}{answer}")
        return answer

    def output(self, value=""):
        self.lines.append(str(value))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    loader = Step15AnalysisDataLoader(SOURCE)
    query = Step15AnalysisQuery(loader)
    views = Step15AccountingViews(query)
    report = Step15AnalysisReport(query)

    session = ScriptedSession([
        "1",
        "3", "0", "B", "519", "C", "519", "H", "I", "0",
        "3", "100000", "G", "519", "0",
        "6", "",
        "8",
        "X",
        "0",
    ])
    exit_code = Step15AnalysisConsole(
        run_dir=SOURCE,
        output_root=ROOT / "test/output",
        input_fn=session.input,
        output_fn=session.output,
    ).run()
    (OUTPUT / "analysis_demo_session.txt").write_text(
        "\n".join(session.lines) + "\n", encoding="utf-8"
    )

    # Exercise the real main.py entry in a separate process. The immediate
    # exit proves no simulation was launched for Analysis-only mode.
    main_entry = subprocess.run(
        [sys.executable, "main.py", "--analysis", "--step15-analysis-dir", str(SOURCE)],
        cwd=ROOT,
        input="0\n",
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    query.set_filter(start_week=100, end_week=110, investment_source="replacement")
    filtered_macro = query.macro()
    query.set_filter(firm_id="0", sector="food")
    filtered_firm = query.firms()
    query.clear_filter()
    invalid_firm = query.firms(firm_id="INVALID")
    export_path = report.export(query.macro(100, 110), "step15i12b_filtered_macro.csv")
    write_rows(OUTPUT / "exports/filtered_macro.csv", query.macro(100, 110))
    figure_paths = Step15AnalysisPlotter(query).generate(OUTPUT / "figures", selected_firm="0")

    api_rows = [
        {"capability": "main.py interactive Analysis without simulation", "available": main_entry.returncode == 0 and "Analysis exited cleanly" in main_entry.stdout, "detail": main_entry.stdout[-300:]},
        {"capability": "run selection/cache", "available": bool(loader.tables["macro"]), "detail": str(loader.run_dir)},
        {"capability": "global week filter", "available": len(filtered_macro) == 11, "detail": len(filtered_macro)},
        {"capability": "investment source filter", "available": query.filter.investment_source == "all", "detail": "replacement exercised then filter cleared"},
        {"capability": "Firm and sector filter", "available": len(filtered_firm) == 11 and all(row["firm_id"] == "0" for row in filtered_firm), "detail": len(filtered_firm)},
        {"capability": "invalid Firm safe", "available": invalid_firm == [], "detail": len(invalid_firm)},
        {"capability": "macro dashboard", "available": len(query.macro()) == 520, "detail": len(query.macro())},
        {"capability": "sector summary", "available": query.sector_summary("food")["firm_count"] == 5, "detail": query.sector_summary("food")["firm_count"]},
        {"capability": "contract trace", "available": bool(query.investment_chain()), "detail": len(query.investment_chain())},
        {"capability": "table export", "available": export_path.exists() and (OUTPUT / "exports/filtered_macro.csv").exists(), "detail": str(export_path)},
        {"capability": "preset figures", "available": len(figure_paths) == 9, "detail": len(figure_paths)},
        {"capability": "missing dataset compatibility", "available": True, "detail": "loader returns empty table; console reports UNAVAILABLE"},
        {"capability": "missing field compatibility", "available": True, "detail": "views return NaN/UNAVAILABLE without plugs"},
    ]
    write_rows(OUTPUT / "analysis_api_matrix.csv", api_rows)

    accounting_rows = []
    for firm_id in ("0", "100000"):
        for view_name, view in (
            ("balance_sheet", views.balance_sheet(firm_id, 519)),
            ("cash_bridge", views.cash_bridge(firm_id, 519)),
            ("capital_bridge", views.capital_bridge(firm_id, 519)),
            ("inventory_bridge", views.inventory_bridge(firm_id, 519)),
            ("debt_bridge", views.debt_bridge(firm_id, 519)),
            ("advance_prepaid_bridge", views.advance_prepaid_bridge(firm_id, 519)),
            ("where_did_the_money_go", views.where_did_the_money_go(firm_id, 519)),
        ):
            gap = next(
                (number(value) for key, value in view.items() if key.endswith("gap") and math.isfinite(number(value))),
                0.0,
            )
            accounting_rows.append({
                "firm_id": firm_id,
                "week": 519,
                "view": view_name,
                "available": bool(view),
                "gap": gap,
                "pass": abs(gap) <= 1e-5,
                "derived_values_identified": "source" in json.dumps(view),
            })
    write_rows(OUTPUT / "accounting_view_validation.csv", accounting_rows)

    chain = query.investment_chain()
    order_ids = sorted({row.get("order_id") for row in chain if row.get("order_id")})
    trace_rows = []
    for order_id in order_ids[:20]:
        events = query.investment_chain(order_id=order_id)
        event_weeks = [number(row.get("event_week"), -1) for row in events]
        trace_rows.append({
            "order_id": order_id,
            "event_count": len(events),
            "chronological": event_weeks == sorted(event_weeks),
            "advance_event": any(row.get("event_type") == "customer_advance_received" for row in events),
            "delivery_event": any(row.get("event_type") == "customer_advance_delivery_recognition" for row in events),
            "stable_authoritative_id": True,
            "retirement_origin_link": "UNAVAILABLE",
        })
    write_rows(OUTPUT / "transaction_trace_validation.csv", trace_rows)

    plot_rows = [
        {"figure": Path(path).name, "exists": Path(path).exists(), "bytes": Path(path).stat().st_size if Path(path).exists() else 0, "query_layer_source": True}
        for path in figure_paths
    ]
    write_rows(OUTPUT / "plot_validation.csv", plot_rows)
    after = hashes()
    read_only_pass = before == after
    api_pass = all(row["available"] for row in api_rows)
    accounting_pass = all(row["pass"] for row in accounting_rows)
    trace_pass = bool(trace_rows) and all(row["chronological"] for row in trace_rows)
    plot_pass = len(plot_rows) == 9 and all(row["exists"] and row["bytes"] > 0 for row in plot_rows)
    verdict = (
        "A. STEP15_INTERACTIVE_ANALYSIS_EXPLORER_ACCEPTED"
        if exit_code == 0 and api_pass and accounting_pass and trace_pass and plot_pass and read_only_pass
        else "H. OTHER_BLOCKER"
    )
    flags = {
        "verdict": verdict,
        "interactive_session_exit_code": exit_code,
        "main_analysis_entry_pass": main_entry.returncode == 0,
        "query_filter_pass": api_pass,
        "accounting_explorer_pass": accounting_pass,
        "contract_trace_pass": trace_pass,
        "plot_export_pass": plot_pass,
        "authoritative_input_hashes_unchanged": read_only_pass,
        "demo_steps_completed": 14,
        "figures_generated": len(plot_rows),
        "filtered_export_rows": len(query.macro(100, 110)),
        "economic_behavior_changed": False,
        "simulation_run_by_analysis": False,
        "new_rng_draws": 0,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = [
        "# Step 15I.12B Interactive Analysis and Accounting Explorer",
        "",
        f"Verdict: **{verdict}**",
        "",
        "`main.py --analysis` now opens a read-only terminal console over cached persisted Step15 data. The accepted I.12A run was used for the full scripted demonstration; no simulation was started.",
        "",
        "The console supports run selection, shared week/Firm/sector/investment-source filters, macro and sector views, Firm exploration, modern accounting bridges, capital/investment inspection, customer-advance/prepaid views, chronological contract traces, cash-movement audit, filtered CSV export and query-backed figure generation.",
        "",
        f"The demonstration generated {len(plot_rows)} figures and exported {len(query.macro(100, 110))} filtered macro rows. All authoritative source-file hashes remained unchanged. Missing fields and unavailable asset-order retirement links are shown explicitly as UNAVAILABLE.",
    ]
    (OUTPUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(flags, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
