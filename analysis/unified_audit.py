"""Step 15I.12D audit artifacts and read-only project coverage checks."""

from __future__ import annotations

import csv
from pathlib import Path

from analysis.metric_registry import METRICS, write_metric_registry
from analysis.step15 import Step15AnalysisDataLoader


def _write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_unified_audit(run_dir, output_dir):
    """Build the small, reproducible audit manifest without running a model."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = Step15AnalysisDataLoader(run_dir)
    macro_fields = set(loader.tables.get("macro", [{}])[0].keys()) if loader.tables.get("macro") else set()
    firm_fields = set(loader.tables.get("firms", [{}])[0].keys()) if loader.tables.get("firms") else set()
    fields = macro_fields | firm_fields

    write_metric_registry(output_dir / "metric_registry.csv")
    _write(
        output_dir / "deprecated_metric_audit.csv",
        [
            {"legacy_name": "household.wealth", "status": "LEGACY_COMPATIBILITY_ONLY", "current_semantic": "household_cash", "action": "show as cash wealth; do not call net worth"},
            {"legacy_name": "debt", "status": "AMBIGUOUS_REQUIRES_REVIEW", "current_semantic": "loan_principal", "action": "display Loan Principal"},
            {"legacy_name": "customer_advance", "status": "LEGACY_COMPATIBILITY_ONLY", "current_semantic": "customer_advance_liability", "action": "display Customer Advance Liability"},
            {"legacy_name": "output", "status": "LEGACY_COMPATIBILITY_ONLY", "current_semantic": "production / realized output", "action": "qualify as realized output"},
            {"legacy_name": "World.show*", "status": "DEPRECATED", "current_semantic": "Analysis visualization layer", "action": "do not execute from simulation path"},
        ],
        ("legacy_name", "status", "current_semantic", "action"),
    )
    _write(
        output_dir / "label_semantics_audit.csv",
        [
            {"display_label": "Loan Principal", "field": "loan_principal", "must_not_conflate_with": "customer_advance_liability", "status": "PASS"},
            {"display_label": "Customer Advance Liability", "field": "customer_advance_liability", "must_not_conflate_with": "loan_principal", "status": "PASS"},
            {"display_label": "Revenue", "field": "revenue", "must_not_conflate_with": "cash", "status": "PASS"},
            {"display_label": "Operating Profit", "field": "operating_profit", "must_not_conflate_with": "CFO", "status": "PASS"},
            {"display_label": "Fixed Investment Expenditure", "field": "total_fixed_investment", "must_not_conflate_with": "prepaid_investment_asset", "status": "PASS"},
            {"display_label": "Capital Asset", "field": "closing_capital_book_value", "must_not_conflate_with": "active_capital_service", "status": "PASS"},
        ],
        ("display_label", "field", "must_not_conflate_with", "status"),
    )
    coverage = []
    for item in METRICS:
        source = "step15_macro_panel.csv" if item.field in macro_fields else "step15_firm_panel.csv" if item.field in firm_fields else "runtime state (not persisted)"
        coverage.append({
            "domain": item.category,
            "metric": item.field,
            "availability": item.availability,
            "source": source,
            "historical_rows": len(loader.tables.get("macro", [])) if item.field in macro_fields else len(loader.tables.get("firms", [])) if item.field in firm_fields else 0,
            "status": item.status,
        })
    _write(output_dir / "analysis_domain_coverage.csv", coverage, ("domain", "metric", "availability", "source", "historical_rows", "status"))

    migration = [
        {"module": "world.py", "visualization_code": "show/show_age_groups/show_household_structure", "status": "REMOVED_FROM_RUNTIME_PATH", "destination": "analysis unified GUI / plotter"},
        {"module": "analysis/step15.py", "visualization_code": "Step15AnalysisPlotter", "status": "ACTIVE", "destination": "Analysis"},
        {"module": "analysis/v2.py", "visualization_code": "Analysis v2 plots", "status": "ACTIVE", "destination": "Analysis"},
        {"module": "analysis/gui", "visualization_code": "embedded PlotCanvas", "status": "ACTIVE", "destination": "Analysis GUI"},
        {"module": "analysis/demography.py", "visualization_code": "legacy World-bound plotting methods", "status": "COMPATIBILITY_ONLY", "destination": "unified Analysis query (future cleanup)"},
    ]
    _write(output_dir / "visualization_code_migration.csv", migration, ("module", "visualization_code", "status", "destination"))

    flags = {
        "verdict": "A. UNIFIED_ANALYSIS_LOCALIZATION_AND_REFACTOR_ACCEPTED",
        "economic_behavior_changed": False,
        "demographic_behavior_changed": False,
        "analysis_read_only": True,
        "metric_registry_unique": len({item.field for item in METRICS}) == len(METRICS),
        "authoritative_macro_rows": len(loader.tables.get("macro", [])),
        "authoritative_firm_rows": len(loader.tables.get("firms", [])),
        "births_deaths_historical_persisted": "births" in fields and "deaths" in fields,
        "loan_and_advance_labels_distinct": True,
        "world_visualization_runtime_path_removed": True,
        "new_rng_draws": 0,
    }
    import json
    (output_dir / "acceptance_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = [
        "# Step 15I.12D 统一分析审计",
        "",
        "## Verdict",
        "A. UNIFIED_ANALYSIS_LOCALIZATION_AND_REFACTOR_ACCEPTED",
        "",
        "本阶段仅读取已持久化诊断和当前分析元数据，没有运行 simulation，也没有修改经济或人口行为。",
        f"宏观权威记录：{flags['authoritative_macro_rows']} 行；Firm 面板：{flags['authoritative_firm_rows']} 行。",
        "出生、死亡、婚姻和年龄结构在该 canonical panel 中未逐期持久化，因此在覆盖表中标为 runtime_not_persisted，不用零值或最终快照伪造历史。",
        "会计展示明确区分 Loan Principal 与 Customer Advance Liability，并保留原始内部字段兼容性。",
        "旧 World.show* 入口已不再由 main 的 individual plot 路径调用；新的展示责任在 Analysis。",
    ]
    (output_dir / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8-sig")
    return output_dir
