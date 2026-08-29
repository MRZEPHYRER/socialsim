"""Passive hard/soft analysis checks; never changes simulation state."""

import json
from pathlib import Path

from analysis.statistics import distribution_summary, safe_mean


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_acceptance(context):
    macro = context.tables["weekly_macro"]
    firms = context.tables["weekly_firm"]
    household = context.tables["household_snapshot"]
    items = []

    def add(domain, metric, value, window, status, source, interpretation):
        items.append({"domain": domain, "metric": metric, "value": value, "window": window, "status": status, "source": source, "interpretation": interpretation})

    violations = sum(str(row.get("invariant_failed", "")).lower() == "true" for row in macro)
    add("ACCOUNTING", "invariant_violations", violations, "full", "PASS" if violations == 0 else "FAIL", "diagnostics_rows.invariant_failed", "Simulation hard invariant signal.")
    for field, label in (("monetary_accounting_gap", "money accounting gap"), ("money_delta_gap", "money delta gap"), ("goods_conservation_gap", "goods conservation gap")):
        values = [abs(_number(row.get(field)) or 0.0) for row in macro]
        maximum = max(values, default=0.0)
        add("MONEY" if "money" in field else "REAL_ECONOMY", label, maximum, "full", "PASS" if maximum <= 1e-6 else "WATCH", f"diagnostics_rows.{field}", "Reported reconciliation residual; threshold is numerical sanity only.")
    negative_cash = sum((_number(row.get("cash")) or 0.0) < -1e-9 for row in firms)
    add("FIRMS", "negative_firm_cash_rows", negative_cash, "full", "PASS" if negative_cash == 0 else "FAIL", "firm_diagnostics.cash", "Cash non-negativity check where applicable.")
    exposure_gaps = []
    affected = set()
    for row in firms:
        principal = _number(row.get("closing_principal"))
        arrears = _number(row.get("closing_interest_arrears"))
        reported = _number(row.get("lender_exposure"))
        if reported is not None and principal is not None and arrears is not None:
            gap = reported - principal - arrears
            exposure_gaps.append(abs(gap))
            if abs(gap) > 1e-8:
                affected.add(str(row.get("firm_id")))
    max_gap = max(exposure_gaps, default=0.0)
    add("INTEREST", "lender_exposure_gap", max_gap, "full", "PASS" if max_gap <= 1e-8 else "WATCH", "firm_diagnostics.lender_exposure", "Reported versus reconstructed principal plus arrears; never corrected silently.")
    add("INTEREST", "affected_firms", ",".join(sorted(affected)) or "none", "full", "INFO", "firm_diagnostics", "Firms with nonzero exposure semantic gap.")
    population = [_number(row.get("population")) for row in context.window("mature")]
    population = [v for v in population if v is not None]
    add("DEMOGRAPHY", "mature_population_mean", safe_mean(population), "mature", "INFO" if population else "WATCH", "diagnostics_rows.population", "Descriptive mature-window signal; no arbitrary pass/fail threshold.")
    add("HOUSEHOLDS", "active_household_snapshot_count", len(household), "final", "INFO", "World.active household objects", "Empty household objects excluded.")
    return {"items": items, "hard_acceptance_status": "PASS" if all(item["status"] != "FAIL" for item in items) else "FAIL"}


def write_acceptance(context, report):
    directory = context.analysis_dir / "acceptance"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "acceptance_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Analysis v2 Acceptance Summary", "", f"Hard status: **{report['hard_acceptance_status']}**", "", "| Domain | Metric | Value | Window | Status |", "|---|---|---:|---|---|"]
    lines.extend(f"| {item['domain']} | {item['metric']} | {item['value']} | {item['window']} | {item['status']} |" for item in report["items"])
    (directory / "acceptance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
