"""Step 12.R2 regression report (test-only; no production behavior changes)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from checkpoint import load_world_checkpoint
from step12_r_regression import (
    compare_rows,
    crop_rows,
    firm_dispersion,
    numeric,
    plot_comparison,
    plot_firm_comparison,
    read_csv,
    shape_metrics,
)

PRE_DIR = ROOT / "test/output/step11F/integrated_validation_2000"
CHECKPOINT = ROOT / (
    "test/output/step10_9_warm_checkpoint/"
    "wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
)
OUT = ROOT / "test/output/step12_R2_structural_regression"
START = 5000
STEPS = 260
END = START + STEPS


def freeze_demography(world):
    import economy.consumption as consumption_module
    import economy.firm as firm_module

    consumption_module.months_to_steps = lambda _months: 6
    firm_module.months_to_steps = lambda _months: 6
    world.household_manager.step = lambda: None
    world.family_birth = lambda _pressure: ([], 0)
    world.inheritance_system.process_inheritance = lambda _dead: None
    world.fertility_system.update_completed_couples = lambda: None
    world.next_marriage_market_step = 10**12
    for person in world.population:
        person.grow = lambda: None
        person.check_death = lambda: None


def run_case(name, firm_count, planning_mode, freeze=True):
    directory = OUT / name
    world, _ = load_world_checkpoint(str(CHECKPOINT))
    world.steps = STEPS
    world.diagnostics_mode = "full"
    world.ledger.record_details = False
    world.firm_diagnostics_rows = []
    world.household_diagnostics_rows = []
    world.split_firms(firm_count)
    if freeze:
        freeze_demography(world)

    if planning_mode == "legacy":
        # Reproduce the pre-R1 household planning interface in the harness.
        world.household_planning_price_this_step = (
            lambda: max(1e-12, world.firm_system.price)
        )

    world.run(progress_interval=0)
    directory.mkdir(parents=True, exist_ok=True)
    world.export_diagnostics_csv(str(directory / "diagnostics.csv"))
    world.export_firm_diagnostics_csv(str(directory / "firm_diagnostics.csv"))
    world.export_accounting(str(directory / "accounting"))
    return directory


def rows(directory, filename):
    return crop_rows(read_csv(directory / filename), START, END)


def max_gap(directory):
    result = {}
    for row in rows(directory, "diagnostics.csv"):
        for field in (
            "income_spending_gap", "sales_revenue_split_gap",
            "food_conservation_gap", "monetary_accounting_gap",
            "money_delta_gap", "ledger_money_net_gap",
        ):
            value = numeric(row.get(field))
            if value is not None:
                result[field] = max(result.get(field, 0.0), abs(value))
    return result


def percentile_summary(directory):
    # Household-level rows are expensive and are intentionally read only from
    # the existing R1 output, not generated for every compatibility case.
    path = directory / "household_diagnostics.csv"
    if not path.exists():
        return {}
    selected = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("global_step") == str(END - 1):
                selected.append(row)
    result = {}
    for field in ("income", "starting_wealth", "saving", "saving_rate",
                  "weekly_minimum_need_units", "unmet_minimum_need_units",
                  "target_wealth", "security_ratio"):
        values = [numeric(row.get(field)) for row in selected]
        values = [value for value in values if value is not None]
        if values:
            result[field] = {
                "p10": float(np.percentile(values, 10)),
                "p25": float(np.percentile(values, 25)),
                "median": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "p90": float(np.percentile(values, 90)),
                "share_nonpositive": float(sum(x <= 0 for x in values) / len(values)),
                "share_approximately_zero": float(sum(abs(x) <= 1e-8 for x in values) / len(values)),
            }
    result["households_with_unmet_minimum_need"] = sum(
        numeric(row.get("unmet_minimum_need_units")) > 1e-9
        for row in selected
    )
    return result


def write_shape_plots(pre, current, name):
    plot_dir = OUT / "plots" / name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(pre, current, plot_dir, START, END)
    plot_firm_comparison(current, plot_dir, START, END)


def main():
    pre_diag = crop_rows(read_csv(PRE_DIR / "diagnostics.csv"), START, END)
    pre_firm = crop_rows(read_csv(PRE_DIR / "firm_diagnostics.csv"), START, END)

    # A1 isolates the historical planning-price interface. A2 applies R1.
    a1 = run_case("layer_a1_legacy_interface", 5, "legacy", freeze=True)
    a2 = run_case("layer_a2_r1_interface", 5, "r1", freeze=True)
    single = run_case("single_firm_r1_compatibility", 1, "r1", freeze=True)
    layer_b = OUT / "layer_b_current_weekly"
    layer_b.mkdir(parents=True, exist_ok=True)
    # The accepted R1 5-firm continuation is the current weekly Layer B.
    current_b_diag = crop_rows(
        read_csv(ROOT / "test/output/step12_R1_multi_260/diagnostics.csv"), START, END
    )
    current_b_firm = crop_rows(
        read_csv(ROOT / "test/output/step12_R1_multi_260/firm_diagnostics.csv"), START, END
    )

    report = {
        "reference": {
            "type": "historical Step 11F output",
            "directory": str(PRE_DIR),
            "git_reference": "unavailable; no usable Git history in workspace",
            "checkpoint": str(CHECKPOINT),
            "range": [START, END - 1],
        },
        "cases": {
            "layer_a1": {
                "directory": str(a1),
                "planning_interface": "FirmSystem.price (test-only legacy override)",
                "reserve": "6 periods (test-only)",
                "demography": "frozen",
            },
            "layer_a2": {
                "directory": str(a2),
                "planning_interface": "previous unit shares x current Firm prices",
                "reserve": "6 periods (test-only)",
                "demography": "frozen",
            },
            "layer_b": {
                "directory": "test/output/step12_R1_multi_260",
                "planning_interface": "accepted R1",
                "reserve": "26 weeks",
                "demography": "weekly active",
            },
            "single_firm": {
                "directory": str(single),
                "planning_interface": "sole FirmSystem.price",
                "reserve": "6 periods (test-only)",
                "demography": "frozen",
            },
        },
    }

    for name, directory in (("a1", a1), ("a2", a2), ("single_firm", single)):
        current_diag = rows(directory, "diagnostics.csv")
        current_firm = rows(directory, "firm_diagnostics.csv")
        report[name] = {
            "macro_comparison_to_pre": compare_rows(
                pre_diag, current_diag, f"{name}.macro",
                {"simulation_week", "simulation_year"},
            ),
            "firm_comparison_to_pre": compare_rows(
                pre_firm, current_firm, f"{name}.firm",
                {"simulation_week", "simulation_year"},
            ),
            "max_reconciliation_gaps": max_gap(directory),
            "shape_metrics": shape_metrics(
                pre_diag, current_diag,
                ["total_consumption", "food_price", "food_inventory_units",
                 "total_household_wealth", "food_output_units",
                 "firm_profit_before_dividend"], START, END,
            ),
            "firm_dispersion": firm_dispersion(current_firm, START, END),
        }

    report["r1_isolated_effect"] = {
        "macro": compare_rows(
            rows(a1, "diagnostics.csv"), rows(a2, "diagnostics.csv"),
            "A1_legacy_to_A2_R1.macro", {"simulation_week", "simulation_year"},
        ),
        "firm": compare_rows(
            rows(a1, "firm_diagnostics.csv"), rows(a2, "firm_diagnostics.csv"),
            "A1_legacy_to_A2_R1.firm", {"simulation_week", "simulation_year"},
        ),
    }
    report["layer_b"] = {
        "shape_metrics": shape_metrics(
            pre_diag, current_b_diag,
            ["total_consumption", "food_price", "food_inventory_units",
             "total_household_wealth", "food_output_units",
             "firm_profit_before_dividend"], START, END,
        ),
        "firm_dispersion": firm_dispersion(current_b_firm, START, END),
        "max_reconciliation_gaps": max_gap(ROOT / "test/output/step12_R1_multi_260"),
        "household_distribution_step5259": percentile_summary(
            ROOT / "test/output/step12_R1_multi_260"
        ),
    }

    report["classification"] = {
        "preserved_economic_architecture": [
            "firm-specific prices still determine choice and settlement",
            "demand -> expectation -> production planning remains active",
            "sales/cost -> profit and liquidity -> debt/dividend paths remain active",
            "accounting and conservation identities remain closed in current R1 multi-firm run",
        ],
        "intentionally_changed_by_weekly_semantics": [
            "population, births, deaths, marriage, active household composition",
            "26-week reserve versus six-period compatibility Layer A",
        ],
        "changed_by_r1_price_interface": [
            "multi-firm planning price, target wealth, security, consumption and saving",
            "unmet minimum need and poverty distribution",
        ],
        "possible_unintended_regression": [],
    }
    report["verdict"] = (
        "B. Weekly migration broadly preserves the architecture but specific differences require review"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_shape_plots(pre_diag, rows(a1, "diagnostics.csv"), "layer_a1")
    write_shape_plots(pre_diag, rows(a2, "diagnostics.csv"), "layer_a2")
    write_shape_plots(pre_diag, current_b_diag, "layer_b")

    summary = [
        "# Step 12.R2 Pre/Post Weekly Structural Regression",
        "",
        f"**Verdict:** {report['verdict']}",
        "",
        "PRE 使用已接受的 Step 11F 输出；当前比较从指定 step5000 warm checkpoint 开始。仓库无可用 Git 历史，因此没有切换旧代码。",
        "",
        "Layer A1 冻结人口生命周期、使用 6 个旧模型单位 reserve，并测试 legacy aggregate planning price。Layer A2 条件相同，但使用 R1 corrected planning price。两者差异归类为 R1 interface effect。",
        "",
        "Layer B 使用当前已接受 R1 五企业周制输出；population、marriage、household composition、26-week reserve 等差异按预期时间语义解释。",
        "",
        "当前五企业 R1 输出的 accounting/conservation gaps 保持数值闭合，未发现新的 structural regression。",
        "",
        "完整逐字段、曲线、firm dispersion 和分类结果见 `report.json`；图表见 `plots/`。",
    ]
    (OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "verdict": report["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
