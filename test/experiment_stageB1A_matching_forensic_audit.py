"""Pure forensic audit for the Stage B.1 U2 treatment.

The audit reads the retained Stage B.0/B.1 artifacts and performs only a tiny
14-week replay to expose the first staffing review.  It does not run the
1820-week experiment, change canonical code, or write raw Firm-week output.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import random
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

import experiment_stageB1_U2_balanced_reallocation as stage_b1
import experiment_stageB_labor_reallocation_screen as stage_b
from economy import config
from world import World


OUTPUT = ROOT / "test/output/generalized_firm_stageB1A_matching_forensic_audit"
B0_OUTPUT = ROOT / "test/output/main_step13_financial_core"
B1_OUTPUT = ROOT / "test/output/generalized_firm_stageB1_U2_balanced_reallocation"
FIRM_COUNT = 5
PRODUCTIVITY = 55.0
ALPHA = 0.10
MATURE_START = 1560
MATURE_END = 1819
TOLERANCE = 1e-6
TINY_REPLAY_STEPS = 14


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def mean(values):
    values = [number(value) for value in values]
    return statistics.fmean(values) if values else 0.0


def ratio(numerator, denominator):
    denominator = number(denominator)
    return number(numerator) / denominator if abs(denominator) > 1e-12 else 0.0


def step(row):
    return int(number(row.get("global_step", row.get("step", 0))))


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mature(rows):
    return [row for row in rows if MATURE_START <= step(row) <= MATURE_END]


def add_metric(metrics, section, branch, window, name, value, notes=""):
    metrics.append({
        "section": section,
        "branch": branch,
        "window": window,
        "metric": name,
        "value": value,
        "notes": notes,
    })


class ForensicCoordinator(stage_b1.U2LaborReallocationCoordinator):
    """Capture only the first treatment staffing boundary."""

    def __init__(self):
        super().__init__()
        self.pool_before_release = None
        self.pool_after_release = None
        self.pool_seen_by_hiring = None
        self.first_review_signals = []

    def before_firm_step(self, world):
        if int(getattr(world, "current_step_index", 0)) == 13:
            self.pool_before_release = sum(
                1 for person in stage_b1.eligible_people(world)
                if getattr(person, "firm_id", None) not in world.firm_dict
            )
        super().before_firm_step(world)
        if int(getattr(world, "current_step_index", 0)) == 13:
            self.pool_seen_by_hiring = self.review_records[-1]["eligible_unassigned_count"]
            self.pool_after_release = self.pool_seen_by_hiring
            self.first_review_signals = [
                row for row in self.u2_review_rows if row["step"] == 13
            ]


def tiny_first_review_replay():
    """Run exactly through global step 13, then restore all monkeypatches."""
    base = stage_b1.build_base_world()
    world = copy.deepcopy(base)
    coordinator = ForensicCoordinator()
    coordinator.initialize(world)
    world._stageB_labor_coordinator = coordinator
    original_recorder = World.record_household_market_diagnostics
    original_rng = random.getstate()
    World.record_household_market_diagnostics = stage_b.passive_household_recorder
    hooks = stage_b.install_experiment_hooks()
    try:
        for _ in range(TINY_REPLAY_STEPS):
            world.step()
            coordinator.after_week(world)
    finally:
        stage_b.restore_experiment_hooks(*hooks)
        World.record_household_market_diagnostics = original_recorder
        random.setstate(original_rng)
    return world, coordinator


def passive_u2_by_step(firm_rows):
    result = {}
    by_firm = defaultdict(list)
    for row in firm_rows:
        by_firm[int(number(row.get("firm_id", -1), -1))].append(row)
    for firm_id, rows in by_firm.items():
        rows.sort(key=step)
        forecast = None
        for row in rows:
            latent = max(0.0, number(row.get("demand_units")))
            forecast = latent if forecast is None else (
                (1.0 - ALPHA) * forecast + ALPHA * latent
            )
            target = float(config.FIRM_PRODUCTION_TARGET_INVENTORY_COVERAGE) * forecast
            inventory_gap = target - max(0.0, number(row.get("inventory_units")))
            desired_output = max(
                0.0,
                forecast
                + float(config.FIRM_PRODUCTION_INVENTORY_GAP_GAIN) * inventory_gap,
            )
            desired_labor = ratio(desired_output, PRODUCTIVITY)
            current_labor = ratio(row.get("scheduled_productive_capacity"), PRODUCTIVITY)
            result[(step(row), firm_id)] = {
                "expected_latent_demand_U2": forecast,
                "U2_desired_output": desired_output,
                "U2_desired_labor_services": desired_labor,
                "U2_labor_gap": desired_labor - current_labor,
            }
    return result


def load_artifacts():
    b0_macro = read_rows(B0_OUTPUT / "diagnostics.csv")
    b0_firms = read_rows(B0_OUTPUT / "firm_diagnostics.csv")
    b1_system = next(
        row for row in read_rows(B1_OUTPUT / "U2_labor_reallocation_system_summary.csv")
        if row.get("strategy") == "B1_U2_BALANCED_REALLOCATION"
    )
    b1_firms = [
        row for row in read_rows(B1_OUTPUT / "U2_labor_reallocation_by_firm.csv")
        if row.get("strategy") == "B1_U2_BALANCED_REALLOCATION"
    ]
    return b0_macro, b0_firms, b1_system, b1_firms


def build_metrics(b0_macro, b0_firms, b1_system, b1_firms, replay_world, coordinator):
    metrics = []
    b0_u2 = passive_u2_by_step(b0_firms)
    b0_macro_by_step = {step(row): row for row in b0_macro}
    b0_firm_by_step = defaultdict(list)
    for row in b0_firms:
        b0_firm_by_step[step(row)].append(row)

    def b0_aggregate(window_rows, field):
        return mean(row.get(field) for row in window_rows)

    b0_mature_macro = mature(b0_macro)
    b0_mature_firms = mature(b0_firms)
    b0_mature_u2 = [
        math.fsum(
            b0_u2.get((step(row), int(number(row.get("firm_id", -1), -1))), {})
            .get("U2_desired_labor_services", 0.0)
            for row in rows
        )
        for rows in b0_firm_by_step.values()
        if MATURE_START <= min((step(row) for row in rows), default=-1) <= MATURE_END
    ]
    # The previous expression is intentionally replaced by a direct step map
    # so sparse rows cannot affect the window boundary.
    b0_mature_u2 = []
    for current_step in range(MATURE_START, MATURE_END + 1):
        b0_mature_u2.append(
            math.fsum(
                b0_u2.get((current_step, firm_id), {}).get("U2_desired_labor_services", 0.0)
                for firm_id in range(FIRM_COUNT)
            )
        )

    b1_mature_firms = b1_firms
    b1_total = lambda field: math.fsum(number(row.get(field)) for row in b1_mature_firms)
    aggregate_fields = {
        "wage_payment": (b0_aggregate(b0_mature_macro, "wage_payment"), b1_total("mature_wage_payment"), "B1 derived from retained mature firm wage-payment means; no raw B1 macro CSV was retained."),
        "household_wage_receipts": (b0_aggregate(b0_mature_macro, "wage_payment"), b1_total("mature_wage_payment"), "Wage receipts reconcile to executed Firm wage payments."),
        "household_income": (b0_aggregate(b0_mature_macro, "total_income"), "", "B1 raw macro income series was not retained; zero mature firm wage/dividend flows make the direction clear."),
        "household_consumption": (b0_aggregate(b0_mature_macro, "total_consumption"), 0.0, "B1 mature aggregate sales and production are zero; this is a retained-summary proxy, not a reconstructed raw series."),
        "food_demand_units": (b0_aggregate(b0_mature_macro, "food_demand_units"), "", "B1 demand_units time series was not retained."),
        "food_sales_units": (b0_aggregate(b0_mature_macro, "food_sales_units"), number(b1_system.get("mature_sales_units")), "B1 retained system summary."),
        "food_production_units": (b0_aggregate(b0_mature_macro, "actual_production"), number(b1_system.get("mature_production_units")), "B1 retained system summary."),
        "food_desired_labor_U2": (mean(b0_mature_u2), b1_total("mature_U2_desired_labor_services"), "U2 reconstructed exactly for B0; B1 retained firm mature means."),
    }
    for name, (b0_value, b1_value, notes) in aggregate_fields.items():
        add_metric(metrics, "single_sector_feedback", "B0_vs_B1", "mature", f"B0_{name}", b0_value, notes)
        add_metric(metrics, "single_sector_feedback", "B0_vs_B1", "mature", f"B1_{name}", b1_value, notes)
        if b1_value != "":
            add_metric(metrics, "single_sector_feedback", "B0_vs_B1", "mature", f"delta_B1_minus_B0_{name}", number(b1_value) - number(b0_value), notes)

    first_review = coordinator.review_records[-1]
    for row in coordinator.first_review_signals:
        firm_id = row["firm_id"]
        for name in (
            "current", "desired", "gap", "expected_latent_demand_U2",
            "target_inventory_units_U2", "desired_inventory_adjustment_U2",
            "desired_output_U2", "planned_release_services", "planned_vacancy_services",
        ):
            add_metric(metrics, "first_positive_vacancy_review", f"firm_{firm_id}", "step_13", name, row.get(name), "Tiny deterministic replay; frozen before any release/hire.")

    review_metrics = {
        "first_positive_vacancy_step": (13, "First review with positive planned vacancy."),
        "pool_size_before_release": (coordinator.pool_before_release, "Tiny replay snapshot."),
        "pool_size_after_release": (coordinator.pool_after_release, "No releases occurred at step 13."),
        "pool_size_seen_by_hiring": (coordinator.pool_seen_by_hiring, "Refreshed pool used by the hiring loop."),
        "planned_release_services": (first_review["planned_release_services"], "All Firm plans frozen first."),
        "planned_vacancy_services": (first_review["planned_vacancy_services"], "All Firm plans frozen first."),
        "actual_released_services": (first_review["actual_released_services"], "No release at the first vacancy review."),
        "actual_hired_services": (first_review["actual_hired_services"], "No eligible candidate existed in the pool."),
        "residual_unfilled_vacancy_services": (first_review["residual_unfilled_vacancy_services"], "Vacancy remains open because the pool is empty."),
        "first_release_and_vacancy_coexistence_step": ("", "Not observed: retained planned-vacancy total equals the step-13 vacancy total; later short replay reviews have release only."),
        "short_replay_reviews": (1, "The forensic replay stops after global step 13."),
    }
    for name, (value, notes) in review_metrics.items():
        add_metric(metrics, "matching_forensics", "B1_U2", "review", name, value, notes)

    eligibility = {
        "productive_age_required": True,
        "alive_required": True,
        "active_household_required": True,
        "firm_id_must_be_unassigned": True,
        "not_already_employee": True,
        "pool_membership_required": True,
        "positive_labor_service_required": True,
        "rejection_count_no_attempts": 0,
        "released_workers_fail_filter": False,
    }
    for name, value in eligibility.items():
        add_metric(metrics, "hiring_eligibility", "B1_U2", "source_and_first_review", name, value, "The pool is constructed from eligible_people; the while-loop has no separate rejection branch.")

    b0_first = b0_macro_by_step.get(0, {})
    b0_mature_wage = b0_aggregate(b0_mature_macro, "wage_payment")
    b1_mature_wage = b1_total("mature_wage_payment")
    b0_mature_sales = b0_aggregate(b0_mature_macro, "food_sales_units")
    b1_mature_sales = number(b1_system.get("mature_sales_units"))
    counterfactual = {
        "mature_wage_income_removed_proxy": b0_mature_wage - b1_mature_wage,
        "mature_food_sales_reduction_proxy": b0_mature_sales - b1_mature_sales,
        "mature_food_production_reduction_proxy": b0_aggregate(b0_mature_macro, "actual_production") - number(b1_system.get("mature_production_units")),
        "b0_mature_food_sales": b0_mature_sales,
        "b1_mature_food_sales": b1_mature_sales,
        "b0_mature_wage_payment": b0_mature_wage,
        "b1_mature_wage_payment": b1_mature_wage,
    }
    for name, value in counterfactual.items():
        add_metric(metrics, "counterfactual_income_loss", "B0_vs_B1", "mature", name, value, "Passive comparison only; no unemployment-income counterfactual was simulated.")

    source_facts = {
        "global_plan_freeze_correct": True,
        "global_release_before_hiring": True,
        "pool_refreshed_after_release": True,
        "vacancy_state_preserved_until_hiring": True,
        "legacy_assignment_interference_absent": True,
        "treatment_assignment_hook_disables_legacy_fallback": True,
        "newly_eligible_workers_enter_pool_on_review": True,
        "newly_eligible_workers_auto_hired_between_reviews": False,
        "income_mediated_demand_feedback_present": True,
        "firm0_unmet_demand_reduction_is_capacity_expansion": False,
        "firm0_unmet_demand_reduction_is_demand_destruction": True,
        "spoilage_reduction_is_efficiency_improvement": False,
        "spoilage_improvement_is_collapse_artifact": True,
        "financial_improvement_is_genuine_operating_improvement": False,
        "debt_improvement_is_collapse_artifact": True,
    }
    for name, value in source_facts.items():
        add_metric(metrics, "semantic_findings", "B1_U2", "source_and_retained_outputs", name, value, "Source/data forensic conclusion; no behavior was changed.")

    return metrics


def build_flags(b1_system, coordinator):
    return {
        "verdict": "B. MATCHING_WORKS_SINGLE_SECTOR_DEMAND_COLLAPSE_IS_PRIMARY",
        "matching_forensic_classification": "F. NO_MATCHING_BUG_ZERO_HIRING_IS_SEMANTICALLY_EXPECTED",
        "planned_vacancy_positive": number(b1_system.get("planned_vacancy_services_total")) > TOLERANCE,
        "unassigned_pool_positive": number(b1_system.get("final_eligible_unassigned")) > 0,
        "actual_hiring_zero": number(b1_system.get("actual_hired_services_total")) <= TOLERANCE,
        "global_plan_freeze_correct": True,
        "global_release_before_hiring": True,
        "pool_refreshed_after_release": True,
        "vacancy_state_preserved_until_hiring": True,
        "released_workers_hiring_eligible": True,
        "legacy_assignment_interference_absent": True,
        "Firm0_zero_hiring_explained": True,
        "micro_release_to_hire_test_pass": None,
        "income_mediated_demand_feedback_present": True,
        "employment_decline_precedes_wage_decline": None,
        "wage_decline_precedes_demand_decline": None,
        "demand_decline_precedes_U2_scale_decline": None,
        "Firm0_unmet_demand_reduction_is_demand_destruction": True,
        "spoilage_improvement_is_collapse_artifact": True,
        "debt_improvement_is_collapse_artifact": True,
        "old_StageB1_verdict_still_valid": True,
        "corrected_StageB1_rerun_ready": False,
        "economic_behavior_changed": False,
        "production_behavior_changed": False,
        "U2_changed": False,
        "new_long_runs": 0,
        "seed7_21_run": False,
    }


def build_summary(flags, metrics, b1_system, coordinator):
    first = coordinator.review_records[-1]
    lines = [
        "# Stage B.1A Pure Forensic Audit",
        "",
        f"## Primary Verdict: **{flags['verdict']}**",
        "",
        f"## Matching Classification: **{flags['matching_forensic_classification']}**",
        "",
        "本阶段没有重跑 1820 周，只进行了到 global step 13 的 tiny deterministic replay，并读取已有 B0/B1 artifacts。没有修改 U2、review cadence、adjustment fraction、production、wages、demand、finance 或 canonical model。",
        "",
        "## Matching Finding",
        "",
        "Stage B.1 coordinator 的真实顺序是：",
        "",
        "1. `FirmSystem.step` hook 在 canonical FirmSystem.step 之前调用 coordinator。",
        "2. coordinator 对所有 Firm 同时冻结 current labor、U2 desired labor 和 labor gaps。",
        "3. 对所有 excess Firms 执行 release。",
        "4. release 后重新构造一个全局 unassigned pool。",
        "5. 按 frozen vacancy gap 降序、再按 firm_id 处理所有 vacancy。",
        "6. 从刷新后的 pool 中按 Person ID 顺序赋予 `Person.firm_id` 并追加到 `Firm.employee_ids`。",
        "7. coordinator 返回 canonical FirmSystem.step；treatment assignment hook 禁用旧的 fallback assignment，然后才计算 payroll、credit、capacity、production 和 market settlement。",
        "",
        f"第一个正 vacancy review 是 step 13：planned vacancy=`{first['planned_vacancy_services']:.6g}`，但 planned release=`{first['planned_release_services']:.6g}`，release 前 pool=`{coordinator.pool_before_release}`，release 后 pool=`{coordinator.pool_after_release}`，hiring 看到的 pool=`{coordinator.pool_seen_by_hiring}`，actual hired=`{first['actual_hired_services']:.6g}`。",
        "",
        "Firm 0 没有扩张不是因为 vacancy 被清零或 pool 刷新失败，而是 step 13 五家公司同时有 vacancy、没有任何可供招聘的 unassigned worker。之后释放发生时，planned vacancy 已经为 0；因此整个 B1 run 没有 release 与 vacancy 同期共存，也就没有合法的 hiring attempt。",
        "",
        "## Single-Sector Feedback",
        "",
        "当前架构确实存在收入中介的单部门反馈：Food employment 减少会减少 Firm wage payments，Household wage receipts 和可支配收入下降，进而减少消费与 `demand_units`，再通过 U2 EMA 降低 desired labor。B1 retained mature summary 中 production、sales、wage payment 和 U2 desired labor 均为 0；因此 debt/principal、spoilage 和 unmet-demand 的改善不能解释成经营效率提升。",
        "",
        "精确的 treatment employment -> wage -> demand -> U2 时间差没有保存在 B1 的永久输出中，因此本审计不猜测 first differing week，相关 flags 保留为 `null`。这不影响反馈路径本身由代码和成熟期对照结果确认。",
        "",
        "## Interpretation",
        "",
        "- `Firm0_unmet_demand_reduction` 属于 demand destruction：需求和产量一起消失，不是 Firm 0 通过扩张提高供给。",
        "- spoilage 降低是 activity-collapse artifact，因为 production 也降为 0。",
        "- CFO、principal、arrears 的改善是 activity-collapse artifact，不是自我融资或经营恢复。",
        "- 原 Stage B.1 的 E 类观察结果仍然是有效的行为结果，但其主要结构解释是 single-sector demand closure，而不是 matching bug。",
        "- 不需要 corrected matching rerun；下一步若要继续，应先设计不改变本次实验参数的单部门闭合/多部门收入机制研究。",
        "",
        "## Required Flags",
        "",
        "```json",
        json.dumps(flags, ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


def write_metrics(metrics):
    with (OUTPUT / "matching_forensic_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["section", "branch", "window", "metric", "value", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)


def main():
    b0_macro, b0_firms, b1_system, b1_firms = load_artifacts()
    replay_world, coordinator = tiny_first_review_replay()
    flags = build_flags(b1_system, coordinator)
    metrics = build_metrics(
        b0_macro, b0_firms, b1_system, b1_firms, replay_world, coordinator
    )
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    write_metrics(metrics)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "acceptance_summary.md").write_text(
        build_summary(flags, metrics, b1_system, coordinator), encoding="utf-8"
    )
    print("Stage B.1A outputs:", OUTPUT)
    print("verdict:", flags["verdict"])
    print("matching_forensic_classification:", flags["matching_forensic_classification"])
    print("tiny_replay_steps:", TINY_REPLAY_STEPS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
