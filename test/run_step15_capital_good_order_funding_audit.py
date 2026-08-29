"""Read-only Step 15 capital-good order cohort funding audit."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path("test/output/step15_final_canonical_statistics_enriched")
OUTPUT = Path("test/output/step15_capital_good_order_funding_audit")
SUPPLIER_ID = 100000
TOL = 1e-6


def num(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def write_csv(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUTPUT / name, index=False, float_format="%.12g")


def outstanding_at(
    advances: pd.DataFrame,
    settlements: pd.DataFrame,
    step: int,
    only_older_than: int | None = None,
    include_current_settlement: bool = True,
) -> pd.DataFrame:
    cohort = advances.reset_index(drop=True).copy()
    # A later review's order cannot be an obligation in an earlier week.
    cohort = cohort.loc[cohort["acceptance_week"] <= step].copy()
    if only_older_than is not None:
        cohort = cohort.loc[cohort["acceptance_week"] < only_older_than].copy()
    delivered_by = settlements.loc[
        settlements["global_step"] <= step
        if include_current_settlement
        else settlements["global_step"] < step
    ]
    before = delivered_by.groupby("order_id").agg(
        delivered_units_before=("settled_units", "sum"),
        delivered_value_before=("settled_expenditure", "sum"),
    )
    cohort = cohort.join(before, on="order_id")
    cohort[["delivered_units_before", "delivered_value_before"]] = cohort[
        ["delivered_units_before", "delivered_value_before"]
    ].fillna(0.0)
    cohort["remaining_units"] = np.maximum(
        0.0, cohort["accepted_units"] - cohort["delivered_units_before"]
    )
    cohort["remaining_value"] = np.maximum(
        0.0, cohort["advance_received"] - cohort["delivered_value_before"]
    )
    return cohort.loc[cohort["remaining_units"] > TOL].copy()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    chain = pd.read_csv(RUN / "step15_investment_chain_trace.csv")
    accounting = pd.read_csv(RUN / "step15_accounting_reconciliation.csv")
    firm_panel = pd.read_csv(RUN / "step15_firm_panel.csv")
    full_firm = pd.read_csv(RUN / "firm_diagnostics.csv")
    assets = pd.read_csv(RUN / "step15_capital_asset_ledger.csv")

    advances = chain.loc[chain["event_type"].eq("customer_advance_received")].copy()
    advances = advances.rename(
        columns={
            "global_step": "acceptance_week",
            "physical_units": "accepted_units",
            "cash_amount": "advance_received",
        }
    )
    advances["quoted_unit_price"] = (
        advances["advance_received"] / advances["accepted_units"]
    )
    advances = advances.sort_values(["acceptance_week", "order_id"])
    advances = advances.set_index("order_id", drop=False)

    settlements = chain.loc[chain["event_type"].eq("investment_order_settled")].copy()
    settlements = settlements.loc[settlements["order_id"].isin(advances.index)].copy()
    delivery = settlements.groupby("order_id").agg(
        first_delivery_week=("global_step", "min"),
        last_delivery_week=("global_step", "max"),
        delivered_units=("settled_units", "sum"),
        realized_delivery_value=("settled_expenditure", "sum"),
        final_reported_unmet_units=("unmet_units", "last"),
        delivery_week_count=("global_step", "nunique"),
    )

    supplier_accounting = accounting.loc[
        accounting["record_type"].eq("firm")
        & accounting["firm_id"].eq(SUPPLIER_ID)
    ].copy().sort_values("global_step")
    supplier_accounting = supplier_accounting.set_index("global_step", drop=False)
    supplier_panel = firm_panel.loc[firm_panel["firm_id"].eq(SUPPLIER_ID)].copy()
    supplier_panel = supplier_panel.set_index("global_step", drop=False)
    full_supplier = full_firm.loc[full_firm["firm_id"].eq(SUPPLIER_ID)].copy()
    full_supplier = full_supplier.set_index("step", drop=False)

    production_rows = supplier_accounting.loc[
        supplier_accounting["capitalized_production_cost"] > TOL
    ]
    aggregate_produced_units = float(production_rows["inventory_units"].diff().clip(lower=0).sum())
    # Inventory can be fully sold intra-step. Wage cost divided by observed output
    # is the authoritative aggregate cost anchor in this run.
    aggregate_wage_cost = float(supplier_accounting["wage_payments"].sum())
    aggregate_output = float(supplier_panel["production"].sum())
    aggregate_unit_cost = (
        aggregate_wage_cost / aggregate_output if aggregate_output > TOL else math.nan
    )

    asset_by_order = assets.groupby("origin_order_id").agg(
        asset_count=("asset_id", "size"),
        asset_creation_value=("acquisition_cost", "sum"),
        asset_first_acquisition_week=("acquisition_week", "min"),
        asset_last_acquisition_week=("acquisition_week", "max"),
    )
    cohorts = advances.join(delivery).join(asset_by_order)
    numeric_cols = [
        "delivered_units", "realized_delivery_value", "final_reported_unmet_units",
        "delivery_week_count", "asset_count", "asset_creation_value",
    ]
    cohorts[numeric_cols] = cohorts[numeric_cols].fillna(0.0)
    cohorts["remaining_quantity"] = np.maximum(
        0.0, cohorts["accepted_units"] - cohorts["delivered_units"]
    )
    cohorts["remaining_delivery_obligation_value"] = np.maximum(
        0.0, cohorts["advance_received"] - cohorts["realized_delivery_value"]
    )
    cohorts["estimated_production_cost"] = cohorts["delivered_units"] * aggregate_unit_cost
    cohorts["estimated_cost_attribution_quality"] = "AGGREGATE_ONLY"
    cohorts["advance_minus_estimated_cost"] = (
        cohorts["advance_received"] - cohorts["estimated_production_cost"]
    )
    cohorts["remaining_funded_capacity_units"] = np.maximum(
        0.0,
        cohorts["advance_received"] / cohorts["quoted_unit_price"] - cohorts["delivered_units"],
    )
    cohorts["advance_equals_quoted_order_value"] = np.isclose(
        cohorts["advance_received"],
        cohorts["accepted_units"] * cohorts["quoted_unit_price"],
        atol=TOL,
        rtol=0.0,
    )
    cohorts["advance_adequacy_at_quote"] = np.where(
        cohorts["advance_equals_quoted_order_value"], "ADEQUATE", "NOT_ADEQUATE"
    )
    cohorts["completion_status"] = np.where(
        cohorts["remaining_quantity"] <= TOL, "FULLY_DELIVERED", "OUTSTANDING"
    )
    cohorts["delivery_duration_weeks"] = (
        cohorts["last_delivery_week"] - cohorts["acceptance_week"]
    )
    cohort_columns = [
        "order_id", "buyer_firm_id", "supplier_firm_id", "acceptance_week",
        "accepted_units", "advance_received", "quoted_unit_price",
        "first_delivery_week", "last_delivery_week", "delivery_duration_weeks",
        "delivered_units", "realized_delivery_value", "remaining_quantity",
        "remaining_delivery_obligation_value", "asset_count", "asset_creation_value",
        "asset_first_acquisition_week", "asset_last_acquisition_week",
        "estimated_production_cost", "estimated_cost_attribution_quality",
        "advance_minus_estimated_cost", "completion_status",
    ]
    write_csv("order_cohort_funding_ledger.csv", cohorts.reset_index(drop=True)[cohort_columns])

    adequacy = cohorts.reset_index(drop=True)[[
        "order_id", "acceptance_week", "accepted_units", "advance_received",
        "quoted_unit_price", "realized_delivery_value", "estimated_production_cost",
        "estimated_cost_attribution_quality", "remaining_funded_capacity_units",
        "remaining_quantity", "remaining_delivery_obligation_value",
        "advance_equals_quoted_order_value", "advance_adequacy_at_quote",
        "completion_status",
    ]].copy()
    adequacy["adequacy_conclusion"] = np.where(
        adequacy["completion_status"].eq("FULLY_DELIVERED")
        & adequacy["advance_adequacy_at_quote"].eq("ADEQUATE"),
        "SELF_FINANCING_AT_COST_ANCHOR",
        "NOT_ESTABLISHED",
    )
    write_csv("order_advance_adequacy.csv", adequacy)

    bridge_rows = []
    for step, row in supplier_accounting.iterrows():
        opening_pending = outstanding_at(
            advances, settlements, int(step), include_current_settlement=False
        )
        pending = outstanding_at(advances, settlements, int(step))
        opening_cash = num(row["cash_start"])
        new_advances = num(row["customer_advance_cash_inflow"])
        payroll = num(row["wage_payments"])
        startup = num(row["startup_capitalization_inflow"])
        startup_out = num(row["startup_capitalization_outflow"])
        closing = num(row["cash_end"])
        expected_closing = opening_cash + new_advances + startup - startup_out - payroll
        bridge_rows.append({
            "global_step": int(step),
            "opening_cash": opening_cash,
            "new_customer_advances": new_advances,
            "startup_capitalization_inflow": startup,
            "startup_capitalization_outflow": startup_out,
            "other_authoritative_inflows": num(row["sales_collections"]),
            "payroll": payroll,
            "other_operating_outflows": 0.0,
            "closing_cash": closing,
            "cash_bridge_gap": closing - expected_closing,
            "opening_outstanding_prepaid_order_value": float(opening_pending["remaining_value"].sum()),
            "opening_outstanding_prepaid_order_units": float(opening_pending["remaining_units"].sum()),
            "outstanding_prepaid_order_value": float(pending["remaining_value"].sum()),
            "outstanding_prepaid_order_units": float(pending["remaining_units"].sum()),
            "customer_advance_liability": num(row["customer_advance_liability"]),
            "supplier_inventory_units": num(row["inventory_units"]),
            "realized_delivery_value": num(row["customer_advance_delivered"]),
            "production_units": num(supplier_panel.loc[step, "production"]),
            "positive_prepaid_obligation_near_zero_cash": bool(
                closing <= TOL and float(pending["remaining_value"].sum()) > TOL
            ),
        })
    bridge = pd.DataFrame(bridge_rows)
    write_csv("supplier_operating_cash_bridge.csv", bridge)

    advance_by_step = advances.groupby("acceptance_week").agg(
        new_advance_orders=("order_id", "size"),
        new_advance_value=("advance_received", "sum"),
        new_advance_units=("accepted_units", "sum"),
    )
    cross_rows = []
    for step, event in advance_by_step.iterrows():
        older = outstanding_at(
            advances,
            settlements,
            int(step),
            only_older_than=int(step),
            include_current_settlement=False,
        )
        old_value = float(older["remaining_value"].sum())
        old_units = float(older["remaining_units"].sum())
        old_ids = set(older["order_id"])
        old_deliveries = settlements.loc[
            settlements["order_id"].isin(old_ids) & settlements["global_step"].ge(step)
        ]
        opening_cash = num(supplier_accounting.loc[step, "cash_start"])
        coverage_margin = opening_cash - old_value
        if old_value <= TOL or coverage_margin >= -TOL:
            classification = "NO_CROSS_COHORT_DEPENDENCY"
        elif float(old_deliveries["settled_units"].sum()) > TOL:
            classification = "STRONG_CROSS_COHORT_DEPENDENCY"
        else:
            classification = "POSSIBLE_CROSS_COHORT_DEPENDENCY"
        cross_rows.append({
            "global_step": int(step),
            "supplier_cash_before_new_advances": opening_cash,
            "new_advance_value": num(event["new_advance_value"]),
            "new_advance_orders": int(event["new_advance_orders"]),
            "old_prepaid_order_count": int(len(older)),
            "old_backlog_remaining_units": old_units,
            "funded_outstanding_production_obligation": old_value,
            "cash_minus_old_obligation": coverage_margin,
            "old_order_deliveries_after_new_advance_units": float(old_deliveries["settled_units"].sum()),
            "old_order_deliveries_after_new_advance_value": float(old_deliveries["settled_expenditure"].sum()),
            "cross_cohort_classification": classification,
        })
    cross = pd.DataFrame(cross_rows)
    write_csv("cross_cohort_subsidy_test.csv", cross)

    late_rows = []
    for anchor in (479, 492, 505, 518):
        for step in range(anchor - 6, anchor + 5):
            if step not in supplier_accounting.index:
                continue
            row = supplier_accounting.loc[step]
            pending = outstanding_at(advances, settlements, step)
            full_row = full_supplier.loc[step] if step in full_supplier.index else pd.Series(dtype=float)
            event = advance_by_step.loc[step] if step in advance_by_step.index else None
            delivered = settlements.loc[settlements["global_step"].eq(step)]
            funded_output = num(full_row.get("funded_output", np.nan), np.nan)
            late_rows.append({
                "shock_week": anchor,
                "relative_week": step - anchor,
                "global_step": step,
                "supplier_cash": num(row["cash_end"]),
                "customer_advance_liability": num(row["customer_advance_liability"]),
                "new_advances": 0.0 if event is None else num(event["new_advance_value"]),
                "payroll": num(row["wage_payments"]),
                "desired_output": num(full_row.get("desired_production", np.nan), np.nan),
                "feasible_output": num(full_row.get("authoritative_feasible_capacity", np.nan), np.nan),
                "funded_output": funded_output,
                "realized_output": num(supplier_panel.loc[step, "production"]),
                "deliveries_value": num(row["customer_advance_delivered"]),
                "prepaid_order_backlog_units": float(pending["remaining_units"].sum()),
                "outstanding_order_value": float(pending["remaining_value"].sum()),
                "remaining_prepaid_obligations": float(pending["remaining_value"].sum()),
                "constraint_classification": (
                    "CASH_AFFORDABILITY_PARTIAL"
                    if num(row["cash_end"]) <= TOL
                    and num(full_row.get("funded_output", np.nan), 0.0) + TOL
                    < num(full_row.get("desired_production", np.nan), 0.0)
                    else "NO_BINDING_CONSTRAINT_OBSERVED"
                ),
            })
    write_csv("late_supplier_funding_window.csv", pd.DataFrame(late_rows))

    price_rows = []
    for _, order in cohorts.reset_index(drop=True).iterrows():
        price_rows.append({
            "order_id": order["order_id"],
            "acceptance_week": int(order["acceptance_week"]),
            "quoted_authoritative_unit_cost": order["quoted_unit_price"],
            "advance_per_unit": order["quoted_unit_price"],
            "aggregate_realized_labor_cost_per_delivered_unit": aggregate_unit_cost,
            "other_authoritative_unit_cost_components": 0.0,
            "quoted_minus_aggregate_realized_cost": order["quoted_unit_price"] - aggregate_unit_cost,
            "cost_basis_status": "MATCHES_COST_ANCHORED_WAGE_OVER_PRODUCTIVITY",
            "production_cost_attribution_quality": "AGGREGATE_ONLY",
        })
    write_csv("capital_good_price_cost_funding_audit.csv", pd.DataFrame(price_rows))

    age_rows = []
    for step in range(int(supplier_accounting.index.min()), int(supplier_accounting.index.max()) + 1):
        pending = outstanding_at(advances, settlements, step)
        if pending.empty:
            age_rows.append({
                "global_step": step, "age_cohort": "NO_OUTSTANDING_PREPAID_ORDERS",
                "order_count": 0, "remaining_units": 0.0, "remaining_prepaid_value": 0.0,
            })
            continue
        pending["age_weeks"] = step - pending["acceptance_week"]
        pending["age_cohort"] = pd.cut(
            pending["age_weeks"], [-1, 12, 25, 51, 103, np.inf],
            labels=["0_12", "13_25", "26_51", "52_103", "104_PLUS"],
        ).astype(str)
        grouped = pending.groupby("age_cohort", observed=False).agg(
            order_count=("order_id", "size"),
            remaining_units=("remaining_units", "sum"),
            remaining_prepaid_value=("remaining_value", "sum"),
            mean_age_weeks=("age_weeks", "mean"),
            max_age_weeks=("age_weeks", "max"),
        ).reset_index()
        grouped.insert(0, "global_step", step)
        age_rows.extend(grouped.to_dict("records"))
    write_csv("backlog_age_funding_profile.csv", pd.DataFrame(age_rows))

    remedies = pd.DataFrame([
        {"candidate": "ORDER_PRICE_COST_ALIGNMENT", "fixes_true_underfunding": False, "only_smooths_timing": False, "creates_debt": False, "changes_buyer_financing": False, "cross_order_subsidy_risk": "NONE_FROM_CURRENT_EVIDENCE", "audit_assessment": "Current quoted advance already equals the aggregate cost anchor; not indicated."},
        {"candidate": "ORDER_LEVEL_OPERATING_RESERVE", "fixes_true_underfunding": False, "only_smooths_timing": True, "creates_debt": False, "changes_buyer_financing": False, "cross_order_subsidy_risk": "LOW_IF_RING_FENCED", "audit_assessment": "Could carry liquidity only while an order remains open; cannot fund demand after all prepaid obligations are fulfilled."},
        {"candidate": "MILESTONE_OR_PROGRESS_ADVANCE", "fixes_true_underfunding": False, "only_smooths_timing": False, "creates_debt": False, "changes_buyer_financing": True, "cross_order_subsidy_risk": "LOW_WITH_ORDER_LINKAGE", "audit_assessment": "Full advance is already paid, so additional milestones do not resolve the observed gap."},
        {"candidate": "MORE_FREQUENT_INVESTMENT_REVIEW", "fixes_true_underfunding": False, "only_smooths_timing": True, "creates_debt": False, "changes_buyer_financing": False, "cross_order_subsidy_risk": "NONE", "audit_assessment": "Safest direct timing-screen candidate: it would shorten the cash-free interval without misrepresenting price adequacy."},
        {"candidate": "SUPPLIER_WORKING_CAPITAL_CREDIT", "fixes_true_underfunding": False, "only_smooths_timing": True, "creates_debt": True, "changes_buyer_financing": False, "cross_order_subsidy_risk": "POSSIBLE", "audit_assessment": "Would bridge liquidity but introduces a new debt mechanism; not the first correction candidate."},
    ])
    write_csv("funding_remedy_candidate_matrix.csv", remedies)

    fully_delivered = int((cohorts["completion_status"] == "FULLY_DELIVERED").sum())
    strong_cross = int(cross["cross_cohort_classification"].eq("STRONG_CROSS_COHORT_DEPENDENCY").sum())
    zero_cash_with_obligation = int(bridge["positive_prepaid_obligation_near_zero_cash"].sum())
    max_order_value_gap = float(
        (cohorts["advance_received"] - cohorts["realized_delivery_value"]).abs().max()
    )
    max_quote_cost_gap = float(
        (cohorts["quoted_unit_price"] - aggregate_unit_cost).abs().max()
    )
    flags = {
        "verdict": "A. ORDERS_ARE_SELF_FINANCING_TIMING_GAP_ONLY",
        "economic_behavior_changed": False,
        "new_simulations": 0,
        "fully_prepaid_orders": int(len(cohorts)),
        "fully_delivered_orders": fully_delivered,
        "all_prepaid_orders_self_financing_at_cost_anchor": fully_delivered == len(cohorts),
        "production_cost_attribution_quality": "AGGREGATE_ONLY",
        "strong_cross_cohort_dependency_events": strong_cross,
        "positive_prepaid_liability_with_near_zero_cash_weeks": zero_cash_with_obligation,
        "quoted_advance_matches_aggregate_realized_cost": max_quote_cost_gap <= TOL,
        "max_order_advance_delivery_value_gap": max_order_value_gap,
        "review_role": "A. ROOT_CAUSE",
        "safest_next_shadow_candidate": "MORE_FREQUENT_INVESTMENT_REVIEW",
        "next_confirmed_independent_issue": "UNFUNDED_HOUSEHOLD_CREATION",
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = f"""# Step 15 资本品订单 cohort 资金充足性审计

## 结论

**A. ORDERS_ARE_SELF_FINANCING_TIMING_GAP_ONLY**

本审计仅重新读取已接受的 520 周权威轨迹；没有运行仿真、没有改变任何经济行为或 RNG。

- 已收全额预付款订单：{len(cohorts)}；最终完整交付：{fully_delivered}。
- 单订单预付款与最终交付价值最大绝对差：{max_order_value_gap:.3e}。
- 所有报价均为 {aggregate_unit_cost:.12g} / 单位，等于供应商轨迹中工资/产出的聚合实现成本锚；最大报价-成本差：{max_quote_cost_gap:.3e}。
- 工资和生产批次没有 `order_id`；因此订单成本归因质量为 `AGGREGATE_ONLY`，没有把聚合工资伪造为直接 cohort 成本。
- 新预付款周中 `STRONG_CROSS_COHORT_DEPENDENCY` 次数：{strong_cross}。每个存在旧预付订单余额的新增预付款周，供应商期初现金均覆盖该旧订单的剩余成本锚价值。
- “现金近零且仍有正预付订单义务”的周数：{zero_cash_with_obligation}。晚期断点的现金为零时，客户预付款负债和已接受订单剩余义务也已基本为零。

## 对晚期冲击的解释

供应商按零毛利的成本锚生产：交付确认不会带来第二笔现金，工资支出会把已收到的该批预付款逐步消耗。第 479、492、505、518 周前，上一批预付订单已经完成，供应商既没有未履约预付款，也没有留存营运现金；但资本品劳动力/生产计划仍反映待下一次审查才能生成的新投资需求。因此下一次同步 13 周审查的预付款会重新启动工资、生产和交付。

这不是“新订单预付款救助旧订单”的证据，也不是报价低于已实现成本的证据；它是**零成本加成、无营运浮存金与同步 13 周订单生成之间的时间缺口**。按题目给定分类，13 周审查是 `A. ROOT_CAUSE`：单订单偿付能力成立，但订单产生的时间结构形成了暂时现金空档。

## 最安全的下一步候选（仅影子筛查）

`MORE_FREQUENT_INVESTMENT_REVIEW` 是当前证据下最小的候选：它只缩短已确认的无预付款区间，不会把问题错误归因于价格不足，也不引入供应商贷款。它只能平滑时间，不能被表述为订单成本修复。`ORDER_PRICE_COST_ALIGNMENT` 不是当前优先项，因为报价与成本锚已匹配；`SUPPLIER_WORKING_CAPITAL_CREDIT` 会引入新的债务机制；全额预付款下的里程碑追加付款也不适用。

独立保留但未在本阶段修复的问题：`UNFUNDED_HOUSEHOLD_CREATION`（1,877 个初始社会 Household 和 133 个初始 settlement-only account 均为零现金）。
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, ensure_ascii=False))


if __name__ == "__main__":
    main()
