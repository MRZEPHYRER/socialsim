"""Presentation-only metric registry for the unified Analysis layer.

The registry deliberately contains no simulation logic.  It describes how
authoritative fields are named, displayed, aggregated, and qualified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path


@dataclass(frozen=True)
class MetricDefinition:
    field: str
    zh_label: str
    en_label: str
    category: str
    unit: str
    aggregation: str
    description: str
    availability: str
    deprecated_aliases: str = ""
    status: str = "AUTHORITATIVE_CURRENT"


METRICS = (
    MetricDefinition("population", "总人口", "Population", "demography", "persons", "stock", "Alive population at the recorded step.", "historical"),
    MetricDefinition("working_age_population", "工作年龄人口", "Working-age population", "demography", "persons", "stock", "Authoritative working-age population.", "historical"),
    MetricDefinition("births", "出生事件", "Birth events", "demography", "events/week", "flow", "Not persisted in the Step 15 macro panel.", "runtime_not_persisted", status="AUTHORITATIVE_RUNTIME_BUT_NOT_PERSISTED"),
    MetricDefinition("deaths", "死亡事件", "Death events", "demography", "events/week", "flow", "Not persisted in the Step 15 macro panel.", "runtime_not_persisted", status="AUTHORITATIVE_RUNTIME_BUT_NOT_PERSISTED"),
    MetricDefinition("marriages", "婚姻事件", "Marriage events", "demography", "events/week", "flow", "Not persisted in the Step 15 macro panel.", "runtime_not_persisted", status="AUTHORITATIVE_RUNTIME_BUT_NOT_PERSISTED"),
    MetricDefinition("children", "儿童人口", "Children", "demography", "persons", "stock", "Age-group history is not persisted in the Step 15 macro panel.", "runtime_not_persisted", status="AUTHORITATIVE_RUNTIME_BUT_NOT_PERSISTED"),
    MetricDefinition("elderly_population", "老年人口", "Elderly population", "demography", "persons", "stock", "Age-group history is not persisted in the Step 15 macro panel.", "runtime_not_persisted", status="AUTHORITATIVE_RUNTIME_BUT_NOT_PERSISTED"),
    MetricDefinition("household_count", "家庭数量", "Household count", "households", "households", "stock", "Not persisted in the Step 15 macro panel.", "runtime_not_persisted", status="AUTHORITATIVE_RUNTIME_BUT_NOT_PERSISTED"),
    MetricDefinition("average_household_size", "平均家庭规模", "Average household size", "households", "persons/household", "derived", "Requires household count and population history.", "unavailable", status="AMBIGUOUS_REQUIRES_REVIEW"),
    MetricDefinition("household_income", "家庭收入", "Household income", "households", "money/week", "aggregate", "Recorded household income flow.", "historical"),
    MetricDefinition("household_consumption", "家庭消费", "Household consumption", "households", "money/week", "aggregate", "Recorded household consumption flow.", "historical"),
    MetricDefinition("household_saving", "家庭储蓄", "Household saving", "households", "money/week", "aggregate", "Recorded household saving flow.", "historical"),
    MetricDefinition("total_employment", "总就业", "Total employment", "labor", "workers", "aggregate", "Assigned workers across active firms.", "historical"),
    MetricDefinition("food_employment", "食品行业就业", "Food employment", "labor", "workers", "sector", "Food-sector assigned workers.", "historical"),
    MetricDefinition("capital_good_employment", "资本品行业就业", "Capital-good employment", "labor", "workers", "sector", "Capital-good-sector assigned workers.", "historical"),
    MetricDefinition("unassigned_labor", "未分配劳动力", "Unassigned eligible labor", "labor", "workers", "aggregate", "Eligible workers without a current firm assignment.", "historical"),
    MetricDefinition("revenue", "企业收入", "Firm revenue", "firms", "money/week", "firm", "Supplier-side recognized revenue.", "historical"),
    MetricDefinition("operating_profit", "经营利润", "Operating profit", "firms", "money/week", "firm", "Profit before dividend routing.", "historical"),
    MetricDefinition("cash", "现金", "Cash", "accounting", "money", "firm", "Firm cash balance.", "historical"),
    MetricDefinition("loan_principal", "贷款本金", "Loan principal", "accounting", "money", "firm", "Contractual loan principal; not customer advances.", "historical", "debt"),
    MetricDefinition("customer_advance_liability", "客户预付款负债", "Customer advance liability", "accounting", "money", "firm", "Unearned advance liability for investment orders.", "historical", "advance_liability"),
    MetricDefinition("prepaid_investment_asset", "预付投资资产", "Prepaid investment asset", "accounting", "money", "firm", "Buyer-side prepaid capital investment balance.", "historical"),
    MetricDefinition("active_capital_service", "在役资本服务", "Active capital service", "capital", "capacity units", "aggregate", "Physical productive service, not book value.", "historical"),
    MetricDefinition("total_fixed_investment", "固定投资", "Fixed investment", "capital", "money/week", "aggregate", "Executed firm fixed investment final demand.", "historical"),
    MetricDefinition("accounting_gap", "会计核对差额", "Accounting reconciliation gap", "reconciliation", "money", "aggregate", "Diagnostic identity gap.", "historical"),
    MetricDefinition("money_gap", "货币核对差额", "Money reconciliation gap", "reconciliation", "money", "aggregate", "Diagnostic money conservation gap.", "historical"),
    MetricDefinition("goods_gap", "商品守恒差额", "Goods conservation gap", "reconciliation", "goods", "aggregate", "Diagnostic goods conservation gap.", "historical"),
    MetricDefinition("production", "实际产量", "Realized production", "firms", "goods/week", "firm", "Actual output after capacity and funding constraints.", "historical", "output"),
    MetricDefinition("desired_labor", "目标劳动力", "Desired labor", "labor", "workers", "firm", "Planned labor requirement.", "historical"),
    MetricDefinition("actual_employment", "实际就业", "Actual employment", "labor", "workers", "firm", "Assigned workers on the firm roster.", "historical", "employment"),
)


METRIC_BY_FIELD = {item.field: item for item in METRICS}


def metric_label(field: str, language: str = "zh") -> str:
    item = METRIC_BY_FIELD.get(field)
    if item is None:
        return field
    return item.zh_label if language == "zh" else item.en_label


def write_metric_registry(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(METRICS[0])))
        writer.writeheader()
        writer.writerows(asdict(item) for item in METRICS)
    return path
