"""Metric semantics registry backed by the frozen Step 15 data contract."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


AVAILABILITY = {
    "AUTHORITATIVE_AND_PERSISTED",
    "DERIVABLE_FROM_PERSISTED_AUTHORITATIVE_DATA",
    "AUTHORITATIVE_RUNTIME_ONLY",
    "NOT_AVAILABLE",
}


@dataclass(frozen=True)
class MetricDefinition:
    internal_name: str
    display_name_zh: str
    display_name_en: str
    definition: str
    unit: str
    kind: str
    aggregation_level: str
    source_dataset: str
    source_field: str
    availability: str
    allowed_aggregations: str
    visualization_type: str
    definition_en: str


@dataclass(frozen=True)
class StatisticalMetricDefinition:
    internal_name: str
    display_name_zh: str
    display_name_en: str
    formula: str
    population: str
    unit_of_analysis: str
    weighting: str
    exclusion_rules: str
    source_fields: str
    time_interpretation: str
    availability: str


def _m(name, zh, en, definition, unit, kind, level, dataset, field,
       availability="AUTHORITATIVE_AND_PERSISTED", aggregations="last,mean,sum,min,max", visualization="line"):
    definition_en = f"{en}, using the authoritative semantics of source field '{field}'."
    return MetricDefinition(name, zh, en, definition, unit, kind, level, dataset, field, availability, aggregations, visualization, definition_en)


METRICS = [
    _m("population", "人口", "Population", "存活人口总数", "人", "stock", "world", "macro", "population"),
    _m("births", "出生", "Births", "当周出生事件数", "人/周", "flow", "world", "demography", "births", visualization="bar"),
    _m("deaths", "死亡", "Deaths", "当周死亡事件数", "人/周", "flow", "world", "demography", "deaths", visualization="bar"),
    _m("marriages", "结婚", "Marriages", "当周权威婚姻匹配事件数", "对/周", "event", "world", "demography", "marriages", visualization="event"),
    _m("age_0_19", "0–19 岁", "Age 0–19", "持久化年龄组人数", "人", "stock", "world", "demography", "age_0_19_count", visualization="area"),
    _m("age_20_39", "20–39 岁", "Age 20–39", "持久化年龄组人数", "人", "stock", "world", "demography", "age_20_39_count", visualization="area"),
    _m("age_40_64", "40–64 岁", "Age 40–64", "持久化年龄组人数", "人", "stock", "world", "demography", "age_40_64_count", visualization="area"),
    _m("age_65_plus", "65 岁及以上", "Age 65+", "持久化年龄组人数", "人", "stock", "world", "demography", "age_65_plus_count", visualization="area"),
    _m("social_households", "社会家庭", "Social Households", "排除 settlement-only 账户的社会家庭数量", "户", "stock", "world", "macro", "social_household_count"),
    _m("settlement_accounts", "临时个人结算账户", "Temporary Individual Settlement Accounts", "为暂未归属社会家庭的成年人提供工资、现金与消费结算；不代表婚姻、亲子或共同居住关系。", "账户", "stock", "world", "macro", "settlement_only_account_count"),
    _m("economic_accounts", "经济账户总数", "Total Economic Accounts", "用于现金结算的 Household 对象总数", "账户", "stock", "world", "macro", "total_economic_accounts"),
    _m("eligible_labor", "可匹配劳动力", "Labor-match Eligible", "具有劳动生产率和有效结算账户的人数", "人", "stock", "world", "macro", "eligible_labor"),
    _m("employment", "就业", "Employment", "全部经营企业的已分配劳动者", "人", "stock", "world", "macro", "total_employment"),
    _m("unassigned_labor", "未分配劳动力", "Unassigned Eligible Labor", "可匹配但未分配至企业的劳动者", "人", "stock", "world", "macro", "unassigned_labor"),
    _m("food_employment", "食品业就业", "Food Employment", "食品企业就业总数", "人", "stock", "sector", "macro", "food_employment"),
    _m("capital_good_employment", "资本品业就业", "Capital-goods Employment", "资本品企业就业总数", "人", "stock", "sector", "macro", "capital_good_employment"),
    _m("household_income", "家庭收入", "Household Income", "家庭当周总收入", "货币/周", "flow", "world", "macro", "household_income"),
    _m("household_consumption", "家庭消费", "Household Consumption", "家庭当周最终消费", "货币/周", "flow", "world", "macro", "household_consumption"),
    _m("household_saving", "家庭储蓄", "Household Saving", "家庭当周收入减消费后的净储蓄", "货币/周", "flow", "world", "macro", "household_saving"),
    _m("household_cash", "家庭现金", "Household Cash", "全部经济结算账户现金", "货币", "stock", "world", "macro", "household_cash"),
    _m("aggregate_firm_cash", "企业现金合计", "Aggregate Firm Cash", "Firm panel 逐周现金求和", "货币", "stock", "world", "firms", "cash", "DERIVABLE_FROM_PERSISTED_AUTHORITATIVE_DATA"),
    _m("legacy_owner_cash", "LegacyOwner 现金", "LegacyOwner Cash", "LegacyOwnershipPool 收款账户现金", "货币", "stock", "world", "macro", "legacy_owner_cash"),
    _m("estate_cash", "遗产账户现金", "Estate Cash", "开放 Estate 账户现金合计", "货币", "stock", "world", "macro", "estate_cash"),
    _m("money_stock", "货币存量", "Money Stock", "权威系统货币存量", "货币", "stock", "world", "macro", "money_stock"),
    _m("food_production", "食品产量", "Food Production", "食品行业当周实际产量", "食品单位/周", "flow", "sector", "macro", "food_production"),
    _m("fixed_investment", "固定投资", "Fixed Investment", "企业当周执行固定投资", "货币/周", "flow", "world", "macro", "total_fixed_investment"),
    _m("expansion_investment", "扩张投资", "Expansion Investment", "提高产能的投资", "货币/周", "flow", "world", "macro", "expansion_investment"),
    _m("replacement_investment", "更新投资", "Replacement Investment", "替代退休资本服务的投资", "货币/周", "flow", "world", "macro", "replacement_investment"),
    _m("capital_good_production", "资本品产量", "Capital-good Production", "资本品企业当周实际产量", "资本品单位/周", "flow", "sector", "macro", "capital_good_production"),
    _m("active_capital_assets", "活跃资本资产", "Active Capital Assets", "当周活跃 CapitalAsset 数量", "项", "stock", "world", "macro", "active_capital_assets"),
    _m("active_capital_service", "活跃资本服务", "Active Capital Service", "活跃资产提供的生产能力", "产能单位", "stock", "world", "macro", "active_capital_service"),
    _m("depreciation", "折旧", "Depreciation", "当周非现金会计折旧", "货币/周", "flow", "world", "macro", "depreciation"),
    _m("customer_advances_received", "收到客户预付款", "Customer Advances Received", "资本品供应方当周收到的客户预付款", "货币/周", "flow", "world", "macro", "customer_advances_received"),
    _m("customer_advance_liability", "客户预付款负债", "Customer Advance Liability", "企业对买方尚未履约的预付款负债", "货币", "stock", "firm", "firms", "customer_advance_liability"),
    _m("prepaid_investment_asset", "预付资本投资资产", "Prepaid Investment Asset", "买方已付款但尚未资本化的投资资产", "货币", "stock", "firm", "firms", "prepaid_investment_asset"),
    _m("loan_principal", "贷款本金", "Loan Principal", "企业未偿贷款本金，与客户预付款负债不同", "货币", "stock", "firm", "firms", "principal"),
    _m("interest_arrears", "利息欠款", "Interest Arrears", "企业累计未付利息合同债权", "货币", "stock", "firm", "firms", "arrears"),
    _m("capital_book_value", "资本资产账面价值", "Capital Asset Book Value", "资本资产折旧后的账面价值", "货币", "stock", "firm", "firms", "capital_book_value"),
    _m("firm_cash", "企业现金", "Firm Cash", "所选企业期末现金", "货币", "stock", "firm", "firms", "cash"),
    _m("firm_inventory", "企业库存", "Firm Inventory", "所选企业实物库存", "单位", "stock", "firm", "firms", "inventory"),
    _m("firm_revenue", "企业收入", "Firm Revenue", "所选企业当周销售收入", "货币/周", "flow", "firm", "firms", "revenue"),
    _m("firm_profit", "企业营业利润", "Firm Operating Profit", "AccountingLayer 营业利润", "货币/周", "flow", "firm", "firms", "operating_profit"),
    _m("firm_cfo", "经营现金流", "Operating Cash Flow", "所选企业经营现金流", "货币/周", "flow", "firm", "firms", "cfo"),
    _m("firm_equity", "企业账面权益", "Firm Book Equity", "资产减负债的 AccountingLayer 权益", "货币", "stock", "firm", "accounting", "equity"),
    _m("production_cost", "生产成本", "Production Cost", "权威制造成本", "货币/周", "flow", "firm", "accounting", "manufacturing_cost"),
    _m("capital_backlog", "资本品积压", "Capital-good Backlog", "本 canonical run 未持久化权威聚合 backlog stock 历史", "资本品单位", "stock", "world", "none", "", "NOT_AVAILABLE", visualization="status"),
    _m("generic_transaction_id", "通用交易 ID", "Generic Transaction ID", "未持久化的通用交易标识", "—", "event", "transaction", "none", "", "NOT_AVAILABLE", visualization="status"),
    _m("generic_counterparty", "通用交易对手方", "Generic Counterparty", "聚合资金流未持久化逐交易对手方", "—", "event", "transaction", "none", "", "NOT_AVAILABLE", visualization="status"),
    _m("money_location_gap", "货币位置差额", "Money-location Gap", "包含 Household、Firm、Public、LegacyOwner 与 Estate 的货币位置桥差额", "货币", "reconciliation", "world", "macro", "money_location_gap"),
    _m("goods_gap", "商品守恒差额", "Goods Gap", "食品实物流守恒差额", "食品单位", "reconciliation", "world", "macro", "goods_gap"),
    _m("advance_prepaid_gap", "预付款核对差额", "Advance/Prepaid Gap", "供应方负债与买方预付资产差额", "货币", "reconciliation", "world", "macro", "advance_prepaid_gap"),
    _m("assignment_violations", "劳动分配违规", "Labor Assignment Violations", "企业 roster 与 Person.firm_id 不一致数", "项", "reconciliation", "world", "macro", "assignment_violations"),
    _m("feasible_capacity_gap", "超可行产能", "Output Above Feasible Capacity", "实际产出超过权威可行产能的数量", "产出单位", "reconciliation", "world", "macro", "output_above_feasible_capacity"),
    _m("full_money_location_gap", "Full Money Location Gap", "Full Money-location Gap", "Canonical stock identity gap; authoritative money acceptance metric.", "money", "reconciliation", "world", "macro", "full_money_location_gap"),
    _m("money_delta_gap", "Money Flow Timing Gap", "Money Flow/Timing Gap", "Located-money change minus net money issued; flow/timing diagnostic, not a stock-location identity.", "money/week", "reconciliation", "world", "macro", "money_delta_gap"),
    _m("private_support_near_zero_share", "家庭近零现金占比", "Household Near-zero-cash Share", "Accepted mature-genealogy private-support experiment near-zero household share", "share", "distribution", "world", "support_elderly", "near_zero_share", aggregations="last,mean,min,max"),
    _m("private_support_elderly_near_zero_share", "老年家庭近零现金占比", "Elderly Near-zero-cash Share", "Near-zero-cash share among elderly parent households", "share", "distribution", "world", "support_elderly", "elderly_near_zero_share", aggregations="last,mean,min,max"),
    _m("private_support_event_count", "家庭支持事件数", "Private-support Events", "Observed private family-support transfer records", "events", "event", "world", "support_events", "transfer_events", visualization="status"),
    _m("private_support_total_value", "家庭支持总额", "Private-support Total Value", "Total realized private family-support transfer value", "money", "flow", "world", "support_events", "total_value", aggregations="last,sum"),
    _m("private_support_unique_payers", "支持付款家庭数", "Unique Support Payers", "Unique payer Households in the accepted support experiment", "households", "stock", "world", "support_events", "unique_payer_households", visualization="status"),
    _m("private_support_unique_recipients", "支持收款家庭数", "Unique Support Recipients", "Unique recipient Households in the accepted support experiment", "households", "stock", "world", "support_events", "unique_recipient_households", visualization="status"),
    _m("private_support_recipient_coverage", "支持收款覆盖率", "Support Recipient Coverage", "Realized recipient share of observation Households", "share", "distribution", "world", "support_coverage", "realized_support_share_of_observation_households", visualization="status"),
    _m("private_support_eligible_parents", "可支持父母家庭数", "Eligible Parent Households", "Mean eligible elderly-parent Households in the treatment branch", "households", "stock", "world", "support_coverage", "support_eligible_parent_households_mean", visualization="status"),
    _m("private_support_eligible_children", "可支持子女家庭数", "Eligible Child Households", "Mean eligible adult-child Households in the treatment branch", "households", "stock", "world", "support_coverage", "support_eligible_child_households_mean", visualization="status"),
    _m("private_support_child_below_share", "子女付款后近零占比", "Child Payer Below-threshold Share", "Observed child-payer transfer records ending below the one-week minimum", "share", "distribution", "world", "support_child_burden", "payer_near_zero_after", availability="DERIVABLE_FROM_PERSISTED_AUTHORITATIVE_DATA", visualization="status"),
    _m("private_support_food_sales", "食品销售", "Food Sales with Support", "Food sales in the support experiment branch", "money/week", "flow", "world", "support_elderly", "food_sales", aggregations="last,mean,sum"),
    _m("private_support_aggregate_firm_cash", "企业现金合计", "Aggregate Firm Cash with Support", "Aggregate Firm cash in the support experiment branch", "money", "stock", "world", "support_elderly", "aggregate_firm_cash", aggregations="last,mean"),]


def _s(name, zh, en, formula, population, unit, sources, time,
       availability="DERIVABLE_FROM_PERSISTED_AUTHORITATIVE_DATA",
       weighting="unweighted", exclusions="missing/non-finite source values"):
    return StatisticalMetricDefinition(
        name, zh, en, formula, population, unit, weighting, exclusions,
        sources, time, availability,
    )


STATISTICAL_METRICS = [
    _s("child_share", "儿童占比", "Child Share", "age_0_19_count / population", "all alive Persons", "Person", "age_0_19_count;population", "selected persisted week"),
    _s("working_age_share", "工作年龄人口占比", "Working-age Share", "(age_20_39_count + age_40_64_count) / population", "all alive Persons", "Person", "age_20_39_count;age_40_64_count;population", "selected persisted week"),
    _s("elderly_share", "老年人口占比", "Elderly Share", "age_65_plus_count / population", "all alive Persons", "Person", "age_65_plus_count;population", "selected persisted week"),
    _s("dependency_ratio", "总抚养比", "Dependency Ratio", "(age_0_19_count + age_65_plus_count) / (age_20_39_count + age_40_64_count)", "persisted broad age groups", "Person", "four persisted age-group counts", "selected persisted week"),
    _s("youth_dependency_ratio", "少儿抚养比", "Youth Dependency Ratio", "age_0_19_count / working_age_count", "persisted broad age groups", "Person", "age_0_19_count;working_age_count", "selected persisted week"),
    _s("old_age_dependency_ratio", "老年抚养比", "Old-age Dependency Ratio", "age_65_plus_count / working_age_count", "persisted broad age groups", "Person", "age_65_plus_count;working_age_count", "selected persisted week"),
    _s("population_growth", "人口增长率", "Population Growth", "population[t] / population[start] - 1", "all alive Persons", "world", "population", "selected filter window"),
    _s("working_age_count", "20–64 岁宽年龄组人数", "Persisted Age 20–64 Count", "age_20_39_count + age_40_64_count", "all alive Persons", "world-week", "age_20_39_count;age_40_64_count", "persisted broad age bands"),
    _s("settlement_valid_person_count", "结算有效人数", "Settlement-valid Person Count", "runtime Person has a valid economic settlement account", "working-age Persons", "world-week", "Person household/settlement links", "persisted week", "NOT_AVAILABLE", exclusions="runtime-only relationship was not aggregated into canonical history"),
    _s("household_income_distribution", "社会家庭收入分布", "Social-household Income Distribution", "micro cross-section required", "social Households only", "social Household", "household_id;income", "authoritative snapshot/week", "NOT_AVAILABLE", exclusions="settlement-only accounts; missing income"),
    _s("household_income_gini", "社会家庭收入基尼系数", "Social-household Income Gini", "Gini(valid social-Household income observations)", "social Households only", "social Household", "household_id;income", "authoritative snapshot/week", "NOT_AVAILABLE", exclusions="settlement-only accounts; no aggregate reconstruction"),
    _s("household_income_lorenz", "社会家庭收入洛伦兹曲线", "Social-household Income Lorenz Curve", "cumulative income share by ranked social Household", "social Households only", "social Household", "household_id;income", "authoritative snapshot/week", "NOT_AVAILABLE", exclusions="settlement-only accounts; no aggregate reconstruction"),
    _s("household_cash_distribution", "社会家庭现金资产分布", "Social-household Cash-asset Distribution", "micro cross-section required", "social Households only", "social Household", "household_id;cash", "authoritative snapshot/week", "NOT_AVAILABLE", exclusions="settlement-only accounts; cash is not total wealth"),
    _s("household_cash_gini", "社会家庭现金资产基尼系数", "Social-household Cash-asset Gini", "Gini(valid social-Household cash observations)", "social Households only", "social Household", "household_id;cash", "authoritative snapshot/week", "NOT_AVAILABLE", exclusions="settlement-only accounts; no aggregate reconstruction"),
    _s("household_financial_assets_distribution", "家庭金融资产分布", "Household Financial-assets Distribution", "cash + authoritatively valued equity + other financial assets", "social Households only", "social Household", "household cash;equity value;other assets", "authoritative snapshot/week", "NOT_AVAILABLE", exclusions="no silent cash-as-wealth substitution"),
    _s("household_saving_distribution", "社会家庭储蓄分布", "Social-household Saving Distribution", "income - consumption by Household", "social Households only", "social Household", "household_id;income;consumption", "authoritative snapshot/week", "NOT_AVAILABLE", exclusions="settlement-only accounts; no aggregate allocation"),
    _s("sales_market_share", "同业已实现销量份额", "Within-sector Realized-sales Share", "Firm realized sales / same-sector realized sales", "active Firms in selected sector", "Firm-week", "firm_id;sector;sales", "persisted week", exclusions="weeks with non-positive sector sales denominator"),
    _s("revenue_share", "同业收入份额", "Within-sector Revenue Share", "Firm revenue / same-sector revenue", "active Firms in selected sector", "Firm-week", "firm_id;sector;revenue", "persisted week", exclusions="weeks with non-positive sector revenue denominator"),
    _s("production_share", "同业产量份额", "Within-sector Production Share", "Firm production / same-sector production", "active Firms in selected sector", "Firm-week", "firm_id;sector;production", "persisted week", exclusions="weeks with non-positive sector production denominator"),
    _s("employment_share", "同业就业份额", "Within-sector Employment Share", "Firm employment / same-sector employment", "active Firms in selected sector", "Firm-week", "firm_id;sector;employment", "persisted week", exclusions="weeks with non-positive sector employment denominator"),
    _s("inventory_share", "同业库存份额", "Within-sector Inventory Share", "Firm inventory / same-sector inventory", "active Firms in selected sector", "Firm-week", "firm_id;sector;inventory", "persisted week", exclusions="weeks with non-positive sector inventory denominator"),
    _s("sales_hhi", "销量 HHI", "Sales HHI", "sum(within-sector realized-sales share^2)", "active Firms in selected sector", "sector-week", "firm_id;sector;sales", "persisted week", exclusions="single supplier labelled separately; non-positive denominator"),
    _s("largest_firm_share", "最大企业份额", "Largest-firm Share", "max(within-sector realized-sales share)", "active Firms in selected sector", "sector-week", "firm_id;sector;sales", "persisted week"),
    _s("top2_share", "前两家企业份额", "Top-2 Share", "sum(two largest within-sector realized-sales shares)", "active Firms in selected sector", "sector-week", "firm_id;sector;sales", "persisted week"),
    _s("top3_share", "前三家企业份额", "Top-3 Share", "sum(three largest within-sector realized-sales shares)", "active Firms in selected sector", "sector-week", "firm_id;sector;sales", "persisted week"),
    _s("cross_firm_cv", "企业横向变异系数", "Cross-firm Coefficient of Variation", "population_sd(metric) / abs(mean(metric))", "active Firms in selected sector", "sector-week", "firm_id;sector;selected metric", "persisted week"),
    _s("profit_margin", "营业利润率", "Operating Profit Margin", "operating_profit / revenue", "selected Firm", "Firm-week", "operating_profit;revenue", "persisted week", exclusions="non-positive revenue"),
    _s("labor_eligibility_rate", "劳动资格率", "Labor Eligibility Rate", "eligible_labor / exact count under the runtime age-productivity eligibility rule", "runtime labor-age Persons", "world-week", "eligible_labor;exact labor-age denominator", "persisted week", "NOT_AVAILABLE", exclusions="the 0-19 broad band cannot isolate eligible ages 18-19; no exact historical denominator"),
    _s("eligible_population_share", "可匹配劳动力占总人口", "Labor-eligible Share of Population", "eligible_labor / population", "all alive Persons", "world-week", "eligible_labor;population", "persisted week"),
    _s("employment_rate", "就业率", "Employment Rate", "total_employment / eligible_labor", "labor-match eligible Persons", "world-week", "total_employment;eligible_labor", "persisted week"),
    _s("unassigned_eligible_rate", "未分配合格劳动力率", "Unassigned Eligible Rate", "unassigned_labor / eligible_labor", "labor-match eligible Persons", "world-week", "unassigned_labor;eligible_labor", "persisted week"),
    _s("sector_employment_share", "行业就业份额", "Sector Employment Share", "sector employment / total employment", "employed Persons", "sector-week", "food_employment;capital_good_employment;total_employment", "persisted week"),
    _s("food_employment_share", "食品业就业份额", "Food Employment Share", "food_employment / total_employment", "employed Persons", "world-week", "food_employment;total_employment", "persisted week"),
    _s("capital_good_employment_share", "资本品业就业份额", "Capital-goods Employment Share", "capital_good_employment / total_employment", "employed Persons", "world-week", "capital_good_employment;total_employment", "persisted week"),
    _s("investment_order_count", "投资订单数", "Investment Order Count", "count(distinct authoritative order_id)", "persisted investment orders", "order", "order_id", "selected filter window"),
    _s("average_order_size", "平均订单规模", "Average Order Size", "mean(requested_units at first authoritative order record)", "persisted investment orders", "order", "order_id;requested_units", "selected filter window", exclusions="duplicate weekly order-state rows"),
    _s("delivery_volume", "交付量", "Delivery Volume", "sum(physical_units on delivery-recognition events)", "persisted deliveries", "delivery event", "event_type;physical_units", "selected filter window"),
    _s("order_delivery_lag", "订单至首次交付时滞", "Order-to-first-delivery Lag", "first delivery week - order/advance week by order_id", "orders with linked authoritative order_id", "order", "order_id;global_step;event_type", "selected filter window", exclusions="events without linkable order_id"),
    _s("capital_acquisition_cohort", "资本取得批次", "Capital Acquisition Cohort", "group assets by acquisition simulation year", "persisted CapitalAssets", "asset", "asset_id;acquisition_week;acquisition_cost;active", "52-week acquisition cohort"),
    _s("sector_cash_distribution", "部门现金分布", "Sectoral Cash Distribution", "Household + Food Firms + capital-good Firms + LegacyOwner + Estate cash", "all persisted cash holders", "sector-week", "macro cash fields;Firm cash", "persisted week"),
    _s("sector_cash_change", "部门现金变化", "Sectoral Cash Change", "current sector cash - prior-week sector cash", "all persisted cash holders", "sector-week", "sector_cash_distribution", "weekly flow derived from adjacent stocks"),
    _s("food_firm_cash", "食品企业现金", "Food-firm Cash", "sum(Firm cash where sector=food)", "Food Firms", "sector-week", "firm_id;sector;cash", "persisted week"),
    _s("capital_good_firm_cash", "资本品企业现金", "Capital-goods Firm Cash", "sum(Firm cash where sector=capital_goods)", "capital-goods Firms", "sector-week", "firm_id;sector;cash", "persisted week"),
    _s("household_cash_change", "家庭现金变化", "Household Cash Change", "household_cash[t] - household_cash[t-1]", "economic settlement accounts", "world-week", "household_cash", "adjacent persisted weeks"),
    _s("food_firm_cash_change", "食品企业现金变化", "Food-firm Cash Change", "food_firm_cash[t] - food_firm_cash[t-1]", "Food Firms", "sector-week", "firm cash", "adjacent persisted weeks"),
    _s("capital_good_firm_cash_change", "资本品企业现金变化", "Capital-goods Firm Cash Change", "capital_good_firm_cash[t] - capital_good_firm_cash[t-1]", "capital-goods Firms", "sector-week", "firm cash", "adjacent persisted weeks"),
    _s("legacy_owner_cash_change", "LegacyOwner 现金变化", "LegacyOwner Cash Change", "legacy_owner_cash[t] - legacy_owner_cash[t-1]", "LegacyOwner", "world-week", "legacy_owner_cash", "adjacent persisted weeks"),
    _s("estate_cash_change", "遗产账户现金变化", "Estate Cash Change", "estate_cash[t] - estate_cash[t-1]", "Estate accounts", "world-week", "estate_cash", "adjacent persisted weeks"),
]


class MetricRegistry:
    def __init__(self, consolidation_dir: Path):
        self.consolidation_dir = Path(consolidation_dir)
        self._metrics = {metric.internal_name: metric for metric in METRICS}
        self._statistics = {metric.internal_name: metric for metric in STATISTICAL_METRICS}
        self.contract_rows = self._read("authoritative_analysis_data_contract.csv")
        self.availability_rows = self._read("analysis_availability_matrix.csv")
        self.household_semantics = self._read("household_semantic_registry.csv")
        unknown = {
            metric.availability
            for metric in [*METRICS, *STATISTICAL_METRICS]
        } - AVAILABILITY
        if unknown:
            raise ValueError(f"Unknown availability values: {sorted(unknown)}")

    def _read(self, filename):
        path = self.consolidation_dir / filename
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def get(self, name):
        if name in self._metrics:
            return self._metrics[name]
        return self._statistics[name]

    def all(self):
        return list(self._metrics.values())

    def rows(self):
        return [asdict(metric) for metric in self.all()]

    def statistical_all(self):
        return list(self._statistics.values())

    def statistical_rows(self):
        return [asdict(metric) for metric in self.statistical_all()]

    def label(self, name, language="zh"):
        metric = self.get(name)
        return metric.display_name_zh if language == "zh" else metric.display_name_en

    def availability(self, name):
        return self.get(name).availability


__all__ = [
    "AVAILABILITY", "METRICS", "STATISTICAL_METRICS", "MetricDefinition",
    "StatisticalMetricDefinition", "MetricRegistry",
]
