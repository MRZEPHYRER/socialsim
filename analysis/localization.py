"""Small, explicit presentation translation layer for Analysis."""

from __future__ import annotations


TRANSLATIONS = {
    "Overview": "总览",
    "Macro": "宏观经济",
    "Sectors": "行业",
    "Firms": "企业",
    "Accounting": "财务与查账",
    "Capital & Investment": "资本与投资",
    "Contracts": "合同与交易",
    "Reconciliation": "守恒与核对",
    "Reports": "报告与导出",
    "Population & Social": "人口与社会",
    "Labor": "劳动市场",
    "Household Income": "家庭收入",
    "Household Consumption": "家庭消费",
    "Employment": "就业",
    "Food Production": "食品产量",
    "Fixed Investment": "固定投资",
    "Capital Service": "资本服务",
    "Backlog": "积压订单",
    "Aggregate Firm Cash": "企业现金总额",
    "Income, Consumption, and Employment": "收入、消费与就业",
    "Selected Macro Metrics": "选定宏观指标",
    "Current values and selected-period trajectories from authoritative persisted diagnostics.": "基于权威持久化诊断的当前值与选定区间轨迹。",
    "Select one or more persisted metrics. The table and chart share the global filter.": "选择一个或多个已持久化指标；表格与图表共享全局筛选条件。",
    "Read-only": "只读",
    "READ-ONLY | authoritative persisted diagnostics": "只读 | 权威持久化诊断",
    "Chinese": "中文",
    "English": "English",
}


def tr(text: str, language: str = "zh") -> str:
    if language == "en":
        return text
    return TRANSLATIONS.get(text, text)

