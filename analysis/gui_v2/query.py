"""Semantic/query layer consumed by GUI V2 widgets."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analysis.statistics import (
    coefficient_of_variation,
    extended_distribution_summary,
    hhi,
    lorenz_curve,
)
from analysis.gui_v2.data import CanonicalDataStore


class AnalysisV2Query:
    STATISTICALLY_ENRICHED_METRICS = {
        "household_income_distribution", "household_income_gini",
        "household_income_lorenz", "household_cash_distribution",
        "household_cash_gini", "household_financial_assets_distribution",
        "household_saving_distribution", "labor_eligibility_rate",
        "settlement_valid_person_count",
    }
    def __init__(self, store: CanonicalDataStore):
        self.store = store
        self.registry = store.registry

    @staticmethod
    def _range(frame, start=None, end=None, field="global_step"):
        result = frame
        if field not in result.columns:
            return result.copy()
        if start is not None:
            result = result[result[field] >= start]
        if end is not None:
            result = result[result[field] <= end]
        return result.copy()

    @staticmethod
    def _numeric(frame, columns):
        result = frame.copy()
        for column in columns:
            if column in result.columns:
                result[column] = pd.to_numeric(result[column], errors="coerce")
        return result

    def time_bounds(self):
        if self.store.has_support_dataset():
            frame = self.store.support_table("support_elderly")
            if "research_week" in frame.columns:
                steps = pd.to_numeric(frame["research_week"], errors="coerce")
            elif "global_step" in frame.columns:
                steps = pd.to_numeric(frame["global_step"], errors="coerce")
            elif "window" in frame.columns:
                # Compact support-comparison artifacts are window summaries,
                # not weekly histories. Map their endpoint labels explicitly.
                steps = frame["window"].map({"first_52": 52, "last_52": 468, "full_520": 520})
            else:
                return 0, 0
            steps = steps.dropna()
            if steps.empty:
                return 0, 0
            return int(steps.min()), int(steps.max())
        macro = self.store.table("macro")
        return int(macro.global_step.min()), int(macro.global_step.max())


    @staticmethod
    def _at_or_before(frame, week, field="global_step"):
        if frame.empty:
            return frame.copy()
        eligible = frame[pd.to_numeric(frame[field], errors="coerce") <= week]
        selected = eligible[field].max() if not eligible.empty else frame[field].min()
        return frame[frame[field] == selected].copy()

    @staticmethod
    def _ratio(numerator, denominator):
        numerator = pd.to_numeric(numerator, errors="coerce")
        denominator = pd.to_numeric(denominator, errors="coerce")
        if np.isscalar(denominator):
            return numerator / denominator if math.isfinite(float(denominator)) and abs(float(denominator)) > 1e-15 else numerator * math.nan
        return numerator.div(denominator.where(denominator.abs() > 1e-15))

    def list_firms(self):
        values = self.store.table("firms")["firm_id"].dropna().astype(str).unique().tolist()
        return sorted(values, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))

    def list_sectors(self):
        return sorted(self.store.table("firms")["sector"].dropna().astype(str).unique().tolist())

    def list_event_types(self):
        return sorted(self.store.table("chain")["event_type"].dropna().astype(str).unique().tolist())

    def get_metric_definition(self, name):
        return self.registry.get(name)

    def get_availability(self, name):
        if self.store.statistics_enriched and name in self.STATISTICALLY_ENRICHED_METRICS:
            return "AUTHORITATIVE_AND_PERSISTED"
        return self.registry.availability(name)

    def get_macro_series(self, metric_names, start=None, end=None):
        base = self._range(self.store.table("macro"), start, end)
        result = base[["global_step"]].copy()
        firm_frame = None
        for name in metric_names:
            metric = self.registry.get(name)
            if metric.availability in {"NOT_AVAILABLE", "AUTHORITATIVE_RUNTIME_ONLY"}:
                result[name] = np.nan
                continue
            if metric.source_dataset == "macro" and metric.source_field in base:
                result[name] = pd.to_numeric(base[metric.source_field], errors="coerce")
            elif name == "aggregate_firm_cash":
                if firm_frame is None:
                    firm_frame = self._range(self.store.table("firms"), start, end)
                grouped = firm_frame.groupby("global_step", as_index=False)["cash"].sum().rename(columns={"cash": name})
                result = result.merge(grouped, on="global_step", how="left")
            else:
                result[name] = np.nan
        return result

    def get_population_series(self, metric_names=None, start=None, end=None):
        names = metric_names or ["population", "births", "deaths", "marriages", "age_0_19", "age_20_39", "age_40_64", "age_65_plus"]
        frame = self._range(self.store.table("demography"), start, end)
        result = frame[["global_step"]].copy()
        for name in names:
            metric = self.registry.get(name)
            result[name] = pd.to_numeric(frame.get(metric.source_field), errors="coerce")
        return result

    def get_population_structure(self, week):
        frame = self.get_population_series(end=week)
        selected = self._at_or_before(frame, week)
        if selected.empty:
            return pd.DataFrame(columns=["global_step", "age_group", "count", "share"])
        row = selected.iloc[-1]
        groups = [
            ("age_0_19", "0-19"),
            ("age_20_39", "20-39"),
            ("age_40_64", "40-64"),
            ("age_65_plus", "65+"),
        ]
        population = float(pd.to_numeric(pd.Series([row.get("population")]), errors="coerce").iloc[0])
        rows = []
        for field, label in groups:
            count = float(pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0])
            rows.append({
                "global_step": int(row.global_step),
                "age_group": label,
                "count": count,
                "share": count / population if population > 0 else math.nan,
            })
        return pd.DataFrame(rows)

    def get_exact_population_structure(self, week):
        columns = ["global_step", "age_year", "sex", "population_count"]
        if not self.store.has_table("age_histogram"):
            result = pd.DataFrame(columns=columns)
            result.attrs["availability"] = "NOT_AVAILABLE"
            return result
        frame = self.store.table("age_histogram")
        selected = self._at_or_before(frame, week)
        result = self._numeric(selected, ["global_step", "age_year", "population_count"])
        result.attrs["availability"] = "AUTHORITATIVE_AND_PERSISTED"
        return result[columns].sort_values(["age_year", "sex"]).reset_index(drop=True)

    def get_exact_age_ratio_series(self, start=None, end=None):
        if not self.store.has_table("age_histogram"):
            return pd.DataFrame(columns=[
                "global_step", "child_share", "working_age_share", "elderly_share",
                "dependency_ratio", "youth_dependency_ratio", "old_age_dependency_ratio",
            ])
        frame = self._range(self.store.table("age_histogram"), start, end)
        frame = self._numeric(frame, ["age_year", "population_count"])
        rows = []
        for step, group in frame.groupby("global_step"):
            child = group.loc[group.age_year < 20, "population_count"].sum()
            working = group.loc[(group.age_year >= 20) & (group.age_year <= 64), "population_count"].sum()
            elderly = group.loc[group.age_year >= 65, "population_count"].sum()
            population = child + working + elderly
            rows.append({
                "global_step": step,
                "child_share": child / population if population else math.nan,
                "working_age_share": working / population if population else math.nan,
                "elderly_share": elderly / population if population else math.nan,
                "dependency_ratio": (child + elderly) / working if working else math.nan,
                "youth_dependency_ratio": child / working if working else math.nan,
                "old_age_dependency_ratio": elderly / working if working else math.nan,
            })
        return pd.DataFrame(rows)

    def get_demographic_ratio_series(self, start=None, end=None):
        frame = self.get_population_series(start=start, end=end)
        result = frame[["global_step"]].copy()
        working = frame["age_20_39"] + frame["age_40_64"]
        result["child_share"] = self._ratio(frame["age_0_19"], frame["population"])
        result["working_age_share"] = self._ratio(working, frame["population"])
        result["elderly_share"] = self._ratio(frame["age_65_plus"], frame["population"])
        result["dependency_ratio"] = self._ratio(frame["age_0_19"] + frame["age_65_plus"], working)
        result["youth_dependency_ratio"] = self._ratio(frame["age_0_19"], working)
        result["old_age_dependency_ratio"] = self._ratio(frame["age_65_plus"], working)
        return result

    def get_household_distribution_availability(self):
        names = [
            "household_income_distribution", "household_income_gini",
            "household_income_lorenz", "household_cash_distribution",
            "household_cash_gini", "household_financial_assets_distribution",
            "household_saving_distribution",
        ]
        return pd.DataFrame([
            {
                "metric": name,
                "availability": self.get_availability(name),
                "population": self.registry.get(name).population,
                "source_fields": self.registry.get(name).source_fields,
                "exclusion_rules": self.registry.get(name).exclusion_rules,
            }
            for name in names
        ])

    def get_household_micro_distribution(self, metric_name, week=None):
        """Return one authoritative social-Household cross-section."""
        metric = self.registry.get(metric_name)
        columns = ["global_step", "household_id", "social_household", "value"]
        availability = self.get_availability(metric_name)
        if availability != "AUTHORITATIVE_AND_PERSISTED" or not self.store.has_table("social_households"):
            result = pd.DataFrame(columns=columns)
            result.attrs["availability"] = availability
            result.attrs["reason"] = metric.exclusion_rules
            return result
        source = {
            "household_income_distribution": "current_week_income",
            "household_income_gini": "current_week_income",
            "household_income_lorenz": "current_week_income",
            "household_cash_distribution": "cash",
            "household_cash_gini": "cash",
            "household_financial_assets_distribution": "total_financial_assets",
            "household_saving_distribution": "current_week_saving",
        }.get(metric_name)
        if source is None:
            raise ValueError(f"Unsupported Household distribution metric: {metric_name}")
        frame = self.store.table("social_households")
        selected_week = frame.global_step.max() if week is None else week
        selected = self._at_or_before(frame, selected_week)
        result = selected[["global_step", "household_id"]].copy()
        result["social_household"] = True
        result["value"] = pd.to_numeric(selected[source], errors="coerce")
        result = result[columns].reset_index(drop=True)
        result.attrs["availability"] = availability
        result.attrs["source_field"] = source
        return result

    def get_household_distribution_summary(self, metric_name, week=None):
        frame = self.get_household_micro_distribution(metric_name, week)
        if frame.empty:
            return pd.DataFrame(columns=["global_step", "count", "mean", "median", "p10", "p25", "p75", "p90", "p95", "gini", "top_10_share", "bottom_50_share", "p90_p10_ratio", "positive_share", "zero_share", "negative_share"])
        summary = extended_distribution_summary(frame.value.tolist())
        summary["global_step"] = int(frame.global_step.iloc[0])
        return pd.DataFrame([summary])

    def get_household_distribution_history(self, metric_name, start=None, end=None):
        if not self.store.has_table("social_households"):
            return pd.DataFrame(columns=["global_step", "gini", "median", "p10", "p90"])
        source = {
            "household_income_distribution": "current_week_income",
            "household_cash_distribution": "cash",
            "household_financial_assets_distribution": "total_financial_assets",
            "household_saving_distribution": "current_week_saving",
        }.get(metric_name)
        if source is None:
            raise ValueError(f"Unsupported Household distribution metric: {metric_name}")
        frame = self._range(self.store.table("social_households"), start, end)
        rows = []
        for step, group in frame.groupby("global_step"):
            summary = extended_distribution_summary(pd.to_numeric(group[source], errors="coerce").tolist())
            summary["global_step"] = step
            rows.append(summary)
        return pd.DataFrame(rows).sort_values("global_step").reset_index(drop=True) if rows else pd.DataFrame()

    def get_household_lorenz(self, metric_name, week=None):
        frame = self.get_household_micro_distribution(metric_name, week)
        population, values = lorenz_curve(frame.value.tolist())
        return pd.DataFrame({"population_share": population, "value_share": values})

    def get_household_ecdf(self, metric_name, week=None):
        frame = self.get_household_micro_distribution(metric_name, week)
        values = np.sort(pd.to_numeric(frame.value, errors="coerce").dropna().to_numpy())
        return pd.DataFrame({
            "value": values,
            "cumulative_share": np.arange(1, len(values) + 1) / len(values) if len(values) else [],
        })

    def get_saving_rate_distribution(self, week=None):
        columns = ["global_step", "household_id", "social_household", "value"]
        if not self.store.has_table("social_households"):
            return pd.DataFrame(columns=columns)
        frame = self.store.table("social_households")
        selected = self._at_or_before(frame, frame.global_step.max() if week is None else week)
        income = pd.to_numeric(selected.current_week_income, errors="coerce")
        saving = pd.to_numeric(selected.current_week_saving, errors="coerce")
        result = selected[["global_step", "household_id"]].copy()
        result["social_household"] = True
        result["value"] = saving.div(income.where(income.abs() > 1e-15))
        return result[columns]

    def get_social_household_series(self, start=None, end=None):
        return self.get_macro_series(["social_households", "settlement_accounts", "economic_accounts"], start, end)

    def get_labor_series(self, start=None, end=None):
        return self.get_macro_series(["eligible_labor", "employment", "unassigned_labor", "food_employment", "capital_good_employment"], start, end)

    def get_labor_quality_series(self, start=None, end=None):
        if self.store.has_table("labor_denominators"):
            denominator = self._range(self.store.table("labor_denominators"), start, end)
            denominator = self._numeric(denominator, [
                "population", "working_age_population_20_60",
                "settlement_valid_working_age_20_60", "labor_age_eligible",
                "labor_match_eligible", "employed_eligible", "unassigned_eligible",
            ])
            result = denominator.rename(columns={
                "working_age_population_20_60": "working_age_count",
                "settlement_valid_working_age_20_60": "settlement_valid_person_count",
                "labor_match_eligible": "eligible_labor",
                "employed_eligible": "employment",
                "unassigned_eligible": "unassigned_labor",
            })
            result["labor_eligibility_rate"] = self._ratio(result["eligible_labor"], result["labor_age_eligible"])
            result["eligible_population_share"] = self._ratio(result["eligible_labor"], result["population"])
            result["employment_rate"] = self._ratio(result["employment"], result["eligible_labor"])
            result["unassigned_eligible_rate"] = self._ratio(result["unassigned_labor"], result["eligible_labor"])
            macro = self.get_labor_series(start, end)[["global_step", "food_employment", "capital_good_employment"]]
            result = result.merge(macro, on="global_step", how="left")
            result["food_employment_share"] = self._ratio(result["food_employment"], result["employment"])
            result["capital_good_employment_share"] = self._ratio(result["capital_good_employment"], result["employment"])
            return result
        labor = self.get_labor_series(start, end)
        ages = self.get_population_series(start=start, end=end)
        result = labor.merge(
            ages[["global_step", "population", "age_20_39", "age_40_64"]],
            on="global_step", how="left",
        )
        result["working_age_count"] = result["age_20_39"] + result["age_40_64"]
        result["labor_eligibility_rate"] = math.nan
        result["eligible_population_share"] = self._ratio(result["eligible_labor"], result["population"])
        result["employment_rate"] = self._ratio(result["employment"], result["eligible_labor"])
        result["unassigned_eligible_rate"] = self._ratio(result["unassigned_labor"], result["eligible_labor"])
        result["food_employment_share"] = self._ratio(result["food_employment"], result["employment"])
        result["capital_good_employment_share"] = self._ratio(result["capital_good_employment"], result["employment"])
        return result

    def get_labor_event_series(self, start=None, end=None):
        event_names = ["hire", "release", "eligibility_entry", "eligibility_exit", "sector_transfer"]
        if not self.store.has_table("labor_events"):
            return pd.DataFrame(columns=["global_step", *event_names, "net_hires", "turnover_rate"])
        events = self._range(self.store.table("labor_events"), start, end)
        if events.empty:
            return pd.DataFrame(columns=["global_step", *event_names, "net_hires", "turnover_rate"])
        grouped = events.groupby(["global_step", "event_type"]).size().unstack(fill_value=0)
        for name in event_names:
            if name not in grouped:
                grouped[name] = 0
        result = grouped[event_names].reset_index()
        result["net_hires"] = result["hire"] - result["release"]
        denominator = self.get_labor_quality_series(start, end)[["global_step", "employment"]]
        result = result.merge(denominator, on="global_step", how="left")
        result["turnover_rate"] = self._ratio(result["hire"] + result["release"], 2 * result["employment"])
        return result

    def get_labor_events(self, start=None, end=None):
        if not self.store.has_table("labor_events"):
            return pd.DataFrame()
        return self._range(self.store.table("labor_events"), start, end)

    def get_firm_capital_history(self, firm_id=None, sector=None, start=None, end=None):
        if not self.store.has_table("firm_capital"):
            return pd.DataFrame()
        frame = self._range(self.store.table("firm_capital"), start, end)
        if firm_id is not None:
            frame = frame[frame.firm_id.astype(str) == str(firm_id)]
        if sector is not None:
            frame = frame[frame.sector_id.astype(str) == str(sector)]
        return frame.reset_index(drop=True)

    def get_firm_capital_comparison(self, week, sector=None):
        frame = self.get_firm_capital_history(sector=sector)
        if frame.empty:
            return frame
        return self._at_or_before(frame, week).sort_values("firm_id").reset_index(drop=True)

    def get_sector_series(self, sector=None, start=None, end=None):
        frame = self._range(self.store.table("firms"), start, end)
        if sector:
            frame = frame[frame["sector"].astype(str) == str(sector)]
        numeric = ["employment", "production", "sales", "revenue", "cash", "operating_profit", "principal", "customer_advance_liability"]
        frame = self._numeric(frame, numeric)
        return frame.groupby("global_step", as_index=False)[numeric].sum(min_count=1)

    def get_firm_series(self, firm_id, start=None, end=None):
        frame = self._range(self.store.table("firms"), start, end)
        if firm_id is not None:
            frame = frame[frame["firm_id"].astype(str) == str(firm_id)]
        return frame.reset_index(drop=True)

    def resolve_comparison_sector(self, sector=None, firm_id=None):
        if sector:
            return str(sector)
        firms = self.store.table("firms")
        if firm_id is not None:
            match = firms[firms["firm_id"].astype(str) == str(firm_id)]
            if not match.empty:
                return str(match.iloc[0]["sector"])
        sectors = self.list_sectors()
        return "food" if "food" in sectors else (sectors[0] if sectors else None)

    def get_firm_share_series(self, sector=None, metric="sales", start=None, end=None):
        sector = self.resolve_comparison_sector(sector)
        frame = self._range(self.store.table("firms"), start, end)
        frame = frame[frame["sector"].astype(str) == str(sector)].copy()
        if metric not in frame.columns:
            return pd.DataFrame(columns=["global_step", "firm_id", "sector", "metric_value", "sector_total", "market_share"])
        frame["metric_value"] = pd.to_numeric(frame[metric], errors="coerce")
        frame["sector_total"] = frame.groupby("global_step")["metric_value"].transform("sum")
        frame["market_share"] = self._ratio(frame["metric_value"], frame["sector_total"])
        return frame[["global_step", "firm_id", "sector", "metric_value", "sector_total", "market_share"]].reset_index(drop=True)

    def get_market_structure_series(self, sector=None, metric="sales", start=None, end=None):
        shares = self.get_firm_share_series(sector, metric, start, end)
        rows = []
        for week, group in shares.groupby("global_step", sort=True):
            valid = group["market_share"].dropna().astype(float).tolist()
            levels = group["metric_value"].dropna().astype(float).tolist()
            ranked = sorted(valid, reverse=True)
            firm_count = int(group["firm_id"].nunique())
            rows.append({
                "global_step": week,
                "sector": str(group["sector"].iloc[0]),
                "share_metric": metric,
                "firm_count": firm_count,
                "sales_hhi": hhi(valid, already_shares=True),
                "largest_firm_share": ranked[0] if ranked else math.nan,
                "top2_share": sum(ranked[:2]) if ranked else math.nan,
                "top3_share": sum(ranked[:3]) if ranked else math.nan,
                "cross_firm_cv": coefficient_of_variation(levels),
                "sector_structure": "SINGLE_SUPPLIER_SECTOR" if firm_count == 1 else "MULTI_FIRM_SECTOR",
            })
        return pd.DataFrame(rows)

    def get_firm_cross_section(self, sector=None, week=None, firm_id=None):
        sector = self.resolve_comparison_sector(sector, firm_id)
        frame = self.store.table("firms")
        frame = frame[frame["sector"].astype(str) == str(sector)].copy()
        if frame.empty:
            return frame
        selected_week = frame["global_step"].max() if week is None else week
        frame = self._at_or_before(frame, selected_week)
        numeric = [
            "cash", "employment", "inventory", "sales", "revenue",
            "operating_profit", "principal", "customer_advance_liability",
            "prepaid_investment_asset", "production", "capital_book_value",
            "capital_asset_count",
        ]
        frame = self._numeric(frame, numeric)
        frame["market_share"] = self._ratio(frame["sales"], frame["sales"].sum())
        frame["profit_margin"] = self._ratio(frame["operating_profit"], frame["revenue"])
        columns = ["global_step", "firm_id", "sector", "market_share", *numeric, "profit_margin"]
        return frame[columns].sort_values("market_share", ascending=False, na_position="last").reset_index(drop=True)

    def get_firm_dispersion(self, sector=None, week=None, metric_names=None):
        frame = self.get_firm_cross_section(sector, week)
        names = metric_names or ["sales", "revenue", "cash", "employment", "inventory", "operating_profit", "principal"]
        rows = []
        for name in names:
            values = pd.to_numeric(frame.get(name), errors="coerce").dropna()
            rows.append({
                "metric": name,
                "mean": values.mean() if len(values) else math.nan,
                "median": values.median() if len(values) else math.nan,
                "minimum": values.min() if len(values) else math.nan,
                "maximum": values.max() if len(values) else math.nan,
                "cv": coefficient_of_variation(values.tolist()),
            })
        return pd.DataFrame(rows)

    def get_firm_indexed_series(self, sector=None, metric="sales", start=None, end=None):
        sector = self.resolve_comparison_sector(sector)
        frame = self._range(self.store.table("firms"), start, end)
        frame = frame[frame["sector"].astype(str) == str(sector)].copy()
        frame["metric_value"] = pd.to_numeric(frame.get(metric), errors="coerce")
        indexed = []
        for firm_id, group in frame.groupby("firm_id", sort=True):
            group = group.sort_values("global_step").copy()
            candidates = group["metric_value"].dropna()
            nonzero = candidates[candidates.abs() > 1e-15]
            base = nonzero.iloc[0] if not nonzero.empty else math.nan
            group["indexed_value"] = group["metric_value"] / base * 100 if math.isfinite(base) else math.nan
            indexed.append(group[["global_step", "firm_id", "sector", "metric_value", "indexed_value"]])
        return pd.concat(indexed, ignore_index=True) if indexed else pd.DataFrame(columns=["global_step", "firm_id", "sector", "metric_value", "indexed_value"])

    def get_balance_sheet(self, firm_id, week=None):
        frame = self.store.table("accounting")
        frame = frame[(frame["record_type"] == "firm") & (frame["firm_id"].astype(str) == str(firm_id))]
        if frame.empty:
            return pd.DataFrame(columns=["component", "classification", "value"])
        selected_week = frame.global_step.max() if week is None else week
        eligible = frame[frame.global_step <= selected_week]
        row = (eligible.iloc[-1] if not eligible.empty else frame.iloc[0])
        components = [
            ("cash", "asset", row.get("cash_end")),
            ("inventory_book_value", "asset", row.get("inventory_book_value")),
            ("prepaid_investment_asset", "asset", row.get("prepaid_capital_investment_asset")),
            ("capital_book_value", "asset", row.get("capital_asset_book_value")),
            ("loan_principal", "liability", row.get("loan_balance")),
            ("interest_arrears", "liability", row.get("post_termout_interest_arrears_claim")),
            ("customer_advance_liability", "liability", row.get("customer_advance_liability")),
            ("equity", "equity", row.get("equity")),
        ]
        return pd.DataFrame(components, columns=["component", "classification", "value"])

    def get_cash_bridge(self, firm_id, start=None, end=None):
        frame = self.store.table("accounting")
        frame = frame[(frame["record_type"] == "firm") & (frame["firm_id"].astype(str) == str(firm_id))]
        frame = self._range(frame, start, end)
        fields = ["global_step", "cash_start", "sales_collections", "customer_advance_cash_inflow", "wage_payments", "prepaid_investment_cash_outflow", "cfo", "cfi", "cff", "cash_end", "cash_flow_gap"]
        return self._numeric(frame[fields], fields[1:]).reset_index(drop=True)

    def get_accounting_series(self, firm_id, start=None, end=None):
        frame = self.store.table("accounting")
        frame = frame[(frame["record_type"] == "firm") & (frame["firm_id"].astype(str) == str(firm_id))]
        return self._range(frame, start, end).reset_index(drop=True)

    def get_capital_assets(self, firm_id=None, active=None, start=None, end=None):
        frame = self.store.table("assets")
        if firm_id is not None:
            frame = frame[frame["owner_firm_id"].astype(str) == str(firm_id)]
        if active is not None:
            truth = frame["active"].astype(str).str.lower().isin({"true", "1"})
            frame = frame[truth == bool(active)]
        if start is not None:
            frame = frame[frame["acquisition_week"] >= start]
        if end is not None:
            frame = frame[frame["acquisition_week"] <= end]
        return frame.reset_index(drop=True)

    def get_investment_chain(self, firm_id=None, event_type=None, start=None, end=None):
        frame = self._range(self.store.table("chain"), start, end)
        if firm_id is not None:
            buyer = frame.get("buyer_firm_id", pd.Series(index=frame.index, dtype=object)).astype(str)
            supplier = frame.get("supplier_firm_id", pd.Series(index=frame.index, dtype=object)).astype(str)
            owner = frame.get("owner_firm_id", pd.Series(index=frame.index, dtype=object)).astype(str)
            frame = frame[(buyer == str(firm_id)) | (supplier == str(firm_id)) | (owner == str(firm_id))]
        if event_type:
            frame = frame[frame["event_type"].astype(str) == str(event_type)]
        return frame.reset_index(drop=True)

    def get_contract_trace(self, order_id):
        frame = self.store.table("chain")
        return frame[frame["order_id"].astype(str) == str(order_id)].sort_values("global_step").reset_index(drop=True)

    def get_capital_pipeline(self, start=None, end=None):
        chain = self._range(self.store.table("chain"), start, end)
        assets = self.store.table("assets")
        macro = self._range(self.store.table("macro"), start, end)
        firms = self._range(self.store.table("firms"), start, end)
        order_rows = chain[chain["order_id"].notna()].sort_values("global_step")
        order_count = int(order_rows["order_id"].nunique())
        requested = (
            order_rows.dropna(subset=["requested_units"])
            .groupby("order_id", as_index=False).first()["requested_units"]
            if "requested_units" in order_rows else pd.Series(dtype=float)
        )
        delivery = chain[chain["event_type"].astype(str) == "customer_advance_delivery_recognition"]
        advance = chain[chain["event_type"].astype(str) == "customer_advance_received"]
        acquired = chain[chain["event_type"].astype(str) == "asset_acquired"]
        retired = chain[chain["event_type"].astype(str) == "asset_retired"]
        latest_firms = self._at_or_before(firms, end if end is not None else firms.global_step.max())
        latest_macro = self._at_or_before(macro, end if end is not None else macro.global_step.max())
        rows = [
            {"pipeline_stage": "investment_order", "event_count": order_count, "physical_units": requested.sum(min_count=1), "cash_amount": math.nan},
            {"pipeline_stage": "customer_advance", "event_count": len(advance), "physical_units": pd.to_numeric(advance.get("physical_units"), errors="coerce").sum(min_count=1), "cash_amount": pd.to_numeric(advance.get("cash_amount"), errors="coerce").sum(min_count=1)},
            {"pipeline_stage": "delivery", "event_count": len(delivery), "physical_units": pd.to_numeric(delivery.get("physical_units"), errors="coerce").sum(min_count=1), "cash_amount": pd.to_numeric(delivery.get("revenue_recognized"), errors="coerce").sum(min_count=1)},
            {"pipeline_stage": "asset_acquisition", "event_count": len(acquired), "physical_units": pd.to_numeric(acquired.get("physical_asset_units"), errors="coerce").sum(min_count=1), "cash_amount": pd.to_numeric(acquired.get("acquisition_cost"), errors="coerce").sum(min_count=1)},
            {"pipeline_stage": "asset_retirement", "event_count": len(retired), "physical_units": pd.to_numeric(retired.get("retired_service_capacity"), errors="coerce").sum(min_count=1), "cash_amount": pd.to_numeric(retired.get("remaining_book_value"), errors="coerce").sum(min_count=1)},
        ]
        summary = {
            "order_count": order_count,
            "average_order_size": pd.to_numeric(requested, errors="coerce").mean() if len(requested) else math.nan,
            "delivery_volume": pd.to_numeric(delivery.get("physical_units"), errors="coerce").sum(min_count=1),
            "outstanding_advances": pd.to_numeric(latest_firms.get("customer_advance_liability"), errors="coerce").sum(min_count=1),
            "asset_acquisitions": len(acquired),
            "asset_retirements": len(retired),
            "active_assets": pd.to_numeric(latest_macro.get("active_capital_assets"), errors="coerce").iloc[-1] if not latest_macro.empty else math.nan,
            "active_capital_service": pd.to_numeric(latest_macro.get("active_capital_service"), errors="coerce").iloc[-1] if not latest_macro.empty else math.nan,
            "ledger_asset_rows": len(assets),
        }
        return pd.DataFrame(rows), summary

    def get_order_delivery_lags(self, start=None, end=None):
        chain = self._range(self.store.table("chain"), start, end)
        linked = chain[chain["order_id"].notna()].copy()
        starts = linked[linked["event_type"].isin(["customer_advance_received", "investment_order_settled"])].groupby("order_id")["global_step"].min()
        deliveries = linked[linked["event_type"] == "customer_advance_delivery_recognition"].groupby("order_id")["global_step"].min()
        result = pd.concat([starts.rename("order_week"), deliveries.rename("delivery_week")], axis=1).dropna()
        result["delivery_lag_weeks"] = result["delivery_week"] - result["order_week"]
        return result.reset_index()

    def get_capital_asset_cohorts(self):
        assets = self.store.table("assets").copy()
        assets["acquisition_week"] = pd.to_numeric(assets["acquisition_week"], errors="coerce")
        assets = assets.dropna(subset=["acquisition_week"])
        assets["acquisition_cohort_year"] = (assets["acquisition_week"] // 52).astype(int)
        assets["active_flag"] = assets["active"].astype(str).str.lower().isin({"true", "1"}).astype(int)
        assets["retired_flag"] = pd.to_numeric(assets["retirement_week"], errors="coerce").notna().astype(int)
        assets["service_capacity"] = (
            pd.to_numeric(assets["physical_asset_units"], errors="coerce")
            * pd.to_numeric(assets["capital_service_capacity_per_unit"], errors="coerce")
        )
        grouped = assets.groupby("acquisition_cohort_year", as_index=False).agg(
            asset_count=("asset_id", "count"),
            active_asset_count=("active_flag", "sum"),
            retirement_count=("retired_flag", "sum"),
            acquisition_cost=("acquisition_cost", "sum"),
            accumulated_depreciation=("accumulated_depreciation", "sum"),
            acquired_service_capacity=("service_capacity", "sum"),
        )
        return grouped

    def get_sector_cash_series(self, start=None, end=None):
        macro = self._range(self.store.table("macro"), start, end)
        firms = self._range(self.store.table("firms"), start, end)
        by_sector = firms.groupby(["global_step", "sector"], as_index=False)["cash"].sum(min_count=1)
        pivot = by_sector.pivot(index="global_step", columns="sector", values="cash").reset_index()
        result = macro[["global_step", "household_cash", "legacy_owner_cash", "estate_cash"]].merge(pivot, on="global_step", how="left")
        result = result.rename(columns={"food": "food_firm_cash", "capital_goods": "capital_good_firm_cash"})
        for name in ("food_firm_cash", "capital_good_firm_cash"):
            if name not in result:
                result[name] = math.nan
        for name in ("household_cash", "food_firm_cash", "capital_good_firm_cash", "legacy_owner_cash", "estate_cash"):
            result[f"{name}_change"] = pd.to_numeric(result[name], errors="coerce").diff()
        return result

    def get_demographic_events(self, start=None, end=None):
        frame = self.store.table("demographic_events")
        return self._range(frame, start, end, "global_step").reset_index(drop=True)

    def get_reconciliation(self, start=None, end=None):
        names = ["money_location_gap", "full_money_location_gap", "money_delta_gap", "goods_gap", "advance_prepaid_gap", "assignment_violations", "feasible_capacity_gap"]
        return self.get_macro_series(names, start, end)

    def get_availability_rows(self):
        return pd.DataFrame(self.registry.rows())

    def get_statistical_registry_rows(self):
        return pd.DataFrame(self.registry.statistical_rows())

    def get_known_boundaries(self):
        return pd.read_csv(self.store.consolidation_dir / "step15_known_boundaries.csv", low_memory=False)

    def reconciliation_status(self, start=None, end=None):
        frame = self.get_reconciliation(start, end)
        # Only stock identities belong in the pass/fail reconciliation panel.
        # money_delta_gap remains a separately labeled flow/timing diagnostic.
        tolerances = {
            "money_location_gap": 1e-5,
            "full_money_location_gap": 1e-5,
            "goods_gap": 1e-6,
            "advance_prepaid_gap": 1e-5,
            "assignment_violations": 0.0,
            "feasible_capacity_gap": 1e-6,
        }
        rows = []
        for name, tolerance in tolerances.items():
            maximum = float(pd.to_numeric(frame[name], errors="coerce").abs().max()) if not frame.empty else math.nan
            status = "PASS" if math.isfinite(maximum) and maximum <= tolerance else "FAIL"
            rows.append({"metric": name, "max_abs": maximum, "tolerance": tolerance, "status": status})
        return pd.DataFrame(rows)


    def has_private_support_dataset(self):
        return self.store.has_support_dataset()

    def get_private_support_series(self, start=None, end=None):
        if not self.has_private_support_dataset():
            return pd.DataFrame()
        if self.store.dataset_kind in {"half_week_buffer_support", "buffer_target_selection"}:
            distribution = self.store.support_table("support_distribution").copy()
            elderly = self.store.support_table("support_elderly").copy()
            food = self.store.support_table("support_food").copy()
            if self.store.dataset_kind == "buffer_target_selection":
                firm = food[["branch", "window"]].copy()
                firm["firm_cash"] = float("nan")
                firm["firm_profit"] = float("nan")
            else:
                firm = self.store.support_table("support_firm").copy()
            branch_map = {
                "CONTROL": "control",
                "CURRENT_WEEKLY": "current_weekly",
                "BUFFER_0_5": "buffer_0_5",
            }
            window_step = {"last_52": 468, "full_520": 520}
            for frame in (distribution, elderly, food, firm):
                frame["branch"] = frame["branch"].astype(str).map(branch_map).fillna(frame["branch"].astype(str).str.lower())
                frame["global_step"] = frame["window"].map(window_step)
            frame = distribution.merge(
                elderly[["branch", "window", "elderly_near_zero_share", "elderly_mean_cash", "elderly_median_cash"]],
                on=["branch", "window"], how="left",
            )
            frame = frame.merge(
                food[["branch", "window", "food_sales", "food_revenue"]],
                on=["branch", "window"], how="left",
            )
            frame = frame.merge(
                firm[["branch", "window", "firm_cash", "firm_profit"]],
                on=["branch", "window"], how="left",
            )
            frame = frame.rename(columns={"firm_cash": "aggregate_firm_cash", "firm_profit": "aggregate_firm_profit"})
            frame["phase"] = frame["window"]
            frame = self._range(frame, start, end)
            return frame.sort_values(["global_step", "branch"]).reset_index(drop=True)
        frame = self.store.support_table("support_elderly").copy()
        frame = frame.rename(columns={"research_week": "global_step"})
        frame["branch"] = frame["phase"].map({
            "control_support_off": "control",
            "treatment_support_on": "treatment",
        }).fillna(frame["phase"])
        frame = self._range(frame, start, end)
        numeric = [
            "near_zero_share", "elderly_near_zero_share", "median_household_cash",
            "household_cash_total", "household_consumption", "household_saving",
            "food_sales", "aggregate_firm_cash", "support_paid", "support_received",
            "support_transfer_count", "support_eligible_parents", "support_eligible_children",
            "elderly_support_received", "elderly_cash_total",
        ]
        for field in numeric:
            if field in frame:
                frame[field] = pd.to_numeric(frame[field], errors="coerce")
        return frame.sort_values(["global_step", "branch"]).reset_index(drop=True)

    def get_private_support_summary(self):
        if not self.has_private_support_dataset():
            return {}
        events = self.store.support_table("support_events").copy()
        coverage = self.store.support_table("support_coverage").copy()
        distribution = self.store.support_table("support_distribution").copy()
        food = self.store.support_table("support_food").copy()
        macro_cash = self.store.support_table("support_macro_cash").copy()
        if self.store.dataset_kind in {"half_week_buffer_support", "buffer_target_selection"}:
            child = self.store.support_table("support_child_burden").copy()
            return {
                "events": events,
                "coverage": coverage,
                "distribution": distribution,
                "food": food,
                "macro_cash": macro_cash,
                "child_burden": child,
            }
        child_path = self.store.support_paths.get("support_child_burden")
        transfer_records = below_count = above_count = unique_payers = 0
        if child_path is not None:
            for chunk in pd.read_csv(child_path, usecols=["payer_household_id", "payer_near_zero_after"], chunksize=100000, low_memory=False):
                flags = chunk["payer_near_zero_after"].astype(str).str.lower()
                transfer_records += len(chunk)
                below_count += int(flags.eq("true").sum())
                above_count += int(flags.eq("false").sum())
                unique_payers += int(chunk["payer_household_id"].nunique())
        observed = below_count + above_count
        return {
            "events": events,
            "coverage": coverage,
            "distribution": distribution,
            "food": food,
            "macro_cash": macro_cash,
            "child_burden": pd.DataFrame([{
                "transfer_records": transfer_records,
                "records_below_one_week_minimum": below_count,
                "records_above_one_week_minimum": above_count,
                "below_share_observed": float(below_count / observed) if observed else float("nan"),
                "unique_payer_households": unique_payers,
            }]),
        }
    def get_private_support_event_rows(self):
        if not self.has_private_support_dataset():
            return pd.DataFrame()
        return self.store.support_table("support_events").copy()
__all__ = ["AnalysisV2Query"]



