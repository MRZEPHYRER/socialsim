"""Independent Analysis GUI V2 pages; widgets consume only query APIs."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QPushButton, QScrollArea, QTabWidget,
    QVBoxLayout, QWidget,
)

from analysis.gui_v2.localization import tr, value_label
from analysis.gui_v2.widgets import ChartCanvas, DataTable, MetricCard, Notice, card_grid
from analysis.gui_v2.liquidity import HouseholdLiquidityExplorer


def _last(frame, field):
    if frame.empty or field not in frame:
        return float("nan")
    return pd.to_numeric(frame[field], errors="coerce").iloc[-1]


def _support_branch_labels(query, language):
    if getattr(query.store, "dataset_kind", "") == "buffer_target_selection":
        return {
            "control": "Control / no support",
            "current_weekly": "Current legacy weekly",
            "buffer_0_25": "Selected 0.25-week buffer",
            "buffer_0_5": "0.5-week reference",
            "buffer_1_00": "1.0-week reference",
        }
    if getattr(query.store, "dataset_kind", "") == "half_week_buffer_support":
        return {
            "control": "Control / no support",
            "current_weekly": "Current weekly",
            "buffer_0_5": "0.5-week buffer",
        }
    return {
        "control": tr("label_control", language),
        "treatment": tr("label_treatment", language),
    }

def _clear(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        child = item.layout()
        if child is not None:
            _clear(child)


class BasePage(QWidget):
    title_key = "page_overview"
    subtitle_key = "subtitle_overview"

    def __init__(self, query, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(5)
        self.title = QLabel()
        self.title.setObjectName("pageTitle")
        self.subtitle = QLabel()
        self.subtitle.setObjectName("pageSubtitle")
        root.addWidget(self.title)
        root.addWidget(self.subtitle)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.content = QWidget()
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(0, 10, 0, 10)
        self.body.setSpacing(10)
        self.body.addStretch(1)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll, 1)
        self.query = query
        self.language = "zh"
        self.state = None

    def heading(self, key):
        label = QLabel(tr(key, self.language))
        label.setObjectName("sectionTitle")
        return label

    def charts(self, left, right):
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(left, 1)
        layout.addWidget(right, 1)
        return holder

    def panel(self, *widgets):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 8, 2, 8)
        layout.setSpacing(10)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        return panel

    def tabs(self, entries):
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setMinimumHeight(670)
        for key, widget in entries:
            tabs.addTab(widget, tr(key, self.language))
        return tabs

    def refresh(self, state, language="zh"):
        self.state = state
        self.language = language
        self.title.setText(tr(self.title_key, language))
        self.subtitle.setText(tr(self.subtitle_key, language))
        _clear(self.body)
        self.build()
        self.body.addStretch(1)

    def build(self):
        raise NotImplementedError


class OverviewPage(BasePage):
    title_key = "page_overview"
    subtitle_key = "subtitle_overview"

    def build(self):
        names = ["population", "employment", "household_income", "household_consumption", "household_cash", "aggregate_firm_cash", "food_production", "fixed_investment", "active_capital_service"]
        frame = self.query.get_macro_series(names, self.state.start, self.state.end)
        status = self.query.reconciliation_status(self.state.start, self.state.end)
        all_pass = not status.empty and (status["status"] == "PASS").all()
        cards = [MetricCard(self.query.registry.label(name, self.language), _last(frame, name)) for name in names]
        cards.append(MetricCard(tr("section_reconciliation_status", self.language), tr("status_pass" if all_pass else "status_fail", self.language)))
        self.body.addWidget(self.heading("section_current_snapshot"))
        self.body.addWidget(card_grid(cards, 5))
        flows = ChartCanvas()
        flows.plot_lines(frame, ["household_income", "household_consumption", "fixed_investment"], tr("chart_household_flows", self.language), self.query.registry, self.language)
        stocks = ChartCanvas()
        stocks.plot_lines(frame, ["household_cash", "aggregate_firm_cash"], tr("chart_cash_stocks", self.language), self.query.registry, self.language)
        self.body.addWidget(self.charts(flows, stocks))


class PopulationPage(BasePage):
    title_key = "page_population"
    subtitle_key = "subtitle_population"

    def build(self):
        frame = self.query.get_population_series(start=self.state.start, end=self.state.end)
        households = self.query.get_social_household_series(self.state.start, self.state.end)
        cards = [
            MetricCard(self.query.registry.label(name, self.language), _last(frame, name))
            for name in ("population", "births", "deaths", "marriages")
        ]
        cards.extend([
            MetricCard(self.query.registry.label("social_households", self.language), _last(households, "social_households")),
            MetricCard(self.query.registry.label("settlement_accounts", self.language), _last(households, "settlement_accounts")),
        ])
        population = ChartCanvas()
        population.plot_lines(frame, ["population"], tr("chart_population_level", self.language), self.query.registry, self.language)
        events = self.query.get_demographic_events(self.state.start, self.state.end)
        columns = [column for column in ("global_step", "event_type", "person_id", "child_id", "mother_id", "father_id", "household_id") if column in events]
        overview = self.panel(
            card_grid(cards, 6),
            population,
            self.heading("table_demographic_events"),
            DataTable(events[columns].tail(100), self.language),
        )

        exact_structure = self.query.get_exact_population_structure(self.state.end)
        structure = exact_structure if not exact_structure.empty else self.query.get_population_structure(self.state.end)
        selected_week = int(structure.global_step.iloc[0]) if not structure.empty else self.state.end
        pyramid = ChartCanvas()
        if not exact_structure.empty:
            pyramid.plot_age_sex_pyramid(
                exact_structure,
                tr("chart_exact_population_pyramid", self.language, week=selected_week),
                self.language,
            )
            pyramid_panel = self.panel(
                Notice(tr("notice_population_exact", self.language)),
                pyramid,
                DataTable(exact_structure, self.language),
            )
        else:
            pyramid.plot_population_pyramid(
                structure.get("age_group", pd.Series(dtype=str)).tolist(),
                pd.to_numeric(structure.get("count"), errors="coerce").fillna(0).tolist(),
                tr("chart_population_pyramid_week", self.language, week=selected_week),
            )
            pyramid_panel = self.panel(
                Notice(tr("notice_population_resolution", self.language)),
                Notice(tr("notice_population_no_sex", self.language), "warning"),
                pyramid,
                DataTable(structure, self.language),
            )

        ages = ChartCanvas()
        ages.plot_lines(frame, ["age_0_19", "age_20_39", "age_40_64", "age_65_plus"], tr("chart_age_groups", self.language), self.query.registry, self.language, stacked=True)
        exact_ratios = self.query.get_exact_age_ratio_series(self.state.start, self.state.end)
        ratios = exact_ratios if not exact_ratios.empty else self.query.get_demographic_ratio_series(self.state.start, self.state.end)
        ratio_chart = ChartCanvas()
        ratio_chart.plot_lines(
            ratios,
            ["child_share", "working_age_share", "elderly_share", "dependency_ratio"],
            tr("chart_demographic_ratios", self.language), self.query.registry, self.language,
        )
        ratio_cards = [
            MetricCard(self.query.registry.label(name, self.language), _last(ratios, name))
            for name in ("child_share", "working_age_share", "elderly_share", "dependency_ratio", "old_age_dependency_ratio", "youth_dependency_ratio")
        ]
        age_panel = self.panel(
            Notice(tr("notice_age_groups", self.language)),
            card_grid(ratio_cards, 6),
            self.charts(ages, ratio_chart),
        )
        self.tab_widget = self.tabs([
            ("tab_overview", overview),
            ("tab_population_pyramid", pyramid_panel),
            ("tab_age_structure", age_panel),
        ])
        self.body.addWidget(self.tab_widget)


class HouseholdPage(BasePage):
    title_key = "page_household"
    subtitle_key = "subtitle_household"

    def build(self):
        frame = self.query.get_social_household_series(self.state.start, self.state.end)
        cash = self.query.get_macro_series(["household_cash"], self.state.start, self.state.end)
        cards = [MetricCard(self.query.registry.label(name, self.language), _last(frame, name)) for name in ("social_households", "settlement_accounts", "economic_accounts")]
        cards.append(MetricCard(self.query.registry.label("household_cash", self.language), _last(cash, "household_cash")))
        chart = ChartCanvas()
        chart.plot_lines(frame, ["social_households", "settlement_accounts"], tr("chart_social_accounts", self.language), self.query.registry, self.language)
        cash_chart = ChartCanvas()
        cash_chart.plot_lines(cash, ["household_cash"], tr("chart_cash_stocks", self.language), self.query.registry, self.language)
        overview = self.panel(
            card_grid(cards, 4),
            self.charts(chart, cash_chart),
            Notice(tr("notice_household_scope", self.language)),
        )

        availability = self.query.get_household_distribution_availability()

        def distribution_panel(metric_name, unavailable_key, include_lorenz=True):
            micro = self.query.get_household_micro_distribution(metric_name, self.state.end)
            if micro.empty:
                rows = availability[availability.metric == metric_name]
                return self.panel(
                    self.heading("section_household_distribution_scope"),
                    Notice(tr("notice_household_population_scope", self.language)),
                    Notice(tr(unavailable_key, self.language), "warning"),
                    DataTable(rows, self.language),
                )
            summary = self.query.get_household_distribution_summary(metric_name, self.state.end)
            history = self.query.get_household_distribution_history(metric_name, self.state.start, self.state.end)
            ecdf = self.query.get_household_ecdf(metric_name, self.state.end)
            histogram = ChartCanvas()
            histogram.plot_histogram(
                micro.value,
                tr("chart_distribution_histogram", self.language),
                self.query.registry.label(metric_name, self.language),
                self.language,
            )
            ecdf_chart = ChartCanvas()
            ecdf_chart.plot_ecdf(
                ecdf.value,
                ecdf.cumulative_share,
                tr("chart_distribution_ecdf", self.language),
                self.query.registry.label(metric_name, self.language),
                self.language,
            )
            widgets = [
                self.heading("section_household_distribution_scope"),
                Notice(tr("notice_household_statistics_ready", self.language)),
                self.charts(histogram, ecdf_chart),
            ]
            if include_lorenz:
                lorenz = self.query.get_household_lorenz(metric_name, self.state.end)
                lorenz_chart = ChartCanvas()
                lorenz_chart.plot_lorenz(
                    lorenz.population_share,
                    lorenz.value_share,
                    tr("chart_distribution_lorenz", self.language),
                    self.language,
                )
                gini_chart = ChartCanvas()
                gini_chart.plot_raw_lines(
                    history,
                    {"gini": self.query.registry.label(
                        "household_income_gini" if "income" in metric_name else "household_cash_gini",
                        self.language,
                    )},
                    tr("chart_gini_history", self.language),
                    self.language,
                )
                widgets.append(self.charts(lorenz_chart, gini_chart))
            else:
                quantile_chart = ChartCanvas()
                quantile_chart.plot_quantile_band(
                    history, "p10", "median", "p90",
                    tr("section_distribution_summary", self.language), self.language,
                )
                widgets.append(quantile_chart)
            widgets.extend([
                self.heading("section_distribution_summary"),
                DataTable(summary, self.language),
            ])
            return self.panel(*widgets)

        income = distribution_panel(
            "household_income_distribution", "notice_income_micro_unavailable", True,
        )
        cash_assets = distribution_panel(
            "household_cash_distribution", "notice_cash_micro_unavailable", True,
        )
        saving = distribution_panel(
            "household_saving_distribution", "notice_saving_micro_unavailable", False,
        )
        support = self.query.get_private_support_series(self.state.start, self.state.end)
        if support.empty:
            private_support = self.panel(
                self.heading("section_private_support"),
                Notice(tr("notice_private_support_unavailable", self.language), "warning"),
            )
        else:
            summary = self.query.get_private_support_summary()
            events = summary["events"]
            is_buffer_support = getattr(self.query.store, "dataset_kind", "") in {"half_week_buffer_support", "buffer_target_selection"}
            if is_buffer_support:
                treatment_events = events
                treatment_coverage = summary["coverage"]
            else:
                treatment_events = events[events["branch"].astype(str).eq("treatment")] if "branch" in events else events.iloc[0:0]
                if treatment_events.empty:
                    treatment_events = events.iloc[-1:]
                coverage = summary["coverage"]
                treatment_coverage = coverage[coverage["branch"].astype(str).eq("treatment")] if "branch" in coverage else coverage.iloc[0:0]
            child = summary["child_burden"]
            if is_buffer_support:
                selected_branch = "buffer_0_5" if getattr(self.query.store, "dataset_kind", "") == "half_week_buffer_support" else "buffer_0_25"
                buffer_support = support[support["branch"].eq(selected_branch)]
                selected_coverage_branch = "BUFFER_0_5" if getattr(self.query.store, "dataset_kind", "") == "half_week_buffer_support" else "BUFFER_0_25"
                buffer_coverage = treatment_coverage[treatment_coverage["branch"].astype(str).eq(selected_coverage_branch)]
                cards = [
                    MetricCard("Support branches", events["branch"].nunique() if "branch" in events else 0),
                    MetricCard("Selected closing buffer", _last(buffer_coverage, "recipient_buffer_050")),
                    MetricCard("Selected near-zero share", _last(buffer_support, "near_zero_share")),
                ]
            else:
                cards = [
                    MetricCard(self.query.registry.label("private_support_event_count", self.language), _last(treatment_events, "transfer_events")),
                    MetricCard(self.query.registry.label("private_support_total_value", self.language), _last(treatment_events, "total_value")),
                    MetricCard(self.query.registry.label("private_support_unique_payers", self.language), _last(treatment_events, "unique_payer_households")),
                    MetricCard(self.query.registry.label("private_support_unique_recipients", self.language), _last(treatment_events, "unique_recipient_households")),
                    MetricCard(self.query.registry.label("private_support_recipient_coverage", self.language), _last(treatment_coverage, "realized_support_share_of_observation_households")),
                    MetricCard(self.query.registry.label("private_support_child_below_share", self.language), _last(child, "below_share_observed")),
                ]
            labels = _support_branch_labels(self.query, self.language)
            liquidity = support.pivot(index="global_step", columns="branch", values="near_zero_share").reset_index()
            elderly = support.pivot(index="global_step", columns="branch", values="elderly_near_zero_share").reset_index()
            for frame in (liquidity, elderly):
                frame.columns.name = None
            liquidity_chart = ChartCanvas()
            liquidity_chart.plot_raw_lines(
                liquidity,
                labels,
                tr("chart_private_support_liquidity", self.language), self.language,
            )
            elderly_chart = ChartCanvas()
            elderly_chart.plot_raw_lines(
                elderly,
                labels,
                tr("chart_private_support_liquidity", self.language) + " / elderly", self.language,
            )
            private_support = self.panel(
                self.heading("section_private_support"),
                Notice(tr("notice_private_support_read_only", self.language)),
                card_grid(cards, 6),
                self.charts(liquidity_chart, elderly_chart),
                DataTable(summary["distribution"], self.language),
                DataTable(summary["events"], self.language),
                DataTable(child, self.language),
            )
        tab_entries = [
            ("tab_overview", overview),
            ("tab_private_support", private_support),
            ("tab_income_inequality", income),
            ("tab_cash_assets_distribution", cash_assets),
            ("tab_saving_distribution", saving),
        ]
        liquidity_root = Path(self.query.store.run_dir)
        liquidity_path = liquidity_root / "household_liquidity_weekly_snapshot.csv"
        long_horizon_path = liquidity_root / "long_horizon_household_snapshots.csv"
        if liquidity_path.exists() or long_horizon_path.exists():
            tab_entries.append((
                "tab_household_liquidity",
                self.panel(
                    Notice("Step17 research snapshot: weekly aggregate liquidity with exact available Household snapshot weeks."),
                    HouseholdLiquidityExplorer(self.query.store.run_dir, self.language),
                ),
            ))
        self.tab_widget = self.tabs(tab_entries)
        self.body.addWidget(self.tab_widget)


class LaborPage(BasePage):
    title_key = "page_labor"
    subtitle_key = "subtitle_labor"

    def build(self):
        frame = self.query.get_labor_series(self.state.start, self.state.end)
        quality = self.query.get_labor_quality_series(self.state.start, self.state.end)
        cards = [
            MetricCard(self.query.registry.label("working_age_count", self.language), _last(quality, "working_age_count")),
            MetricCard(
                self.query.registry.label("settlement_valid_person_count", self.language),
                _last(quality, "settlement_valid_person_count")
                if "settlement_valid_person_count" in quality else tr("label_not_persisted", self.language),
            ),
            *[MetricCard(self.query.registry.label(name, self.language), _last(frame, name)) for name in ("eligible_labor", "employment", "unassigned_labor", "food_employment", "capital_good_employment")],
        ]
        self.body.addWidget(card_grid(cards, 5))
        levels = ChartCanvas()
        levels.plot_lines(frame, ["eligible_labor", "employment", "unassigned_labor"], tr("chart_labor_levels", self.language), self.query.registry, self.language)
        sectors = ChartCanvas()
        sectors.plot_lines(frame, ["food_employment", "capital_good_employment"], tr("chart_sector_employment", self.language), self.query.registry, self.language)
        self.body.addWidget(self.charts(levels, sectors))
        quality_chart = ChartCanvas()
        quality_chart.plot_lines(
            quality,
            ["labor_eligibility_rate", "employment_rate", "unassigned_eligible_rate"],
            tr("chart_labor_quality", self.language), self.query.registry, self.language,
        )
        self.body.addWidget(self.heading("section_labor_quality"))
        self.body.addWidget(quality_chart)
        events = self.query.get_labor_event_series(self.state.start, self.state.end)
        if events.empty:
            self.body.addWidget(Notice(tr("notice_labor_transition_limit", self.language), "warning"))
        else:
            event_chart = ChartCanvas()
            event_chart.plot_raw_lines(
                events,
                {
                    "hire": "招聘" if self.language == "zh" else "Hires",
                    "release": "释放" if self.language == "zh" else "Releases",
                    "eligibility_entry": "资格进入" if self.language == "zh" else "Eligibility Entries",
                    "eligibility_exit": "资格退出" if self.language == "zh" else "Eligibility Exits",
                },
                tr("chart_labor_events", self.language),
                self.language,
            )
            self.body.addWidget(self.heading("section_labor_events"))
            self.body.addWidget(Notice(tr("notice_labor_events_ready", self.language)))
            self.body.addWidget(event_chart)
            self.body.addWidget(DataTable(self.query.get_labor_events(self.state.start, self.state.end).tail(100), self.language))


class MacroPage(BasePage):
    title_key = "page_macro"
    subtitle_key = "subtitle_macro"

    def build(self):
        names = ["household_income", "household_consumption", "household_saving", "household_cash", "aggregate_firm_cash", "legacy_owner_cash", "estate_cash", "money_stock"]
        frame = self.query.get_macro_series(names, self.state.start, self.state.end)
        flows = ChartCanvas()
        flows.plot_lines(frame, names[:3], tr("chart_macro_flows", self.language), self.query.registry, self.language)
        stocks = ChartCanvas()
        stocks.plot_lines(frame, names[3:], tr("chart_macro_stocks", self.language), self.query.registry, self.language)
        self.body.addWidget(self.charts(flows, stocks))
        sector_cash = self.query.get_sector_cash_series(self.state.start, self.state.end)
        cash_stocks = ChartCanvas()
        cash_stocks.plot_lines(
            sector_cash,
            ["household_cash", "food_firm_cash", "capital_good_firm_cash", "legacy_owner_cash", "estate_cash"],
            tr("chart_sector_cash_stocks", self.language), self.query.registry, self.language,
        )
        cash_changes = ChartCanvas()
        cash_changes.plot_lines(
            sector_cash,
            ["household_cash_change", "food_firm_cash_change", "capital_good_firm_cash_change", "legacy_owner_cash_change", "estate_cash_change"],
            tr("chart_sector_cash_changes", self.language), self.query.registry, self.language,
        )
        self.body.addWidget(self.heading("section_sector_cash"))
        self.body.addWidget(self.charts(cash_stocks, cash_changes))
        support = self.query.get_private_support_series(self.state.start, self.state.end)
        if support.empty:
            self.body.addWidget(self.heading("section_private_support"))
            self.body.addWidget(Notice(tr("notice_private_support_unavailable", self.language), "warning"))
        else:
            support_summary = self.query.get_private_support_summary()
            is_buffer_support = getattr(self.query.store, "dataset_kind", "") in {"half_week_buffer_support", "buffer_target_selection"}
            labels = _support_branch_labels(self.query, self.language)
            if is_buffer_support:
                branch = "buffer_0_5" if getattr(self.query.store, "dataset_kind", "") == "half_week_buffer_support" else "buffer_0_25"
                event_branch = "BUFFER_0_5" if getattr(self.query.store, "dataset_kind", "") == "half_week_buffer_support" else "BUFFER_0_25"
                event_row = support_summary["events"][support_summary["events"]["branch"].astype(str).eq(event_branch)].tail(1)
            else:
                branch = "treatment"
                event_row = support_summary["events"]
                event_row = event_row[event_row["branch"].astype(str).eq("treatment")] if "branch" in event_row else event_row
            food = support.pivot(index="global_step", columns="branch", values="food_sales").reset_index()
            firm_cash = support.pivot(index="global_step", columns="branch", values="aggregate_firm_cash").reset_index()
            for frame in (food, firm_cash):
                frame.columns.name = None
            food_chart = ChartCanvas()
            food_chart.plot_raw_lines(food, labels, tr("chart_private_support_food", self.language), self.language)
            firm_chart = ChartCanvas()
            firm_chart.plot_raw_lines(firm_cash, labels, tr("chart_private_support_food", self.language) + " / Firm cash", self.language)
            if is_buffer_support:
                cards = [
                    MetricCard("Buffer transfer events", _last(event_row, "transfer_events")),
                    MetricCard(self.query.registry.label("private_support_food_sales", self.language), _last(support[support["branch"].eq(branch)], "food_sales")),
                    MetricCard(self.query.registry.label("private_support_aggregate_firm_cash", self.language), _last(support[support["branch"].eq(branch)], "aggregate_firm_cash")),
                ]
            else:
                cards = [
                    MetricCard(self.query.registry.label("private_support_total_value", self.language), _last(event_row, "total_value")),
                    MetricCard(self.query.registry.label("private_support_food_sales", self.language), _last(support[support["branch"].eq(branch)], "food_sales")),
                    MetricCard(self.query.registry.label("private_support_aggregate_firm_cash", self.language), _last(support[support["branch"].eq(branch)], "aggregate_firm_cash")),
                ]
            self.body.addWidget(self.heading("section_private_support"))
            self.body.addWidget(Notice(tr("notice_private_support_read_only", self.language)))
            self.body.addWidget(card_grid(cards, 3))
            self.body.addWidget(self.charts(food_chart, firm_chart))
        self.body.addWidget(self.heading("section_boundary"))
        self.body.addWidget(Notice(tr("notice_boundary", self.language), "warning"))


class SectorPage(BasePage):
    title_key = "page_sector"
    subtitle_key = "subtitle_sector"

    def build(self):
        frame = self.query.get_sector_series(self.state.sector, self.state.start, self.state.end)
        renamed = frame.rename(columns={"revenue": "firm_revenue", "operating_profit": "firm_profit", "cash": "aggregate_firm_cash", "principal": "loan_principal", "customer_advance_liability": "customer_advance_liability"})
        operations = ChartCanvas()
        operations.plot_lines(renamed, ["firm_revenue", "firm_profit"], tr("chart_firm_operations", self.language), self.query.registry, self.language)
        employment = ChartCanvas()
        employment.plot_lines(renamed.rename(columns={"employment": "employment"}), ["employment"], tr("chart_sector_employment", self.language), self.query.registry, self.language)
        self.body.addWidget(self.charts(operations, employment))
        stocks = ChartCanvas()
        stocks.plot_lines(renamed, ["aggregate_firm_cash", "loan_principal", "customer_advance_liability"], tr("chart_firm_balance", self.language), self.query.registry, self.language)
        self.body.addWidget(stocks)


class FirmPage(BasePage):
    title_key = "page_firm"
    subtitle_key = "subtitle_firm"

    def build(self):
        frame = self.query.get_firm_series(self.state.firm_id, self.state.start, self.state.end)
        renamed = frame.rename(columns={"revenue": "firm_revenue", "operating_profit": "firm_profit", "cfo": "firm_cfo", "cash": "firm_cash", "principal": "loan_principal", "inventory": "firm_inventory"})
        cards = [MetricCard(self.query.registry.label(name, self.language), _last(renamed, name)) for name in ("firm_cash", "firm_inventory", "firm_revenue", "firm_profit", "loan_principal", "customer_advance_liability", "prepaid_investment_asset")]
        operations = ChartCanvas()
        operations.plot_lines(renamed, ["firm_revenue", "firm_profit", "firm_cfo"], tr("chart_firm_operations", self.language), self.query.registry, self.language)
        cash = ChartCanvas()
        cash.plot_lines(renamed, ["firm_cash"], tr("chart_cash_stocks", self.language), self.query.registry, self.language)
        liabilities = ChartCanvas()
        liabilities.plot_lines(renamed, ["loan_principal", "customer_advance_liability", "prepaid_investment_asset"], tr("chart_firm_balance", self.language), self.query.registry, self.language)
        single = self.panel(
            self.heading("section_firm_snapshot"),
            card_grid(cards, 4),
            self.charts(operations, cash),
            liabilities,
        )

        sector = self.query.resolve_comparison_sector(self.state.sector, self.state.firm_id)
        cross = self.query.get_firm_cross_section(sector, self.state.end, self.state.firm_id)
        indexed = self.query.get_firm_indexed_series(sector, "sales", self.state.start, self.state.end)
        multiples = ChartCanvas()
        multiples.plot_small_multiples(
            indexed, "firm_id", "indexed_value", tr("chart_firm_indexed_sales", self.language),
            tr("firm", self.language), self.language,
        )
        ranked = ChartCanvas()
        ranked.plot_ranked_bars(
            [f"{tr('firm', self.language)} {firm_id}" for firm_id in cross.get("firm_id", pd.Series(dtype=str))],
            pd.to_numeric(cross.get("sales"), errors="coerce").fillna(0).tolist(),
            tr("chart_ranked_sales", self.language),
        )
        dispersion = self.query.get_firm_dispersion(sector, self.state.end)
        comparison_widgets = [
            Notice(tr("notice_market_share_denominator", self.language)),
            self.charts(multiples, ranked),
            self.heading("section_firm_health"),
            DataTable(cross, self.language),
        ]
        capital_history = self.query.get_firm_capital_history(
            sector=sector, start=self.state.start, end=self.state.end,
        )
        if capital_history.empty:
            comparison_widgets.append(Notice(tr("notice_firm_service_unavailable", self.language), "warning"))
        else:
            capital_chart = ChartCanvas()
            capital_chart.plot_small_multiples(
                capital_history,
                "firm_id",
                "active_capital_service",
                tr("chart_firm_capital_service", self.language),
                tr("firm", self.language),
                self.language,
            )
            comparison_widgets.extend([
                self.heading("section_firm_capital_history"),
                Notice(tr("notice_firm_capital_ready", self.language)),
                capital_chart,
                DataTable(self.query.get_firm_capital_comparison(self.state.end, sector), self.language),
            ])
        comparison_widgets.extend([
            self.heading("section_dispersion"),
            DataTable(dispersion, self.language),
        ])
        comparison = self.panel(*comparison_widgets)

        shares = self.query.get_firm_share_series(sector, "sales", self.state.start, self.state.end)
        structure = self.query.get_market_structure_series(sector, "sales", self.state.start, self.state.end)
        if not structure.empty and int(structure.firm_count.max()) == 1:
            market = self.panel(
                Notice(tr("notice_single_supplier", self.language), "warning"),
                DataTable(structure.tail(20), self.language),
            )
        else:
            share_chart = ChartCanvas()
            share_chart.plot_share_area(
                shares, "firm_id", "market_share", tr("chart_firm_market_share", self.language),
                tr("firm", self.language), self.language,
            )
            concentration = ChartCanvas()
            concentration.plot_lines(
                structure, ["sales_hhi", "largest_firm_share", "top2_share"],
                tr("chart_market_concentration", self.language), self.query.registry, self.language,
            )
            market = self.panel(
                Notice(tr("notice_market_share_denominator", self.language)),
                self.charts(share_chart, concentration),
                self.heading("section_market_concentration"),
                DataTable(structure.tail(100), self.language),
            )
        self.tab_widget = self.tabs([
            ("tab_single_firm", single),
            ("tab_cross_firm", comparison),
            ("tab_market_structure", market),
        ])
        self.body.addWidget(self.tab_widget)


class AccountingPage(BasePage):
    title_key = "page_accounting"
    subtitle_key = "subtitle_accounting"

    def build(self):
        balance = self.query.get_balance_sheet(self.state.firm_id, self.state.end)
        cash = self.query.get_cash_bridge(self.state.firm_id, self.state.start, self.state.end)
        localized = balance.copy()
        component_map = {
            "cash": self.query.registry.label("firm_cash", self.language),
            "inventory_book_value": self.query.registry.label("firm_inventory", self.language),
            "prepaid_investment_asset": self.query.registry.label("prepaid_investment_asset", self.language),
            "capital_book_value": self.query.registry.label("capital_book_value", self.language),
            "loan_principal": self.query.registry.label("loan_principal", self.language),
            "interest_arrears": self.query.registry.label("interest_arrears", self.language),
            "customer_advance_liability": self.query.registry.label("customer_advance_liability", self.language),
            "equity": self.query.registry.label("firm_equity", self.language),
        }
        localized["component"] = localized["component"].map(component_map).fillna(localized["component"])
        localized["classification"] = localized["classification"].map({
            "asset": tr("assets", self.language),
            "liability": tr("liabilities", self.language),
            "equity": tr("equity", self.language),
        }).fillna(localized["classification"])
        self.body.addWidget(self.heading("table_balance_sheet"))
        self.body.addWidget(DataTable(localized, self.language))
        balance_chart = ChartCanvas()
        balance_chart.plot_bars(localized["component"].tolist(), pd.to_numeric(localized["value"], errors="coerce").fillna(0).tolist(), tr("chart_balance_sheet", self.language), horizontal=True)
        if cash.empty:
            bridge_values = [0, 0, 0, 0]
        else:
            bridge_values = [cash.cash_start.iloc[0], cash.sales_collections.sum() + cash.customer_advance_cash_inflow.sum(), -(cash.wage_payments.sum() + cash.prepaid_investment_cash_outflow.sum()), cash.cash_end.iloc[-1]]
        bridge_chart = ChartCanvas()
        bridge_chart.plot_bars([tr("opening_cash", self.language), tr("cash_inflows", self.language), tr("cash_outflows", self.language), tr("closing_cash", self.language)], bridge_values, tr("chart_cash_bridge", self.language))
        self.body.addWidget(self.charts(balance_chart, bridge_chart))


class CapitalPage(BasePage):
    title_key = "page_capital"
    subtitle_key = "subtitle_capital"

    def build(self):
        names = ["fixed_investment", "expansion_investment", "replacement_investment", "capital_good_production", "active_capital_assets", "active_capital_service", "depreciation"]
        frame = self.query.get_macro_series(names, self.state.start, self.state.end)
        flows = ChartCanvas()
        flows.plot_lines(frame, names[:4], tr("chart_investment_flows", self.language), self.query.registry, self.language)
        stocks = ChartCanvas()
        stocks.plot_lines(frame, ["active_capital_assets", "active_capital_service"], tr("chart_capital_stocks", self.language), self.query.registry, self.language)
        overview_widgets = [self.charts(flows, stocks)]
        availability = self.query.get_availability("capital_backlog")
        if availability == "NOT_AVAILABLE":
            overview_widgets.append(Notice(tr("availability_not_available", self.language) + ": " + self.query.registry.label("capital_backlog", self.language), "warning"))
        assets = self.query.get_capital_assets(self.state.firm_id, start=self.state.start, end=self.state.end)
        columns = [column for column in ("asset_id", "owner_firm_id", "origin_order_id", "investment_source", "acquisition_week", "acquisition_cost", "quantity", "active", "accumulated_depreciation", "closing_book_value", "retirement_week") if column in assets]
        overview_widgets.extend([self.heading("table_assets"), DataTable(assets[columns].tail(250), self.language)])
        overview = self.panel(*overview_widgets)

        pipeline, summary = self.query.get_capital_pipeline(self.state.start, self.state.end)
        pipeline_cards = [
            MetricCard(self.query.registry.label("investment_order_count", self.language), summary["order_count"]),
            MetricCard(self.query.registry.label("average_order_size", self.language), summary["average_order_size"]),
            MetricCard(self.query.registry.label("delivery_volume", self.language), summary["delivery_volume"]),
            MetricCard(self.query.registry.label("customer_advance_liability", self.language), summary["outstanding_advances"]),
            MetricCard(self.query.registry.label("active_capital_assets", self.language), summary["active_assets"]),
            MetricCard(self.query.registry.label("active_capital_service", self.language), summary["active_capital_service"]),
        ]
        stage_labels = [str(value_label("pipeline_stage", value, self.language)) for value in pipeline.pipeline_stage]
        event_chart = ChartCanvas()
        event_chart.plot_bars(stage_labels, pipeline.event_count.tolist(), tr("chart_pipeline_event_counts", self.language))
        unit_chart = ChartCanvas()
        unit_chart.plot_bars(stage_labels, pd.to_numeric(pipeline.physical_units, errors="coerce").fillna(0).tolist(), tr("chart_pipeline_physical_units", self.language))
        lags = self.query.get_order_delivery_lags(self.state.start, self.state.end)
        lag_chart = ChartCanvas()
        lag_chart.plot_histogram(lags.get("delivery_lag_weeks", []), tr("chart_delivery_lag", self.language), tr("section_delivery_lag", self.language), self.language)
        pipeline_panel = self.panel(
            card_grid(pipeline_cards, 6),
            self.charts(event_chart, unit_chart),
            Notice(tr("notice_pipeline_link_coverage", self.language)),
            lag_chart,
            DataTable(lags.tail(250), self.language),
        )

        cohorts = self.query.get_capital_asset_cohorts()
        cohort_labels = [str(value) for value in cohorts.get("acquisition_cohort_year", [])]
        cohort_chart = ChartCanvas()
        cohort_chart.plot_cohort(
            cohort_labels,
            cohorts.get("asset_count", pd.Series(dtype=float)).tolist(),
            cohorts.get("active_asset_count", pd.Series(dtype=float)).tolist(),
            cohorts.get("retirement_count", pd.Series(dtype=float)).tolist(),
            tr("chart_asset_cohorts", self.language), self.language,
        )
        cohort_cost = ChartCanvas()
        cohort_cost.plot_grouped_bars(
            cohort_labels,
            [
                (
                    tr("label_acquisition_cost", self.language),
                    pd.to_numeric(cohorts.get("acquisition_cost"), errors="coerce").fillna(0).tolist(),
                ),
                (
                    tr("label_accumulated_depreciation", self.language),
                    pd.to_numeric(cohorts.get("accumulated_depreciation"), errors="coerce").fillna(0).tolist(),
                ),
            ],
            tr("chart_cohort_cost", self.language),
        )
        cohort_panel = self.panel(self.charts(cohort_chart, cohort_cost), DataTable(cohorts, self.language))
        self.tab_widget = self.tabs([
            ("tab_overview", overview),
            ("tab_capital_pipeline", pipeline_panel),
            ("tab_asset_cohorts", cohort_panel),
        ])
        self.body.addWidget(self.tab_widget)


class ContractPage(BasePage):
    title_key = "page_contract"
    subtitle_key = "subtitle_contract"

    def build(self):
        self.body.addWidget(Notice(tr("notice_no_transaction_id", self.language), "warning"))
        frame = self.query.get_investment_chain(self.state.firm_id, self.state.event_type, self.state.start, self.state.end)
        columns = [column for column in ("global_step", "event_type", "order_id", "buyer_firm_id", "supplier_firm_id", "asset_id", "investment_source", "physical_units", "cash_amount", "settled_expenditure", "capital_asset_created") if column in frame]
        self.body.addWidget(self.heading("table_contracts"))
        self.body.addWidget(DataTable(frame[columns].tail(500), self.language))


class ReconciliationPage(BasePage):
    title_key = "page_reconciliation"
    subtitle_key = "subtitle_reconciliation"

    def build(self):
        status = self.query.reconciliation_status(self.state.start, self.state.end)
        localized = status.copy()
        localized["metric"] = localized["metric"].map(lambda name: self.query.registry.label(name, self.language))
        localized["status"] = localized["status"].map(lambda value: tr("status_pass" if value == "PASS" else "status_fail", self.language))
        cards = [MetricCard(row.metric, row.max_abs, row.status) for row in localized.itertuples()]
        self.body.addWidget(card_grid(cards, 5))
        self.body.addWidget(self.heading("section_reconciliation_status"))
        self.body.addWidget(DataTable(localized, self.language))
        frame = self.query.get_reconciliation(self.state.start, self.state.end)
        stock_chart = ChartCanvas()
        stock_chart.plot_lines(frame, ["money_location_gap", "full_money_location_gap", "goods_gap", "advance_prepaid_gap", "feasible_capacity_gap"], tr("chart_reconciliation", self.language), self.query.registry, self.language)
        self.body.addWidget(stock_chart)
        flow_chart = ChartCanvas()
        flow_chart.plot_lines(frame, ["money_delta_gap"], "Money flow / timing diagnostic", self.query.registry, self.language)
        self.body.addWidget(flow_chart)


class ModelDiagnosticsPage(BasePage):
    title_key = "page_diagnostics"
    subtitle_key = "subtitle_diagnostics"

    def build(self):
        population = self.query.get_population_series(start=self.state.start, end=self.state.end)
        ratios = self.query.get_demographic_ratio_series(self.state.start, self.state.end)
        labor = self.query.get_labor_quality_series(self.state.start, self.state.end)
        market = self.query.get_market_structure_series("food", "sales", self.state.start, self.state.end)
        capital = self.query.get_macro_series(
            ["fixed_investment", "active_capital_assets", "active_capital_service", "depreciation"],
            self.state.start, self.state.end,
        )
        reconciliation = self.query.reconciliation_status(self.state.start, self.state.end)
        growth = float("nan")
        if len(population) >= 2 and float(population.population.iloc[0]) != 0:
            growth = float(population.population.iloc[-1] / population.population.iloc[0] - 1)
        reconciliation_pass = not reconciliation.empty and (reconciliation.status == "PASS").all()
        cards = [
            MetricCard(self.query.registry.label("population_growth", self.language), growth),
            MetricCard(self.query.registry.label("dependency_ratio", self.language), _last(ratios, "dependency_ratio")),
            MetricCard(self.query.registry.label("household_income_gini", self.language), tr("label_not_supported", self.language)),
            MetricCard(self.query.registry.label("household_cash_gini", self.language), tr("label_not_supported", self.language)),
            MetricCard(self.query.registry.label("employment_rate", self.language), _last(labor, "employment_rate")),
            MetricCard(self.query.registry.label("unassigned_eligible_rate", self.language), _last(labor, "unassigned_eligible_rate")),
            MetricCard(self.query.registry.label("sales_hhi", self.language), _last(market, "sales_hhi")),
            MetricCard(self.query.registry.label("cross_firm_cv", self.language), _last(market, "cross_firm_cv")),
            MetricCard(self.query.registry.label("fixed_investment", self.language), pd.to_numeric(capital.fixed_investment, errors="coerce").sum()),
            MetricCard(self.query.registry.label("active_capital_service", self.language), _last(capital, "active_capital_service")),
            MetricCard(tr("section_reconciliation_status", self.language), tr("status_pass" if reconciliation_pass else "status_fail", self.language)),
        ]
        self.body.addWidget(Notice(tr("notice_model_no_targets", self.language)))
        self.body.addWidget(card_grid(cards, 4))

        demography_chart = ChartCanvas()
        demography_chart.plot_lines(
            ratios, ["child_share", "working_age_share", "elderly_share", "dependency_ratio"],
            tr("chart_model_demography", self.language), self.query.registry, self.language,
        )
        labor_chart = ChartCanvas()
        labor_chart.plot_lines(
            labor, ["eligible_population_share", "employment_rate", "unassigned_eligible_rate"],
            tr("chart_model_labor", self.language), self.query.registry, self.language,
        )
        firm_chart = ChartCanvas()
        firm_chart.plot_lines(
            market, ["sales_hhi", "largest_firm_share", "cross_firm_cv"],
            tr("chart_model_firms", self.language), self.query.registry, self.language,
        )
        capital_chart = ChartCanvas()
        capital_chart.plot_lines(
            capital, ["fixed_investment", "depreciation"],
            tr("chart_model_capital", self.language), self.query.registry, self.language,
        )
        self.body.addWidget(self.charts(demography_chart, labor_chart))
        self.body.addWidget(self.charts(firm_chart, capital_chart))
        self.body.addWidget(Notice(tr("notice_income_micro_unavailable", self.language), "warning"))
        self.body.addWidget(Notice(tr("notice_boundary", self.language), "warning"))


class ReportsPage(BasePage):
    title_key = "page_reports"
    subtitle_key = "subtitle_reports"

    def build(self):
        frame = self.query.get_availability_rows()
        frame["availability"] = frame["availability"].map({
            "AUTHORITATIVE_AND_PERSISTED": tr("availability_persisted", self.language),
            "DERIVABLE_FROM_PERSISTED_AUTHORITATIVE_DATA": tr("availability_derivable", self.language),
            "AUTHORITATIVE_RUNTIME_ONLY": tr("availability_runtime", self.language),
            "NOT_AVAILABLE": tr("availability_not_available", self.language),
        })
        frame["localized_definition"] = frame["definition"] if self.language == "zh" else frame["definition_en"]
        columns = ["internal_name", "display_name_zh" if self.language == "zh" else "display_name_en", "localized_definition", "unit", "kind", "source_dataset", "source_field", "availability", "visualization_type"]
        self.body.addWidget(self.heading("section_registry"))
        self.body.addWidget(DataTable(frame[columns], self.language))
        statistical = self.query.get_statistical_registry_rows()
        self.body.addWidget(self.heading("section_statistical_registry"))
        self.body.addWidget(DataTable(statistical, self.language))
        export_button = QPushButton(tr("export", self.language))
        export_button.clicked.connect(lambda: self._export(frame))
        self.body.addWidget(export_button)
        self.body.addWidget(self.heading("section_boundary"))
        self.body.addWidget(Notice(tr("notice_boundary", self.language), "warning"))

    def _export(self, frame):
        filename, _ = QFileDialog.getSaveFileName(
            self, tr("export", self.language), "analysis_v2_metrics.csv",
            tr("export_csv_filter", self.language),
        )
        if filename:
            frame.to_csv(filename, index=False, encoding="utf-8-sig")


PAGE_CLASSES = [
    OverviewPage, PopulationPage, HouseholdPage, LaborPage, MacroPage, SectorPage,
    FirmPage, AccountingPage, CapitalPage, ContractPage, ReconciliationPage,
    ModelDiagnosticsPage, ReportsPage,
]


__all__ = ["PAGE_CLASSES", "BasePage"]
