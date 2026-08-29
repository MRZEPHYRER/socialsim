import csv
import json
import math
import os
import random
from collections import defaultdict
from household import Household
from productivity import age_productivity
from person import Person
from marriage import MarriageSystem
from household_manager import HouseholdManager
from economy.income import IncomeSystem
from economy.private_family_support import PrivateFamilySupportSystem
from economy.social_insurance import PaygPensionSystem, SocialInsuranceFund
from economy.public_budget import GovernmentPublicBudget, collect_firm_tax
from economy.income_tax import settle_personal_income_tax, year_end_reconcile
from economy.intergenerational_wealth_transfer import IntergenerationalWealthTransferSystem
from economy.inheritance import InheritanceSystem
from economy import config as economy_config
from central_bank import config as central_bank_config
from economy.goods import GoodSpec, GoodsCatalog
from economy.ledger import Ledger
from economy.ledger import object_account
from economy.needs import NeedsSystem
from economy.consumption import ConsumptionSystem
from economy.household_demand import HouseholdDemandSystem
from economy.firm import FirmSystem
from economy.multi_firm import safe_ratio
from economy.multi_firm import split_single_firm
from economy.multi_firm import ensure_single_firm_view
from economy.multi_firm import allocate_inventory_constrained_purchase
from economy.interest import settle_interest
from economy.interest import annual_to_weekly_rate
from economy.accounting import AccountingLayer
from economy.dividend_routing import ensure_dividend_routing_state, route_declared_dividend
from economy.equity_transition import GradualPersonEquityTransition
from economy.autonomous_secondary_equity import (
    AutonomousSecondaryEquityPurchaseSystem,
)
from economy.ownership_accounting import (
    CapTable,
    EquityOwnershipSystem,
    HouseholdBalanceSheetView,
)
from economy.shareholder_estate import ShareholderEstateSystem
from economy.founder_ownership import assign_bootstrap_founders as assign_founders
from economy.endogenous_firm_entry import EndogenousCapitalGoodsEntrySystem
from economy.operating_contracts import StageAOperatingContractAdapter
from economy.multisector import MultiSectorFoundation
from economy.multisector import default_firm_identity
from economy.default_bookkeeping import (
    ensure_default_state,
    update_firm_default_bookkeeping,
)
from economy.restructuring import (
    begin_restructuring_week,
    ensure_restructuring_state,
    finalize_restructuring_week,
)
from fertility import FertilitySystem
from time_system import SimulationClock
from time_system import MARRIAGE_MARKET_INTERVAL_WEEKS

class World:


    def __init__(
        self,
        initial_population=15000,
        base_resource=50000,
        consumption_per_person=5,
        seed=None,
        record_ledger_details=False,
        scenario_name="baseline",
        scenario_overrides=None,
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
        observability_mode="FULL_DIAGNOSTIC",
    ):

        # =========================
        # Parameters
        # =========================

        self.initial_population = initial_population

        self.seed = seed
        self.scenario_name = scenario_name
        self.scenario_overrides = scenario_overrides or {}
        self.person_founder_bootstrap_enabled = bool(
            self.scenario_overrides.get(
                "PERSON_FOUNDER_BOOTSTRAP_ENABLED",
                getattr(economy_config, "PERSON_FOUNDER_BOOTSTRAP_ENABLED", False),
            )
        )
        # This context is only for fresh World formation.
        self.new_world_formation_context = True
        self.founder_assignment_rows = []
        self.founder_assigned_firm_ids = set()
        self.diagnostics_mode = diagnostics_mode
        self.observability_mode = str(
            observability_mode or "FULL_DIAGNOSTIC"
        ).upper()
        self.diagnostic_micro_snapshot_cadence = 13
        self.private_family_support_enabled = bool(
            self.scenario_overrides.get(
                "PRIVATE_FAMILY_SUPPORT_ENABLED",
                getattr(economy_config, "PRIVATE_FAMILY_SUPPORT_ENABLED", False),
            )
        )
        # Step17.D research institutions remain disabled in the canonical model.
        self.payg_pension_enabled = bool(self.scenario_overrides.get("PAYG_PENSION_ENABLED", getattr(economy_config, "PAYG_PENSION_ENABLED", False)))
        self.payg_contribution_rate = float(self.scenario_overrides.get("PAYG_CONTRIBUTION_RATE", getattr(economy_config, "PAYG_CONTRIBUTION_RATE", 0.0)))
        self.payg_pension_target_multiplier = float(self.scenario_overrides.get("PAYG_PENSION_TARGET_MULTIPLIER", getattr(economy_config, "PAYG_PENSION_TARGET_MULTIPLIER", 0.0)))
        self.payg_employer_contribution_rate = float(self.scenario_overrides.get("PAYG_EMPLOYER_CONTRIBUTION_RATE", 0.0))
        self.public_revenue_tax_enabled = bool(self.scenario_overrides.get("PUBLIC_REVENUE_TAX_ENABLED", False))
        self.public_revenue_tax_base = str(self.scenario_overrides.get("PUBLIC_REVENUE_TAX_BASE", "FIRM_SALES_TAX"))
        self.public_revenue_tax_rate = float(self.scenario_overrides.get("PUBLIC_REVENUE_TAX_RATE", 0.0))
        self.personal_income_tax_enabled = bool(self.scenario_overrides.get("PERSONAL_INCOME_TAX_ENABLED", False))
        self.personal_income_tax_brackets = list(self.scenario_overrides.get("PERSONAL_INCOME_TAX_BRACKETS", []))
        self.corporate_profit_tax_enabled = bool(self.scenario_overrides.get("CORPORATE_PROFIT_TAX_ENABLED", False))
        self.corporate_profit_tax_rate = float(self.scenario_overrides.get("CORPORATE_PROFIT_TAX_RATE", 0.0))
        self.personal_tax_year_reconciliation = []
        self.retirement_runtime_enabled = bool(self.scenario_overrides.get("RETIREMENT_RUNTIME_ENABLED", False))
        self.retirement_age = float(self.scenario_overrides.get("RETIREMENT_AGE", 65.0))
        self.intergenerational_wealth_transfer_enabled = bool(self.scenario_overrides.get("INTERGENERATIONAL_WEALTH_TRANSFER_ENABLED", getattr(economy_config, "INTERGENERATIONAL_WEALTH_TRANSFER_ENABLED", False)))
        self.intergenerational_reserve_weeks = float(self.scenario_overrides.get("INTERGENERATIONAL_RESERVE_WEEKS", getattr(economy_config, "INTERGENERATIONAL_RESERVE_WEEKS", 13.0)))
        self.intergenerational_donor_surplus_share = float(self.scenario_overrides.get("INTERGENERATIONAL_DONOR_SURPLUS_SHARE", getattr(economy_config, "INTERGENERATIONAL_DONOR_SURPLUS_SHARE", 0.25)))
        self.intergenerational_recipient_target_weeks = float(self.scenario_overrides.get("INTERGENERATIONAL_RECIPIENT_TARGET_WEEKS", getattr(economy_config, "INTERGENERATIONAL_RECIPIENT_TARGET_WEEKS", 1.0)))
        self.family_link_observability_enabled = bool(
            self.scenario_overrides.get(
                "FAMILY_LINK_OBSERVABILITY_ENABLED",
                getattr(economy_config, "FAMILY_LINK_OBSERVABILITY_ENABLED", False),
            )
        )
        self.family_link_observability_cadence = max(
            1,
            int(
                self.scenario_overrides.get(
                    "FAMILY_LINK_OBSERVABILITY_CADENCE",
                    getattr(economy_config, "FAMILY_LINK_OBSERVABILITY_CADENCE", 13),
                )
            ),
        )
        self.family_link_observability_rows = []
        self.generalized_firm_operating_contracts = bool(
            self.scenario_overrides.get(
                "GENERALIZED_FIRM_OPERATING_CONTRACTS",
                getattr(
                    economy_config,
                    "GENERALIZED_FIRM_OPERATING_CONTRACTS",
                    False,
                ),
            )
        )
        self.multisector_foundation_enabled = bool(
            self.scenario_overrides.get(
                "MULTISECTOR_FOUNDATION_ENABLED",
                self.scenario_overrides.get(
                    "multisector_foundation_enabled",
                    getattr(
                        economy_config,
                        "MULTISECTOR_FOUNDATION_ENABLED",
                        False,
                    ),
                ),
            )
        )
        self.shadow_multigood_household_demand_enabled = bool(
            self.scenario_overrides.get(
                "SHADOW_MULTIGOOD_HOUSEHOLD_DEMAND_ENABLED",
                getattr(
                    economy_config,
                    "SHADOW_MULTIGOOD_HOUSEHOLD_DEMAND_ENABLED",
                    False,
                ),
            )
        )
        self.service_firm_runtime_enabled = bool(
            self.scenario_overrides.get(
                "SERVICE_FIRM_RUNTIME_ENABLED",
                getattr(economy_config, "SERVICE_FIRM_RUNTIME_ENABLED", False),
            )
        )
        self.canonical_investment_enabled = bool(
            self.scenario_overrides.get(
                "CANONICAL_INVESTMENT_ENABLED",
                getattr(
                    economy_config,
                    "CANONICAL_INVESTMENT_ENABLED",
                    False,
                ),
            )
        )
        self.endogenous_capital_goods_entry_enabled = bool(
            self.scenario_overrides.get(
                "ENDOGENOUS_CAPITAL_GOODS_ENTRY_ENABLED",
                getattr(economy_config, "ENDOGENOUS_CAPITAL_GOODS_ENTRY_ENABLED", False),
            )
        )
        self.deterministic_review_phase_staggering_enabled = bool(
            self.scenario_overrides.get(
                "DETERMINISTIC_FIRM_REVIEW_PHASE_STAGGERING_ENABLED",
                getattr(
                    economy_config,
                    "DETERMINISTIC_FIRM_REVIEW_PHASE_STAGGERING_ENABLED",
                    False,
                ),
            )
        )
        self.adult_settlement_labor_eligibility_enabled = bool(
            self.scenario_overrides.get(
                "ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED",
                getattr(
                    economy_config,
                    "ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED",
                    False,
                ),
            )
        )
        self.age_labor_participation_contract_mode = str(
            self.scenario_overrides.get(
                "AGE_LABOR_PARTICIPATION_CONTRACT_MODE",
                getattr(
                    economy_config,
                    "AGE_LABOR_PARTICIPATION_CONTRACT_MODE",
                    "current",
                ),
            )
        ).strip().lower()
        self.age_labor_hard_exit_age = float(
            self.scenario_overrides.get(
                "AGE_LABOR_HARD_EXIT_AGE",
                getattr(economy_config, "AGE_LABOR_HARD_EXIT_AGE", 65.0),
            )
        )
        self.age_labor_gradual_transition_start_age = float(
            self.scenario_overrides.get(
                "AGE_LABOR_GRADUAL_TRANSITION_START_AGE",
                getattr(
                    economy_config,
                    "AGE_LABOR_GRADUAL_TRANSITION_START_AGE", 60.0),
            )
        )
        self.age_labor_gradual_exit_age = float(
            self.scenario_overrides.get(
                "AGE_LABOR_GRADUAL_EXIT_AGE",
                getattr(economy_config, "AGE_LABOR_GRADUAL_EXIT_AGE", 75.0),
            )
        )
        if self.age_labor_participation_contract_mode not in {
            "current", "hard_exit", "gradual_participation",
        }:
            raise ValueError("unknown age-labor participation contract mode")
        self.age_labor_participation_release_events = []
        self.retirement_events = []
        self.initial_household_one_week_consumption_buffer_enabled = bool(
            self.scenario_overrides.get(
                "INITIAL_HOUSEHOLD_ONE_WEEK_CONSUMPTION_BUFFER_ENABLED",
                getattr(
                    economy_config,
                    "INITIAL_HOUSEHOLD_ONE_WEEK_CONSUMPTION_BUFFER_ENABLED",
                    False,
                ),
            )
        )
        self.initial_household_opening_buffer_total = 0.0
        self.capital_good_customer_advance_enabled = bool(
            self.scenario_overrides.get(
                "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED",
                getattr(
                    economy_config,
                    "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED",
                    False,
                ),
            )
        )
        self.capital_capacity_runtime_enabled = bool(
            self.scenario_overrides.get(
                "CAPITAL_CAPACITY_RUNTIME_ENABLED",
                getattr(
                    economy_config,
                    "CAPITAL_CAPACITY_RUNTIME_ENABLED",
                    False,
                ),
            )
        )
        self.capital_lifecycle_enabled = bool(
            self.scenario_overrides.get(
                "CAPITAL_LIFECYCLE_ENABLED",
                getattr(economy_config, "CAPITAL_LIFECYCLE_ENABLED", False),
            )
        )
        self.capital_lifecycle_useful_life_weeks = max(
            1.0,
            float(
                self.scenario_overrides.get(
                    "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS",
                    getattr(
                        economy_config,
                        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS",
                        52.0,
                    ),
                )
            ),
        )
        self.capital_good_firm_count = max(
            1,
            int(
                self.scenario_overrides.get(
                    "CAPITAL_GOOD_FIRM_COUNT",
                    getattr(economy_config, "CAPITAL_GOOD_FIRM_COUNT", 1),
                )
            ),
        )
        self.capital_good_startup_cash_fraction = max(
            0.0,
            float(
                self.scenario_overrides.get(
                    "CAPITAL_GOOD_STARTUP_CASH_FRACTION",
                    getattr(
                        economy_config,
                        "CAPITAL_GOOD_STARTUP_CASH_FRACTION",
                        0.001,
                    ),
                )
            ),
        )
        self.capital_good_labor_share = max(
            0.0,
            min(
                1.0,
                float(
                    self.scenario_overrides.get(
                        "CAPITAL_GOOD_LABOR_SHARE",
                        getattr(economy_config, "CAPITAL_GOOD_LABOR_SHARE", 0.05),
                    )
                ),
            ),
        )
        self.capital_good_fixture_productivity_per_labor = max(
            0.0,
            float(
                self.scenario_overrides.get(
                    "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR",
                    getattr(
                        economy_config,
                        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR",
                        1.0,
                    ),
                )
            ),
        )
        self.capital_good_unit_price = max(
            1e-12,
            float(
                self.scenario_overrides.get(
                    "CAPITAL_GOOD_UNIT_PRICE",
                    getattr(economy_config, "CAPITAL_GOOD_UNIT_PRICE", 10.0),
                )
            ),
        )
        self.capital_good_offer_price_mode = str(
            self.scenario_overrides.get(
                "CAPITAL_GOOD_OFFER_PRICE_MODE",
                "fixed_engineering",
            )
        ).strip().lower()
        self.capital_good_investment_review_interval_weeks = max(
            1,
            int(
                self.scenario_overrides.get(
                    "CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS",
                    getattr(
                        economy_config,
                        "CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS",
                        13,
                    ),
                )
            ),
        )
        self.capital_good_backlog_service_horizon_weeks = max(
            1,
            int(
                self.scenario_overrides.get(
                    "CAPITAL_GOOD_BACKLOG_SERVICE_HORIZON_WEEKS",
                    getattr(
                        economy_config,
                        "CAPITAL_GOOD_BACKLOG_SERVICE_HORIZON_WEEKS",
                        13,
                    ),
                )
            ),
        )
        self.capital_good_committed_capital_planner_enabled = bool(
            self.scenario_overrides.get(
                "CAPITAL_GOOD_COMMITTED_CAPITAL_PLANNER_ENABLED",
                getattr(
                    economy_config,
                    "CAPITAL_GOOD_COMMITTED_CAPITAL_PLANNER_ENABLED",
                    False,
                ),
            )
        )
        self.capital_good_pipeline_depletion_early_review_enabled = bool(
            self.scenario_overrides.get(
                "CAPITAL_GOOD_PIPELINE_DEPLETION_EARLY_REVIEW_ENABLED",
                getattr(
                    economy_config,
                    "CAPITAL_GOOD_PIPELINE_DEPLETION_EARLY_REVIEW_ENABLED",
                    False,
                ),
            )
        )
        self.capital_good_firms = []
        self.capital_good_firm_ids = set()
        self.temporary_interest_relief_enabled = bool(
            getattr(
                central_bank_config,
                "CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_ENABLED",
                False,
            )
        )
        self.temporary_interest_relief_fraction = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        central_bank_config,
                        "CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_FRACTION",
                        0.0,
                    )
                ),
            ),
        )
        self.temporary_interest_relief_eligibility_weeks = max(
            1,
            int(
                getattr(
                    central_bank_config,
                    "CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_ELIGIBILITY_WEEKS",
                    26,
                )
            ),
        )
        self.temporary_interest_relief_max_weeks = max(
            1,
            int(
                getattr(
                    central_bank_config,
                    "CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_MAX_WEEKS",
                    26,
                )
            ),
        )
        self.snapshot_legacy_arrears_termout_enabled = bool(
            self.scenario_overrides.get(
                "SNAPSHOT_LEGACY_ARREARS_TERMOUT_ENABLED",
                getattr(
                    central_bank_config,
                    "CENTRAL_BANK_SNAPSHOT_LEGACY_ARREARS_TERMOUT_ENABLED",
                    False,
                ),
            )
        )
        if initial_age_phase_mode not in {"distributed", "synchronized"}:
            raise ValueError(
                "initial_age_phase_mode must be 'distributed' or 'synchronized'"
            )
        self.initial_age_phase_mode = initial_age_phase_mode
        self.initial_age_phase_schema = "dedicated_initial_age_phase_rng_v1"
        self.age_phase_rng = random.Random(
            f"{seed}:demographic_initial_age_phase:v1"
            if seed is not None
            else None
        )
        self.current_step_index = 0
        self.clock = SimulationClock(0)
        self.next_marriage_market_step = MARRIAGE_MARKET_INTERVAL_WEEKS
        self.marriage_schedule_schema = "annual_equivalent_marriage_schedule_v1"
        self.legacy_marriage_schedule_migration = False
        self.marriage_market_execution_count = 0
        self.marriage_market_executed = False
        self.marriage_market_last_result = 0
        self.marriage_market_last_eligible_males = 0
        self.marriage_market_last_eligible_females = 0
        self.marriage_market_last_unmatched_males = 0
        self.marriage_market_last_unmatched_females = 0
        self.market_rng = random.Random(
            f"{seed}:market"
            if seed is not None
            else None
        )
        self.price_choice_sensitivity = (
            economy_config.PRICE_CHOICE_SENSITIVITY
        )
        self.initial_firm_relative_price_spread = (
            economy_config.INITIAL_FIRM_RELATIVE_PRICE_SPREAD
        )

        if seed is not None:
            random.seed(seed)

        self.base_resource = base_resource

        self.productivity_value = 10

        self.next_person_id=0

        self.consumption_per_person = consumption_per_person

        self.steps = 5000

        self.households=[]
        self.settlement_household_by_person = {}
        self.settlement_household_ids = set()

        self.public_wealth = 0
        self.public_wealth_audit_sources = [
            "death",
            "no_heir",
            "household_dissolution",
            "public_inventory_income",
            "loan_interest",
            "other",
        ]
        self.pending_public_wealth_sources = {
            source: 0.0
            for source in self.public_wealth_audit_sources
        }
        self.current_public_wealth_audit = {}
        self.cumulative_public_wealth_audit = {
            "public_wealth_from_death": 0.0,
            "public_wealth_from_no_heir": 0.0,
            "public_wealth_from_household_dissolution": 0.0,
            "public_wealth_from_other": 0.0,
            "central_bank_public_income_from_death": 0.0,
            "central_bank_public_income_from_no_heir": 0.0,
            "central_bank_public_income_from_household_dissolution": 0.0,
            "central_bank_public_income_from_public_inventory": 0.0,
            "central_bank_public_income_from_loan_interest": 0.0,
            "central_bank_public_income_from_other": 0.0,
            "inherited_wealth_to_spouse_or_survivors": 0.0,
            "inherited_wealth_to_children": 0.0,
        }

        self.pending_household_formation_wealth = 0.0

        self.next_household_id=0

        self.marriage_system = MarriageSystem(self)

        self.household_manager = HouseholdManager(self)

        self.income_system = IncomeSystem(self)
        self.private_family_support_system = PrivateFamilySupportSystem(self)
        self.social_insurance_fund = SocialInsuranceFund()
        self.public_budget = GovernmentPublicBudget()
        self.public_budget_weekly_history = []
        self.payg_pension_system = PaygPensionSystem(self, self.social_insurance_fund)
        self.intergenerational_wealth_transfer_system = IntergenerationalWealthTransferSystem(self)
        self.active_social_policy_history = []
        self.payg_weekly_history = []
        self.wealth_transfer_weekly_history = []
        self.recipient_policy_instrumentation_enabled = False
        self.active_social_policy_branch_name = "UNSPECIFIED"
        self.active_social_recipient_events = []
        self._active_social_policy_observation = None
        self.payg_cash_safety_audit_enabled = False
        self.payg_cash_safety_audit_rows = []
        self._payg_cash_safety_audit_current = {}
        self.household_liquidity_snapshot_enabled = False
        self.household_liquidity_snapshot_rows = []
        self.long_horizon_liquidity_enabled = False
        self.long_horizon_liquidity_snapshot_cadence = 13
        self.long_horizon_weekly_liquidity_rows = []
        self.long_horizon_household_snapshot_rows = []
        self.long_horizon_household_snapshot_stream_path = None
        self.long_horizon_transition_rows = []
        self.long_horizon_lifecycle_rows = []
        self._long_horizon_previous_liquidity = {}
        self._long_horizon_previous_households = set()

        self.inheritance_system = InheritanceSystem(self)

        self.ledger = Ledger(
            self,
            record_details=record_ledger_details,
        )

        self.goods_catalog = GoodsCatalog(
            [
                GoodSpec(
                    id=economy_config.BASIC_CONSUMPTION_GOOD_ID,
                    name=economy_config.BASIC_CONSUMPTION_GOOD_NAME,
                    is_essential=True,
                    is_storable=True,
                    perish_rate=economy_config.FOOD_INVENTORY_SPOILAGE_RATE,
                    unit=economy_config.BASIC_CONSUMPTION_GOOD_UNIT,
                    physical=True,
                    service=False,
                    inventory_policy_compatibility="food_inventory_v1",
                )
            ]
        )

        self.needs_system = NeedsSystem(self)

        self.consumption_system = ConsumptionSystem(self)

        self.household_demand_system = None

        self.firm_system = FirmSystem(self)
        self.equity_ownership_system = EquityOwnershipSystem()
        self.shareholder_estate_system = ShareholderEstateSystem(self)
        self.estate_accounts = {}
        self.shareholder_estate_by_deceased = {}
        self.shareholder_estate_events = []
        ensure_dividend_routing_state(self)
        self.person_equity_transition_enabled = bool(
            self.scenario_overrides.get(
                "PERSON_EQUITY_TRANSITION_ENABLED",
                getattr(economy_config, "PERSON_EQUITY_TRANSITION_ENABLED", False),
            )
        )
        self.equity_transition_system = GradualPersonEquityTransition(
            self,
            issuance_price=getattr(economy_config, "PERSON_EQUITY_ISSUANCE_PRICE", 10.0),
            review_interval_weeks=getattr(
                economy_config, "PERSON_EQUITY_REVIEW_INTERVAL_WEEKS", 52
            ),
            max_household_available_cash_fraction=getattr(
                economy_config,
                "PERSON_EQUITY_MAX_HOUSEHOLD_AVAILABLE_CASH_FRACTION",
                0.01,
            ),
            max_firm_opening_base_fraction=getattr(
                economy_config,
                "PERSON_EQUITY_MAX_FIRM_OPENING_BASE_FRACTION",
                0.001,
            ),
            max_buyers_per_firm=getattr(
                economy_config, "PERSON_EQUITY_MAX_BUYERS_PER_FIRM", 3
            ),
        )
        self.last_equity_transition_result = None
        self.autonomous_secondary_equity_enabled = bool(
            self.scenario_overrides.get(
                "AUTONOMOUS_SECONDARY_EQUITY_ENABLED",
                getattr(
                    economy_config,
                    "AUTONOMOUS_SECONDARY_EQUITY_ENABLED",
                    False,
                ),
            )
        )
        self.autonomous_secondary_equity_system = (
            AutonomousSecondaryEquityPurchaseSystem(
                self,
                price_per_share=getattr(
                    economy_config,
                    "SECONDARY_EQUITY_TRANSFER_PRICE",
                    10.0,
                ),
                review_interval_weeks=getattr(
                    economy_config,
                    "SECONDARY_EQUITY_REVIEW_INTERVAL_WEEKS",
                    52,
                ),
                max_available_cash_fraction=getattr(
                    economy_config,
                    "SECONDARY_EQUITY_MAX_AVAILABLE_CASH_FRACTION",
                    0.01,
                ),
                max_buyers_per_firm=getattr(
                    economy_config,
                    "SECONDARY_EQUITY_MAX_BUYERS_PER_FIRM",
                    10,
                ),
                max_household_equity_share_of_assets=getattr(
                    economy_config,
                    "SECONDARY_EQUITY_MAX_HOUSEHOLD_EQUITY_SHARE_OF_ASSETS",
                    0.10,
                ),
                max_person_ownership_fraction=getattr(
                    economy_config,
                    "SECONDARY_EQUITY_MAX_PERSON_OWNERSHIP_FRACTION",
                    0.05,
                ),
                reference_price_mode=getattr(
                    economy_config,
                    "AUTONOMOUS_SECONDARY_EQUITY_PRICE_MODE",
                    "engineering_fixed",
                ),
            )
        )
        self.last_autonomous_secondary_purchase_result = None

        self.fertility_system = FertilitySystem(self)

        self.diagnostics_rows = []
        self.firm_diagnostics_rows = []
        # P1.1: same-week read-only labor-capacity reuse. The cache key
        # includes every eligibility/employment input used below.
        self._labor_capacity_cache = {}
        self._labor_capacity_cache_step = None
        self.labor_capacity_cache_hits = 0
        self.labor_capacity_cache_misses = 0
        self.household_diagnostics_rows = []
        self.household_wealth_instrumentation_enabled = False
        self.household_employer_exposure_instrumentation_enabled = False
        self.household_employer_exposure_rows = []
        self.household_employer_exposure_weekly_rows = []
        self.household_active_denominator_instrumentation_enabled = False
        self.household_active_denominator_rows = []
        self._household_employer_weekly_state = {}
        self.household_wealth_micro_trace_rows = []
        self.household_wealth_weekly_summary_rows = []
        self.household_low_wealth_state = {}
        self.household_low_wealth_history = defaultdict(list)
        self.demographic_events = []
        self.marriage_market_diagnostics = []
        self.initial_age_phase_diagnostics_rows = []
        self.age_transition_diagnostics = []
        # Event-level capital provenance is diagnostic state only.  It is
        # intentionally separate from aggregate economic diagnostics.
        self.capital_asset_event_rows = []
        self.entered_worker_age_this_step = 0
        self.entered_elderly_age_this_step = 0
        self.accounting = AccountingLayer(self)
        # Optional appendable canonical diagnostics. It is configured before
        # running and remains downstream of all economic events.
        self.diagnostic_persistence = None
        # The first checkpoint created by this version knows only the
        # bookkeeping history simulated in this World.  A legacy warm
        # checkpoint cannot reconstruct Default events before its boundary.
        self.default_bookkeeping_history_complete = False
        self.household_lifecycle_events = []
        self.household_lifecycle_transfer_events = []
        self.next_lifecycle_transfer_event_id = 0

        self.invariant_violations = []

        # =========================
        # Population
        # =========================

        self.population = []

        self.person_dict = {}
        self.person_by_id = self.person_dict

        self.household_dict = {}
        self.household_by_id = self.household_dict
        self.active_households_cache = []


        for _ in range(self.initial_population):

            age = random.randint(0,80)

            sex = random.choice(
                ["M","F"]
            )

            person = Person(
                self.next_person_id,
                age,
                sex
            )

            if self.initial_age_phase_mode == "distributed":
                person.age_weeks += self.age_phase_rng.randrange(52)

            self.initial_age_phase_diagnostics_rows.append({
                "person_id": person.id,
                "initial_age_years": age,
                "initial_age_weeks": person.age_weeks,
                "age_week_phase": person.age_weeks % 52,
                "initial_age_group": self.age_labor_group(person.age_weeks),
                "initial_age_phase_mode": self.initial_age_phase_mode,
            })

            self.next_person_id += 1

            self.population.append(
                person
            )

            self.person_dict[person.id] = person

        self.initialize_households()
        self.refresh_active_households()
        self.ensure_adult_settlement_labor_eligibility_state()
        self.ensure_labor_settlement_households()
        self.apply_initial_household_liquidity_buffer()

        self.initial_private_money_stock = (
            self.firm_system.cash
            +
            sum(h.wealth for h in self.households)
        )
        self.firms = []
        self.firm_dict = {}
        self.firm_by_id = self.firm_dict
        self.firm_count = 1
        split_single_firm(self, 1)
        self.assign_bootstrap_founders()
        self.ensure_ownership_state()
        self.ensure_multisector_foundation_contracts()
        from economy.canonical_investment import CanonicalInvestmentSystem

        self.canonical_investment_system = CanonicalInvestmentSystem(self)
        self.endogenous_capital_goods_entry_system = EndogenousCapitalGoodsEntrySystem(self)
        self.pending_capital_wages = {}
        self.pending_capital_person_wages = {}
        self.last_canonical_investment_week = None

            


        # =========================
        # History
        # =========================

        self.population_history=[]

        self.birth_history=[]

        self.death_history=[]

        self.birth_rate_history=[]

        self.death_rate_history=[]

        self.average_age_history=[]

        self.resource_history=[]

        self.labor_history=[]

        self.pressure_history=[]

        # =========================
        # Economy History
        # =========================

        self.gdp_history=[]

        self.income_history=[]

        self.consumption_history=[]

        self.saving_history=[]

        self.wealth_history=[]

        self.consumption_rate_history=[]

        self.saving_rate_history=[]

        self.wealth_per_capita_history=[]

        self.firm_cash_history=[]

        self.firm_inventory_history=[]

        self.firm_net_worth_history=[]

        self.firm_profit_history=[]

        self.food_price_history=[]
        self.unit_labor_cost_history=[]
        self.unit_labor_cost_growth_history=[]
        self.price_inventory_gap_history=[]
        self.price_cost_growth_history=[]
        self.price_inflation_signal_history=[]
        self.price_log_adjustment_history=[]
        self.demand_pressure_history=[]
        self.available_food_supply_units_history=[]

        self.firm_sales_income_ratio_history=[]

        self.food_output_units_history=[]

        self.food_demand_units_history=[]

        self.food_sales_units_history=[]

        self.food_inventory_units_history=[]

        self.food_inventory_demand_ratio_history=[]

        self.unmet_food_demand_units_history=[]

        self.money_issued_history=[]

        self.working_capital_loan_issued_history=[]
        self.working_capital_target_cash_history=[]
        self.working_capital_funding_gap_history=[]

        self.working_capital_loan_repaid_history=[]

        self.working_capital_interest_paid_history=[]

        self.working_capital_loan_balance_history=[]

        self.cumulative_money_issued_history=[]

        self.inventory_monetized_units_history=[]

        self.cumulative_inventory_monetized_units_history=[]

        self.central_bank_inventory_purchase_history=[]

        self.central_bank_market_release_revenue_history=[]

        self.central_bank_poverty_subsidy_value_history=[]

        self.central_bank_public_income_history=[]

        self.central_bank_public_income_used_history=[]

        self.central_bank_public_income_balance_history=[]

        self.central_bank_net_money_issued_history=[]

        self.private_cash_stock_history=[]

        self.public_wealth_history=[]

        self.expected_money_stock_history=[]

        self.located_money_stock_history=[]

        self.monetary_accounting_gap_history=[]

        self.central_bank_food_inventory_units_history=[]

        self.central_bank_food_purchase_units_history=[]

        self.central_bank_food_release_units_history=[]

        self.central_bank_food_subsidy_units_history=[]


        # 鏂板

        self.age_distribution_history=[]

        self.age_group_history=[]

        self.household_size_history=[]

        self.married_household_history=[]

        self.single_parent_history=[]

        self.empty_household_history=[]



    def time_metadata(self, step_index=None):
        """Return passive weekly-clock metadata for diagnostics."""
        if not hasattr(self, "clock"):
            self.clock = SimulationClock(0)
        if step_index is None:
            step_index = self.current_step_index
        self.clock.set_step(step_index)
        return self.clock.metadata()

    @staticmethod
    def age_labor_group(age_weeks):
        """Return the canonical continuous-age diagnostics group."""
        if int(age_weeks) < 20 * 52:
            return "child"
        if int(age_weeks) <= 60 * 52:
            return "worker"
        return "elderly"

    def initial_age_phase_histogram(self):
        rows = getattr(self, "initial_age_phase_diagnostics_rows", [])
        output = []
        for phase in range(52):
            phase_rows = [row for row in rows if row["age_week_phase"] == phase]
            output.append({
                "age_week_phase": phase,
                "initial_population_count": len(phase_rows),
                "initial_children_count": sum(
                    row["initial_age_group"] == "child" for row in phase_rows
                ),
                "initial_workers_count": sum(
                    row["initial_age_group"] == "worker" for row in phase_rows
                ),
                "initial_elderly_count": sum(
                    row["initial_age_group"] == "elderly" for row in phase_rows
                ),
                "initial_age_phase_mode": getattr(
                    self, "initial_age_phase_mode", "legacy_synchronized"
                ),
            })
        return output

    def record_age_transitions(self, stages_before):
        entered_worker = 0
        entered_elderly = 0
        for person in self.population:
            before = stages_before.get(person.id)
            after = self.age_labor_group(person.age_weeks)
            if before == "child" and after == "worker":
                entered_worker += 1
            elif before == "worker" and after == "elderly":
                entered_elderly += 1

        self.entered_worker_age_this_step = entered_worker
        self.entered_elderly_age_this_step = entered_elderly
        if not hasattr(self, "age_transition_diagnostics"):
            self.age_transition_diagnostics = []
        self.age_transition_diagnostics.append({
            "global_step": self.current_step_index,
            "simulation_week": self.current_step_index,
            "simulation_year": self.current_step_index / 52.0,
            "week_of_year": self.current_step_index % 52,
            "entered_worker_age": entered_worker,
            "entered_elderly_age": entered_elderly,
        })

    def marriage_diagnostics(self):
        return {
            "marriage_market_executed": self.marriage_market_executed,
            "next_marriage_market_step": self.next_marriage_market_step,
            "marriage_market_execution_count": self.marriage_market_execution_count,
            "eligible_males": self.marriage_market_last_eligible_males,
            "eligible_females": self.marriage_market_last_eligible_females,
            "matches_formed": self.marriage_market_last_result,
            "unmatched_males": self.marriage_market_last_unmatched_males,
            "unmatched_females": self.marriage_market_last_unmatched_females,
        }

    def demographic_structure_diagnostics(self):
        people = list(self.population)
        population = len(people)
        groups = {
            "0_19": sum(0 <= int(p.age_years) < 20 for p in people),
            "20_39": sum(20 <= int(p.age_years) < 40 for p in people),
            "40_64": sum(40 <= int(p.age_years) < 65 for p in people),
            "65_plus": sum(int(p.age_years) >= 65 for p in people),
        }
        reciprocity = 0
        membership = 0
        parent_links = 0
        for person in people:
            if person.partner_id is not None:
                partner = self.person_dict.get(person.partner_id)
                if partner is None or partner.partner_id != person.id:
                    reciprocity += 1
            household = self.household_dict.get(person.household_id)
            if household is None:
                membership += 1
            elif person.id not in household.parents and person.id not in household.children:
                membership += 1
            for child_id in person.children_ids:
                child = self.person_dict.get(child_id)
                if child is None or person.id not in child.parent_ids:
                    parent_links += 1
        return {
            "age_0_19_count": groups["0_19"],
            "age_20_39_count": groups["20_39"],
            "age_40_64_count": groups["40_64"],
            "age_65_plus_count": groups["65_plus"],
            "age_0_19_share": self.safe_ratio(groups["0_19"], population),
            "age_20_39_share": self.safe_ratio(groups["20_39"], population),
            "age_40_64_share": self.safe_ratio(groups["40_64"], population),
            "age_65_plus_share": self.safe_ratio(groups["65_plus"], population),
            "reproductive_age_female_count": sum(
                p.sex == "F" and 20 <= p.age_years <= 40 for p in people
            ),
            "working_age_population": sum(20 <= p.age_years <= 60 for p in people),
            "elderly_population": groups["65_plus"],
            "partner_relationship_violation_count": reciprocity,
            "household_membership_violation_count": membership,
            "parent_child_link_violation_count": parent_links,
        }

    def record_demographic_event(self, event_type, **fields):
        row = {
            "global_step": self.current_step_index,
            "simulation_week": self.current_step_index,
            "simulation_year": self.current_step_index / 52.0,
            "week_of_year": self.current_step_index % 52,
            "event_type": event_type,
        }
        row.update(fields)
        self.demographic_events.append(row)

    def record_marriage_market_diagnostic(self, eligible_males, eligible_females, matches):
        self.marriage_market_diagnostics.append({
            "global_step": self.current_step_index,
            "simulation_week": self.current_step_index,
            "simulation_year": self.current_step_index / 52.0,
            "week_of_year": self.current_step_index % 52,
            "eligible_males": eligible_males,
            "eligible_females": eligible_females,
            "matches_formed": matches,
            "unmatched_males": max(0, eligible_males - matches),
            "unmatched_females": max(0, eligible_females - matches),
            "male_match_rate": self.safe_ratio(matches, eligible_males),
            "female_match_rate": self.safe_ratio(matches, eligible_females),
            "next_marriage_market_step": self.next_marriage_market_step,
        })

    def family_birth(self, pressure):

        return self.fertility_system.family_birth(
            pressure
        )

    def create_household(self):

        household = Household(
            self.next_household_id,
            self
        )
        household.settlement_only = False

        self.next_household_id += 1

        self.households.append(
            household
        )

        self.household_dict[household.id]=household

        return household

    def apply_initial_household_liquidity_buffer(self):
        """Fund one opening consumption week from initial Food-Firm cash.

        This runs once after initial Social and settlement-only accounts exist
        and before the first Food-Firm split or payroll.  It is an opening
        balance-sheet transfer, not income, credit, or a consumption-rule
        change.
        """
        if not getattr(
            self,
            "initial_household_one_week_consumption_buffer_enabled",
            False,
        ):
            return 0.0

        allocations = []
        for household in self.households:
            need = max(
                0.0,
                float(
                    self.needs_system.household_minimum_need_units(household)
                ),
            )
            if need > 0.0:
                allocations.append((household, need))

        total = math.fsum(amount for _, amount in allocations)
        available = max(0.0, float(getattr(self.firm_system, "cash", 0.0)))
        if total > available + 1e-9:
            raise ValueError(
                "initial Household liquidity buffer exceeds opening Firm cash"
            )

        for household, amount in allocations:
            household.wealth += amount

        self.firm_system.cash = available - total
        self.initial_household_opening_buffer_total = total
        return total

    def authoritative_money_location_components(self):
        """Return every authoritative runtime cash holder in one scope."""
        firms = math.fsum(
            float(getattr(firm, "cash", 0.0))
            for firm in self.operating_firms()
        )
        households = math.fsum(
            float(getattr(household, "wealth", 0.0))
            for household in getattr(self, "households", [])
        )
        legacy_owner = float(getattr(self, "legacy_owner_cash", 0.0))
        estate = math.fsum(
            float(getattr(account, "cash", 0.0))
            for account in getattr(self, "estate_accounts", {}).values()
        )
        public = float(getattr(self, "public_wealth", 0.0))
        government = float(getattr(getattr(self, "public_budget", None), "cash", 0.0))
        central_bank = getattr(getattr(self, "firm_system", None), "central_bank", None)
        central_public = float(
            getattr(central_bank, "public_income_balance", 0.0)
        )
        pending_formation = float(
            getattr(self, "pending_household_formation_wealth", 0.0)
        )
        return {
            "household_cash": households,
            "settlement_only_cash": math.fsum(
                float(getattr(household, "wealth", 0.0))
                for household in getattr(self, "households", [])
                if getattr(household, "settlement_only", False)
            ),
            "social_household_cash": math.fsum(
                float(getattr(household, "wealth", 0.0))
                for household in getattr(self, "households", [])
                if not getattr(household, "settlement_only", False)
            ),
            "firm_cash": firms,
            "legacy_owner_cash": legacy_owner,
            "estate_cash": estate,
            "social_insurance_fund_cash": float(getattr(getattr(self, "social_insurance_fund", None), "cash", 0.0)),
            "public_wealth": public,
            "government_budget_cash": government,
            "central_public_income_balance": central_public,
            "pending_household_formation_wealth": pending_formation,
            "located_money_stock": (
                households
                + firms
                + legacy_owner
                + estate
                + float(getattr(getattr(self, "social_insurance_fund", None), "cash", 0.0))
                + public
                + government
                + central_public
                + pending_formation
            ),
        }

    def authoritative_money_stock(self):
        """Return the stock implied by opening money plus net CB issuance."""
        central_bank = getattr(getattr(self, "firm_system", None), "central_bank", None)
        return float(getattr(self, "initial_private_money_stock", 0.0)) + float(
            getattr(central_bank, "money_supply", 0.0)
        )

    def add_person_to_household(
        self,
        person,
        household,
        role
    ):

        person.household_id = household.id

        if role == "parent":

            household.add_parent(
                person.id
            )

        elif role == "child":

            household.add_child(
                person.id
        )

    def remove_person_relationship(self, person):
        # Keep employer rosters consistent when a worker is removed from the
        # population.  This is lifecycle bookkeeping only: the person is
        # already dead and cannot participate in payroll or matching.
        for firm in self.operating_firms():
            employee_ids = getattr(firm, "employee_ids", None)
            if employee_ids is not None and person.id in employee_ids:
                firm.employee_ids = [
                    person_id for person_id in employee_ids
                    if person_id != person.id
                ]
            if getattr(person, "firm_id", None) == getattr(firm, "firm_id", None):
                person.firm_id = None
        partner = self.get_person_by_id(
            person.partner_id
        )

        if partner is not None:

            if partner.partner_id == person.id:

                partner.partner_id = None

                if partner.alive and partner.age >= 20:

                    partner.is_seeking_partner = True

        # 娓呯悊鐖舵瘝鍏崇郴
        for parent_id in person.parent_ids:
            parent = self.get_person_by_id(parent_id)
            if parent is not None:
                if person.id in parent.children_ids:
                    parent.children_ids.remove(
                        person.id
                    )
        # 娓呯悊瀛愬コ鍏崇郴
        for child_id in person.children_ids:
            child = self.get_person_by_id(child_id)
            if child is not None:
                if person.id in child.parent_ids:
                    child.parent_ids.remove(
                        person.id
                    )

    def get_person_by_id(self, person_id):
        lookup = getattr(self, "person_by_id", None)
        if not isinstance(lookup, dict):
            lookup = getattr(self, "person_dict", {})
        return lookup.get(person_id)

    def rebuild_runtime_id_indexes(self):
        """Rebuild derived ID indexes from authoritative collections."""
        self.person_dict = {
            person.id: person
            for person in getattr(self, "population", [])
        }
        self.person_by_id = self.person_dict
        self.household_dict = {
            household.id: household
            for household in getattr(self, "households", [])
        }
        self.household_by_id = self.household_dict
        firms = [
            *getattr(self, "firms", []),
            *getattr(self, "capital_good_firms", []),
        ]
        self.firm_dict = {
            getattr(firm, "firm_id", index): firm
            for index, firm in enumerate(firms)
        }
        self.firm_by_id = self.firm_dict
        return self.validate_runtime_id_indexes()

    def validate_runtime_id_indexes(self):
        """Return lightweight index consistency diagnostics for tests/debugging."""
        collections = {
            "persons": list(getattr(self, "population", [])),
            "households": list(getattr(self, "households", [])),
            "firms": [
                *getattr(self, "firms", []),
                *getattr(self, "capital_good_firms", []),
            ],
        }
        indexes = {
            "persons": getattr(self, "person_by_id", {}),
            "households": getattr(self, "household_by_id", {}),
            "firms": getattr(self, "firm_by_id", {}),
        }
        result = {}
        for kind, objects in collections.items():
            index = indexes[kind]
            expected_ids = [
                getattr(obj, "id", getattr(obj, "firm_id", None))
                for obj in objects
            ]
            exact = all(
                index.get(object_id) is obj
                for object_id, obj in zip(expected_ids, objects)
            )
            result[kind] = {
                "collection_count": len(objects),
                "index_count": len(index),
                "exact_object_identity": exact,
                "consistent": (
                    len(expected_ids) == len(set(expected_ids))
                    and exact
                ),
            }
        return result
    def get_single_adults(self):


        singles=[]


        for person in self.population:


            if (

                person.age >=20

                and

                person.partner_id is None

                and

                person.alive

            ):

                singles.append(person)



        return singles

    def leave_household(self, person):


        if person.household_id is None:

            return 0.0


        household = self.get_household(
            person.household_id
        )


        if household is None:

            return 0.0

        size_before = household.size()
        moving_wealth = 0.0

        if size_before > 0 and abs(household.wealth) > 0:

            wealth_before = household.wealth
            pending_before = self.pending_household_formation_wealth

            moving_wealth = household.wealth / size_before

            self.transfer_signed_balance(
                source_obj=household,
                source_attr="wealth",
                source_name=f"household.{household.id}.wealth",
                destination_obj=self,
                destination_attr="pending_household_formation_wealth",
                destination_name="world.pending_household_formation_wealth",
                amount=moving_wealth,
                reason="leave_household_wealth_transfer",
            )

            self.record_household_lifecycle_event(
                household_id=household.id,
                event_type="leave_household",
                wealth_before=wealth_before,
                wealth_after=household.wealth,
                wealth_transferred_out=moving_wealth,
            )
            self.record_lifecycle_transfer_event(
                event_type="adult_separation_to_pending_formation",
                source_household_id=household.id,
                person_id=person.id,
                amount=moving_wealth,
                source_wealth_before=wealth_before,
                source_wealth_after=household.wealth,
                destination_wealth_before=pending_before,
                destination_wealth_after=self.pending_household_formation_wealth,
                source_account=f"household.{household.id}.wealth",
                destination_account="world.pending_household_formation_wealth",
            )



        if person.id in household.children:

            household.remove_child(
                person.id
            )


        if person.id in household.parents:

            household.parents.remove(
                person.id
            )


        person.household_id = None

        return moving_wealth

    def transfer_signed_balance(
        self,
        source_obj,
        source_attr,
        source_name,
        destination_obj,
        destination_attr,
        destination_name,
        amount,
        reason,
    ):
        """Move a signed balance while preserving both account totals."""
        if amount >= 0:
            return self.ledger.transfer_attrs(
                payer_obj=source_obj,
                payer_attr=source_attr,
                payer_name=source_name,
                receiver_obj=destination_obj,
                receiver_attr=destination_attr,
                receiver_name=destination_name,
                amount=amount,
                reason=reason,
            )

        return self.ledger.transfer_attrs(
            payer_obj=destination_obj,
            payer_attr=destination_attr,
            payer_name=destination_name,
            receiver_obj=source_obj,
            receiver_attr=source_attr,
            receiver_name=source_name,
            amount=-amount,
            reason=f"{reason}: signed_negative_balance",
        )

    def record_household_lifecycle_event(
        self,
        household_id,
        event_type,
        wealth_before=0.0,
        wealth_after=0.0,
        wealth_transferred_in=0.0,
        wealth_transferred_out=0.0,
        wealth_to_heirs=0.0,
        wealth_to_public=0.0,
        wealth_removed_on_deletion=0.0,
        negative_wealth_removed=0.0,
    ):
        if not hasattr(self, "household_lifecycle_events"):
            self.household_lifecycle_events = []
        lifecycle_cash_gap = (
            wealth_after
            - wealth_before
            - wealth_transferred_in
            + wealth_transferred_out
            + wealth_to_heirs
            + wealth_to_public
            + wealth_removed_on_deletion
        )
        self.household_lifecycle_events.append({
            "step": len(self.population_history),
            "household_id": household_id,
            "event_type": event_type,
            "wealth_before": wealth_before,
            "wealth_after": wealth_after,
            "wealth_transferred_in": wealth_transferred_in,
            "wealth_transferred_out": wealth_transferred_out,
            "wealth_to_heirs": wealth_to_heirs,
            "wealth_to_public": wealth_to_public,
            "wealth_removed_on_deletion": wealth_removed_on_deletion,
            "negative_wealth_removed": negative_wealth_removed,
            "lifecycle_cash_gap": lifecycle_cash_gap,
        })

    def record_lifecycle_transfer_event(
        self,
        event_type,
        source_household_id=None,
        destination_household_id=None,
        person_id=None,
        amount=0.0,
        source_wealth_before=0.0,
        source_wealth_after=0.0,
        destination_wealth_before=0.0,
        destination_wealth_after=0.0,
        public_sector_amount=0.0,
        source_account=None,
        destination_account=None,
    ):
        """Record an already-executed lifecycle wealth transfer.

        This is passive attribution only. It deliberately does not perform or
        repeat the transfer. The account-total identity also covers the
        intermediate pending-formation account used by adult separation.
        """
        if not getattr(self, "household_wealth_instrumentation_enabled", False):
            return
        self.next_lifecycle_transfer_event_id = getattr(
            self, "next_lifecycle_transfer_event_id", 0
        ) + 1
        gap = (
            source_wealth_before
            + destination_wealth_before
            - source_wealth_after
            - destination_wealth_after
            - public_sector_amount
        )
        self.household_lifecycle_transfer_events.append({
            "week": self.current_step_index,
            "event_id": self.next_lifecycle_transfer_event_id,
            "event_type": event_type,
            "source_household_id": source_household_id,
            "destination_household_id": destination_household_id,
            "person_id": person_id,
            "amount": amount,
            "source_wealth_before": source_wealth_before,
            "source_wealth_after": source_wealth_after,
            "destination_wealth_before": destination_wealth_before,
            "destination_wealth_after": destination_wealth_after,
            "public_sector_amount": public_sector_amount,
            "source_account": source_account,
            "destination_account": destination_account,
            "lifecycle_conservation_gap": gap,
        })


    def initialize_households(self):


        # =========================
        # Separate adults by sex
        # =========================

        adult_males = []

        adult_females = []


        for person in self.population:


            if person.age >= 20:


                if person.sex == "M":

                    adult_males.append(person)


                else:

                    adult_females.append(person)



        # shuffle to randomize matching

        random.shuffle(adult_males)

        random.shuffle(adult_females)



        # =========================
        # Create couples
        # =========================

        pair_number = min(

            len(adult_males),

            len(adult_females)

        )



        for i in range(pair_number):


            male = adult_males[i]

            female = adult_females[i]



            # create household

            household = self.create_household()



            # add parents

            self.add_person_to_household(

                male,

                household,

                "parent"

            )


            self.add_person_to_household(

                female,

                household,

                "parent"

            )



            # establish partner relationship

            male.partner_id = female.id

            female.partner_id = male.id


    def record_family_link_observability(self, step):
        if not getattr(self, "family_link_observability_enabled", False):
            return
        cadence = max(1, int(getattr(self, "family_link_observability_cadence", 13)))
        if step % cadence != 0:
            return
        rows = getattr(self, "family_link_observability_rows", None)
        if rows is None:
            self.family_link_observability_rows = []
            rows = self.family_link_observability_rows
        for person in sorted(getattr(self, "population", []), key=lambda item: item.id):
            if not getattr(person, "alive", False):
                continue
            household = self.get_household(getattr(person, "household_id", None))
            if household is None or getattr(household, "settlement_only", False):
                social_household_id = ""
            else:
                social_household_id = household.id
            rows.append({
                "global_step": step,
                "person_id": person.id,
                "social_household_id": social_household_id,
                "parent_ids": json.dumps(sorted(set(getattr(person, "parent_ids", []))), separators=(",", ":")),
                "children_ids": json.dumps(sorted(set(getattr(person, "children_ids", []))), separators=(",", ":")),
                "age": float(getattr(person, "age", 0.0)),
                "alive": True,
            })
    def get_household(self, household_id):
        lookup = getattr(self, "household_by_id", None)
        if not isinstance(lookup, dict):
            lookup = getattr(self, "household_dict", {})
        return lookup.get(household_id)

    def ensure_adult_settlement_labor_eligibility_state(self):
        """Backfill passive settlement fields without altering social relations."""
        if not hasattr(self, "adult_settlement_labor_eligibility_enabled"):
            self.adult_settlement_labor_eligibility_enabled = False
        if not hasattr(self, "settlement_household_by_person"):
            self.settlement_household_by_person = {}
        if not hasattr(self, "settlement_household_ids"):
            self.settlement_household_ids = set()
        for household in getattr(self, "households", []):
            if not hasattr(household, "settlement_only"):
                household.settlement_only = False
            if household.settlement_only:
                self.settlement_household_ids.add(household.id)
        for person in getattr(self, "population", []):
            if not hasattr(person, "settlement_household_id"):
                person.settlement_household_id = None

    def settlement_household_for_person(self, person):
        """Return the economic cash/consumption Household, not social status."""
        social = self.get_household(getattr(person, "household_id", None))
        if social is not None:
            return social
        if not getattr(self, "adult_settlement_labor_eligibility_enabled", False):
            return None
        settlement_id = getattr(person, "settlement_household_id", None)
        if settlement_id is None:
            settlement_id = self.settlement_household_by_person.get(person.id)
        return self.get_household(settlement_id)

    def age_labor_participation_factor(self, age):
        """Return a deterministic participation intensity, separate from productivity."""
        if age_productivity(age) <= 0.0:
            return 0.0
        mode = getattr(self, "age_labor_participation_contract_mode", "current")
        if mode == "current":
            return 1.0
        if mode == "hard_exit":
            return 1.0 if float(age) < self.age_labor_hard_exit_age else 0.0
        start = self.age_labor_gradual_transition_start_age
        exit_age = self.age_labor_gradual_exit_age
        if float(age) < start:
            return 1.0
        if float(age) >= exit_age:
            return 0.0
        width = max(1e-12, exit_age - start)
        return max(0.0, min(1.0, (exit_age - float(age)) / width))

    def update_retirement_status(self):
        """Apply deterministic retirement before labor allocation and payroll."""
        if not getattr(self, "retirement_runtime_enabled", False):
            return []
        retirement_age = float(getattr(self, "retirement_age", 65.0))
        events = []
        for person in list(getattr(self, "population", [])):
            if not getattr(person, "alive", False) or getattr(person, "retired", False) or float(person.age) < retirement_age:
                continue
            former_firm_id = getattr(person, "firm_id", None)
            person.retired = True
            for firm in self.operating_firms():
                employee_ids = list(getattr(firm, "employee_ids", []))
                if person.id in employee_ids:
                    firm.employee_ids = [pid for pid in employee_ids if pid != person.id]
            person.firm_id = None
            event = {"global_step": int(getattr(self, "current_step_index", 0)), "person_id": person.id, "household_id": getattr(person, "household_id", None), "age": float(person.age), "former_firm_id": former_firm_id, "effective_labor_removed": float(age_productivity(person.age))}
            events.append(event)
        if not hasattr(self, "retirement_events"):
            self.retirement_events = []
        self.retirement_events.extend(events)
        return events

    def labor_formally_eligible(self, person):
        return bool(
            getattr(person, "alive", False)
            and age_productivity(person.age) > 0.0
            and not getattr(person, "retired", False)
            and not (
                getattr(self, "age_labor_participation_contract_mode", "current")
                == "hard_exit"
                and float(person.age) >= self.age_labor_hard_exit_age
            )
        )

    def labor_participates(self, person):
        """Select a stable deterministic participant without consuming RNG."""
        if not self.labor_formally_eligible(person):
            return False
        factor = self.age_labor_participation_factor(person.age)
        if factor <= 0.0:
            return False
        if factor >= 1.0:
            return True
        try:
            person_key = int(person.id)
        except (TypeError, ValueError):
            person_key = sum(ord(char) for char in str(person.id))
        rank = ((person_key * 2654435761) % 10000) / 10000.0
        return rank < factor

    def refresh_age_labor_participation(self):
        """Release non-participants before matching; the current mode is a no-op."""
        if getattr(self, "age_labor_participation_contract_mode", "current") == "current":
            return 0
        released = 0
        for firm in self.operating_firms():
            retained = []
            for person_id in getattr(firm, "employee_ids", []):
                person = self.person_dict.get(person_id)
                if person is not None and self.labor_participates(person):
                    retained.append(person_id)
                    continue
                if person is not None and getattr(person, "firm_id", None) == firm.firm_id:
                    person.firm_id = None
                released += 1
                self.age_labor_participation_release_events.append({
                    "global_step": self.current_step_index,
                    "person_id": person_id,
                    "firm_id": firm.firm_id,
                    "age": float(getattr(person, "age", 0.0)) if person is not None else math.nan,
                    "reason": "age_labor_participation_contract",
                })
            firm.employee_ids = retained
        return released

    def has_valid_settlement_household(self, person):
        return self.settlement_household_for_person(person) is not None

    def ensure_labor_settlement_households(self):
        """Create deterministic zero-cash settlement accounts for eligible adults."""
        self.ensure_adult_settlement_labor_eligibility_state()
        if not getattr(self, "adult_settlement_labor_eligibility_enabled", False):
            return 0
        created = 0
        for person in self.population:
            if not getattr(person, "alive", False) or not self.labor_formally_eligible(person):
                continue
            if self.settlement_household_for_person(person) is not None:
                continue
            household = self.create_household()
            household.settlement_only = True
            household.add_parent(person.id)
            person.settlement_household_id = household.id
            self.settlement_household_by_person[person.id] = household.id
            self.settlement_household_ids.add(household.id)
            created += 1
        if created:
            self.refresh_active_households()
        return created

    def merge_settlement_household_into_social_household(self, person, destination):
        """Move a temporary settlement balance into a newly formed social Household."""
        settlement_id = getattr(person, "settlement_household_id", None)
        source = self.get_household(settlement_id)
        if source is None or not getattr(source, "settlement_only", False):
            return 0.0
        amount = float(getattr(source, "wealth", 0.0))
        if abs(amount) > 1e-12:
            source_before = amount
            destination_before = float(getattr(destination, "wealth", 0.0))
            self.transfer_signed_balance(
                source_obj=source,
                source_attr="wealth",
                source_name=f"household.{source.id}.wealth",
                destination_obj=destination,
                destination_attr="wealth",
                destination_name=f"household.{destination.id}.wealth",
                amount=amount,
                reason="settlement_household_to_social_household",
            )
            self.record_lifecycle_transfer_event(
                event_type="settlement_to_social_household",
                source_household_id=source.id,
                destination_household_id=destination.id,
                person_id=person.id,
                amount=amount,
                source_wealth_before=source_before,
                source_wealth_after=float(getattr(source, "wealth", 0.0)),
                destination_wealth_before=destination_before,
                destination_wealth_after=float(getattr(destination, "wealth", 0.0)),
                source_account=f"household.{source.id}.wealth",
                destination_account=f"household.{destination.id}.wealth",
            )
        if person.id in source.parents:
            source.parents.remove(person.id)
        if person.id in source.children:
            source.children.remove(person.id)
        self.settlement_household_by_person.pop(person.id, None)
        person.settlement_household_id = None
        self.settlement_household_ids.discard(source.id)
        if source.id in self.household_dict:
            del self.household_dict[source.id]
        if source in self.households:
            self.households.remove(source)
        return amount

    def refresh_active_households(self):
        self.active_households_cache = [
            household
            for household in self.households
            if household.parents or household.children
        ]

        return self.active_households_cache

    def active_households(self):
        return self.active_households_cache

    def household_planning_price_index(self):
        """Return the ex-ante price households use before this market settles."""
        firms = list(getattr(self, "firms", []))
        if not firms:
            return max(1e-12, getattr(self.firm_system, "price", 1.0))

        if len(firms) == 1:
            # Preserve the exact legacy single-firm market price.  The
            # compatibility FirmSlice is a diagnostic mirror and may lag
            # while FirmSystem performs its own price update.
            price = max(1e-12, self.firm_system.price)
            for firm in firms:
                firm.pre_market_posted_price = price
                firm.planning_weight = 1.0
                firm.planning_price_component = price
            return price

        prices = [max(1e-12, float(firm.price)) for firm in firms]
        raw_weights = [
            float(getattr(firm, "unit_market_share", 0.0))
            for firm in firms
        ]
        valid_weights = (
            all(math.isfinite(weight) and weight >= 0 for weight in raw_weights)
            and sum(raw_weights) > 1e-12
        )
        if valid_weights:
            total = math.fsum(raw_weights)
            weights = [weight / total for weight in raw_weights]
        else:
            weights = [1.0 / len(firms) for _ in firms]

        # Passive observability for the ex-ante planning interface. These
        # values are captured before household transactions and price review.
        for firm, weight, price in zip(firms, weights, prices):
            firm.pre_market_posted_price = price
            firm.planning_weight = weight
            firm.planning_price_component = weight * price

        return max(
            1e-12,
            math.fsum(weight * price for weight, price in zip(weights, prices)),
        )

    def household_planning_price_this_step(self):
        """Snapshot one ex-ante planning price for the current market step."""
        step = getattr(self, "current_step_index", 0)
        if getattr(self, "_household_planning_price_step", None) != step:
            self._household_planning_price_step = step
            self._household_planning_price_value = (
                self.household_planning_price_index()
            )
        return self._household_planning_price_value

    def scenario_multiplier(
        self,
        step_key,
        multiplier_key,
        default=1.0,
    ):
        shock_step = self.scenario_overrides.get(step_key)

        if shock_step is None:
            return default

        if self.current_step_index < shock_step:
            return default

        return self.scenario_overrides.get(multiplier_key, default)

    def scenario_window_multiplier(
        self,
        start_key,
        end_key,
        multiplier_key,
        default=1.0,
    ):
        start_step = self.scenario_overrides.get(start_key)

        if start_step is None:
            return default

        if self.current_step_index < start_step:
            return default

        end_step = self.scenario_overrides.get(end_key)

        if end_step is not None and self.current_step_index >= end_step:
            return default

        return self.scenario_overrides.get(multiplier_key, default)

    def reset_public_wealth_audit_step(self):
        self.current_public_wealth_audit = {
            "public_wealth_from_death": 0.0,
            "public_wealth_from_no_heir": 0.0,
            "public_wealth_from_household_dissolution": 0.0,
            "public_wealth_from_other": 0.0,
            "central_bank_public_income_from_death": 0.0,
            "central_bank_public_income_from_no_heir": 0.0,
            "central_bank_public_income_from_household_dissolution": 0.0,
            "central_bank_public_income_from_public_inventory": 0.0,
            "central_bank_public_income_from_loan_interest": 0.0,
            "central_bank_public_income_from_other": 0.0,
            "inherited_wealth_to_spouse_or_survivors": 0.0,
            "inherited_wealth_to_children": 0.0,
        }

    def add_public_wealth_audit(self, key, amount):
        amount = max(0.0, amount)

        if amount <= 0:
            return

        self.current_public_wealth_audit[key] += amount
        self.cumulative_public_wealth_audit[key] += amount

    def record_public_wealth_source(self, source, amount):
        amount = max(0.0, amount)

        if amount <= 0:
            return

        if source not in self.pending_public_wealth_sources:
            source = "other"

        self.pending_public_wealth_sources[source] += amount

        if source == "no_heir":
            self.add_public_wealth_audit(
                "public_wealth_from_death",
                amount,
            )
            self.add_public_wealth_audit(
                "public_wealth_from_no_heir",
                amount,
            )
        elif source == "household_dissolution":
            self.add_public_wealth_audit(
                "public_wealth_from_household_dissolution",
                amount,
            )
        else:
            self.add_public_wealth_audit(
                "public_wealth_from_other",
                amount,
            )

    def record_inherited_wealth_to_children(self, amount):
        self.add_public_wealth_audit(
            "inherited_wealth_to_children",
            amount,
        )

    def record_inherited_wealth_to_spouse_or_survivors(self, amount):
        self.add_public_wealth_audit(
            "inherited_wealth_to_spouse_or_survivors",
            amount,
        )

    def record_public_wealth_collection(self, amount):
        remaining = max(0.0, amount)

        for source in self.public_wealth_audit_sources:
            if remaining <= 0:
                break

            available = self.pending_public_wealth_sources.get(source, 0.0)
            collected = min(available, remaining)

            if collected <= 0:
                continue

            self.pending_public_wealth_sources[source] -= collected
            remaining -= collected
            self.record_central_bank_public_income_source(
                source,
                collected,
            )

        if remaining > 0:
            self.record_central_bank_public_income_source(
                "other",
                remaining,
            )

    def record_central_bank_public_income_source(self, source, amount):
        amount = max(0.0, amount)

        if amount <= 0:
            return

        key_by_source = {
            "death": "central_bank_public_income_from_death",
            "no_heir": "central_bank_public_income_from_no_heir",
            "household_dissolution": (
                "central_bank_public_income_from_household_dissolution"
            ),
            "public_inventory_income": (
                "central_bank_public_income_from_public_inventory"
            ),
            "loan_interest": "central_bank_public_income_from_loan_interest",
            "other": "central_bank_public_income_from_other",
        }
        key = key_by_source.get(
            source,
            "central_bank_public_income_from_other",
        )

        self.add_public_wealth_audit(
            key,
            amount,
        )

    def split_firms(self, firm_count):
        self.ensure_market_rng()
        if (
            firm_count > 1
            and len(getattr(self, "firms", [])) == firm_count
            and getattr(self, "multi_firm_bootstrap_complete", False)
        ):
            return
        previous_count = getattr(
            self,
            "multi_firm_bootstrap_execution_count",
            0,
        )
        split_single_firm(self, firm_count)
        self.assign_bootstrap_founders()
        if self.deterministic_review_phase_staggering_enabled:
            from economy.scheduling import deterministic_review_phase

            production_cadence = max(
                1,
                int(economy_config.FIRM_PRODUCTION_REVIEW_INTERVAL_WEEKS),
            )
            investment_cadence = max(
                1,
                int(
                    economy_config.CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS
                ),
            )
            for firm in self.firms:
                firm.production_review_phase_offset = deterministic_review_phase(
                    firm.firm_id,
                    production_cadence,
                    "production",
                )
                firm.production_review_timer = firm.production_review_phase_offset
                firm.production_review_count = 0
                firm.investment_review_phase_offset = deterministic_review_phase(
                    firm.firm_id,
                    investment_cadence,
                    "investment",
                )
                firm.investment_reviewed_this_step = False
                firm.investment_review_count = 0
        if firm_count <= 1 and getattr(self, "firms", None):
            # A one-firm continuation is the legacy path: the compatibility
            # FirmSlice must not inherit multi-firm relative-price noise.
            self.firms[0].price = self.firm_system.price
        self.multi_firm_bootstrap_complete = (
            firm_count <= 1 or self.current_step_index > 0
        )
        self.multi_firm_bootstrap_execution_count = (
            previous_count if self.multi_firm_bootstrap_complete else 0
        )
        self.ensure_multisector_foundation_contracts()

    def sync_firms_from_aggregate(self, firm_result=None):
        """Mirror the aggregate compatibility Firm into FirmSlice objects."""
        firms = list(getattr(self, "firms", []))
        if not firms:
            return
        aggregate = getattr(self, "firm_system", None)
        if aggregate is None:
            return
        result = firm_result or {}
        split_fields = {
            "cash": "firm_cash", "inventory_units": "food_inventory_units",
            "inventory_value": "firm_inventory", "wage_bill": "wage_bill",
            "sales": "sales", "production": "production",
            "actual_production": "actual_production",
            "profit": "profit_before_dividend", "loan_balance": "loan_balance",
            "dividend_payment": "dividend",
            "person_dividend_paid": "person_dividend_paid",
            "legacy_dividend_entitlement": "legacy_dividend_entitlement",
            "estate_dividend_paid": "estate_dividend_paid",
            "dividend_routing_gap": "dividend_routing_gap",
            "sales_revenue": "sales_revenue", "wage_payment": "executed_wage_bill",
            "cash_start": "cash_start", "cash_end": "cash_end",
            "funding_gap": "funding_gap", "loan_issued": "loan_issued",
            "loan_repaid": "loan_repaid",
        }
        for firm in firms:
            share = 1.0 if len(firms) == 1 else max(0.0, float(getattr(firm, "share", 0.0)))
            for target, source in split_fields.items():
                if source in result:
                    value = result[source]
                elif hasattr(aggregate, source):
                    value = getattr(aggregate, source)
                else:
                    continue
                if isinstance(value, (int, float)) and len(firms) > 1 and target not in {"cash_start", "cash_end"}:
                    value = float(value) * share
                setattr(firm, target, value)
            if len(firms) == 1:
                firm.price = float(getattr(aggregate, "price", firm.price))
                firm.expected_demand = float(getattr(aggregate, "food_demand_units", firm.expected_demand))
                firm.desired_production = float(getattr(aggregate, "food_output_units", firm.desired_production))
                firm.production_plan = firm.desired_production
                firm.actual_production = float(getattr(aggregate, "food_output_units", firm.actual_production))
                firm.previous_actual_production = firm.actual_production
            firm.actual_market_share = float(getattr(firm, "share", 1.0))
            firm.unit_market_share = firm.actual_market_share
        self.ensure_ownership_state()

    def calculate_multi_firm_dividends(self, aggregate_dividend, wage_bill=0.0):
        """Route aggregate dividend across existing FirmSlice cap tables."""
        total = max(0.0, float(aggregate_dividend))
        firms = list(getattr(self, "firms", []))
        if not firms or total <= 0.0:
            return 0.0
        share_total = math.fsum(max(0.0, float(getattr(firm, "share", 0.0))) for firm in firms)
        if share_total <= 0.0:
            share_total = float(len(firms))
        paid = 0.0
        for firm in firms:
            fraction = max(0.0, float(getattr(firm, "share", 0.0))) / share_total
            amount = total * fraction
            if amount <= 0.0:
                continue
            result = route_declared_dividend(self, firm, amount)
            firm.dividend_payment = result.declared_dividend
            paid += result.declared_dividend
        self._last_dividend_routing_person_paid = math.fsum(float(getattr(firm, "person_dividend_paid", 0.0)) for firm in firms)
        self._last_dividend_routing_legacy_entitlement = math.fsum(float(getattr(firm, "legacy_dividend_entitlement", 0.0)) for firm in firms)
        self._last_dividend_routing_estate_paid = math.fsum(float(getattr(firm, "estate_dividend_paid", 0.0)) for firm in firms)
        self._last_dividend_routing_gap = math.fsum(float(getattr(firm, "dividend_routing_gap", 0.0)) for firm in firms)
        return paid
    def assign_bootstrap_founders(self):
        """Formation-layer entry point; Firm constructors remain owner-neutral."""
        return assign_founders(self)

    def bootstrap_fresh_multi_firm_state(self, firm_result, shares):
        """Transfer the first aggregate market's physical supply once."""
        if (
            len(self.firms) <= 1
            or getattr(self, "multi_firm_bootstrap_complete", True)
        ):
            return

        aggregate = self.firm_system
        opening_inventory = max(
            0.0,
            firm_result.get(
                "pre_market_inventory_units",
                getattr(aggregate, "pre_market_inventory_units", 0.0),
            ),
        )
        production = max(
            0.0,
            firm_result.get(
                "pre_market_production_units",
                getattr(aggregate, "pre_market_production_units", 0.0),
            ),
        )
        expected_demand = max(
            0.0,
            firm_result.get("food_demand_units", 0.0),
        )

        inventory_values = self.split_total_by_shares(opening_inventory, shares)
        production_values = self.split_total_by_shares(production, shares)
        demand_values = self.split_total_by_shares(expected_demand, shares)

        self.bootstrap_cash_gap = 0.0
        self.bootstrap_inventory_units_gap = (
            opening_inventory - math.fsum(inventory_values)
        )
        self.bootstrap_inventory_book_value_gap = 0.0
        self.bootstrap_capacity_gap = 0.0
        self.bootstrap_wage_bill_gap = 0.0
        self.bootstrap_expected_demand_gap = (
            expected_demand - math.fsum(demand_values)
        )
        self.bootstrap_production_gap = (
            production - math.fsum(production_values)
        )
        self.bootstrap_loan_gap = 0.0

        for index, firm in enumerate(self.firms):
            firm.inventory_units = inventory_values[index]
            firm.inventory_value = inventory_values[index] * firm.price
            firm.expected_demand = demand_values[index]
            firm.observed_demand = demand_values[index]
            firm.production_plan = production_values[index]
            firm.desired_production = production_values[index]
            firm.actual_production = production_values[index]
            firm.previous_actual_production = production_values[index]
            firm.bootstrap_production_units = production_values[index]

        self.multi_firm_bootstrap_complete = True
        self.multi_firm_bootstrap_execution_count = (
            getattr(self, "multi_firm_bootstrap_execution_count", 0) + 1
        )

    def _labor_capacity_cache_key(self, firm):
        """Build a cheap same-week key for unchanged employee lists."""
        employee_ids = getattr(firm, "employee_ids", [])
        return (
            getattr(self, "current_step_index", None),
            tuple(employee_ids),
        )
    def firm_employee_capacity(self, firm):
        cache = getattr(self, "_labor_capacity_cache", None)
        if cache is None:
            cache = {}
            self._labor_capacity_cache = cache
        step = getattr(self, "current_step_index", None)
        if getattr(self, "_labor_capacity_cache_step", None) != step:
            cache.clear()
            self._labor_capacity_cache_step = step

        cache_key = self._labor_capacity_cache_key(firm)
        firm_key = getattr(firm, "firm_id", id(firm))
        cached = cache.get(firm_key)
        if cached is not None and cached[0] == cache_key:
            self.labor_capacity_cache_hits = getattr(self, "labor_capacity_cache_hits", 0) + 1
            return cached[1]

        self.labor_capacity_cache_misses = getattr(self, "labor_capacity_cache_misses", 0) + 1
        capacity = 0.0
        alive_employee_ids = []

        for person_id in firm.employee_ids:
            person = self.person_dict.get(person_id)

            if (
                person is None
                or not person.alive
                or not self.has_valid_settlement_household(person)
                or not self.labor_participates(person)
            ):
                continue

            productivity = age_productivity(person.age)

            if productivity <= 0:
                continue

            alive_employee_ids.append(person_id)
            capacity += productivity

        firm.employee_ids = alive_employee_ids
        # Store under the post-cleanup key so invalid legacy ids do not force
        # a second equivalent scan in the same week.
        post_key = self._labor_capacity_cache_key(firm)
        cache[firm_key] = (post_key, capacity)
        return capacity
    def get_firm_by_id(self, firm_id, default=None):
        """Return a Firm by stable ID through the authoritative index."""
        lookup = getattr(self, "firm_by_id", None)
        if not isinstance(lookup, dict):
            lookup = getattr(self, "firm_dict", {})
        return lookup.get(firm_id, default)
    def operating_firms(self):
        """Return Food and optionally active capital-good Firms."""
        return [*getattr(self, "firms", []), *getattr(self, "capital_good_firms", [])]

    def apply_active_personal_income_tax(self, step):
        if not getattr(self, "personal_income_tax_enabled", False):
            return {"scheduled": 0.0, "actual": 0.0, "shortfall": 0.0, "records": []}
        self.public_budget.reset_step()
        result = settle_personal_income_tax(
            self, step, getattr(self, "personal_income_tax_brackets", [])
        )
        if (int(step) + 1) % 52 == 0:
            year_end_reconcile(self, step)
        return result

    def collect_public_tax(self, step):
        if getattr(self, "corporate_profit_tax_enabled", False):
            base = "FIRM_POSITIVE_TAXABLE_PROFIT"
            rate = getattr(self, "corporate_profit_tax_rate", 0.0)
        else:
            base = self.public_revenue_tax_base
            rate = self.public_revenue_tax_rate
        reset = not getattr(self, "personal_income_tax_enabled", False)
        budget = collect_firm_tax(self, step, base, rate, reset=reset)
        self.public_budget_weekly_history = budget.weekly_history
        food_ids = {getattr(firm, "firm_id", None) for firm in getattr(self, "firms", [])}
        food_tax = sum(row["actual_tax"] for row in budget.tax_records if row.get("firm_id") in food_ids)
        if not getattr(self, "corporate_profit_tax_enabled", False) and not getattr(self, "personal_income_tax_enabled", False):
            self.firm_system.cash = max(0.0, float(getattr(self.firm_system, "cash", 0.0)) - food_tax)
        return budget

    def transfer_public_pension(self, step, scheduled_amount):
        """Transfer only the current pension residual from Government to Fund."""
        budget = self.public_budget
        scheduled = max(0.0, float(scheduled_amount or 0.0))
        actual = min(scheduled, max(0.0, float(getattr(budget, "cash", 0.0))))
        if actual > 1e-12:
            self.ledger.transfer_attrs(
                payer_obj=budget, payer_attr="cash",
                payer_name=budget.account_id, receiver_obj=self.social_insurance_fund,
                receiver_attr="cash", receiver_name=self.social_insurance_fund.fund_id,
                amount=actual, reason="government_public_pension_transfer"
            )
        shortfall = max(0.0, scheduled - actual)
        budget.scheduled_public_pension_transfer_this_step = scheduled
        budget.actual_public_pension_transfer_this_step = actual
        budget.public_pension_transfer_shortfall_this_step = shortfall
        budget.public_expenditure_outflow_this_step = actual
        if budget.weekly_history:
            row = budget.weekly_history[-1]
            opening = float(row.get("opening_government_cash", 0.0))
            row["scheduled_public_pension_transfer"] = scheduled
            row["actual_public_pension_transfer"] = actual
            row["public_pension_transfer_shortfall"] = shortfall
            row["public_expenditure"] = actual
            row["closing_government_cash"] = float(budget.cash)
            row["government_stock_flow_gap"] = float(budget.cash) - opening - float(row.get("actual_tax", 0.0)) + actual
        return {"scheduled": scheduled, "actual": actual, "shortfall": shortfall}

    def prepare_canonical_investment_week(self, step_index):
        system = getattr(self, "canonical_investment_system", None)
        if system is None:
            self.pending_capital_wages = {}
            self.pending_capital_person_wages = {}
            return None
        result = system.prepare_week(step_index)
        self.pending_capital_wages = dict(system.pending_household_wages)
        self.pending_capital_person_wages = dict(
            getattr(system, "pending_person_wages", {})
        )
        self.last_canonical_investment_week = result
        return result

    def add_pending_capital_wages(self):
        system = getattr(self, "canonical_investment_system", None)
        if system is None:
            return 0.0
        return system.add_pending_wages()

    def ensure_multisector_foundation_contracts(self):
        """Backfill passive 14A identities and optionally build its views.

        This is deliberately safe for pre-14A checkpoints.  The OFF path only
        adds neutral identity defaults; the ON path constructs read-only
        registries and views but never invokes an economic executor.
        """
        if not hasattr(self, "multisector_foundation_enabled"):
            self.multisector_foundation_enabled = False
        if not getattr(self, "firms", None):
            # Historical single-firm checkpoints predate the passive
            # FirmSlice view.  Reconstruct the existing compatibility view;
            # this does not create a new firm or change economic state.
            ensure_single_firm_view(self)
        for firm in getattr(self, "firms", []):
            defaults = default_firm_identity()
            for name, value in defaults.items():
                if not hasattr(firm, name):
                    setattr(firm, name, value)
        if not self.multisector_foundation_enabled:
            self.multisector_foundation = None
            return None
        foundation = getattr(self, "multisector_foundation", None)
        if foundation is None or getattr(foundation, "VERSION", None) != 1:
            foundation = MultiSectorFoundation(
                self.goods_catalog,
                self.operating_firms(),
            )
            self.multisector_foundation = foundation
        else:
            foundation.refresh_firm_registry(self.operating_firms())
        return foundation

    def ensure_household_demand_system(self):
        """Lazily expose the passive Step 14C demand boundary."""
        if not hasattr(self, "shadow_multigood_household_demand_enabled"):
            self.shadow_multigood_household_demand_enabled = False
        if not self.shadow_multigood_household_demand_enabled:
            self.household_demand_system = None
            return None
        system = getattr(self, "household_demand_system", None)
        if system is None or getattr(system, "VERSION", None) != 1:
            system = HouseholdDemandSystem(self)
            self.household_demand_system = system
        return system

    def ensure_stageA_operating_contract_adapter(self):
        """Lazily reconstruct the passive Stage A adapter for old checkpoints."""
        if not getattr(self, "generalized_firm_operating_contracts", False):
            return None
        adapter = getattr(self, "stageA_operating_contract_adapter", None)
        if adapter is None or getattr(adapter, "VERSION", None) != 1:
            good = self.goods_catalog.get(
                economy_config.BASIC_CONSUMPTION_GOOD_ID
            )
            adapter = StageAOperatingContractAdapter(
                good,
                self.firm_system.food_productivity,
            )
            self.stageA_operating_contract_adapter = adapter
        return adapter

    def assign_unassigned_workers_to_firms(self, include_capital_goods=False):
        if not getattr(self, "firms", None):
            return

        self.firm_dict = {
            getattr(firm, "firm_id", index): firm
            for index, firm in enumerate([*self.firms, *getattr(self, "capital_good_firms", [])])
        }
        self.firm_by_id = self.firm_dict

        capacities = [
            self.firm_employee_capacity(firm)
            for firm in self.firms
        ]

        # The canonical capital-good executor may request workers after its
        # inventory backlog is observed.  It uses this same deterministic
        # unassigned-worker pool before the ordinary Food balancing fallback;
        # the default path remains byte-for-byte unchanged.
        capital_firms = (
            list(getattr(self, "capital_good_firms", []))
            if include_capital_goods
            else []
        )
        capital_capacities = {
            firm.firm_id: self.firm_employee_capacity(firm)
            for firm in capital_firms
        }
        capital_hiring_order = sorted(
            capital_firms,
            key=lambda firm: str(getattr(firm, "firm_id", "")),
        )

        for person in self.population:
            if (
                not person.alive
                or not self.has_valid_settlement_household(person)
                or not self.labor_participates(person)
            ):
                continue

            productivity = age_productivity(person.age)

            # ``firm_id=None`` is the explicit unassigned state used by the
            # capital-good labor release boundary.  Older paths with a valid
            # employer id retain exactly the previous behavior.
            if (
                productivity <= 0
                or not self.labor_participates(person)
                or getattr(person, "firm_id", None) is not None
            ):
                continue

            if capital_hiring_order:
                candidates = [
                    firm for firm in capital_hiring_order
                    if capital_capacities[firm.firm_id]
                    + 1e-12
                    < max(0.0, float(getattr(firm, "desired_labor", 0.0)))
                ]
                if candidates:
                    firm = min(
                        candidates,
                        key=lambda item: (
                            max(0.0, float(getattr(item, "desired_labor", 0.0)))
                            - capital_capacities[item.firm_id],
                            str(getattr(item, "firm_id", "")),
                        ),
                    )
                    person.firm_id = firm.firm_id
                    firm.employee_ids.append(person.id)
                    capital_capacities[firm.firm_id] += productivity
                    continue

            index = min(range(len(self.firms)), key=lambda item: capacities[item])
            person.firm_id = self.firms[index].firm_id
            self.firms[index].employee_ids.append(person.id)
            capacities[index] += productivity
    def split_total_by_shares(self, total, shares):
        values = []
        assigned = 0.0

        for index, share in enumerate(shares):
            if index == len(shares) - 1:
                value = total - assigned
            else:
                value = total * share
                assigned += value

            values.append(value)

        return values

    def ensure_market_rng(self):
        if not hasattr(self, "market_rng"):
            self.market_rng = random.Random(
                f"{self.seed}:market"
                if self.seed is not None
                else None
            )

        if not hasattr(self, "price_choice_sensitivity"):
            self.price_choice_sensitivity = (
                economy_config.PRICE_CHOICE_SENSITIVITY
            )

        if not hasattr(self, "initial_firm_relative_price_spread"):
            self.initial_firm_relative_price_spread = (
                economy_config.INITIAL_FIRM_RELATIVE_PRICE_SPREAD
            )

        for firm in getattr(self, "firms", []):
            ensure_default_state(firm)
            ensure_restructuring_state(firm)
            if not hasattr(firm, "firm_rng") or firm.firm_rng is None:
                firm.firm_rng = random.Random(
                    f"{self.seed}:firm:{firm.firm_id}"
                    if self.seed is not None
                    else None
                )

            defaults = {
                "expected_profit": 0.0,
                "expected_share": getattr(firm, "share", 0.0),
                "last_price_direction": 0,
                "last_review_price": getattr(firm, "price", 0.0),
                "last_review_profit": getattr(firm, "profit", 0.0),
                "last_review_share": getattr(firm, "actual_market_share", 0.0),
                "last_review_inventory_gap": 0.0,
                "price_reviewed": False,
                "price_decision": "none",
                "price_change": 0.0,
                "unit_labor_cost": 0.0,
                "margin": 0.0,
                "inventory_coverage": 0.0,
                "inventory_gap": 0.0,
                "smoothed_profit": getattr(firm, "profit", 0.0),
                "baseline_profit": getattr(firm, "profit", 0.0),
                "evaluated_profit": getattr(firm, "profit", 0.0),
                "estimated_gradient": 0.0,
                "evaluation_timer": 0,
                "last_test_price": getattr(firm, "price", 0.0),
                "last_direction": 0,
                "last_profit_delta": 0.0,
                "cash_start": getattr(firm, "cash", 0.0),
                "sales_revenue": getattr(firm, "sales", 0.0),
                "wage_payment": getattr(firm, "wage_bill", 0.0),
                "target_cash": 0.0,
                "funding_gap": 0.0,
                "dividend_payment": 0.0,
                "dividend_eligible_profit": 0.0,
                "dividend_payout_ratio": 0.0,
                "retained_profit_after_dividend": 0.0,
                "dividend_funding_source": "none",
                "cash_before_dividend": getattr(firm, "cash", 0.0),
                "cash_after_dividend": getattr(firm, "cash", 0.0),
                "public_sector_cash_inflow": 0.0,
                "public_sector_cash_outflow": 0.0,
                "loan_interest_paid": 0.0,
                "other_cash_inflow": 0.0,
                "other_cash_outflow": 0.0,
                "cash_end": getattr(firm, "cash", 0.0),
                "cash_bridge_gap": 0.0,
                "cash_depletion_stage": "not_depleted",
                "credit_allocation_mode": "not_recorded",
                "observed_demand": 0.0,
                "target_inventory_units": 0.0,
                "inventory_gap_units": 0.0,
                "production_plan": getattr(firm, "production_plan", 0.0),
                "desired_production": getattr(firm, "desired_production", 0.0),
                "actual_production": getattr(firm, "actual_production", 0.0),
                "capacity_utilization": 0.0,
                "production_reviewed": False,
                "production_change": 0.0,
                "production_review_timer": 0,
                "production_review_phase_offset": 0,
                "production_review_count": 0,
                "investment_review_phase_offset": 0,
                "investment_reviewed_this_step": False,
                "investment_review_count": 0,
                "previous_actual_production": getattr(firm, "previous_actual_production", 0.0),
            }

            for name, value in defaults.items():
                if not hasattr(firm, name):
                    setattr(firm, name, value)

    def begin_restructuring_week(self):
        return begin_restructuring_week(
            self.operating_firms(),
            getattr(self, "temporary_interest_relief_enabled", False),
            getattr(self, "temporary_interest_relief_eligibility_weeks", 26),
            getattr(self, "temporary_interest_relief_max_weeks", 26),
            getattr(self, "snapshot_legacy_arrears_termout_enabled", False),
        )

    def update_default_bookkeeping(self, step):
        results = []
        for firm in self.operating_firms():
            results.append(update_firm_default_bookkeeping(firm, step))
        finalize_restructuring_week(
            self.operating_firms(),
            getattr(self, "temporary_interest_relief_enabled", False),
            getattr(self, "temporary_interest_relief_eligibility_weeks", 26),
            getattr(self, "temporary_interest_relief_max_weeks", 26),
        )
        return results

    def ensure_default_bookkeeping_state(self):
        """Backfill passive Default fields for old World/Firm pickles."""
        for firm in getattr(self, "firms", []):
            ensure_default_state(firm)
            ensure_restructuring_state(firm)

    def ensure_capital_stock_state(self):
        """Backfill the passive Step 15B capital container on old pickles."""
        from economy.investment_contracts import CapitalStock

        if not hasattr(self, "capital_good_customer_advance_enabled"):
            self.capital_good_customer_advance_enabled = bool(
                getattr(self, "scenario_overrides", {}).get(
                    "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED", False
                )
            )

        aggregate = getattr(self, "firm_system", None)
        if aggregate is not None and not hasattr(aggregate, "capital_stock"):
            aggregate.capital_stock = CapitalStock(owner_firm_id=0)

        all_firms = [
            *getattr(self, "firms", []),
            *getattr(self, "capital_good_firms", []),
        ]
        for index, firm in enumerate(all_firms):
            firm_id = getattr(firm, "firm_id", index)
            if not hasattr(firm, "paid_in_equity"):
                firm.paid_in_equity = 0.0
            stock = getattr(firm, "capital_stock", None)
            if stock is None:
                firm.capital_stock = CapitalStock(owner_firm_id=firm_id)
            elif getattr(stock, "owner_firm_id", firm_id) != firm_id:
                raise ValueError(
                    "Checkpoint capital stock owner does not match firm_id"
                )
            for name, default in {
                "customer_advance_liability": 0.0,
                "customer_advance_received_this_step": 0.0,
                "customer_advance_delivered_this_step": 0.0,
                "prepaid_capital_investment_asset": 0.0,
                "prepaid_capital_investment_paid_this_step": 0.0,
                "prepaid_capital_investment_capitalized_this_step": 0.0,
                "sales_collections_this_step": 0.0,
                "investment_cash_outflow_this_step": 0.0,
            }.items():
                if not hasattr(firm, name):
                    setattr(firm, name, default)
        system = getattr(self, "canonical_investment_system", None)
        if system is not None:
            if not hasattr(system, "customer_advance_enabled"):
                system.customer_advance_enabled = bool(
                    self.capital_good_customer_advance_enabled
                )
            if not hasattr(system, "customer_advance_fraction"):
                system.customer_advance_fraction = 1.0
            if not hasattr(system, "customer_advance_pending_orders"):
                system.customer_advance_pending_orders = {}
            if not hasattr(system, "customer_advance_ledger"):
                system.customer_advance_ledger = []
        if not hasattr(self, "capital_asset_event_rows"):
            self.capital_asset_event_rows = []

    def ensure_ownership_state(self):
        """Backfill passive Legacy-only CapTables on old/current Worlds."""
        ensure_dividend_routing_state(self)
        self.ensure_shareholder_estate_state()
        if not hasattr(self, "equity_ownership_system"):
            self.equity_ownership_system = EquityOwnershipSystem()

        aggregate = getattr(self, "firm_system", None)
        if aggregate is not None:
            aggregate_table = getattr(aggregate, "cap_table", None)
            if aggregate_table is None:
                aggregate.cap_table = CapTable.initial_legacy_owned(0)
            elif aggregate_table.firm_id != 0:
                raise ValueError("aggregate Firm cap table must use firm_id=0")

        for index, firm in enumerate([*getattr(self, "firms", []), *getattr(self, "capital_good_firms", [])]):
            firm_id = getattr(firm, "firm_id", index)
            table = getattr(firm, "cap_table", None)
            if table is None:
                table = CapTable.initial_legacy_owned(firm_id)
                firm.cap_table = table
            elif table.firm_id != firm_id:
                raise ValueError("Firm cap table owner does not match firm_id")
            table.validate()
            self.equity_ownership_system.cap_tables[firm_id] = table
        for household in getattr(self, "households", []):
            if not hasattr(household, "equity_asset_value"):
                household.equity_asset_value = 0.0
        for person in getattr(self, "population", []):
            if not hasattr(person, "equity_holdings"):
                person.equity_holdings = {}
            if not hasattr(person, "equity_cost_basis"):
                person.equity_cost_basis = {}
        return self.equity_ownership_system

    def ensure_shareholder_estate_state(self):
        """Backfill Estate holders without inventing historical ownership."""
        if not hasattr(self, "estate_accounts"):
            self.estate_accounts = {}
        if not hasattr(self, "shareholder_estate_by_deceased"):
            self.shareholder_estate_by_deceased = {}
        if not hasattr(self, "shareholder_estate_events"):
            self.shareholder_estate_events = []
        system = getattr(self, "shareholder_estate_system", None)
        if system is None or getattr(system, "world", None) is not self:
            self.shareholder_estate_system = ShareholderEstateSystem(self)
        for person in getattr(self, "population", []):
            if not hasattr(person, "equity_holdings"):
                person.equity_holdings = {}
            if not hasattr(person, "equity_cost_basis"):
                person.equity_cost_basis = {}
        return self.shareholder_estate_system

    def ensure_active_social_policy_state(self):
        """Migrate old checkpoints to the neutral Step17.D/E state."""
        if not hasattr(self, "social_insurance_fund"):
            self.social_insurance_fund = SocialInsuranceFund()
        if not hasattr(self, "public_budget") or self.public_budget is None:
            self.public_budget = GovernmentPublicBudget()
        if not hasattr(self, "public_budget_weekly_history") or self.public_budget_weekly_history is None:
            self.public_budget_weekly_history = []
        if not hasattr(self, "payg_pension_system"):
            self.payg_pension_system = PaygPensionSystem(self, self.social_insurance_fund)
        else:
            self.payg_pension_system.world = self
            self.payg_pension_system.fund = self.social_insurance_fund
        if not hasattr(self, "intergenerational_wealth_transfer_system"):
            self.intergenerational_wealth_transfer_system = IntergenerationalWealthTransferSystem(self)
        else:
            self.intergenerational_wealth_transfer_system.world = self
        if not hasattr(self, "active_social_policy_history"):
            self.active_social_policy_history = []
        if not hasattr(self, "payg_weekly_history"):
            self.payg_weekly_history = []
        if not hasattr(self, "wealth_transfer_weekly_history"):
            self.wealth_transfer_weekly_history = []
        if not hasattr(self, "recipient_policy_instrumentation_enabled"):
            self.recipient_policy_instrumentation_enabled = False
        if not hasattr(self, "active_social_policy_branch_name"):
            self.active_social_policy_branch_name = "UNSPECIFIED"
        if not hasattr(self, "active_social_recipient_events"):
            self.active_social_recipient_events = []
        if not hasattr(self, "_active_social_policy_observation"):
            self._active_social_policy_observation = None
        if not hasattr(self, "payg_cash_safety_audit_enabled"):
            self.payg_cash_safety_audit_enabled = False
        if not hasattr(self, "payg_cash_safety_audit_rows"):
            self.payg_cash_safety_audit_rows = []
        if not hasattr(self, "_payg_cash_safety_audit_current"):
            self._payg_cash_safety_audit_current = {}
        if not hasattr(self, "household_liquidity_snapshot_enabled"):
            self.household_liquidity_snapshot_enabled = False
        if not hasattr(self, "household_liquidity_snapshot_rows"):
            self.household_liquidity_snapshot_rows = []
        if not hasattr(self, "long_horizon_liquidity_enabled"):
            self.long_horizon_liquidity_enabled = False
        if not hasattr(self, "long_horizon_liquidity_snapshot_cadence"):
            self.long_horizon_liquidity_snapshot_cadence = 13
        if not hasattr(self, "long_horizon_weekly_liquidity_rows"):
            self.long_horizon_weekly_liquidity_rows = []
        if not hasattr(self, "long_horizon_household_snapshot_rows"):
            self.long_horizon_household_snapshot_rows = []
        if not hasattr(self, "long_horizon_household_snapshot_stream_path"):
            self.long_horizon_household_snapshot_stream_path = None
        if not hasattr(self, "long_horizon_transition_rows"):
            self.long_horizon_transition_rows = []
        if not hasattr(self, "long_horizon_lifecycle_rows"):
            self.long_horizon_lifecycle_rows = []
        if not hasattr(self, "_long_horizon_previous_liquidity"):
            self._long_horizon_previous_liquidity = {}
        if not hasattr(self, "_long_horizon_previous_households"):
            self._long_horizon_previous_households = set()
        for name, default in {
            "payg_pension_enabled": False,
            "payg_contribution_rate": 0.0,
            "payg_pension_target_multiplier": 0.0,
            "payg_employer_contribution_rate": 0.0,
            "public_pension_transfer_enabled": False,
            "public_revenue_tax_enabled": False,
            "public_revenue_tax_base": "FIRM_SALES_TAX",
            "public_revenue_tax_rate": 0.0,
            "personal_income_tax_enabled": False,
            "personal_income_tax_brackets": [],
            "corporate_profit_tax_enabled": False,
            "corporate_profit_tax_rate": 0.0,
            "personal_tax_year_reconciliation": [],
            "intergenerational_wealth_transfer_enabled": False,
            "intergenerational_reserve_weeks": 13.0,
            "intergenerational_donor_surplus_share": 0.25,
            "intergenerational_recipient_target_weeks": 1.0,
        }.items():
            if not hasattr(self, name):
                setattr(self, name, default)
    def _active_social_household_metrics(self, household):
        try:
            minimum_units = float(self.needs_system.household_minimum_need_units(household))
            price = float(self.household_planning_price_this_step())
            minimum_cost = max(0.0, minimum_units * price)
        except (AttributeError, TypeError, ValueError):
            minimum_cost = 0.0
        cash = float(getattr(household, "wealth", 0.0))
        liquidity = cash / minimum_cost if minimum_cost > 1e-12 else float("nan")
        return cash, liquidity, minimum_cost
    def _active_social_household_meta(self, household):
        members = [self.get_person_by_id(person_id) for person_id in [*getattr(household, "parents", []), *getattr(household, "children", [])]]
        members = [person for person in members if person is not None and getattr(person, "alive", False)]
        elderly = any(float(getattr(person, "age", 0.0)) >= 65.0 for person in members)
        neighbors = set()
        for person in members:
            for relative_id in [*getattr(person, "parent_ids", []), *getattr(person, "children_ids", [])]:
                relative = self.get_person_by_id(relative_id)
                relative_household = self.get_household(getattr(relative, "household_id", None)) if relative is not None else None
                if relative_household is not None and relative_household.id != household.id and not getattr(relative_household, "settlement_only", False):
                    neighbors.add(relative_household.id)
        return elderly, elderly and not neighbors
    def _active_social_snapshot(self):
        snapshot = {}
        for household in getattr(self, "households", []):
            cash, liquidity, minimum_cost = self._active_social_household_metrics(household)
            snapshot[household.id] = {"cash": cash, "liquidity": liquidity, "minimum_cost": minimum_cost}
        return snapshot
    def begin_active_social_recipient_observation(self, step):
        if not getattr(self, "recipient_policy_instrumentation_enabled", False):
            self._active_social_policy_observation = None
            return
        self._active_social_policy_observation = {
            "step": step,
            "branch": getattr(self, "active_social_policy_branch_name", "UNSPECIFIED"),
            "before": self._active_social_snapshot(),
            "meta": {household.id: self._active_social_household_meta(household) for household in getattr(self, "households", [])},
        }
    def _update_active_social_followups(self, step):
        for event in getattr(self, "active_social_recipient_events", []):
            event_step = int(event.get("week", -1))
            for offset, cash_key, liquidity_key, status_key in ((1, "next_week_cash", "next_week_liquidity", "next_week_status"), (4, "plus4_week_cash", "plus4_week_liquidity", "plus4_week_status")):
                if step != event_step + offset or event.get(status_key) != "PENDING":
                    continue
                household = self.get_household(event.get("household_id"))
                if household is None or getattr(household, "settlement_only", False):
                    event[status_key] = "NOT_AVAILABLE_LIFECYCLE"
                    continue
                cash, liquidity, _ = self._active_social_household_metrics(household)
                event[cash_key] = cash
                event[liquidity_key] = liquidity
                event[status_key] = "AVAILABLE"
    def finalize_active_social_recipient_observation(self, step):
        observation = getattr(self, "_active_social_policy_observation", None)
        if observation is None:
            return
        self._update_active_social_followups(step)
        payg_result = getattr(self.payg_pension_system, "last_result", None)
        transfer_result = getattr(self.intergenerational_wealth_transfer_system, "last_result", None)
        pension_by_household = {}
        for row in getattr(payg_result, "pension_records", []) if payg_result is not None else []:
            pension_by_household[row["household_id"]] = pension_by_household.get(row["household_id"], 0.0) + float(row.get("actual_pension", 0.0))
        transfer_by_household = {}
        for row in getattr(transfer_result, "transfers", []) if transfer_result is not None else []:
            household_id = row["recipient_household_id"]
            transfer_by_household[household_id] = transfer_by_household.get(household_id, 0.0) + float(row.get("amount", 0.0))
        after_pension = observation.get("after_pension", {})
        after_private = observation.get("after_private", {})
        before = observation.get("before", {})
        meta = observation.get("meta", {})
        for household_id in sorted(set(pension_by_household) | set(transfer_by_household)):
            household = self.get_household(household_id)
            if household is None:
                continue
            pre = before.get(household_id, {})
            post_pension = after_pension.get(household_id, pre)
            post_private = after_private.get(household_id, post_pension)
            cash, liquidity, minimum_cost = self._active_social_household_metrics(household)
            elderly, uncovered = meta.get(household_id, (False, False))
            pension = pension_by_household.get(household_id, 0.0)
            private = transfer_by_household.get(household_id, 0.0)
            event = {
                "branch": observation.get("branch", "UNSPECIFIED"),
                "week": step,
                "household_id": household_id,
                "elderly_household": elderly,
                "genealogy_uncovered_flag": uncovered,
                "pension_received": pension,
                "private_transfer_received": private,
                "total_policy_receipt": pension + private,
                "cash_before_policy": pre.get("cash"),
                "liquidity_before_policy": pre.get("liquidity"),
                "cash_after_pension": post_pension.get("cash"),
                "liquidity_after_pension": post_pension.get("liquidity"),
                "cash_after_private_transfer": post_private.get("cash"),
                "liquidity_after_private_transfer": post_private.get("liquidity"),
                "cash_before_consumption": post_private.get("cash"),
                "liquidity_before_consumption": post_private.get("liquidity"),
                "same_week_consumption": float(getattr(household, "consumption_this_step", 0.0))
                    ,
                "end_week_cash": cash,
                "end_week_liquidity": liquidity,
                "minimum_need_cost": minimum_cost,
                "next_week_cash": None,
                "next_week_liquidity": None,
                "plus4_week_cash": None,
                "plus4_week_liquidity": None,
                "next_week_status": "PENDING",
                "plus4_week_status": "PENDING",
                "event_type": ("BOTH_RECEIPTS" if pension > 1e-12 and private > 1e-12 else "PENSION_ONLY_RECEIPT" if pension > 1e-12 else "PRIVATE_ONLY_RECEIPT"),
                "retention_semantics": "descriptive cash-position change; not tagged-money tracing",
            }
            self.active_social_recipient_events.append(event)
        self._active_social_policy_observation = None
    @staticmethod
    def _long_horizon_percentile(values, q):
        ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
        if not ordered:
            return float("nan")
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        lower = int(math.floor(position))
        upper = min(len(ordered) - 1, lower + 1)
        weight = position - lower
        return ordered[lower] + weight * (ordered[upper] - ordered[lower])

    @staticmethod
    def _long_horizon_gini(values):
        ordered = sorted(max(0.0, float(value)) for value in values if math.isfinite(float(value)))
        total = math.fsum(ordered)
        if not ordered or total <= 1e-12:
            return 0.0
        count = len(ordered)
        return math.fsum((2 * index - count - 1) * value for index, value in enumerate(ordered, 1)) / (count * total)

    def record_long_horizon_liquidity(self, firm_result):
        """Record compact end-of-week observations for Step17.G."""
        if not getattr(self, "long_horizon_liquidity_enabled", False):
            return
        step = int(getattr(self, "current_step_index", 0))
        social = [household for household in getattr(self, "households", []) if not getattr(household, "settlement_only", False)]
        records = []
        snapshots = []
        cadence = max(1, int(getattr(self, "long_horizon_liquidity_snapshot_cadence", 13)))
        for household in social:
            cash, _, minimum_cost = self._active_social_household_metrics(household)
            liquidity = cash / minimum_cost if minimum_cost > 1e-12 else float("nan")
            records.append((household.id, cash, liquidity))
            if step % cadence == 0:
                members = [self.get_person_by_id(pid) for pid in [*getattr(household, "parents", []), *getattr(household, "children", [])]]
                members = [person for person in members if person is not None and getattr(person, "alive", False)]
                elderly, uncovered = self._active_social_household_meta(household)
                snapshots.append({
                    "global_step": step, "household_id": household.id, "closing_cash": cash,
                    "weekly_minimum_consumption_cost": minimum_cost, "liquidity_weeks": liquidity,
                    "household_size": len(members), "elderly_household": bool(elderly),
                    "employed_member": any(getattr(person, "firm_id", None) is not None for person in members),
                    "genealogy_uncovered_elderly": bool(uncovered),
                    "wage_income": float(getattr(household, "wage_income_this_step", 0.0)),
                    "total_income": float(getattr(household, "income_this_step", 0.0)),
                    "consumption": float(getattr(household, "consumption_this_step", 0.0)),
                    "saving": float(getattr(household, "saving_this_step", 0.0)),
                })
        cash_values = [row[1] for row in records]
        liquidity_values = [row[2] for row in records if math.isfinite(float(row[2]))]
        sorted_cash = sorted(cash_values)
        cash_total = math.fsum(cash_values)
        count = len(records)
        share = lambda threshold: sum(math.isfinite(float(row[2])) and float(row[2]) < threshold for row in records) / count if count else float("nan")
        eligible = [person for person in getattr(self, "population", []) if getattr(person, "alive", False) and 20 * 52 <= int(getattr(person, "age_weeks", 0)) <= 60 * 52]
        employed = sum(getattr(person, "firm_id", None) is not None for person in eligible)
        alive_people = [person for person in getattr(self, "population", []) if getattr(person, "alive", False)]
        elderly_people = sum(float(getattr(person, "age", 0.0)) >= 65.0 for person in alive_people)
        working_age_people = sum(20 * 52 <= int(getattr(person, "age_weeks", 0)) <= 60 * 52 for person in alive_people)
        employed_households = {getattr(person, "household_id", None) for person in eligible if getattr(person, "firm_id", None) is not None}
        components = self.authoritative_money_location_components()
        central_bank = getattr(getattr(self, "firm_system", None), "central_bank", None)
        income = float(self.income_history[-1]) if self.income_history else 0.0
        consumption = float(self.consumption_history[-1]) if self.consumption_history else 0.0
        saving = float(self.saving_history[-1]) if self.saving_history else income - consumption
        self.long_horizon_weekly_liquidity_rows.append({
            "global_step": step, "population": len(getattr(self, "population", [])), "social_households": count,
            "household_cash_total": cash_total, "household_cash_median": self._long_horizon_percentile(cash_values, .5),
            "liquidity_share_lt_025": share(.25), "liquidity_share_lt_05": share(.5), "liquidity_share_lt_1": share(1.0), "liquidity_share_lt_2": share(2.0),
            "liquidity_median": self._long_horizon_percentile(liquidity_values, .5), "liquidity_p10": self._long_horizon_percentile(liquidity_values, .1),
            "liquidity_p25": self._long_horizon_percentile(liquidity_values, .25), "liquidity_p75": self._long_horizon_percentile(liquidity_values, .75), "liquidity_p90": self._long_horizon_percentile(liquidity_values, .9),
            "cash_gini": self._long_horizon_gini(cash_values), "cash_bottom50_share": math.fsum(sorted_cash[:max(1, count // 2)]) / cash_total if cash_total > 1e-12 else 0.0,
            "cash_top10_share": math.fsum(sorted_cash[max(0, count - max(1, math.ceil(count * .1))):]) / cash_total if cash_total > 1e-12 else 0.0,
            "cash_top1_share": math.fsum(sorted_cash[max(0, count - max(1, math.ceil(count * .01))):]) / cash_total if cash_total > 1e-12 else 0.0,
            "total_employment": sum(len(getattr(firm, "employee_ids", []) or []) for firm in self.operating_firms()), "unassigned_labor": max(0, len(eligible) - employed),
            "elderly_share": elderly_people / len(alive_people) if alive_people else float("nan"),
            "working_age_share": working_age_people / len(alive_people) if alive_people else float("nan"),
            "employed_household_share": len(employed_households) / count if count else float("nan"),
            "mean_household_size": (sum(len([pid for pid in [*getattr(h, "parents", []), *getattr(h, "children", [])] if self.get_person_by_id(pid) is not None and getattr(self.get_person_by_id(pid), "alive", False)]) for h in social) / count) if count else float("nan"),
            "wage_income": income, "household_consumption": consumption, "household_saving": saving,
            "firm_cash": components["firm_cash"], "located_money_stock": components["located_money_stock"], "central_bank_money": float(getattr(central_bank, "money_supply", 0.0)),
            "food_sales": float(firm_result.get("sales", 0.0)), "firm_production": float(firm_result.get("production", 0.0)), "firm_revenue": float(firm_result.get("firm_sales_revenue", firm_result.get("sales", 0.0))),
            "firm_cfo": float(firm_result.get("cfo", firm_result.get("cash_flow_from_operations", 0.0))), "loans": float(firm_result.get("loan_balance", 0.0)),
        })
        if self.diagnostics_rows:
            diagnostic = self.diagnostics_rows[-1]
            self.long_horizon_weekly_liquidity_rows[-1].update({
                "accounting_gap": float(diagnostic.get("accounting_gap", diagnostic.get("monetary_accounting_gap", 0.0)) or 0.0),
                "money_gap": float(diagnostic.get("money_delta_gap", diagnostic.get("monetary_accounting_gap", 0.0)) or 0.0),
                "goods_gap": float(diagnostic.get("food_conservation_gap", 0.0) or 0.0),
                "fixed_investment": float(firm_result.get("fixed_investment", 0.0) or 0.0),
                "expansion_investment": float(firm_result.get("expansion_investment", 0.0) or 0.0),
                "replacement_investment": float(firm_result.get("replacement_investment", 0.0) or 0.0),
                "money_created": float(getattr(central_bank, "money_issued_this_step", 0.0) or 0.0),
                "money_destroyed": float(getattr(central_bank, "money_destroyed_this_step", 0.0) or 0.0),
            })
        current = {row[0]: row[2] for row in records}
        previous = dict(getattr(self, "_long_horizon_previous_liquidity", {}) or {})
        previous_ids = set(previous); current_ids = set(current); matched = previous_ids & current_ids
        entries = sum(previous[hid] >= .25 and current[hid] < .25 for hid in matched)
        exits = sum(previous[hid] < .25 and current[hid] >= .25 for hid in matched)
        persistent = sum(previous[hid] < .25 and current[hid] < .25 for hid in matched)
        previous_low = sum(previous[hid] < .25 for hid in matched); current_low = sum(current[hid] < .25 for hid in matched)
        self.long_horizon_transition_rows.append({"global_step": step, "threshold": .25, "matched_households": len(matched), "entry_count": entries, "exit_count": exits, "persistent_low_count": persistent, "previous_low_count": previous_low, "current_low_count": current_low, "disappeared_households": len(previous_ids - current_ids), "new_households": len(current_ids - previous_ids)})
        prior_households = set(getattr(self, "_long_horizon_previous_households", set()) or set())
        self.long_horizon_lifecycle_rows.append({"global_step": step, "same_households": len(prior_households & current_ids), "household_formations": len(current_ids - prior_households), "household_dissolutions": len(prior_households - current_ids), "same_household_low_entries": entries, "same_household_low_exits": exits})
        self._long_horizon_previous_liquidity = current
        self._long_horizon_previous_households = current_ids
        stream_path = getattr(self, "long_horizon_household_snapshot_stream_path", None)
        if stream_path:
            snapshot_fields = [
                "global_step", "household_id", "closing_cash", "weekly_minimum_consumption_cost",
                "liquidity_weeks", "household_size", "elderly_household", "employed_member",
                "genealogy_uncovered_elderly", "wage_income", "total_income", "consumption", "saving",
            ]
            stream = os.fspath(stream_path)
            os.makedirs(os.path.dirname(stream) or ".", exist_ok=True)
            exists = os.path.exists(stream) and os.path.getsize(stream) > 0
            with open(stream, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=snapshot_fields, extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                writer.writerows(snapshots)
        else:
            self.long_horizon_household_snapshot_rows.extend(snapshots)
    def record_payg_cash_safety_phase(self, phase):
        """Collect diagnostics-only Household cash states around PAYG/consumption."""
        if not getattr(self, "payg_cash_safety_audit_enabled", False):
            return
        step = int(getattr(self, "current_step_index", 0))
        if phase == "opening":
            self._payg_cash_safety_audit_current = {}
        pension_by_household = defaultdict(float)
        payg_result = getattr(getattr(self, "payg_pension_system", None), "last_result", None)
        for item in getattr(payg_result, "pension_records", []) if payg_result is not None else []:
            pension_by_household[item.get("household_id")] += float(item.get("actual_pension", 0.0))
        for household in getattr(self, "households", []):
            household_id = household.id
            row = self._payg_cash_safety_audit_current.setdefault(household_id, {
                "week": step,
                "household_id": household_id,
            })
            row[f"cash_{phase}"] = float(getattr(household, "wealth", 0.0))
            if phase == "opening":
                members = [self.get_person_by_id(pid) for pid in [*getattr(household, "parents", []), *getattr(household, "children", [])]]
                members = [person for person in members if person is not None and getattr(person, "alive", False)]
                row.update({
                    "elderly_members": sum(float(getattr(person, "age", 0.0)) >= 65.0 for person in members),
                    "employed_members": sum(getattr(person, "firm_id", None) is not None for person in members),
                    "household_size": len(members),
                })
            if phase == "after_payg_contribution":
                scheduled = sum(float(item.get("scheduled_contribution", 0.0)) for item in getattr(payg_result, "contribution_records", []) if item.get("household_id") == household_id) if payg_result is not None else 0.0
                row["scheduled_payg_contribution"] = scheduled
                row["actual_payg_contribution"] = float(getattr(household, "payg_contribution_this_step", 0.0))
            if phase == "after_pension":
                row["pension_received"] = pension_by_household.get(household_id, 0.0)
            if phase == "after_legacy_private_support":
                row["legacy_private_transfer_received"] = float(getattr(household, "private_support_received_this_step", 0.0))
                row["legacy_private_transfer_paid"] = float(getattr(household, "private_support_paid_this_step", 0.0))
            if phase == "after_private_transfer":
                row["private_transfer_received"] = float(getattr(household, "wealth_transfer_received_this_step", 0.0))
                row["private_transfer_paid"] = float(getattr(household, "wealth_transfer_paid_this_step", 0.0))
            if phase == "after_consumption":
                row.update({
                    "wage_income_received": float(getattr(household, "wage_income_this_step", 0.0)),
                    "dividends_received": float(getattr(household, "dividend_income_this_step", 0.0)),
                    "income_this_step": float(getattr(household, "income_this_step", 0.0)),
                    "cash_before_consumption": float(getattr(household, "cash_before_consumption_audit", row.get("cash_before_consumption", getattr(household, "wealth", 0.0)))),
                    "minimum_consumption": float(getattr(household, "necessary_consumption_this_step", 0.0)),
                    "planned_consumption": float(getattr(household, "desired_consumption_this_step", 0.0)),
                    "actual_consumption": float(getattr(household, "consumption_this_step", 0.0)),
                    "affordable_consumption": float(getattr(household, "affordable_consumption_this_step", 0.0)),
                    "budget_feasible_consumption": float(getattr(household, "budget_feasible_consumption_this_step", 0.0)),
                    "current_available_cash": float(getattr(household, "current_available_cash_this_step", 0.0)),
                    "cash_settleable_consumption": float(getattr(household, "cash_settleable_consumption_this_step", 0.0)),
                    "consumption_cash_constraint_binding": bool(getattr(household, "consumption_cash_constraint_binding", False)),
                    "planned_minus_realized_consumption": float(getattr(household, "planned_minus_realized_consumption_this_step", 0.0)),
                    "cash_constraint_suppressed_consumption": float(getattr(household, "cash_constraint_suppressed_consumption_this_step", 0.0)),
                    "cash_after_consumption": float(getattr(household, "wealth", 0.0)),
                    "closing_cash": float(getattr(household, "wealth", 0.0)),
                })
                self.payg_cash_safety_audit_rows.append(dict(row))
                if getattr(self, "household_liquidity_snapshot_enabled", False):
                    members = [
                        self.get_person_by_id(pid)
                        for pid in [*getattr(household, "parents", []), *getattr(household, "children", [])]
                    ]
                    members = [person for person in members if person is not None and getattr(person, "alive", False)]
                    elderly, uncovered = self._active_social_household_meta(household)
                    minimum_cost = float(row.get("minimum_consumption", 0.0))
                    self.household_liquidity_snapshot_rows.append({
                        "global_step": step,
                        "branch": getattr(self, "active_social_policy_branch_name", "UNSPECIFIED"),
                        "household_id": household_id,
                        "settlement_only": bool(getattr(household, "settlement_only", False)),
                        "closing_cash": float(getattr(household, "wealth", 0.0)),
                        "minimum_consumption_cost": minimum_cost,
                        "liquidity_weeks": (
                            float(row.get("cash_after_consumption", 0.0)) / minimum_cost
                            if minimum_cost > 1e-12 else None
                        ),
                        "household_size": len(members),
                        "elderly_household": bool(elderly),
                        "employed_member": any(getattr(person, "firm_id", None) is not None for person in members),
                        "genealogy_uncovered_elderly": bool(uncovered),
                        "pension_recipient_this_week": float(getattr(household, "pension_income_this_step", 0.0)) > 1e-12,
                        "private_transfer_recipient_this_week": float(getattr(household, "wealth_transfer_received_this_step", 0.0)) > 1e-12,
                        "payg_contributor_this_week": float(getattr(household, "payg_contribution_this_step", 0.0)) > 1e-12,
                        "wage_income": float(getattr(household, "wage_income_this_step", 0.0)),
                        "total_income": float(getattr(household, "income_this_step", 0.0)),
                        "realized_consumption": float(getattr(household, "consumption_this_step", 0.0)),
                        "saving": float(getattr(household, "saving_this_step", 0.0)),
                    })
    def record_payg_cash_safety_consumption_plan(self, household, cash_before):
        if not getattr(self, "payg_cash_safety_audit_enabled", False):
            return
        row = self._payg_cash_safety_audit_current.setdefault(household.id, {"week": int(getattr(self, "current_step_index", 0)), "household_id": household.id})
        row["cash_before_consumption"] = float(cash_before)
        household.cash_before_consumption_audit = float(cash_before)
    def apply_active_social_policy(self, step):
        """Settle Step17.D policies after payroll/dividends and before food."""
        if not (getattr(self, "payg_pension_enabled", False) or getattr(self, "intergenerational_wealth_transfer_enabled", False)):
            self._active_social_policy_observation = None
            return {
                "payg": None, "wealth_transfer": None, "payg_contribution": 0.0,
                "pension_paid": 0.0, "wealth_transfer_paid": 0.0, "wealth_transfer_received": 0.0,
            }
        for firm in self.operating_firms():
            firm.actual_employer_contribution_this_step = 0.0
        self.begin_active_social_recipient_observation(step)
        payg_result = self.payg_pension_system.settle(step, defer_pension=getattr(self, "public_pension_transfer_enabled", False))
        for contribution in getattr(payg_result, "employer_contribution_records", []):
            firm = self.get_firm_by_id(contribution.get("firm_id"))
            if firm is not None:
                firm.actual_employer_contribution_this_step = float(contribution.get("actual_employer_contribution", 0.0))
        if getattr(self, "recipient_policy_instrumentation_enabled", False) and self._active_social_policy_observation is not None:
            self._active_social_policy_observation["after_pension"] = self._active_social_snapshot()
        transfer_result = self.intergenerational_wealth_transfer_system.execute(step)
        self.record_payg_cash_safety_phase("after_private_transfer")
        if getattr(self, "recipient_policy_instrumentation_enabled", False) and self._active_social_policy_observation is not None:
            self._active_social_policy_observation["after_private"] = self._active_social_snapshot()
        row = {
            "global_step": step, "payg_contribution": payg_result.actual_contribution,
            "pension_paid": payg_result.actual_pension, "wealth_transfer_paid": transfer_result.total_transfer,
            "wealth_transfer_received": transfer_result.total_transfer, "fund_cash": self.social_insurance_fund.cash,
        }
        self.active_social_policy_history.append(row)
        self.payg_weekly_history.append(self.payg_pension_system.weekly_row(payg_result))
        self.wealth_transfer_weekly_history.append(self.intergenerational_wealth_transfer_system.weekly_row(transfer_result))
        return {
            "payg": payg_result, "wealth_transfer": transfer_result,
            "payg_contribution": payg_result.actual_contribution, "pension_paid": payg_result.actual_pension,
            "wealth_transfer_paid": transfer_result.total_transfer, "wealth_transfer_received": transfer_result.total_transfer,
        }
    def step(self):

        self.current_step_index = len(self.population_history)
        self.time_metadata(self.current_step_index)
        stageA_adapter = self.ensure_stageA_operating_contract_adapter()
        if stageA_adapter is not None:
            stageA_adapter.begin_step(self.current_step_index)

        self.ledger.begin_step(
            self.current_step_index
        )
        self.reset_public_wealth_audit_step()
        self.record_payg_cash_safety_phase("opening")
        # Step 13.8B: begin any already-eligible relief contract before this
        # week's interest obligation is computed.  Eligibility is based on
        # completed active-contract weeks from prior settlement.
        self.begin_restructuring_week()

        if getattr(self, "household_wealth_instrumentation_enabled", False):
            self._household_wealth_opening = {
                household.id: float(getattr(household, "wealth", 0.0))
                for household in self.households
            }
            self._household_ids_before_step = set(self._household_wealth_opening)
            self._household_lifecycle_event_start = len(
                getattr(self, "household_lifecycle_events", [])
            )

        births=0

        deaths=0



        # -------------------------
        # Aging
        # -------------------------

        age_stages_before = {
            person.id: self.age_labor_group(person.age_weeks)
            for person in self.population
        }
        for person in self.population:

            person.grow()
        self.record_age_transitions(age_stages_before)
        retirement_events_this_step = self.update_retirement_status()
        self.household_manager.step()
        self.ensure_labor_settlement_households()
        self.refresh_age_labor_participation()
        # The initial world may have queried capacity before week 0.
        # Clear that pre-aging value before weekly economic decisions.
        if getattr(self, "_labor_capacity_cache", None) is not None:
            self._labor_capacity_cache.clear()
            self._labor_capacity_cache_step = self.current_step_index
        self.marriage_market_executed = False
        self.marriage_market_last_result = 0
        self.marriage_market_last_eligible_males = 0
        self.marriage_market_last_eligible_females = 0
        self.marriage_market_last_unmatched_males = 0
        self.marriage_market_last_unmatched_females = 0
        if self.current_step_index >= self.next_marriage_market_step:
            eligible_males, eligible_females = (
                self.marriage_system.get_marriage_pool()
            )
            marriages = self.marriage_system.process_marriage()
            self.marriage_market_executed = True
            self.marriage_market_execution_count += 1
            self.marriage_market_last_result = marriages
            self.marriage_market_last_eligible_males = len(eligible_males)
            self.marriage_market_last_eligible_females = len(eligible_females)
            self.marriage_market_last_unmatched_males = max(
                0,
                len(eligible_males) - marriages,
            )
            self.marriage_market_last_unmatched_females = max(
                0,
                len(eligible_females) - marriages,
            )
            self.next_marriage_market_step += MARRIAGE_MARKET_INTERVAL_WEEKS
            self.record_marriage_market_diagnostic(
                len(eligible_males),
                len(eligible_females),
                marriages,
            )
        self.refresh_active_households()

        # Step17.Y is an opt-in pre-production capital-goods entry decision.
        entry_system = getattr(self, "endogenous_capital_goods_entry_system", None)
        if entry_system is None:
            entry_system = EndogenousCapitalGoodsEntrySystem(self)
            self.endogenous_capital_goods_entry_system = entry_system
        entry_system.before_production(self.current_step_index)

        # -------------------------
        # Production, wages, consumption, and firm sales
        # -------------------------

        use_firm_specific_credit = len(getattr(self, "firms", [])) > 1
        capital_investment_week = self.prepare_canonical_investment_week(
            self.current_step_index
        )
        firm_result = self.firm_system.step()

        if capital_investment_week is not None and getattr(
            self,
            "canonical_investment_enabled",
            False,
        ):
            capital_labor = math.fsum(
                self.firm_employee_capacity(firm)
                for firm in getattr(self, "capital_good_firms", [])
            )
            capital_wages = capital_investment_week.capital_good_wages
            firm_result["executed_wage_bill"] = (
                firm_result.get("executed_wage_bill", firm_result["wage_bill"])
                + capital_wages
            )
            firm_result["wage_bill"] = firm_result.get("wage_bill", 0.0) + capital_wages
            firm_result["total_labor"] = firm_result.get("total_labor", 0.0) + capital_labor
            firm_result["capital_good_wages"] = capital_wages
            firm_result["capital_good_production"] = capital_investment_week.capital_good_production
            firm_result["capital_good_sales"] = capital_investment_week.capital_good_sales
            firm_result["capital_good_revenue"] = capital_investment_week.capital_good_revenue
            firm_result["fixed_investment"] = capital_investment_week.fixed_investment
            firm_result["replacement_investment"] = capital_investment_week.replacement_investment
            firm_result["expansion_investment"] = capital_investment_week.expansion_investment
            firm_result["capital_depreciation_expense"] = capital_investment_week.depreciation_expense
            firm_result["retired_capacity"] = capital_investment_week.retired_capacity
            firm_result["firm_cash"] = math.fsum(
                float(getattr(firm, "cash", 0.0))
                for firm in self.operating_firms()
            )
            firm_result["firm_net_worth"] = math.fsum(
                float(getattr(firm, "cash", 0.0))
                + float(getattr(firm, "inventory_value", 0.0))
                + float(
                    getattr(
                        getattr(firm, "capital_stock", None),
                        "total_remaining_book_value",
                        0.0,
                    )
                )
                + float(getattr(firm, "prepaid_capital_investment_asset", 0.0))
                - float(getattr(firm, "loan_balance", 0.0))
                - float(getattr(firm, "customer_advance_liability", 0.0))
                for firm in self.operating_firms()
            )

        firms_synced_this_step = False

        if use_firm_specific_credit:
            aggregate_dividend = firm_result.get("dividend", 0.0)
            total_firm_dividend = self.calculate_multi_firm_dividends(
                aggregate_dividend,
                firm_result.get("wage_bill", 0.0),
            )
            # The aggregate FirmSystem no longer pays a temporary dividend;
            # keep its cash mirror consistent with the FirmSlice settlements.
            self.firm_system.cash -= total_firm_dividend
            firm_result["dividend"] = total_firm_dividend
            firm_result["household_dividend"] = getattr(
                self,
                "_last_dividend_routing_person_paid",
                0.0,
            )
            firm_result["person_dividend_paid"] = firm_result["household_dividend"]
            firm_result["legacy_dividend_entitlement"] = getattr(
                self,
                "_last_dividend_routing_legacy_entitlement",
                0.0,
            )
            firm_result["estate_dividend_paid"] = getattr(
                self,
                "_last_dividend_routing_estate_paid",
                0.0,
            )
            firm_result["dividend_routing_gap"] = getattr(
                self,
                "_last_dividend_routing_gap",
                0.0,
            )
            self.sync_firms_from_aggregate(firm_result)
            firms_synced_this_step = True

        personal_tax_result = self.apply_active_personal_income_tax(self.current_step_index)
        firm_result["scheduled_personal_income_tax"] = personal_tax_result.get("scheduled", 0.0)
        firm_result["actual_personal_income_tax"] = personal_tax_result.get("actual", 0.0)
        firm_result["personal_income_tax_shortfall"] = personal_tax_result.get("shortfall", 0.0)

        public_budget = self.collect_public_tax(self.current_step_index)
        firm_result["scheduled_public_tax"] = public_budget.scheduled_tax_this_step
        firm_result["actual_public_tax"] = public_budget.actual_tax_this_step
        firm_result["public_tax_shortfall"] = public_budget.tax_shortfall_this_step
        firm_result["public_tax_base"] = ("FIRM_POSITIVE_TAXABLE_PROFIT" if getattr(self, "corporate_profit_tax_enabled", False) else self.public_revenue_tax_base)
        firm_result["personal_income_tax"] = public_budget.personal_tax_actual_this_step
        firm_result["corporate_profit_tax"] = public_budget.corporate_tax_actual_this_step
        if getattr(self, "public_pension_transfer_enabled", False):
            completed_payg = self.payg_pension_system.complete_deferred_public_pension(self.current_step_index)
            if completed_payg is not None:
                firm_result["pension_paid"] = completed_payg.actual_pension
                firm_result["payg_contribution"] = completed_payg.actual_contribution
                firm_result["employer_pension_contribution"] = completed_payg.actual_employer_contribution
                firm_result["public_pension_transfer"] = completed_payg.actual_public_pension_transfer
                firm_result["scheduled_public_pension_transfer"] = completed_payg.scheduled_public_pension_transfer
                firm_result["public_pension_transfer_shortfall"] = completed_payg.public_pension_transfer_shortfall
                if self.active_social_policy_history and self.active_social_policy_history[-1].get("global_step") == self.current_step_index:
                    self.active_social_policy_history[-1].update({
                        "pension_paid": completed_payg.actual_pension,
                        "public_pension_transfer": completed_payg.actual_public_pension_transfer,
                        "fund_cash": self.social_insurance_fund.cash,
                    })
        firm_result["firm_cash"] = math.fsum(float(getattr(firm, "cash", 0.0)) for firm in self.operating_firms())
        labor = firm_result["total_labor"]

        total_income = (
            firm_result.get("executed_wage_bill", firm_result["wage_bill"])
            + firm_result.get("household_dividend", firm_result["dividend"])
            + firm_result.get("private_support_received", 0.0)
            - firm_result.get("private_support_paid", 0.0)
            + firm_result.get("pension_paid", 0.0)
            + firm_result.get("wealth_transfer_received", 0.0)
            - firm_result.get("payg_contribution", 0.0)
            - firm_result.get("wealth_transfer_paid", 0.0)
        )

        total_consumption = firm_result["sales"]


        # =========================
        # Accounting Identity
        # =========================

        total_saving = (
            total_income
            -
            total_consumption
            -
            firm_result.get("actual_personal_income_tax", 0.0)
        )



        if total_income > 0:

            consumption_rate = (
                total_consumption
                /
                total_income
            )

            saving_rate = (
                total_saving
                /
                total_income
            )

        else:

            consumption_rate = 0

            saving_rate = 0


        expected_money_stock = (
            self.initial_private_money_stock
            +
            firm_result.get("cumulative_money_issued", 0.0)
        )

        population = len(self.population)

        self.gdp_history.append(
            total_income
        )

        self.income_history.append(
            total_income
        )

        self.consumption_history.append(
            total_consumption
        )

        self.saving_history.append(
            total_saving
        )

        # =========================
        # Economy Ratio History
        # =========================


        self.consumption_rate_history.append(
            consumption_rate
        )


        self.saving_rate_history.append(
            saving_rate
        )



        resource = firm_result["nominal_output_value"]



        # -------------------------
        # Consumption
        # -------------------------

        demand=(

            len(self.population)

            *

            self.consumption_per_person

        )


        labor_resource = (
            self.base_resource
            +
            labor * self.productivity_value
        )

        pressure = (
            demand
            /
            labor_resource
            if labor_resource > 0
            else 1.0
        )



        # -------------------------
        # Birth
        # -------------------------

        

        newborns, births = self.family_birth(

            pressure

        )



        # -------------------------
        # Death
        # -------------------------

        # -------------------------
        # Death
        # -------------------------

        survivors=[]

        dead_people=[]


        for person in self.population:


            mortality = person.mortality_diagnostics()
            person.check_death()


            if person.alive:

                survivors.append(person)


            else:

                deaths += 1

                dead_people.append(person)
                self.record_demographic_event(
                    "death",
                    person_id=person.id,
                    sex=person.sex,
                    household_id=person.household_id,
                    **mortality,
                )



        # =========================
        # Inheritance
        # =========================

        self.inheritance_system.process_inheritance(
            dead_people
        )



        # =========================
        # Remove dead people
        # =========================

        for person in dead_people:


            self.remove_person_relationship(
                person
            )


            if person.household_id is not None:

                household = self.get_household(
                    person.household_id
                )

                if household is not None:

                    if person.id in household.parents:

                        household.parents.remove(
                            person.id
                        )


                    if person.id in household.children:

                        household.children.remove(
                            person.id
                        )



            del self.person_dict[
                person.id
            ]

        self.fertility_system.update_completed_couples()



        self.population = survivors + newborns
        active_households = self.refresh_active_households()

        # Step 15F.2: optional end-of-week primary-equity review.  It is
        # deliberately after consumption settlement, so the current week's
        # consumption is unchanged; any subscription affects the next week.
        if getattr(self, "person_equity_transition_enabled", False):
            transition_result = self.equity_transition_system.review(
                self.current_step_index
            )
            firm_result["equity_issuance_cash"] = transition_result.total_issued
            firm_result["equity_issuance_shares"] = (
                transition_result.total_shares_issued
            )
            firm_result["equity_issuance_buyer_count"] = len(
                transition_result.fills
            )
            if len(getattr(self, "firms", [])) <= 1:
                firm_result["firm_cash"] = self.firm_system.cash
        else:
            transition_result = None
            for household in self.households:
                household.equity_purchase_cash_outflow_this_step = 0.0
            for target in [self.firm_system, *getattr(self, "firms", [])]:
                target.equity_issuance_cash_this_step = 0.0
                target.equity_issuance_shares_this_step = 0.0
                target.equity_issuance_buyer_count = 0



        # Step 15G.1: optional deterministic secondary-equity purchase screen.
        # Accounting sees the Household cash-to-equity reallocation in the
        # same completed week. The default remains OFF.
        if getattr(self, "autonomous_secondary_equity_enabled", False):
            secondary_result = self.autonomous_secondary_equity_system.review(
                self.current_step_index
            )
            for target in getattr(self, "firms", []):
                target.secondary_equity_sale_proceeds_this_step = 0.0
                target.secondary_equity_shares_sold_this_step = 0.0
                target.secondary_equity_buyer_count = 0
                target.secondary_decision_step = secondary_result.step
                target.secondary_valuation_step = secondary_result.valuation_step
                target.secondary_book_equity_used = secondary_result.book_equity_used
                target.secondary_total_shares_used = secondary_result.total_shares_used
                target.secondary_reference_price_used = secondary_result.reference_price_used
                target.secondary_valuation_status = secondary_result.valuation_status
            for transfer in secondary_result.transfer_results:
                target = self.get_firm_by_id(transfer.firm_id, self.firm_system)
                target.secondary_equity_sale_proceeds_this_step += (
                    transfer.total_payment
                )
                target.secondary_equity_shares_sold_this_step += (
                    transfer.total_shares_sold
                )
                target.secondary_equity_buyer_count += sum(
                    fill["shares_filled"] > 1e-12
                    for fill in transfer.fills
                )
            firm_result["secondary_equity_sale_proceeds"] = (
                secondary_result.total_payment
            )
            firm_result["secondary_equity_shares_sold"] = (
                secondary_result.total_shares_sold
            )
            firm_result["secondary_equity_buyer_count"] = (
                secondary_result.buyer_count
            )
            firm_result["secondary_decision_step"] = secondary_result.step
            firm_result["secondary_valuation_step"] = secondary_result.valuation_step
            firm_result["secondary_book_equity_used"] = secondary_result.book_equity_used
            firm_result["secondary_total_shares_used"] = secondary_result.total_shares_used
            firm_result["secondary_reference_price_used"] = secondary_result.reference_price_used
            firm_result["secondary_valuation_status"] = secondary_result.valuation_status
        else:
            secondary_result = None

        # =================================================
        # Record
        # =================================================


        step_index = len(self.population_history)

        population=len(self.population)

        total_wealth = sum(
            h.wealth
            for h in self.households
        )
        public_wealth = getattr(self, "public_wealth", 0.0)
        central_bank_public_income_balance = firm_result.get(
            "central_bank_public_income_balance",
            0.0,
        )
        authoritative_money_stock = expected_money_stock
        money_components = self.authoritative_money_location_components()
        if getattr(self, "canonical_investment_enabled", False):
            firm_result["firm_cash"] = money_components["firm_cash"]
            firm_result["firm_net_worth"] = math.fsum(
                float(getattr(firm, "cash", 0.0))
                + float(getattr(firm, "inventory_value", 0.0))
                + float(
                    getattr(
                        getattr(firm, "capital_stock", None),
                        "total_remaining_book_value",
                        0.0,
                    )
                )
                - float(getattr(firm, "loan_balance", 0.0))
                for firm in self.operating_firms()
            )
        private_cash_stock = (
            money_components["firm_cash"]
            + money_components["household_cash"]
            + money_components["legacy_owner_cash"]
            + money_components["estate_cash"]
            + money_components["pending_household_formation_wealth"]
        )
        located_money_stock = money_components["located_money_stock"]
        monetary_accounting_gap = (
            authoritative_money_stock - located_money_stock
        )

        if population > 0:

            wealth_per_capita = (
                total_wealth
                /
                population
            )

        else:

            wealth_per_capita = 0

        self.wealth_history.append(
            total_wealth
        )

        self.wealth_per_capita_history.append(
            wealth_per_capita
        )


        self.population_history.append(

            population

        )

        # -------------------------
        # Household structure
        # -------------------------

        household_sizes=[]

        married_households=0

        single_parent_households=0

        empty_households=0


        for household in self.households:


            size = household.size()

            household_sizes.append(size)


            if len(household.parents)==2:

                married_households +=1


            elif len(household.parents)==1:

                single_parent_households +=1


            else:

                empty_households +=1



        self.household_size_history.append(
            household_sizes
        )


        self.married_household_history.append(
            married_households
        )


        self.single_parent_history.append(
            single_parent_households
        )


        self.empty_household_history.append(
            empty_households
        )


        # -------------------------
        # Age distribution
        # -------------------------

        age_distribution=[0]*101


        for person in self.population:


            age=min(
                int(person.age),
                100
            )


            age_distribution[age]+=1



        self.age_distribution_history.append(

            age_distribution

        )



        # -------------------------
        # Age groups
        # -------------------------

        children=0

        workers=0

        elderly=0


        for person in self.population:


            if person.age < 20:

                children+=1


            elif person.age <=60:

                workers+=1


            else:

                elderly+=1



        self.age_group_history.append(

            {

            "children":children,

            "workers":workers,

            "elderly":elderly

            }

        )



        # -------------------------
        # Birth death
        # -------------------------

        self.birth_history.append(

            births

        )


        self.death_history.append(

            deaths

        )


        if population>0:

            birth_rate=births/population

            death_rate=deaths/population

        else:

            birth_rate=0

            death_rate=0



        self.birth_rate_history.append(

            birth_rate

        )


        self.death_rate_history.append(

            death_rate

        )



        # -------------------------
        # Economy
        # -------------------------

        self.resource_history.append(

            resource

        )


        self.labor_history.append(

            labor

        )


        self.pressure_history.append(

            pressure

        )

        self.firm_cash_history.append(
            firm_result["firm_cash"]
        )

        self.firm_inventory_history.append(
            firm_result["firm_inventory"]
        )

        self.firm_net_worth_history.append(
            firm_result["firm_net_worth"]
        )

        self.firm_profit_history.append(
            firm_result["profit_before_dividend"]
        )

        self.food_price_history.append(
            firm_result["food_price"]
        )

        self.unit_labor_cost_history.append(
            firm_result.get("unit_labor_cost", 0.0)
        )

        self.unit_labor_cost_growth_history.append(
            firm_result.get("unit_labor_cost_growth", 0.0)
        )

        self.price_inventory_gap_history.append(
            firm_result.get("price_inventory_gap", 0.0)
        )

        self.price_cost_growth_history.append(
            firm_result.get("price_cost_growth", 0.0)
        )

        self.price_inflation_signal_history.append(
            firm_result.get("price_inflation_signal", 0.0)
        )

        self.price_log_adjustment_history.append(
            firm_result.get("price_log_adjustment", 0.0)
        )

        self.demand_pressure_history.append(
            firm_result.get("demand_pressure", 0.0)
        )

        self.available_food_supply_units_history.append(
            firm_result.get("available_food_supply_units", 0.0)
        )

        self.firm_sales_income_ratio_history.append(
            firm_result["sales_income_ratio"]
        )

        self.food_output_units_history.append(
            firm_result.get("food_output_units", 0.0)
        )

        self.food_demand_units_history.append(
            firm_result.get("food_demand_units", 0.0)
        )

        self.food_sales_units_history.append(
            firm_result.get("food_sales_units", 0.0)
        )

        self.food_inventory_units_history.append(
            firm_result.get("food_inventory_units", 0.0)
        )

        self.food_inventory_demand_ratio_history.append(
            firm_result.get("inventory_demand_ratio", 0.0)
        )

        self.unmet_food_demand_units_history.append(
            firm_result.get("unmet_food_demand_units", 0.0)
        )

        self.money_issued_history.append(
            firm_result.get("money_issued", 0.0)
        )

        self.working_capital_loan_issued_history.append(
            firm_result.get("working_capital_loan_issued", 0.0)
        )

        self.working_capital_target_cash_history.append(
            firm_result.get("working_capital_target_cash", 0.0)
        )

        self.working_capital_funding_gap_history.append(
            firm_result.get("working_capital_funding_gap", 0.0)
        )

        self.working_capital_loan_repaid_history.append(
            firm_result.get("working_capital_loan_repaid", 0.0)
        )

        self.working_capital_interest_paid_history.append(
            firm_result.get("working_capital_interest_paid", 0.0)
        )

        self.working_capital_loan_balance_history.append(
            firm_result.get("working_capital_loan_balance", 0.0)
        )

        self.cumulative_money_issued_history.append(
            firm_result.get("cumulative_money_issued", 0.0)
        )

        self.inventory_monetized_units_history.append(
            firm_result.get("inventory_monetized_units", 0.0)
        )

        self.cumulative_inventory_monetized_units_history.append(
            firm_result.get("cumulative_inventory_monetized_units", 0.0)
        )

        self.central_bank_inventory_purchase_history.append(
            firm_result.get("central_bank_inventory_purchase", 0.0)
        )

        self.central_bank_market_release_revenue_history.append(
            firm_result.get("central_bank_market_release_revenue", 0.0)
        )

        self.central_bank_poverty_subsidy_value_history.append(
            firm_result.get("central_bank_poverty_subsidy_value", 0.0)
        )

        self.central_bank_public_income_history.append(
            firm_result.get("central_bank_public_income", 0.0)
        )

        self.central_bank_public_income_used_history.append(
            firm_result.get("central_bank_public_income_used", 0.0)
        )

        self.central_bank_public_income_balance_history.append(
            central_bank_public_income_balance
        )

        self.central_bank_net_money_issued_history.append(
            firm_result.get("central_bank_net_money_issued", 0.0)
        )

        self.private_cash_stock_history.append(
            private_cash_stock
        )

        self.public_wealth_history.append(
            public_wealth
        )

        self.expected_money_stock_history.append(
            expected_money_stock
        )

        located_money_stock = money_components["located_money_stock"]

        self.located_money_stock_history.append(
            located_money_stock
        )

        self.monetary_accounting_gap_history.append(
            monetary_accounting_gap
        )

        self.central_bank_food_inventory_units_history.append(
            firm_result.get("central_bank_food_inventory_units", 0.0)
        )

        self.central_bank_food_purchase_units_history.append(
            firm_result.get("central_bank_food_purchase_units", 0.0)
        )

        self.central_bank_food_release_units_history.append(
            firm_result.get("central_bank_food_release_units", 0.0)
        )

        self.central_bank_food_subsidy_units_history.append(
            firm_result.get("central_bank_food_subsidy_units", 0.0)
        )



        # -------------------------
        # Average age
        # -------------------------

        if population>0:

            avg_age=sum(

                p.age

                for p in self.population

            )/population


        else:

            avg_age=0



        self.average_age_history.append(

            avg_age

        )

        if not firms_synced_this_step:
            self.sync_firms_from_aggregate(firm_result)
        if getattr(self, "canonical_investment_enabled", False):
            money_components = self.authoritative_money_location_components()
            firm_result["firm_cash"] = money_components["firm_cash"]
            firm_result["firm_net_worth"] = math.fsum(
                float(getattr(firm, "cash", 0.0))
                + float(getattr(firm, "inventory_value", 0.0))
                + float(
                    getattr(
                        getattr(firm, "capital_stock", None),
                        "total_remaining_book_value",
                        0.0,
                    )
                )
                - float(getattr(firm, "loan_balance", 0.0))
                for firm in self.operating_firms()
            )
            private_cash_stock = (
                money_components["firm_cash"]
                + money_components["household_cash"]
                + money_components["legacy_owner_cash"]
                + money_components["estate_cash"]
            )
            located_money_stock = money_components["located_money_stock"]
            monetary_accounting_gap = (
                authoritative_money_stock - located_money_stock
            )
            if self.private_cash_stock_history:
                self.private_cash_stock_history[-1] = private_cash_stock
            if self.monetary_accounting_gap_history:
                self.monetary_accounting_gap_history[-1] = monetary_accounting_gap
            if self.located_money_stock_history:
                self.located_money_stock_history[-1] = located_money_stock
        if not hasattr(self, "accounting"):
            self.accounting = AccountingLayer(self)
        self.accounting.record_step(
            step_index,
            self.operating_firms(),
            self.firm_system,
        )
        # Step 13.6E is an end-of-week, record-only state update.  It runs
        # after financial, payroll, credit and operating results are final.
        self.update_default_bookkeeping(step_index)
        stageA_adapter = self.ensure_stageA_operating_contract_adapter()
        if stageA_adapter is not None:
            stageA_adapter.build_current_bundles(self)
        self.record_firm_diagnostics(step_index)

        self.record_diagnostics(
            step_index=step_index,
            firm_result=firm_result,
            population=population,
            births=births,
            deaths=deaths,
            birth_rate=birth_rate,
            death_rate=death_rate,
            average_age=avg_age,
            children=children,
            workers=workers,
            elderly=elderly,
            household_sizes=household_sizes,
            married_households=married_households,
            single_parent_households=single_parent_households,
            empty_households=empty_households,
            total_income=total_income,
            total_consumption=total_consumption,
            total_saving=total_saving,
            consumption_rate=consumption_rate,
            saving_rate=saving_rate,
            total_wealth=total_wealth,
            wealth_per_capita=wealth_per_capita,
            public_wealth=public_wealth,
            central_bank_public_income_balance=central_bank_public_income_balance,
            private_cash_stock=private_cash_stock,
            expected_money_stock=expected_money_stock,
            located_money_stock=located_money_stock,
            monetary_accounting_gap=monetary_accounting_gap,
            labor=labor,
            resource=resource,
            pressure=pressure,
            active_households=active_households,
        )

        entry_system.record_week(step_index)

        if getattr(self, "long_horizon_liquidity_enabled", False):
            self.record_long_horizon_liquidity(firm_result)

        # Keep experimental private-support metrics separate from production
        # income; aggregate net transfer is zero by construction.
        if self.diagnostics_rows:
            self.diagnostics_rows[-1].update({
                "private_support_paid": firm_result.get("private_support_paid", 0.0),
                "private_support_received": firm_result.get("private_support_received", 0.0),
                "private_support_transfer_count": firm_result.get("private_support_transfer_count", 0),
            })
        # Strictly downstream of all economic and accounting updates.
        self.record_family_link_observability(step_index)
        self.persist_diagnostics_for_step(step_index)

        self.ledger.end_step()



    # =================================================
    # Diagnostics
    # =================================================

    def configure_diagnostic_persistence(
        self,
        output_dir,
        cadence=1,
        statistical_observability=False,
        statistical_output_dir=None,
        statistical_snapshot_cadence=13,
        statistical_age_cadence=13,
        observability_mode=None,
    ):
        """Enable lightweight end-of-week canonical diagnostic persistence."""
        from diagnostic_persistence import CanonicalDiagnosticPersistence

        mode = str(
            observability_mode or getattr(
                self, "observability_mode", "FULL_DIAGNOSTIC"
            )
        ).upper()
        self.observability_mode = mode
        self.diagnostic_micro_snapshot_cadence = max(
            1, int(statistical_snapshot_cadence)
        )
        self.diagnostic_persistence = CanonicalDiagnosticPersistence(
            output_dir,
            cadence=cadence,
            statistical_observability=statistical_observability,
            statistical_output_dir=statistical_output_dir,
            statistical_snapshot_cadence=statistical_snapshot_cadence,
            statistical_age_cadence=statistical_age_cadence,
            observability_mode=mode,
        )
        self.diagnostic_persistence.initialize(self)

    def persist_diagnostics_for_step(self, step_index):
        persistence = getattr(self, "diagnostic_persistence", None)
        if persistence is not None:
            persistence.write_step(self, step_index)

    def safe_ratio(self, numerator, denominator):
        if denominator == 0:
            return 0.0

        return numerator / denominator

    def median(self, values):
        values = sorted(values)

        if not values:
            return 0.0

        mid = len(values) // 2

        if len(values) % 2 == 1:
            return values[mid]

        return (values[mid - 1] + values[mid]) / 2

    def percentile(self, values, percentile):
        values = sorted(values)

        if not values:
            return 0.0

        if len(values) == 1:
            return values[0]

        rank = (len(values) - 1) * percentile / 100
        lower = int(rank)
        upper = min(lower + 1, len(values) - 1)
        weight = rank - lower

        return values[lower] * (1 - weight) + values[upper] * weight

    def record_firm_diagnostics(self, step_index):
        """Persist the current FirmSlice panel from authoritative runtime state."""
        rows = getattr(self, "firm_diagnostics_rows", None)
        if rows is None:
            self.firm_diagnostics_rows = []
            rows = self.firm_diagnostics_rows
        for firm in self.operating_firms():
            table = getattr(firm, "cap_table", None)
            stock = getattr(firm, "capital_stock", None)
            rows.append({
                "global_step": step_index,
                "firm_id": getattr(firm, "firm_id", None),
                "sector_id": getattr(firm, "sector_id", "food"),
                "technology_id": getattr(firm, "technology_id", "labor_only_food_v1"),
                "employment": len(getattr(firm, "employee_ids", []) or []),
                "desired_labor": float(getattr(firm, "desired_labor", 0.0)),
                "revenue": float(getattr(firm, "sales_revenue", getattr(firm, "sales", 0.0))),
                "operating_profit": float(getattr(firm, "profit", 0.0)),
                "cash": float(getattr(firm, "cash", 0.0)),
                "principal": float(getattr(firm, "loan_balance", 0.0)),
                "arrears": float(getattr(firm, "interest_arrears", 0.0)),
                "legacy_ownership_fraction": table.ownership_fraction("legacy") if table is not None else 0.0,
                "person_ownership_fraction": table.ownership_fraction("person") if table is not None else 0.0,
                "person_shareholder_count": table.read_only_view().get("person_shareholder_count", 0) if table is not None else 0,
                "capital_book_value": float(getattr(stock, "total_remaining_book_value", 0.0)) if stock is not None else 0.0,
                "investment_expenditure": float(getattr(firm, "investment_cash_outflow_this_step", 0.0)),
            })

    def record_diagnostics(
        self,
        step_index,
        firm_result,
        population,
        births,
        deaths,
        birth_rate,
        death_rate,
        average_age,
        children,
        workers,
        elderly,
        household_sizes,
        married_households,
        single_parent_households,
        empty_households,
        total_income,
        total_consumption,
        total_saving,
        consumption_rate,
        saving_rate,
        total_wealth,
        wealth_per_capita,
        public_wealth,
        central_bank_public_income_balance,
        private_cash_stock,
        expected_money_stock,
        located_money_stock,
        monetary_accounting_gap,
        labor,
        resource,
        pressure,
        active_households=None,
    ):
        # The diagnostic row uses the same authoritative stock definition as
        # the runtime reconciliation: opening private money plus net central
        # bank issuance.  Keep this local explicit so compact and full modes
        # cannot accidentally fall back to a located-cash proxy.
        authoritative_money_stock = expected_money_stock
        money_components = self.authoritative_money_location_components()
        if self.diagnostics_mode == "compact":
            if len(self.firm_cash_history) >= 2:
                firm_cash_change = (
                    self.firm_cash_history[-1]
                    -
                    self.firm_cash_history[-2]
                )
            else:
                firm_cash_change = 0.0

            if len(self.firm_inventory_history) >= 2:
                firm_inventory_value_change = (
                    self.firm_inventory_history[-1]
                    -
                    self.firm_inventory_history[-2]
                )
            else:
                firm_inventory_value_change = 0.0

            total_money_stock = (
                private_cash_stock
                +
                public_wealth
                +
                central_bank_public_income_balance
            )
            public_money_stock = (
                public_wealth
                +
                central_bank_public_income_balance
            )
            row = {
                "step": step_index,
                "seed": self.seed if self.seed is not None else "",
                "scenario": self.scenario_name,
                "population": population,
                "firm_cash": firm_result.get("firm_cash", 0.0),
                "firm_cash_change": firm_cash_change,
                "firm_sales_revenue": firm_result.get(
                    "firm_sales_revenue",
                    0.0,
                ),
                "wage_bill": firm_result.get("wage_bill", 0.0),
                "wage_payment": firm_result.get("wage_bill", 0.0),
                "dividend": firm_result.get("dividend", 0.0),
                "firm_inventory_value": firm_result.get(
                    "firm_inventory",
                    0.0,
                ),
                "firm_inventory_value_change": firm_inventory_value_change,
                "production_inventory_investment": (
                    firm_inventory_value_change
                ),
                "firm_cash_to_wage_bill": self.safe_ratio(
                    firm_result.get("firm_cash", 0.0),
                    firm_result.get("wage_bill", 0.0),
                ),
                "working_capital_loan_issued": firm_result.get(
                    "working_capital_loan_issued",
                    0.0,
                ),
                "working_capital_target_cash": firm_result.get(
                    "working_capital_target_cash",
                    0.0,
                ),
                "working_capital_funding_gap": firm_result.get(
                    "working_capital_funding_gap",
                    0.0,
                ),
                "target_cash": firm_result.get("target_cash", 0.0),
                "funding_gap": firm_result.get("funding_gap", 0.0),
                "loan_issued": firm_result.get("loan_issued", 0.0),
                "working_capital_loan_repaid": firm_result.get(
                    "working_capital_loan_repaid",
                    0.0,
                ),
                "working_capital_loan_balance": firm_result.get(
                    "working_capital_loan_balance",
                    0.0,
                ),
                "loan_balance": firm_result.get("loan_balance", 0.0),
                "public_money_share": self.safe_ratio(
                    public_money_stock,
                    total_money_stock,
                ),
                "central_bank_inventory_purchase": firm_result.get(
                    "central_bank_inventory_purchase",
                    0.0,
                ),
                "central_bank_market_release_revenue": firm_result.get(
                    "central_bank_market_release_revenue",
                    0.0,
                ),
                "central_bank_public_income": firm_result.get(
                    "central_bank_public_income",
                    0.0,
                ),
                "central_bank_public_income_used": firm_result.get(
                    "central_bank_public_income_used",
                    0.0,
                ),
                "public_money_stock": public_money_stock,
                "public_wealth": public_wealth,
                "central_bank_public_income_balance": (
                    central_bank_public_income_balance
                ),
                 "cumulative_public_wealth_from_no_heir": (
                     self.cumulative_public_wealth_audit[
                         "public_wealth_from_no_heir"
                     ]
                 ),
                 "default_event_count_this_step": sum(
                     bool(getattr(firm, "default_event_this_week", False))
                     for firm in getattr(self, "firms", [])
                 ),
                 "active_default_firm_count": sum(
                     bool(getattr(firm, "active_default_episode", False))
                     for firm in getattr(self, "firms", [])
                 ),
                 "active_contract_default_firm_count": sum(
                     bool(getattr(firm, "active_contract_default", False))
                     for firm in getattr(self, "firms", [])
                 ),
                 "restructuring_active_firm_count": sum(
                     bool(getattr(firm, "restructuring_active", False))
                     for firm in getattr(self, "firms", [])
                 ),
                 "restructuring_start_count_this_step": sum(
                     bool(getattr(firm, "restructuring_start_this_week", False))
                     for firm in getattr(self, "firms", [])
                 ),
                 "weekly_interest_relief_amount": sum(
                     float(getattr(firm, "weekly_interest_relief_amount", 0.0))
                     for firm in getattr(self, "firms", [])
                 ),
                 "cumulative_interest_relief_amount": sum(
                     float(getattr(firm, "cumulative_interest_relief_amount", 0.0))
                     for firm in getattr(self, "firms", [])
                 ),
                 "contract_cure_count_this_step": sum(
                     bool(getattr(firm, "contract_cure_this_week", False))
                     for firm in getattr(self, "firms", [])
                 ),
                 "acute_default_phase_exit_count_this_step": sum(
                     bool(
                         getattr(
                             firm,
                             "acute_default_phase_exit_this_week",
                             False,
                         )
                     )
                     for firm in getattr(self, "firms", [])
                 ),
                 "default_history_count_total": sum(
                     int(getattr(firm, "default_history_count", 0))
                     for firm in getattr(self, "firms", [])
                 ),
             }
            row.update(self.time_metadata(step_index))
            row.update(self.marriage_diagnostics())
            self.diagnostics_rows.append(row)
            return

        if active_households is None:
            active_households = self.active_households()

        active_household_count = len(active_households)
        wealth_values = []
        income_values = []
        security_ratio_values = []
        reserve_weeks_values = []
        poverty_households = 0
        households_below_target = 0
        unmet_minimum_need_units = 0.0
        target_wealth_total = 0.0
        wealth_gap_total = 0.0
        optional_consumption_total = 0.0
        optional_multiplier_total = 0.0

        for household in active_households:
            wealth_values.append(household.wealth)
            income_values.append(
                getattr(household, "income_this_step", 0.0)
            )

            need_gap = getattr(
                household,
                "food_need_gap_units_this_step",
                0.0,
            )

            if need_gap > 0:
                poverty_households += 1

            unmet_minimum_need_units += need_gap
            target_wealth = getattr(
                household,
                "target_wealth_this_step",
                0.0,
            )
            target_wealth_total += target_wealth
            reserve_weeks_values.append(
                getattr(household, "target_wealth_reserve_weeks", 0.0)
            )

            if target_wealth > 0:
                security_ratio = household.wealth / target_wealth
            else:
                security_ratio = 1.0

            security_ratio_values.append(security_ratio)

            if security_ratio < 1.0:
                households_below_target += 1

            wealth_gap_total += getattr(
                household,
                "wealth_gap_this_step",
                0.0,
            )
            optional_consumption_total += getattr(
                household,
                "optional_consumption_this_step",
                0.0,
            )
            optional_multiplier_total += getattr(
                household,
                "optional_consumption_multiplier_this_step",
                1.0,
            )

        opening_system_food_units = (
            firm_result.get("opening_food_inventory_units", 0.0)
            +
            firm_result.get("opening_central_bank_food_inventory_units", 0.0)
        )
        closing_system_food_units = (
            firm_result.get("food_inventory_units", 0.0)
            +
            firm_result.get("central_bank_food_inventory_units", 0.0)
        )
        food_conservation_left = (
            opening_system_food_units
            +
            firm_result.get("food_output_units", 0.0)
        )
        food_conservation_right = (
            firm_result.get("food_sales_units", 0.0)
            +
            firm_result.get("central_bank_food_subsidy_units", 0.0)
            +
            firm_result.get("food_spoilage_units", 0.0)
            +
            closing_system_food_units
        )
        food_conservation_gap = (
            food_conservation_left
            -
            food_conservation_right
        )

        previous_located_money_stock = (
            self.located_money_stock_history[-2]
            if len(self.located_money_stock_history) >= 2
            else self.initial_private_money_stock
        )
        located_money_delta = located_money_stock - previous_located_money_stock
        net_money_issued = firm_result.get("central_bank_net_money_issued", 0.0)
        money_delta_gap = located_money_delta - net_money_issued
        ledger_summary = self.ledger.summary_for_step(step_index)
        ledger_money_net = (
            ledger_summary["ledger_money_created"]
            -
            ledger_summary["ledger_money_destroyed"]
        )
        ledger_money_net_gap = ledger_money_net - net_money_issued
        household_consumption_sum = sum(
            getattr(household, "consumption_this_step", 0.0)
            for household in self.households
        )
        market_sales_revenue = firm_result.get("sales", 0.0)
        firm_sales_revenue = firm_result.get("firm_sales_revenue", 0.0)
        central_bank_release_revenue = firm_result.get(
            "central_bank_market_release_revenue",
            0.0,
        )
        income_spending_gap = household_consumption_sum - market_sales_revenue
        sales_revenue_split_gap = (
            market_sales_revenue
            -
            firm_sales_revenue
            -
            central_bank_release_revenue
        )
        net_new_money = expected_money_stock - self.initial_private_money_stock
        credit_money_outstanding = firm_result.get(
            "working_capital_loan_balance",
            0.0,
        )
        total_money_stock = authoritative_money_stock
        government_budget_cash = float(getattr(getattr(self, "public_budget", None), "cash", 0.0))
        public_money_stock = public_wealth + central_bank_public_income_balance + government_budget_cash
        if len(self.firm_cash_history) >= 2:
            firm_cash_change = (
                self.firm_cash_history[-1]
                -
                self.firm_cash_history[-2]
            )
        else:
            firm_cash_change = 0.0

        if len(self.firm_inventory_history) >= 2:
            firm_inventory_value_change = (
                self.firm_inventory_history[-1]
                -
                self.firm_inventory_history[-2]
            )
        else:
            firm_inventory_value_change = 0.0

        money_income_ratio = self.safe_ratio(
            total_money_stock,
            total_income,
        )
        money_consumption_ratio = self.safe_ratio(
            total_money_stock,
            total_consumption,
        )
        simplified_money_velocity = self.safe_ratio(
            total_consumption,
            total_money_stock,
        )
        household_money_share = self.safe_ratio(
            total_wealth,
            total_money_stock,
        )
        firm_money_share = self.safe_ratio(
            firm_result.get("firm_cash", 0.0),
            total_money_stock,
        )
        public_money_share = self.safe_ratio(
            public_money_stock,
            total_money_stock,
        )
        firm_cash_to_wage_bill = self.safe_ratio(
            firm_result.get("firm_cash", 0.0),
            firm_result.get("wage_bill", 0.0),
        )
        net_household_saving = total_income - total_consumption
        net_household_saving_rate = self.safe_ratio(
            net_household_saving,
            total_income,
        )
        household_wealth_to_target_wealth = self.safe_ratio(
            total_wealth,
            target_wealth_total,
        )

        tolerance = 1e-4
        invariant_failed = (
            abs(food_conservation_gap) > tolerance
            or
            abs(monetary_accounting_gap) > tolerance
            or
            abs(money_delta_gap) > tolerance
            or
            abs(ledger_money_net_gap) > tolerance
            or
            abs(income_spending_gap) > tolerance
            or
            abs(sales_revenue_split_gap) > tolerance
            or
            firm_result.get("food_inventory_units", 0.0) < -tolerance
            or
            firm_result.get("central_bank_food_inventory_units", 0.0) < -tolerance
        )

        if invariant_failed:
            self.invariant_violations.append(
                {
                    "step": step_index,
                    "food_conservation_gap": food_conservation_gap,
                    "monetary_accounting_gap": monetary_accounting_gap,
                    "money_delta_gap": money_delta_gap,
                    "ledger_money_net_gap": ledger_money_net_gap,
                    "income_spending_gap": income_spending_gap,
                    "sales_revenue_split_gap": sales_revenue_split_gap,
                }
            )

        row = {
            "step": step_index,
            "seed": self.seed if self.seed is not None else "",
            "scenario": self.scenario_name,
            "good_id": firm_result.get("good_id", ""),
            "good_name": firm_result.get("good_name", ""),
            "population": population,
            "births": births,
            "deaths": deaths,
            "birth_rate": birth_rate,
            "death_rate": death_rate,
            "average_age": average_age,
            "average_age_weeks": average_age * 52.0,
            "average_age_years": average_age,
            "children": children,
            "workers": workers,
            "elderly": elderly,
            "child_share": self.safe_ratio(children, population),
            "worker_share": self.safe_ratio(workers, population),
            "elderly_share": self.safe_ratio(elderly, population),
            "entered_worker_age": getattr(
                self, "entered_worker_age_this_step", 0
            ),
            "entered_elderly_age": getattr(
                self, "entered_elderly_age_this_step", 0
            ),
            "households": len(self.households),
            "active_households": active_household_count,
            "average_household_size": self.safe_ratio(
                sum(household_sizes),
                len(household_sizes),
            ),
            "married_households": married_households,
            "single_parent_households": single_parent_households,
            "empty_households": empty_households,
            "total_income": total_income,
            "total_consumption": total_consumption,
            "household_consumption_sum": household_consumption_sum,
            "market_sales_revenue": market_sales_revenue,
            "firm_sales_revenue": firm_sales_revenue,
            "central_bank_release_revenue": central_bank_release_revenue,
            "income_spending_gap": income_spending_gap,
            "sales_revenue_split_gap": sales_revenue_split_gap,
            "total_saving": total_saving,
            "net_household_saving": net_household_saving,
            "net_household_saving_rate": net_household_saving_rate,
            "consumption_rate": consumption_rate,
            "saving_rate": saving_rate,
            "total_household_wealth": total_wealth,
            "average_household_wealth": self.safe_ratio(
                total_wealth,
                active_household_count,
            ),
            "median_household_wealth": self.median(wealth_values),
            "median_household_income": self.median(income_values),
            "wealth_per_capita": wealth_per_capita,
            "average_target_wealth": self.safe_ratio(
                target_wealth_total,
                active_household_count,
            ),
            "target_wealth_reserve_months": getattr(
                economy_config,
                "TARGET_WEALTH_RESERVE_MONTHS",
                0.0,
            ),
            "target_wealth_reserve_weeks": self.median(reserve_weeks_values),
            "household_wealth_to_target_wealth": (
                household_wealth_to_target_wealth
            ),
            "median_security_ratio": self.median(security_ratio_values),
            "mean_security_ratio": self.safe_ratio(
                sum(security_ratio_values),
                len(security_ratio_values),
            ),
            "p10_security_ratio": self.percentile(
                security_ratio_values,
                10,
            ),
            "p90_security_ratio": self.percentile(
                security_ratio_values,
                90,
            ),
            "share_households_below_target": self.safe_ratio(
                households_below_target,
                active_household_count,
            ),
            "average_wealth_gap": self.safe_ratio(
                wealth_gap_total,
                active_household_count,
            ),
            "total_optional_consumption": optional_consumption_total,
            "average_optional_consumption_multiplier": self.safe_ratio(
                optional_multiplier_total,
                active_household_count,
            ),
            "poverty_household_ratio": self.safe_ratio(
                poverty_households,
                active_household_count,
            ),
            "unmet_minimum_need_units": unmet_minimum_need_units,
            "public_support_value": firm_result.get(
                "central_bank_poverty_subsidy_value",
                0.0,
            ),
            "public_support_units": firm_result.get(
                "central_bank_food_subsidy_units",
                0.0,
            ),
            "labor": labor,
            "gdp_nominal_output_value": resource,
            "pressure": pressure,
            "firm_cash": firm_result.get("firm_cash", 0.0),
            "firm_cash_change": firm_cash_change,
            "firm_inventory_value": firm_result.get("firm_inventory", 0.0),
            "firm_inventory_value_change": firm_inventory_value_change,
            "production_inventory_investment": firm_inventory_value_change,
            "firm_net_worth": firm_result.get("firm_net_worth", 0.0),
            "firm_profit_before_dividend": firm_result.get(
                "profit_before_dividend",
                0.0,
            ),
            "aggregate_legacy_profit_before_dividend": firm_result.get(
                "profit_before_dividend", 0.0
            ),
            "sum_firm_profit_before_dividend": math.fsum(
                firm.profit for firm in getattr(self, "firms", [])
            ),
            "firm_profit_aggregation_gap": firm_result.get(
                "profit_before_dividend", 0.0
            ) - math.fsum(
                firm.profit for firm in getattr(self, "firms", [])
            ),
            "total_dividend": math.fsum(
                firm.dividend_payment for firm in getattr(self, "firms", [])
            ),
            "total_household_dividends": sum(
                getattr(household, "dividend_income_this_step", 0.0)
                for household in self.households
            ),
            "total_person_dividends": math.fsum(
                getattr(firm, "person_dividend_paid", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "legacy_dividend_entitlement": math.fsum(
                getattr(firm, "legacy_dividend_entitlement", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "estate_dividend_paid": math.fsum(
                getattr(firm, "estate_dividend_paid", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "legacy_owner_cash": getattr(self, "legacy_owner_cash", 0.0),
            "estate_cash": math.fsum(
                getattr(account, "cash", 0.0)
                for account in getattr(self, "estate_accounts", {}).values()
            ),
            "estate_dividend_income": math.fsum(
                getattr(account, "estate_dividend_income", 0.0)
                for account in getattr(self, "estate_accounts", {}).values()
            ),
            "equity_issuance_cash": math.fsum(
                getattr(firm, "equity_issuance_cash_this_step", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "equity_issuance_shares": math.fsum(
                getattr(firm, "equity_issuance_shares_this_step", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "equity_issuance_buyer_count": sum(
                getattr(firm, "equity_issuance_buyer_count", 0)
                for firm in getattr(self, "firms", [])
            ),
            "secondary_equity_sale_proceeds": firm_result.get(
                "secondary_equity_sale_proceeds", 0.0
            ),
            "secondary_equity_shares_sold": firm_result.get(
                "secondary_equity_shares_sold", 0.0
            ),
            "secondary_equity_buyer_count": firm_result.get(
                "secondary_equity_buyer_count", 0
            ),
            "secondary_decision_step": firm_result.get(
                "secondary_decision_step", float("nan")
            ),
            "secondary_valuation_step": firm_result.get(
                "secondary_valuation_step", float("nan")
            ),
            "secondary_book_equity_used": firm_result.get(
                "secondary_book_equity_used", float("nan")
            ),
            "secondary_total_shares_used": firm_result.get(
                "secondary_total_shares_used", float("nan")
            ),
            "secondary_reference_price_used": firm_result.get(
                "secondary_reference_price_used", float("nan")
            ),
            "secondary_valuation_status": firm_result.get(
                "secondary_valuation_status", "INACTIVE"
            ),
            "dividend_reconciliation_gap": (
                sum(
                    getattr(household, "dividend_income_this_step", 0.0)
                    for household in self.households
                )
                + math.fsum(
                    getattr(firm, "legacy_dividend_entitlement", 0.0)
                    for firm in getattr(self, "firms", [])
                )
                + math.fsum(
                    getattr(firm, "estate_dividend_paid", 0.0)
                    for firm in getattr(self, "firms", [])
                )
                - math.fsum(
                    firm.dividend_payment
                    for firm in getattr(self, "firms", [])
                )
            ),
            "household_dividend_reconciliation_gap": (
                sum(
                    getattr(household, "dividend_income_this_step", 0.0)
                    for household in self.households
                )
                - math.fsum(
                    getattr(firm, "person_dividend_paid", 0.0)
                    for firm in getattr(self, "firms", [])
                )
            ),
            "wage_bill": firm_result.get("wage_bill", 0.0),
            "scheduled_aggregate_wage_bill": math.fsum(
                getattr(firm, "scheduled_wage_bill", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "executed_aggregate_wage_bill": math.fsum(
                getattr(firm, "executed_wage_bill", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "wage_payment": firm_result.get(
                "executed_wage_bill",
                firm_result.get("wage_bill", 0.0),
            ),
            "dividend": firm_result.get("dividend", 0.0),
            "food_price": firm_result.get("food_price", 0.0),
            "legacy_food_price": firm_result.get("food_price", 0.0),
            "household_planning_price_index": (
                getattr(self, "_household_planning_price_value", None)
                if getattr(self, "_household_planning_price_value", None) is not None
                else self.household_planning_price_index()
            ),
            "realized_transaction_price_index": safe_ratio(
                firm_result.get("sales", 0.0),
                firm_result.get("food_sales_units", 0.0),
            ),
            "price_expectation_error": (
                safe_ratio(
                    firm_result.get("sales", 0.0),
                    firm_result.get("food_sales_units", 0.0),
                )
                -
                (
                    getattr(self, "_household_planning_price_value", None)
                    if getattr(self, "_household_planning_price_value", None) is not None
                    else self.household_planning_price_index()
                )
            ),
            "price_expectation_ratio": safe_ratio(
                safe_ratio(
                    firm_result.get("sales", 0.0),
                    firm_result.get("food_sales_units", 0.0),
                ),
                (
                    getattr(self, "_household_planning_price_value", None)
                    if getattr(self, "_household_planning_price_value", None) is not None
                    else self.household_planning_price_index()
                ),
            ),
            "unit_labor_cost": firm_result.get("unit_labor_cost", 0.0),
            "unit_labor_cost_growth": firm_result.get(
                "unit_labor_cost_growth",
                0.0,
            ),
            "price_inventory_gap": firm_result.get(
                "price_inventory_gap",
                0.0,
            ),
            "price_cost_growth": firm_result.get(
                "price_cost_growth",
                0.0,
            ),
            "price_inflation_signal": firm_result.get(
                "price_inflation_signal",
                0.0,
            ),
            "price_log_adjustment": firm_result.get(
                "price_log_adjustment",
                0.0,
            ),
            "demand_pressure": firm_result.get("demand_pressure", 0.0),
            "available_food_supply_units": firm_result.get(
                "available_food_supply_units",
                0.0,
            ),
            "food_output_units": firm_result.get("food_output_units", 0.0),
            "scheduled_productive_capacity": math.fsum(
                getattr(firm, "scheduled_productive_capacity", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "funded_productive_capacity": math.fsum(
                getattr(firm, "funded_productive_capacity", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "actual_production": firm_result.get("food_output_units", 0.0),
            "total_requested_credit": math.fsum(
                getattr(firm, "requested_credit", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "total_executed_credit": math.fsum(
                getattr(firm, "executed_credit", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "total_denied_credit": math.fsum(
                getattr(firm, "denied_credit", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "snapshot_termout_enabled": getattr(
                self, "snapshot_legacy_arrears_termout_enabled", False
            ),
            "termout_start_count_this_step": sum(
                bool(getattr(firm, "termout_start_this_week", False))
                for firm in getattr(self, "firms", [])
            ),
            "total_snapshot_termout_amount": math.fsum(
                getattr(firm, "termout_snapshot_amount", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "total_legacy_arrears_term_claim": math.fsum(
                getattr(firm, "legacy_arrears_term_claim", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "total_post_termout_interest_arrears": math.fsum(
                getattr(firm, "interest_arrears", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "total_revolving_credit_exposure": math.fsum(
                getattr(
                    firm,
                    "revolving_credit_exposure",
                    getattr(firm, "loan_balance", 0.0)
                    + getattr(firm, "interest_arrears", 0.0),
                )
                for firm in getattr(self, "firms", [])
            ),
            "total_lender_exposure": math.fsum(
                getattr(
                    firm,
                    "total_lender_exposure",
                    getattr(firm, "loan_balance", 0.0)
                    + getattr(firm, "interest_arrears", 0.0),
                )
                for firm in getattr(self, "firms", [])
            ),
            "total_legacy_term_claim_payment": math.fsum(
                getattr(firm, "legacy_term_claim_payment", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "total_post_termout_arrears_payment": math.fsum(
                getattr(firm, "post_termout_arrears_payment", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "credit_binding_firm_count": sum(
                bool(getattr(firm, "credit_limit_binding", False))
                for firm in getattr(self, "firms", [])
            ),
            "payroll_constrained_firm_count": sum(
                getattr(firm, "payroll_funding_ratio", 1.0) < 1.0 - 1e-9
                for firm in getattr(self, "firms", [])
            ),
            "current_interest_due": math.fsum(
                getattr(firm, "current_interest_due", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "interest_paid": math.fsum(
                getattr(firm, "interest_paid", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "current_interest_unpaid": math.fsum(
                getattr(firm, "current_interest_unpaid", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "total_interest_arrears": math.fsum(
                getattr(firm, "interest_arrears", 0.0)
                for firm in getattr(self, "firms", [])
            ),
            "central_bank_interest_receivable": getattr(
                self.firm_system.central_bank, "interest_receivable", 0.0
            ),
            "cumulative_interest_income": getattr(
                self.firm_system.central_bank, "cumulative_interest_income", 0.0
            ),
            "food_demand_units": firm_result.get("food_demand_units", 0.0),
            "food_sales_units": firm_result.get("food_sales_units", 0.0),
            "food_inventory_units": firm_result.get("food_inventory_units", 0.0),
            "food_spoilage_units": firm_result.get("food_spoilage_units", 0.0),
            "unmet_food_demand_units": firm_result.get(
                "unmet_food_demand_units",
                0.0,
            ),
            "inventory_demand_ratio": firm_result.get(
                "inventory_demand_ratio",
                0.0,
            ),
            "central_bank_food_inventory_units": firm_result.get(
                "central_bank_food_inventory_units",
                0.0,
            ),
            "central_bank_food_purchase_units": firm_result.get(
                "central_bank_food_purchase_units",
                0.0,
            ),
            "central_bank_food_purchase_requested_units": firm_result.get(
                "central_bank_food_purchase_requested_units",
                firm_result.get("central_bank_food_purchase_units", 0.0),
            ),
            "central_bank_food_purchase_executed_units": firm_result.get(
                "central_bank_food_purchase_executed_units",
                firm_result.get("central_bank_food_purchase_units", 0.0),
            ),
            "central_bank_food_purchase_unfilled_units": firm_result.get(
                "central_bank_food_purchase_unfilled_units",
                0.0,
            ),
            "central_bank_food_release_units": firm_result.get(
                "central_bank_food_release_units",
                0.0,
            ),
            "central_bank_food_subsidy_units": firm_result.get(
                "central_bank_food_subsidy_units",
                0.0,
            ),
            "central_bank_inventory_purchase": firm_result.get(
                "central_bank_inventory_purchase",
                0.0,
            ),
            "central_bank_market_release_revenue": firm_result.get(
                "central_bank_market_release_revenue",
                0.0,
            ),
            "central_bank_market_release_money_destroyed": firm_result.get(
                "central_bank_market_release_money_destroyed",
                0.0,
            ),
            "central_bank_market_release_public_income": firm_result.get(
                "central_bank_market_release_public_income",
                0.0,
            ),
            "central_bank_public_income_balance": (
                central_bank_public_income_balance
            ),
            "government_budget_cash": government_budget_cash,
            "government_tax_revenue": float(getattr(getattr(self, "public_budget", None), "actual_tax_this_step", 0.0)),
            "government_tax_shortfall": float(getattr(getattr(self, "public_budget", None), "tax_shortfall_this_step", 0.0)),
            "central_bank_public_income": firm_result.get(
                "central_bank_public_income",
                0.0,
            ),
            "central_bank_public_income_used": firm_result.get(
                "central_bank_public_income_used",
                0.0,
            ),
            "central_bank_public_income_from_death": (
                self.current_public_wealth_audit[
                    "central_bank_public_income_from_death"
                ]
            ),
            "central_bank_public_income_from_no_heir": (
                self.current_public_wealth_audit[
                    "central_bank_public_income_from_no_heir"
                ]
            ),
            "central_bank_public_income_from_household_dissolution": (
                self.current_public_wealth_audit[
                    "central_bank_public_income_from_household_dissolution"
                ]
            ),
            "central_bank_public_income_from_public_inventory": (
                self.current_public_wealth_audit[
                    "central_bank_public_income_from_public_inventory"
                ]
            ),
            "central_bank_public_income_from_loan_interest": (
                self.current_public_wealth_audit[
                    "central_bank_public_income_from_loan_interest"
                ]
            ),
            "central_bank_public_income_from_other": (
                self.current_public_wealth_audit[
                    "central_bank_public_income_from_other"
                ]
            ),
            "cumulative_central_bank_public_income_from_death": (
                self.cumulative_public_wealth_audit[
                    "central_bank_public_income_from_death"
                ]
            ),
            "cumulative_central_bank_public_income_from_no_heir": (
                self.cumulative_public_wealth_audit[
                    "central_bank_public_income_from_no_heir"
                ]
            ),
            "cumulative_central_bank_public_income_from_household_dissolution": (
                self.cumulative_public_wealth_audit[
                    "central_bank_public_income_from_household_dissolution"
                ]
            ),
            "cumulative_central_bank_public_income_from_public_inventory": (
                self.cumulative_public_wealth_audit[
                    "central_bank_public_income_from_public_inventory"
                ]
            ),
            "cumulative_central_bank_public_income_from_loan_interest": (
                self.cumulative_public_wealth_audit[
                    "central_bank_public_income_from_loan_interest"
                ]
            ),
            "cumulative_central_bank_public_income_from_other": (
                self.cumulative_public_wealth_audit[
                    "central_bank_public_income_from_other"
                ]
            ),
            "central_bank_net_money_issued": net_money_issued,
            "net_new_money": net_new_money,
            "credit_money_outstanding": credit_money_outstanding,
            "total_money_stock": total_money_stock,
            "public_money_stock": public_money_stock,
            "money_income_ratio": money_income_ratio,
            "money_consumption_ratio": money_consumption_ratio,
            "simplified_money_velocity": simplified_money_velocity,
            "household_money_share": household_money_share,
            "firm_money_share": firm_money_share,
            "public_money_share": public_money_share,
            "firm_cash_to_wage_bill": firm_cash_to_wage_bill,
            "working_capital_loan_issued": firm_result.get(
                "working_capital_loan_issued",
                0.0,
            ),
            "working_capital_target_cash": firm_result.get(
                "working_capital_target_cash",
                0.0,
            ),
            "working_capital_funding_gap": firm_result.get(
                "working_capital_funding_gap",
                0.0,
            ),
            "target_cash": firm_result.get("target_cash", 0.0),
            "funding_gap": firm_result.get("funding_gap", 0.0),
            "loan_issued": firm_result.get("loan_issued", 0.0),
            "working_capital_loan_repaid": firm_result.get(
                "working_capital_loan_repaid",
                0.0,
            ),
            "working_capital_interest_paid": firm_result.get(
                "working_capital_interest_paid",
                0.0,
            ),
            "working_capital_loan_balance": firm_result.get(
                "working_capital_loan_balance",
                0.0,
            ),
            "loan_balance": firm_result.get("loan_balance", 0.0),
            "ledger_transaction_count": ledger_summary[
                "ledger_transaction_count"
            ],
            "ledger_transfer_amount": ledger_summary[
                "ledger_transfer_amount"
            ],
            "ledger_money_created": ledger_summary[
                "ledger_money_created"
            ],
            "ledger_money_destroyed": ledger_summary[
                "ledger_money_destroyed"
            ],
            "ledger_money_net": ledger_money_net,
            "ledger_money_net_gap": ledger_money_net_gap,
            "ledger_loans_created": ledger_summary[
                "ledger_loans_created"
            ],
            "ledger_loans_repaid": ledger_summary[
                "ledger_loans_repaid"
            ],
            "private_cash_stock": private_cash_stock,
            "social_household_cash": money_components["social_household_cash"],
            "settlement_only_cash": money_components["settlement_only_cash"],
            "legacy_owner_cash": money_components["legacy_owner_cash"],
            "estate_cash": money_components["estate_cash"],
            "pending_household_formation_wealth": money_components[
                "pending_household_formation_wealth"
            ],
            "public_wealth": public_wealth,
            "public_wealth_from_death": self.current_public_wealth_audit[
                "public_wealth_from_death"
            ],
            "public_wealth_from_no_heir": self.current_public_wealth_audit[
                "public_wealth_from_no_heir"
            ],
            "public_wealth_from_household_dissolution": (
                self.current_public_wealth_audit[
                    "public_wealth_from_household_dissolution"
                ]
            ),
            "public_wealth_from_other": self.current_public_wealth_audit[
                "public_wealth_from_other"
            ],
            "inherited_wealth_to_spouse_or_survivors": (
                self.current_public_wealth_audit[
                    "inherited_wealth_to_spouse_or_survivors"
                ]
            ),
            "inherited_wealth_to_children": self.current_public_wealth_audit[
                "inherited_wealth_to_children"
            ],
            "cumulative_public_wealth_from_death": (
                self.cumulative_public_wealth_audit[
                    "public_wealth_from_death"
                ]
            ),
            "cumulative_public_wealth_from_no_heir": (
                self.cumulative_public_wealth_audit[
                    "public_wealth_from_no_heir"
                ]
            ),
            "cumulative_public_wealth_from_household_dissolution": (
                self.cumulative_public_wealth_audit[
                    "public_wealth_from_household_dissolution"
                ]
            ),
            "cumulative_public_wealth_from_other": (
                self.cumulative_public_wealth_audit[
                    "public_wealth_from_other"
                ]
            ),
            "cumulative_inherited_wealth_to_spouse_or_survivors": (
                self.cumulative_public_wealth_audit[
                    "inherited_wealth_to_spouse_or_survivors"
                ]
            ),
            "cumulative_inherited_wealth_to_children": (
                self.cumulative_public_wealth_audit[
                    "inherited_wealth_to_children"
                ]
            ),
            "expected_money_stock": expected_money_stock,
            "authoritative_money_stock": authoritative_money_stock,
            "located_money_stock": located_money_stock,
            "monetary_accounting_gap": monetary_accounting_gap,
            "full_money_location_gap": monetary_accounting_gap,
            "located_money_delta": located_money_delta,
            "money_delta_gap": money_delta_gap,
            "food_conservation_gap": food_conservation_gap,
             "invariant_failed": invariant_failed,
             "default_event_count_this_step": sum(
                 bool(getattr(firm, "default_event_this_week", False))
                 for firm in getattr(self, "firms", [])
             ),
             "active_default_firm_count": sum(
                 bool(getattr(firm, "active_default_episode", False))
                 for firm in getattr(self, "firms", [])
             ),
             "active_contract_default_firm_count": sum(
                 bool(getattr(firm, "active_contract_default", False))
                 for firm in getattr(self, "firms", [])
             ),
             "restructuring_active_firm_count": sum(
                 bool(getattr(firm, "restructuring_active", False))
                 for firm in getattr(self, "firms", [])
             ),
             "restructuring_start_count_this_step": sum(
                 bool(getattr(firm, "restructuring_start_this_week", False))
                 for firm in getattr(self, "firms", [])
             ),
             "weekly_interest_relief_amount": sum(
                 float(getattr(firm, "weekly_interest_relief_amount", 0.0))
                 for firm in getattr(self, "firms", [])
             ),
             "cumulative_interest_relief_amount": sum(
                 float(getattr(firm, "cumulative_interest_relief_amount", 0.0))
                 for firm in getattr(self, "firms", [])
             ),
             "contract_cure_count_this_step": sum(
                 bool(getattr(firm, "contract_cure_this_week", False))
                 for firm in getattr(self, "firms", [])
             ),
             "acute_default_phase_exit_count_this_step": sum(
                 bool(
                     getattr(
                         firm,
                         "acute_default_phase_exit_this_week",
                         False,
                     )
                 )
                 for firm in getattr(self, "firms", [])
             ),
             "default_history_count_total": sum(
                 int(getattr(firm, "default_history_count", 0))
                 for firm in getattr(self, "firms", [])
             ),
         }

        row.update(
            {
                "multi_firm_bootstrap_complete": getattr(
                    self,
                    "multi_firm_bootstrap_complete",
                    True,
                ),
                "multi_firm_bootstrap_execution_count": getattr(
                    self,
                    "multi_firm_bootstrap_execution_count",
                    0,
                ),
                "bootstrap_cash_gap": getattr(self, "bootstrap_cash_gap", 0.0),
                "bootstrap_inventory_units_gap": getattr(
                    self,
                    "bootstrap_inventory_units_gap",
                    0.0,
                ),
                "bootstrap_inventory_book_value_gap": getattr(
                    self,
                    "bootstrap_inventory_book_value_gap",
                    0.0,
                ),
                "bootstrap_capacity_gap": getattr(
                    self,
                    "bootstrap_capacity_gap",
                    0.0,
                ),
                "bootstrap_wage_bill_gap": getattr(
                    self,
                    "bootstrap_wage_bill_gap",
                    0.0,
                ),
                "bootstrap_expected_demand_gap": getattr(
                    self,
                    "bootstrap_expected_demand_gap",
                    0.0,
                ),
                "bootstrap_production_gap": getattr(
                    self,
                    "bootstrap_production_gap",
                    0.0,
                ),
                "bootstrap_loan_gap": getattr(
                    self,
                    "bootstrap_loan_gap",
                    0.0,
                ),
            }
        )

        row.update(self.time_metadata(step_index))
        row.update(self.demographic_structure_diagnostics())
        row.update(self.marriage_diagnostics())

        self.diagnostics_rows.append(row)

    def export_diagnostics_csv(self, path):
        if not path:
            return

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        fieldnames = []

        for row in self.diagnostics_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.diagnostics_rows)

    def export_firm_diagnostics_csv(self, path):
        if not path:
            return

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        fieldnames = (
            list(self.firm_diagnostics_rows[0].keys())
            if self.firm_diagnostics_rows
            else []
        )

        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if fieldnames:
                writer.writeheader()
                writer.writerows(self.firm_diagnostics_rows)

    def export_household_diagnostics_csv(self, path):
        if not path:
            return

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        fieldnames = []
        for row in getattr(self, "household_diagnostics_rows", []):
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(self.household_diagnostics_rows)

    def export_capital_provenance_csv(self, directory):
        """Persist event-level asset/order provenance for Analysis only."""
        directory = str(directory)
        os.makedirs(directory, exist_ok=True)
        asset_fields = (
            "asset_id", "asset_class", "owner_firm_id", "acquisition_week", "acquisition_cost",
            "quantity", "physical_asset_units", "capital_service_capacity_per_unit",
            "accumulated_depreciation", "closing_book_value", "useful_life_weeks", "age_weeks",
            "active", "retirement_week", "origin_order_id", "investment_source",
            "replaced_asset_id", "replacement_trigger_id", "replacement_order_id",
        )
        event_fields = (
            "global_step", "event_type", "asset_id", "order_id", "buyer_firm_id",
            "supplier_firm_id", "owner_firm_id", "investment_source", "replacement_trigger_id",
            "replacement_order_id", "replaced_asset_id", "requested_units", "settled_units",
            "unmet_units", "settled_expenditure", "capital_asset_created", "acquisition_cost",
            "physical_asset_units",
        )
        assets = []
        events = list(getattr(self, "capital_asset_event_rows", []))
        for firm in [*getattr(self, "firms", []), *getattr(self, "capital_good_firms", [])]:
            stock = getattr(firm, "capital_stock", None)
            for asset in getattr(stock, "assets", []) if stock is not None else []:
                metadata = dict(getattr(asset, "capacity_metadata", {}) or {})
                assets.append({
                    "asset_id": asset.asset_id,
                    "asset_class": asset.asset_class,
                    "owner_firm_id": asset.owner_firm_id,
                    "acquisition_week": metadata.get("acquisition_week", "UNAVAILABLE"),
                    "acquisition_cost": asset.acquisition_cost,
                    "quantity": asset.quantity,
                    "physical_asset_units": metadata.get("physical_asset_units", asset.quantity),
                    "capital_service_capacity_per_unit": metadata.get("capital_service_capacity_per_unit", "UNAVAILABLE"),
                    "accumulated_depreciation": asset.accumulated_depreciation,
                    "closing_book_value": asset.closing_book_value,
                    "useful_life_weeks": asset.useful_life_weeks if hasattr(asset, "useful_life_weeks") else "UNAVAILABLE",
                    "age_weeks": asset.age_weeks if hasattr(asset, "age_weeks") else asset.age,
                    "active": asset.is_active,
                    "retirement_week": metadata.get("retirement_week", "UNAVAILABLE"),
                    "origin_order_id": metadata.get("origin_order_id", "UNAVAILABLE"),
                    "investment_source": metadata.get("investment_source", "UNAVAILABLE"),
                    "replaced_asset_id": metadata.get("replaced_asset_id", "UNAVAILABLE"),
                    "replacement_trigger_id": metadata.get("replacement_trigger_id", "UNAVAILABLE"),
                    "replacement_order_id": metadata.get("replacement_order_id", "UNAVAILABLE"),
                })

        def write(name, rows, fields):
            path = os.path.join(directory, name)
            fields = list(fields)
            for row in rows:
                for field in row:
                    if field not in fields:
                        fields.append(field)
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            return path

        return {
            "asset_ledger": write("capital_asset_ledger.csv", assets, asset_fields),
            "provenance_events": write("capital_provenance_events.csv", events, event_fields),
        }

    def export_household_wealth_instrumentation(self, directory):
        """Export optional household micro-wealth diagnostics only."""
        os.makedirs(directory, exist_ok=True)

        def write(name, rows):
            path = os.path.join(directory, name)
            fields = []
            for row in rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                if fields:
                    writer.writeheader()
                    writer.writerows(rows)
            return path

        trace = list(getattr(self, "household_wealth_micro_trace_rows", []))
        write("household_wealth_weekly_summary.csv", getattr(self, "household_wealth_weekly_summary_rows", []))
        write("household_wealth_micro_trace.csv", trace)

        histories = defaultdict(list)
        for row in trace:
            histories[row["household_id"]].append(row)
        persistence = []
        transitions = defaultdict(int)
        for household_id, rows in histories.items():
            rows.sort(key=lambda row: int(row["week"]))
            low_weeks = [int(row["week"]) for row in rows if row.get("low_wealth_flag")]
            if not low_weeks:
                continue
            spells = []
            current = 0
            previous = None
            for week in low_weeks:
                if previous is not None and week == previous + 1:
                    current += 1
                else:
                    if current:
                        spells.append(current)
                    current = 1
                previous = week
            if current:
                spells.append(current)
            entries = sum(bool(row.get("entered_low_wealth_this_week")) for row in rows)
            exits = sum(bool(row.get("exited_low_wealth_this_week")) for row in rows)
            persistence.append({
                "household_id": household_id,
                "first_low_wealth_week": min(low_weeks),
                "last_low_wealth_week": max(low_weeks),
                "cumulative_low_wealth_weeks": len(low_weeks),
                "longest_continuous_low_wealth_spell": max(spells),
                "low_wealth_entry_count": entries,
                "low_wealth_exit_count": exits,
                "trailing_52_low_wealth_weeks": sum(week >= max(low_weeks) - 51 for week in low_weeks),
                "trailing_260_low_wealth_weeks": sum(week >= max(low_weeks) - 259 for week in low_weeks),
                "classification": "persistent_low_wealth" if max(spells) >= 260 else ("recurrent_low_wealth" if entries > 1 else "transient_visitor"),
            })
            for before, after in zip(rows, rows[1:]):
                if int(after["week"]) != int(before["week"]) + 1:
                    continue
                before_state = "LOW" if before.get("low_wealth_flag") else "HIGH"
                after_state = "LOW" if after.get("low_wealth_flag") else ("100+" if float(after.get("closing_wealth", 0)) >= 100 else "NEG/HIGH")
                transitions[f"{before_state}->{after_state}"] += 1
        write("household_low_wealth_persistence.csv", persistence)
        write("low_wealth_transition_matrix.csv", [{"transition": key, "count": value} for key, value in sorted(transitions.items())])
        new_rows = [row for row in trace if row.get("created_this_week")]
        write("new_household_initial_wealth.csv", new_rows)
        lifecycle = getattr(self, "household_lifecycle_events", [])
        write("lifecycle_event_wealth_analysis.csv", lifecycle)
        write(
            "household_lifecycle_transfer_events.csv",
            getattr(self, "household_lifecycle_transfer_events", []),
        )
        return directory

    def export_household_employer_exposure(self, directory):
        """Export diagnostics-only worker/employer wage exposure records."""
        os.makedirs(directory, exist_ok=True)

        def write(name, rows):
            path = os.path.join(directory, name)
            fields = []
            for row in rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                if fields:
                    writer.writeheader()
                    writer.writerows(rows)
            return path

        write(
            "household_employer_exposure_micro.csv",
            getattr(self, "household_employer_exposure_rows", []),
        )
        write(
            "household_employer_exposure_weekly_summary.csv",
            getattr(self, "household_employer_exposure_weekly_rows", []),
        )
        return directory

    def export_household_active_denominator(self, directory):
        """Export compact all-active household-week risk-set diagnostics."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "all_active_household_week_denominator.csv")
        stream_path = getattr(
            self, "household_active_denominator_stream_path", None
        )
        stream = getattr(self, "_household_active_denominator_stream", None)
        if stream is not None:
            stream.flush()
            stream.close()
            self._household_active_denominator_stream = None
            return stream_path
        rows = getattr(self, "household_active_denominator_rows", [])
        fields = [
            "week", "household_id", "wealth_open", "wealth_close",
            "negative_open", "negative_close", "low_wealth_open",
            "low_wealth_close", "negative_entry", "negative_exit",
            "employed_worker_count", "scheduled_wage_income",
            "executed_wage_income", "wage_shortfall", "wage_execution_ratio",
            "exposed_to_borrowing_firm", "exposed_to_credit_binding_firm",
            "exposed_to_payroll_underfunded_firm", "borrowing_worker_count",
            "binding_worker_count", "payroll_underfunded_worker_count",
            "total_income", "public_support", "actual_consumption",
            "canonical_net_cash_flow", "recovery_margin", "household_size",
            "child_count", "elderly_count",
        ]
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if fields:
                writer.writeheader()
                writer.writerows(dict(zip(fields, row)) for row in rows)
        return path

    def export_accounting(self, directory):
        if not hasattr(self, "accounting"):
            self.accounting = AccountingLayer(self)
        return self.accounting.export_csv(directory)

    def export_demographic_diagnostics(self, directory):
        os.makedirs(directory, exist_ok=True)

        def write_rows(filename, rows):
            path = os.path.join(directory, filename)
            fields = []
            for row in rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            return path

        events_path = write_rows("demographic_events.csv", self.demographic_events)
        markets_path = write_rows(
            "marriage_market_diagnostics.csv",
            self.marriage_market_diagnostics,
        )
        annual_path = write_rows(
            "demographic_annual_summary.csv",
            self.build_demographic_annual_summary(),
        )
        initial_age_path = write_rows(
            "initial_age_phase_diagnostics.csv",
            getattr(self, "initial_age_phase_diagnostics_rows", []),
        )
        initial_histogram_path = write_rows(
            "initial_age_phase_histogram.csv",
            self.initial_age_phase_histogram(),
        )
        age_transitions_path = write_rows(
            "age_transition_diagnostics.csv",
            getattr(self, "age_transition_diagnostics", []),
        )
        return (
            events_path,
            markets_path,
            annual_path,
            initial_age_path,
            initial_histogram_path,
            age_transitions_path,
        )

    def build_demographic_annual_summary(self):
        output = []
        for year in range((len(self.diagnostics_rows) + 51) // 52):
            end = min((year + 1) * 52, len(self.diagnostics_rows)) - 1
            if end < 0:
                continue
            start = year * 52
            current = self.diagnostics_rows[end]
            events = [
                event for event in self.demographic_events
                if start <= int(event["global_step"]) <= end
            ]
            births = sum(event["event_type"] == "birth" for event in events)
            deaths = sum(event["event_type"] == "death" for event in events)
            marriages = sum(event["event_type"] == "marriage" for event in events)
            fertility_records = [
                record for record in getattr(
                    self.fertility_system,
                    "fertility_probability_records",
                    [],
                )
                if start <= record["global_step"] <= end
            ]
            intervals = [
                event["inter_birth_interval_weeks"]
                for event in events
                if event["event_type"] == "birth"
                and event.get("inter_birth_interval_weeks") is not None
            ]
            population = int(current.get("population", 0))
            output.append({
                "simulation_year": year,
                "population": population,
                "household_count": current.get("households", 0),
                "age_0_19_count": current.get("age_0_19_count", ""),
                "age_20_39_count": current.get("age_20_39_count", ""),
                "age_40_64_count": current.get("age_40_64_count", ""),
                "age_65_plus_count": current.get("age_65_plus_count", ""),
                "age_0_19_share": current.get("age_0_19_share", ""),
                "age_20_39_share": current.get("age_20_39_share", ""),
                "age_40_64_share": current.get("age_40_64_share", ""),
                "age_65_plus_share": current.get("age_65_plus_share", ""),
                "mean_age": current.get("average_age", 0),
                "median_age": "",
                "reproductive_age_female_count": current.get("reproductive_age_female_count", ""),
                "working_age_population": current.get("working_age_population", ""),
                "elderly_population": current.get("elderly_population", ""),
                "births": births,
                "deaths": deaths,
                "marriages": marriages,
                "crude_birth_rate": self.safe_ratio(births, population),
                "crude_death_rate": self.safe_ratio(deaths, population),
                "natural_population_change": births - deaths,
                "successful_births": births,
                "fertility_draws": len(fertility_records),
                "inter_birth_interval_min": min(intervals) if intervals else "",
                "inter_birth_interval_mean": sum(intervals) / len(intervals) if intervals else "",
                "count_interval_lt_52": sum(interval < 52 for interval in intervals),
                "count_interval_eq_52": sum(interval == 52 for interval in intervals),
                "fertility_annual_probability_mean": (
                    sum(record["annual"] for record in fertility_records) / len(fertility_records)
                    if fertility_records else ""
                ),
                "fertility_weekly_probability_mean": (
                    sum(record["weekly"] for record in fertility_records) / len(fertility_records)
                    if fertility_records else ""
                ),
            })
        return output

    def diagnostics_summary(self):
        if not self.diagnostics_rows:
            return {
                "rows": 0,
                "invariant_violations": 0,
                "max_abs_monetary_accounting_gap": 0.0,
                "max_abs_food_conservation_gap": 0.0,
                "max_abs_money_delta_gap": 0.0,
                "max_abs_ledger_money_net_gap": 0.0,
                "max_abs_income_spending_gap": 0.0,
                "max_abs_sales_revenue_split_gap": 0.0,
            }

        def max_abs(field):
            return max(
                abs(row.get(field, 0.0))
                for row in self.diagnostics_rows
            )

        return {
            "rows": len(self.diagnostics_rows),
            "invariant_violations": len(self.invariant_violations),
            "max_abs_monetary_accounting_gap": max_abs(
                "monetary_accounting_gap"
            ),
            "max_abs_food_conservation_gap": max_abs(
                "food_conservation_gap"
            ),
            "max_abs_money_delta_gap": max_abs("money_delta_gap"),
            "max_abs_ledger_money_net_gap": max_abs("ledger_money_net_gap"),
            "max_abs_income_spending_gap": max_abs("income_spending_gap"),
            "max_abs_sales_revenue_split_gap": max_abs(
                "sales_revenue_split_gap"
            ),
            "final_population": self.diagnostics_rows[-1]["population"],
            "final_located_money_stock": self.diagnostics_rows[-1].get(
                "located_money_stock",
                0.0,
            ),
        }

    def goods_summary(self):
        return [
            {
                "id": good.id,
                "name": good.name,
                "is_essential": good.is_essential,
                "is_storable": good.is_storable,
                "perish_rate": good.perish_rate,
                "unit": good.unit,
            }
            for good in self.goods_catalog.all()
        ]

    # =================================================
    # Run
    # =================================================

    def run(self, progress_interval=100):

        for step in range(self.steps):

            self.step()

            # 姣?00姝ユ樉绀轰竴娆¤繘搴?
            if progress_interval and step % progress_interval == 0:

                population = len(self.population)

                total_wealth = sum(
                    h.wealth
                    for h in self.households
                )

                print(
                    f"Elapsed week {step}/{self.steps} | "
                    f"Population: {population} | "
                    f"Households: {len(self.households)} | "
                    f"Wealth: {total_wealth:.2f}"
                )

        print("Simulation finished.")




