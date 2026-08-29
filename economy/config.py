# economy/config.py

# 收入转换系数
PRODUCTIVITY_TO_INCOME = 100.0

# 给父母的赡养比例
PARENT_SUPPORT_RATIO = 0.15

# Step 15 experimental private child-Household -> parent-Household transfer.
# The legacy IncomeSystem remains disconnected; this mechanism is default OFF.
PRIVATE_FAMILY_SUPPORT_ENABLED = False
# Selected Step15 research contract. This target is declarative only; support
# remains default OFF and must be enabled explicitly by a research scenario.
PRIVATE_FAMILY_SUPPORT_BUFFER_WEEKS = 0.25
FAMILY_LINK_OBSERVABILITY_ENABLED = False
FAMILY_LINK_OBSERVABILITY_CADENCE = 13

# 最高赡养父母人数
MAX_SUPPORTED_PARENTS = 2

# =====================================================
# Consumption Parameters
# =====================================================

# 每个家庭每回合固定消费
BASE_CONSUMPTION = 20.0

# 每位家庭成员带来的基础消费
CONSUMPTION_PER_MEMBER = 15.0

# 财富消费率（消费一小部分存量财富）
WEALTH_CONSUMPTION_RATE = 0.002

# 老年人消费系数
ELDERLY_CONSUMPTION_FACTOR = 1

# 儿童消费系数
CHILD_CONSUMPTION_FACTOR = 0.75

# 成年人消费系数
ADULT_CONSUMPTION_FACTOR = 1.5

# 每回合消费不能超过收入的比例
MAX_CONSUMPTION_RATIO = 0.95

# =====================================================
# New Consumption Function Parameters
# =====================================================

NEW_BASE_CONSUMPTION = 12.0
NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER = 28.0

FIRST_ADULT_WEIGHT = 1.0
SECOND_ADULT_WEIGHT = 0.65
NEW_CHILD_WEIGHT = 0.45
NEW_ELDERLY_WEIGHT = 0.75

CHILD_EXTRA_COST = 8.0
ELDERLY_EXTRA_COST = 10.0

DISCRETIONARY_INCOME_RATE = 0.25
NEW_WEALTH_CONSUMPTION_RATE = 0.0015
WEALTH_DRAWDOWN_RATE = 0.08

TARGET_WEALTH_RESERVE_MONTHS = 6.0
TARGET_WEALTH_GAP_SENSITIVITY = 1.0
MIN_OPTIONAL_CONSUMPTION_MULTIPLIER = 0.35
EXCESS_WEALTH_CONSUMPTION_RATE = 0.0015

# =====================================================
# Goods Catalog
# =====================================================

BASIC_CONSUMPTION_GOOD_ID = "basic_consumption_good"
BASIC_CONSUMPTION_GOOD_NAME = "Basic Consumption Good"
BASIC_CONSUMPTION_GOOD_UNIT = "unit"

# Stage A generalized Firm operating contracts.  This flag enables passive
# shadow views only; the existing operating and financial executors remain
# authoritative.
GENERALIZED_FIRM_OPERATING_CONTRACTS = False

# Step 14A passive sector/good/market contracts.  The canonical Food
# executor remains authoritative in both modes.
MULTISECTOR_FOUNDATION_ENABLED = False

# Step 14C is a passive household-budget shadow boundary.  The canonical Food
# executor remains authoritative; this flag never activates Service behavior.
SHADOW_MULTIGOOD_HOUSEHOLD_DEMAND_ENABLED = False

# Step 14D exposes active-capable Service runtime plumbing only to isolated
# fixtures. It does not activate Service Firms in the canonical World.
SERVICE_FIRM_RUNTIME_ENABLED = False

# Step 15I.4 canonical investment screen.  The default remains OFF; these
# values are explicit engineering screen controls, not final calibration.
CANONICAL_INVESTMENT_ENABLED = False
# Step17.Y endogenous capital-goods entry is opt-in; default behavior is unchanged.
ENDOGENOUS_CAPITAL_GOODS_ENTRY_ENABLED = False
CAPITAL_GOOD_FIRM_COUNT = 1
CAPITAL_GOOD_STARTUP_CASH_FRACTION = 0.001
CAPITAL_GOOD_LABOR_SHARE = 0.05
CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR = 1.0
CAPITAL_GOOD_UNIT_PRICE = 10.0
CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS = 13
# The buyer investment-review cadence and the supplier's backlog-service
# horizon are distinct clocks.  They share the historical 13-week default,
# preserving the accepted canonical path unless a controlled screen overrides
# one of them explicitly.
CAPITAL_GOOD_BACKLOG_SERVICE_HORIZON_WEEKS = 13

# Step 15 committed-capital planner screen.  Default OFF preserves the
# accepted installed-capital-only investment planner until its pipeline
# treatment has been validated independently.
CAPITAL_GOOD_COMMITTED_CAPITAL_PLANNER_ENABLED = False

# Step 15 pipeline-depletion screen.  Default OFF retains scheduled 13-week
# investment reviews.  The treatment may add a Firm-specific review only
# after the accepted-order pipeline has fully cleared.
CAPITAL_GOOD_PIPELINE_DEPLETION_EARLY_REVIEW_ENABLED = True

# Step 15 accepted opening-balance-sheet contract.  The initial one-week
# minimum-consumption buffer is funded by reallocating opening Food-Firm cash;
# it does not create money and is disabled only by an explicit scenario
# override for historical control fixtures.
INITIAL_HOUSEHOLD_ONE_WEEK_CONSUMPTION_BUFFER_ENABLED = True

# Post-Step15 deterministic scheduler screen. Disabled by default. When
# enabled, Firm review formulas and cadences are unchanged; only stable
# Firm-specific phases differ.
DETERMINISTIC_FIRM_REVIEW_PHASE_STAGGERING_ENABLED = False

# Step 15 adult settlement boundary: keep social Household membership and
# economic wage/consumption settlement distinct. Default OFF preserves the
# accepted pre-fix trajectory.
ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED = False

# Step 15 joint age-labor contract screen.  The default retains the historic
# gate (positive age productivity); treatment modes are scenario-local.
AGE_LABOR_PARTICIPATION_CONTRACT_MODE = "current"
AGE_LABOR_HARD_EXIT_AGE = 65.0
AGE_LABOR_GRADUAL_TRANSITION_START_AGE = 60.0
AGE_LABOR_GRADUAL_EXIT_AGE = 75.0

# Step 15I.11D: full-order customer advances for the capital-good supplier.
# Disabled by default so the accepted pre-advance investment path is unchanged.
CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED = False

# Step 15I.7 runtime treatment switch.  When disabled, capital capacity is
# diagnostic-only and the accepted labor-only Food executor is unchanged.
CAPITAL_CAPACITY_RUNTIME_ENABLED = False

# Step 15I.9 controlled lifecycle screen.  This is an explicitly selected
# engineering horizon for the screen, not a permanent useful-life calibration.
CAPITAL_LIFECYCLE_ENABLED = False
CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS = 52.0

# =====================================================
# Single Firm / Food Market Parameters
# =====================================================

FIRM_INITIAL_CASH = 10_000_000.0
FIRM_COUNT = 1
PRICE_CHOICE_SENSITIVITY = 2.0
INITIAL_FIRM_RELATIVE_PRICE_SPREAD = 0.02
FIRM_PRICE_REVIEW_PROBABILITY = 0.20
FIRM_PRICE_TRIAL_STEP_SMALL = 0.005
FIRM_PRICE_TRIAL_STEP_LARGE = 0.010
FIRM_PRICE_KEEP_PROBABILITY = 0.20
FIRM_PRICE_DIRECTION_PERSIST_PROBABILITY = 0.65
FIRM_PRICE_DIRECTION_REVERSE_PROBABILITY = 0.65
FIRM_PRICE_INVENTORY_WORSEN_TOLERANCE = 0.05
FIRM_PRICE_MIN_MARKUP = 1.02
FIRM_PRICE_EVALUATION_WINDOW = 12
FIRM_PRICE_EVALUATION_WINDOW_WEEKS = FIRM_PRICE_EVALUATION_WINDOW
FIRM_PRICE_PROFIT_EMA_ALPHA = 0.20
FIRM_PRICE_EXPLORATION_PROBABILITY = 0.08
FIRM_PRODUCTION_EXPECTED_DEMAND_ALPHA = 0.10
FIRM_PRODUCTION_TARGET_INVENTORY_COVERAGE = 15.0
FIRM_PRODUCTION_TARGET_INVENTORY_COVERAGE_WEEKS = (
    FIRM_PRODUCTION_TARGET_INVENTORY_COVERAGE
)
FIRM_PRODUCTION_INVENTORY_GAP_GAIN = 0.25
FIRM_PRODUCTION_PLAN_ADJUSTMENT_RATE = 0.08
FIRM_PRODUCTION_MAX_RELATIVE_CHANGE = 0.05
FIRM_PRODUCTION_REVIEW_INTERVAL = 5
FIRM_PRODUCTION_REVIEW_INTERVAL_WEEKS = FIRM_PRODUCTION_REVIEW_INTERVAL
FIRM_WAGE_SHARE = 0.45
FIRM_DIVIDEND_SHARE = 0.15

# Step 17.V newly formed Firms may opt into deterministic Person founder ownership.\r\n# Default OFF preserves legacy-compatible canonical behavior.\r\nPERSON_FOUNDER_BOOTSTRAP_ENABLED = False\r\n\r\n# Step 15F.2 gradual Person primary-equity transition.  The default is OFF;
# these are engineering guardrails, not ownership-distribution targets.
PERSON_EQUITY_TRANSITION_ENABLED = False
PERSON_EQUITY_ISSUANCE_PRICE = 10.0
PERSON_EQUITY_REVIEW_INTERVAL_WEEKS = 52
PERSON_EQUITY_MAX_HOUSEHOLD_AVAILABLE_CASH_FRACTION = 0.01
PERSON_EQUITY_MAX_FIRM_OPENING_BASE_FRACTION = 0.0001
PERSON_EQUITY_MAX_BUYERS_PER_FIRM = 3

# Step 15G.1 controlled autonomous secondary-equity screen.  Defaults remain
# OFF; these are deliberately small engineering guardrails, not ownership
# distribution targets.
AUTONOMOUS_SECONDARY_EQUITY_ENABLED = False
# Engineering settlement remains the default. Book pricing is an explicit
# Step 15G.4 experiment and never activates implicitly.
AUTONOMOUS_SECONDARY_EQUITY_PRICE_MODE = "engineering_fixed"
SECONDARY_EQUITY_TRANSFER_PRICE = 10.0
SECONDARY_EQUITY_REVIEW_INTERVAL_WEEKS = 52
SECONDARY_EQUITY_MAX_AVAILABLE_CASH_FRACTION = 0.01
SECONDARY_EQUITY_MAX_BUYERS_PER_FIRM = 10
SECONDARY_EQUITY_MAX_HOUSEHOLD_EQUITY_SHARE_OF_ASSETS = 0.10
SECONDARY_EQUITY_MAX_PERSON_OWNERSHIP_FRACTION = 0.05
FIRM_INVENTORY_DEPRECIATION = 0.005
FIRM_EXPECTED_SALES_ALPHA = 0.10
FIRM_SALES_CONSTRAINT_WEIGHT = 0.35
FIRM_WAGE_FLOOR_SHARE = 0.70
FIRM_TARGET_WAGE_TO_SALES = 0.92

FOOD_INITIAL_PRICE = 1.0
FOOD_PRICE_ADJUSTMENT_RATE = 0.08
FOOD_TARGET_SALES_INCOME_RATIO = 1.0
FOOD_PRICE_CASH_WINDOW = 80
FOOD_MIN_PRICE = 0.5
FOOD_MAX_PRICE = 2.5
FOOD_PRICE_INVENTORY_GAP_WEIGHT = 0.06
FOOD_PRICE_COST_GROWTH_WEIGHT = 0.50
FOOD_PRICE_MAX_LOG_CHANGE = 0.08
FOOD_PRICE_EPSILON = 1e-9

# =====================================================
# Temporary Real-Food Market + Inventory-Backed Issuance
# =====================================================

# This is the current provisional closed-loop economy. The firm both produces
# food and temporarily issues money against excess inventory. Later versions
# should split money issuance into a bank/monetary authority.

FOOD_PRODUCTIVITY_PER_LABOR = 55.0
FOOD_TARGET_INVENTORY_DAYS = 15.0
FOOD_SHORTAGE_PRICE_WEIGHT = 1.0
FOOD_INVENTORY_PRICE_WEIGHT = 0.50
FOOD_PRODUCTION_ADJUSTMENT_RATE = 0.04
FOOD_MIN_PRODUCTION_SCALE = 0.60
FOOD_MAX_PRODUCTION_SCALE = 1.30
FOOD_INVENTORY_SPOILAGE_RATE = 0.01
FOOD_INVENTORY_SPOILAGE_RATE_PER_WEEK = FOOD_INVENTORY_SPOILAGE_RATE

FIRM_WAGE_PER_LABOR = 41.0
WAGE_MULTIPLIER = 1.0
FIRM_WAGE_SALES_FEEDBACK = 0.60
FIRM_WAGE_CASH_FEEDBACK = 6.0

# Kept for backward compatibility. The main model now routes inventory-backed
# money creation through the separate central_bank module.
TEMP_INVENTORY_BACKED_MONEY_ISSUANCE_ENABLED = False
TEMP_MINT_INVENTORY_RATE = 0.011
TEMP_MINT_COLLATERAL_RATIO = 1.00
TEMP_MINT_MIN_EXCESS_INVENTORY_DAYS = -5.0

# Step17.D active research institutions. Default OFF; enabled only by
# controlled branch overrides.
PAYG_PENSION_ENABLED = False
PAYG_CONTRIBUTION_RATE = 0.0
PAYG_PENSION_TARGET_MULTIPLIER = 0.0
INTERGENERATIONAL_WEALTH_TRANSFER_ENABLED = False
INTERGENERATIONAL_RESERVE_WEEKS = 13.0
INTERGENERATIONAL_DONOR_SURPLUS_SHARE = 0.25
INTERGENERATIONAL_RECIPIENT_TARGET_WEEKS = 1.0
INTERGENERATIONAL_TRANSFER_DIRECTION = "CHILD_TO_PARENT"
