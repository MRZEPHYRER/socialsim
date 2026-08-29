# central_bank/config.py

# Provisional single-good monetary authority parameters.
# The central bank is separated from the producing firm. It can issue money by
# purchasing excess food inventory, provide in-kind poverty subsidies, and
# release its inventory to dampen price pressure.

CENTRAL_BANK_ENABLED = True

CENTRAL_BANK_INITIAL_MONEY_SUPPLY = 0.0

CENTRAL_BANK_INVENTORY_PURCHASE_ENABLED = True
CENTRAL_BANK_FOOD_PURCHASE_RATE = 0.008
CENTRAL_BANK_FOOD_PURCHASE_PRICE_HAIRCUT = 0.70
CENTRAL_BANK_FOOD_PURCHASE_EXCESS_INVENTORY_DAYS = -5.0
CENTRAL_BANK_INVENTORY_PURCHASE_MONEY_ISSUE_ENABLED = False

# Stop buffer-stock purchases once the firm already has enough cash.
CENTRAL_BANK_FIRM_CASH_PURCHASE_GATE_ENABLED = True
CENTRAL_BANK_TARGET_FIRM_CASH = 10_000_000.0
CENTRAL_BANK_FIRM_CASH_BUFFER = 250_000.0

CENTRAL_BANK_POVERTY_SUBSIDY_ENABLED = True
CENTRAL_BANK_FOOD_POVERTY_SUBSIDY_RATE = 0.015

CENTRAL_BANK_MARKET_RELEASE_ENABLED = True
CENTRAL_BANK_FOOD_RELEASE_RATE = 0.60
CENTRAL_BANK_FOOD_PRICE_ANCHOR = 1.50
CENTRAL_BANK_SHORTAGE_FIRST_RELEASE_ENABLED = True

# In the current simplified model the central bank also acts as the credit
# bank. A future commercial-bank layer can take over these functions.
CENTRAL_BANK_FIRM_CREDIT_ENABLED = True
CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = 150_000.0
CENTRAL_BANK_FIRM_CREDIT_WAGE_BUFFER = 1.0
CENTRAL_BANK_FIRM_LOAN_REPAYMENT_RATE = 0.35
CENTRAL_BANK_FIRM_WEEKLY_PRINCIPAL_REPAYMENT_TARGET_RATE = (
    CENTRAL_BANK_FIRM_LOAN_REPAYMENT_RATE
)
CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER = 150_000.0
# Runtime-only experiment switch. The accepted baseline is base_buffer;
# full_target is used only by Pre-Step13.3B.3 experiments.
CENTRAL_BANK_FIRM_REPAYMENT_RESERVE_MODE = "base_buffer"
CENTRAL_BANK_FIRM_LOAN_INTEREST_RATE = 0.0
# Step 13.4 behavioral interest policy.  This is an annual nominal research
# input; the weekly effective rate is derived by the simulation.
CENTRAL_BANK_FIRM_LOAN_INTEREST_ANNUAL_RATE = 0.0

# Step 13.8B: temporary current-interest relief is disabled in the baseline.
# When enabled, it changes only the current contractual interest obligation for
# an eligible restructuring episode; it does not alter principal, arrears,
# credit limits, payment priority, or money flows.
CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_ENABLED = False
CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_FRACTION = 0.0
CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_ELIGIBILITY_WEEKS = 26
CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_MAX_WEEKS = 26

# Step 13.8F: snapshot-only legacy-arrears term-out.  Disabled by default so
# the accepted baseline has no behavioral change.
CENTRAL_BANK_SNAPSHOT_LEGACY_ARREARS_TERMOUT_ENABLED = False

# Step 13.2 experimental credit-capacity treatment.  The default remains
# unconstrained so the accepted baseline is unchanged until a scenario opts in.
CENTRAL_BANK_CREDIT_CAPACITY_ENABLED = False
CENTRAL_BANK_CREDIT_CAPACITY_K_WEEKS = 11.59038588550643
CENTRAL_BANK_CREDIT_CAPACITY_SMOOTHING_WEEKS = 26
