"""Step 15 demographic burn-in / mature initial population design audit.

Design-only artifact generation.  It intentionally performs no simulation and
does not mutate demographic, economic, ownership, support, or inheritance
runtime state.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "test/output/step15_mature_population_burnin_design_audit"
PRIOR_AUDIT = ROOT / "test/output/step15_initial_genealogy_reachability_audit"
I12A = ROOT / "test/output/step15I12A_final_integrated_validation"
SEED = 42
POPULATION = 5000
OBSERVATION_WEEKS = 520


def num(value, default=math.nan):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "UNAVAILABLE"}])


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def observed_canonical_tendency():
    rows = read_csv(I12A / "step15_macro_panel.csv")
    if not rows:
        rows = read_csv(I12A / "macro_diagnostics.csv")
    populations = [num(row.get("population")) for row in rows if math.isfinite(num(row.get("population")))]
    births = read_csv(PRIOR_AUDIT / "runtime_birth_genealogy.csv")
    birth_rows = [row for row in births if row.get("record_type") != "SUMMARY"]
    final_population = populations[-1] if populations else math.nan
    return {
        "observed_source": str(I12A),
        "observed_weeks": len(rows),
        "opening_population": populations[0] if populations else POPULATION,
        "final_population": final_population,
        "population_change": final_population - populations[0] if populations else math.nan,
        "runtime_birth_count_from_prior_audit": len(birth_rows),
        "population_tendency_interpretation": "near_stationary_over_520_weeks" if populations and abs(final_population - populations[0]) / max(1.0, populations[0]) < 0.10 else "material_change_or_unavailable",
    }


def generate():
    tendency = observed_canonical_tendency()

    # The current model uses age 20 for labor/adult household eligibility and
    # an annual-equivalent marriage review.  These are lower bounds, not
    # calibrated forecasts of coverage.
    horizon = [
        {"milestone": "biological_adulthood", "threshold_years": 18, "threshold_weeks": 936, "theoretical_lower_bound_weeks": 936, "practical_note": "Birth at observation-origin week 0 reaches 18 after 936 weeks."},
        {"milestone": "labor_adulthood_current_contract", "threshold_years": 20, "threshold_weeks": 1040, "theoretical_lower_bound_weeks": 1040, "practical_note": "Current adult labor/settlement boundary is age 20."},
        {"milestone": "independent_social_household", "threshold_years": 20, "threshold_weeks": 1040, "theoretical_lower_bound_weeks": 1092, "practical_note": "Age 20 plus up to one annual marriage-market interval; actual timing is stochastic and relationship-dependent."},
        {"milestone": "elderly_parent_with_adult_child_older_parent_case", "threshold_years": 20, "threshold_weeks": 1040, "theoretical_lower_bound_weeks": 1040, "practical_note": "Conditional lower bound when a parent is already about 45 at birth; current paternal age rules permit older fathers."},
        {"milestone": "elderly_maternal_parent_with_adult_child", "threshold_years": 25, "threshold_weeks": 1300, "theoretical_lower_bound_weeks": 1300, "practical_note": "Conservative maternal bound using the current mother age upper fertility boundary of 40 and elderly threshold 65."},
    ]

    mature_horizon = [
        {"population_condition": "elderly_parent_plus_adult_child", "theoretical_lower_bound_weeks": 1040, "practical_screen_horizon_weeks": 1560, "mature_generation_horizon_weeks": 2080, "interpretation": "Non-zero is possible around 20-25 years; meaningful coverage needs multiple birth and survival cohorts."},
        {"population_condition": "elderly_parent_plus_adult_child_in_separate_household", "theoretical_lower_bound_weeks": 1092, "practical_screen_horizon_weeks": 1560, "mature_generation_horizon_weeks": 2080, "interpretation": "Requires adult eligibility plus marriage/independent household formation."},
        {"population_condition": "grandparent_plus_adult_child_plus_grandchild", "theoretical_lower_bound_weeks": 1092, "practical_screen_horizon_weeks": 1560, "mature_generation_horizon_weeks": 2080, "interpretation": "The grandchild need only be born; meaningful counts require subsequent cohort turnover."},
        {"population_condition": "inheritance_to_adult_heir", "theoretical_lower_bound_weeks": 1040, "practical_screen_horizon_weeks": 1560, "mature_generation_horizon_weeks": 2080, "interpretation": "Requires adult identifiable heirs and stochastic death; no exact deterministic date."},
        {"population_condition": "stable_multi_generation_age_and_household_distribution", "theoretical_lower_bound_weeks": 1560, "practical_screen_horizon_weeks": 2080, "mature_generation_horizon_weeks": 2600, "interpretation": "A 30-50 year demographic screen is more defensible than a 20-year single transition window."},
    ]

    target_contract = [
        {"target": "elderly_with_living_adult_children", "minimum_observable_threshold": ">0", "preferred_screen_threshold": "1% elderly parent Households", "not_calibrated": True, "reason": "Detectability threshold only; not a policy or fertility target."},
        {"target": "adult_children_in_separate_social_households", "minimum_observable_threshold": ">0", "preferred_screen_threshold": "1% elderly parent Households", "not_calibrated": True, "reason": "Confirms support can resolve across economic households."},
        {"target": "inheritance_reachability", "minimum_observable_threshold": ">0 identifiable-heir deaths", "preferred_screen_threshold": "multiple events across windows", "not_calibrated": True, "reason": "Event observability threshold only."},
        {"target": "multiple_active_birth_cohorts", "minimum_observable_threshold": ">=3 cohorts", "preferred_screen_threshold": ">=4 cohorts", "not_calibrated": True, "reason": "Supports cohort and age-structure analysis."},
        {"target": "stable_household_composition", "minimum_observable_threshold": "two consecutive diagnostic windows with no material drift", "preferred_screen_threshold": "predeclared stability test", "not_calibrated": True, "reason": "Must be tested, not assumed from elapsed time."},
    ]

    demographic_risks = [
        {"risk": "cash_wealth", "severity": "HIGH", "failure_mode": "Genealogy evolves but Household cash has no earned-history counterpart after reset.", "required_contract": "Explicit economic initialization/reset at observation boundary."},
        {"risk": "employment_and_settlement", "severity": "HIGH", "failure_mode": "Age-eligible Persons may enter the research window without authoritative employer/settlement state.", "required_contract": "Rebuild labor/settlement assignments from preserved Persons before observation."},
        {"risk": "firm_state", "severity": "HIGH", "failure_mode": "Demography-only population is paired with arbitrary Firm cash, inventory, debt and capital.", "required_contract": "Declare Firms as observation-window initial conditions, not historical burn-in balances."},
        {"risk": "inheritance_and_support", "severity": "HIGH", "failure_mode": "Transfers can be economically meaningful only if recipient and payer balances are initialized coherently.", "required_contract": "Preserve genealogy and explicitly initialize account balances before activation."},
        {"risk": "age_dependent_income_history", "severity": "MEDIUM", "failure_mode": "No historical wages, savings or ownership claims exist for the pre-observation period.", "required_contract": "Treat pre-window income history as unavailable unless separately modeled."},
    ]

    full_economy_risks = [
        {"risk": "firm_cash_depletion_or_leakage", "severity": "HIGH", "safe_for_20y": False, "safe_for_30y": False, "safe_for_40y": False, "reason": "Known Step15 Firm cash/profitability boundaries make decades of economic history an uncontrolled initial-condition generator."},
        {"risk": "food_profitability_and_demand_feedback", "severity": "HIGH", "safe_for_20y": False, "safe_for_30y": False, "safe_for_40y": False, "reason": "Long-run income/demand feedback may dominate demographic maturation."},
        {"risk": "debt_and_capital_drift", "severity": "HIGH", "safe_for_20y": False, "safe_for_30y": False, "safe_for_40y": False, "reason": "Debt, capital lifecycle and replacement behavior are not validated as stationary over multi-decade windows."},
        {"risk": "government_absence", "severity": "HIGH", "safe_for_20y": False, "safe_for_30y": False, "safe_for_40y": False, "reason": "No mature public-income/pension counterpart exists for interpreting long-run elderly households."},
        {"risk": "population_explosion_or_collapse", "severity": "MEDIUM", "safe_for_20y": "UNKNOWN", "safe_for_30y": "UNKNOWN", "safe_for_40y": "UNKNOWN", "reason": "The 520-week screen is near-stationary but cannot establish multi-decade demographic stationarity."},
    ]

    staged = [
        {"stage": "A_demographic_genealogy_warmup", "duration_guidance": "At least 1560 weeks for a screen; 2080-2600 weeks for mature multi-cohort research baseline.", "active_state": "age, births, deaths, marriage, fertility, household structure, genealogy, RNG state", "economic_state": "not authoritative research history", "exit_gate": "predeclared support/inheritance reachability and age/household stability checks"},
        {"stage": "B_economic_initialization", "duration_guidance": "Explicit observation-window setup, not an implied historical continuation.", "active_state": "Household cash, Firm cash, debt, inventory, capital, ownership according to declared experiment", "economic_state": "initialized/reset under documented contract", "exit_gate": "money, accounting, goods, assignment and settlement invariants pass"},
        {"stage": "C_short_economic_stabilization", "duration_guidance": "Short pre-observation stabilization only after economic initialization is validated.", "active_state": "accepted Step15 economy with selected mechanisms", "economic_state": "authoritative only after stabilization boundary", "exit_gate": "no material startup transient in declared metrics"},
        {"stage": "D_research_observation", "duration_guidance": "Predeclared observation weeks; burn-in excluded from statistics by default.", "active_state": "all preserved demographic and economic state", "economic_state": "authoritative research sample", "exit_gate": "analysis labels observation week 0 and absolute week separately"},
    ]

    reset = [
        {"state": "person_id", "reset_allowed": False, "preserve": True, "reason": "Identity is required for genealogy, inheritance and longitudinal joins."},
        {"state": "parent_ids_children_ids", "reset_allowed": False, "preserve": True, "reason": "Core mature-population genealogy."},
        {"state": "social_household_and_partner", "reset_allowed": False, "preserve": True, "reason": "Household support and marriage semantics depend on current membership."},
        {"state": "age_alive_status", "reset_allowed": False, "preserve": True, "reason": "Demographic state is the purpose of burn-in."},
        {"state": "household_cash", "reset_allowed": True, "preserve": False, "reason": "May be initialized at observation boundary, but must be labeled as new initial condition and reconciled."},
        {"state": "firm_cash_debt_inventory_capital", "reset_allowed": True, "preserve": False, "reason": "Demography-only burn-in cannot claim these as historical; use an explicit economic initialization contract."},
        {"state": "equity_ownership", "reset_allowed": True, "preserve": False, "reason": "Keep off or initialize explicitly; do not silently carry ownership from an absent economic history."},
        {"state": "rng_state", "reset_allowed": False, "preserve": True, "reason": "Preserve continuous reproducibility or save a deterministic post-warmup checkpoint."},
    ]

    checkpoint = [
        {"field": "person_id", "required": True, "source": "demographic runtime", "note": "Never regenerate identifiers at transition."},
        {"field": "parent_ids", "required": True, "source": "demographic runtime", "note": "Preserve reciprocal links."},
        {"field": "children_ids", "required": True, "source": "demographic runtime", "note": "Preserve reciprocal links."},
        {"field": "household_id_and_membership", "required": True, "source": "current Social Household", "note": "Resolve to valid Household at observation week 0."},
        {"field": "partner_id", "required": True, "source": "demographic runtime", "note": "Preserve spouse relationship if alive."},
        {"field": "age_weeks_and_alive", "required": True, "source": "demographic runtime", "note": "Canonical age storage is weeks."},
        {"field": "rng_state", "required": True, "source": "all dedicated/global RNG streams", "note": "Either continuous stream or deterministic checkpoint state; never silently reset."},
        {"field": "economic_state_metadata", "required": True, "source": "transition manifest", "note": "Record whether balances were preserved or initialized/reset."},
    ]

    support_target = [
        {"metric": "elderly_parent_with_identifiable_adult_child_share", "threshold": 0.0, "screen_levels": ">0%, >=1%, >=5%, >=10%", "interpretation": "Observability thresholds, not calibration targets."},
        {"metric": "adult_child_in_separate_social_household_share", "threshold": 0.0, "screen_levels": ">0%, >=1%, >=5%, >=10%", "interpretation": "Confirms payer/recipient household resolution."},
        {"metric": "positive_child_capacity_share", "threshold": 0.0, "screen_levels": ">0%, >=1%, >=5%", "interpretation": "Capacity depends on economic initialization and must not be inferred from genealogy alone."},
    ]
    inheritance_target = [
        {"metric": "deaths_with_identifiable_heirs", "threshold": ">0", "screen_levels": "first event, multiple events, events across cohorts", "interpretation": "Empirical reachability only."},
        {"metric": "Estate_to_heir_events", "threshold": ">0 when ownership is active", "screen_levels": "first event, repeated events", "interpretation": "Requires ownership activation and valid heirs; do not activate in this audit."},
        {"metric": "multi_generation_asset_transmission", "threshold": ">0", "screen_levels": "first observed transfer, repeated transfers", "interpretation": "Requires coherent economic balances and ownership state."},
    ]

    ownership = [
        {"phase": "demographic_burnin", "ownership_required": False, "recommendation": "Keep ownership off unless the research question explicitly studies ownership during burn-in.", "reason": "Demographic genealogy can mature without inventing historical equity claims."},
        {"phase": "economic_initialization", "ownership_required": False, "recommendation": "Initialize Legacy 100% / Person 0% or another explicitly declared state.", "reason": "Do not silently carry absent pre-window ownership history."},
        {"phase": "research_window", "ownership_required": "question-dependent", "recommendation": "Activate only in ownership/intergenerational-wealth experiments with separate controls.", "reason": "Ownership changes household financial capacity and inheritance interpretation."},
    ]
    observation = [
        {"period": "burnin", "week_label": "absolute_negative_or_prewindow", "included_in_research_statistics": False, "analysis_semantics": "Demographic warmup only; preserve absolute simulation week."},
        {"period": "stabilization", "week_label": "pre_observation", "included_in_research_statistics": False, "analysis_semantics": "Startup transition excluded unless explicitly requested."},
        {"period": "observation", "week_label": "observation_week_0", "included_in_research_statistics": True, "analysis_semantics": "Display both observation week and absolute week."},
    ]

    options = [
        {"option": "A_LONG_FULL_ECONOMY_BURNIN", "demographic_coherence": "HIGH", "economic_coherence": "LOW", "implementation_complexity": "MEDIUM", "runtime_cost": "VERY_HIGH", "calibration_burden": "VERY_HIGH", "intergenerational_suitability": "LOW_CURRENTLY", "assessment": "Not safe while known economic closure boundaries remain."},
        {"option": "B_DEMOGRAPHIC_ONLY_BURNIN", "demographic_coherence": "HIGH", "economic_coherence": "LOW_AFTER_RESET", "implementation_complexity": "MEDIUM", "runtime_cost": "HIGH", "calibration_burden": "MEDIUM", "intergenerational_suitability": "MEDIUM", "assessment": "Useful only with an explicit economic reset/initialization contract."},
        {"option": "C_STAGED_DEMOGRAPHIC_BURNIN_PLUS_ECONOMIC_INITIALIZATION", "demographic_coherence": "HIGH", "economic_coherence": "MEDIUM_HIGH_IF_EXPLICIT", "implementation_complexity": "HIGH", "runtime_cost": "HIGH", "calibration_burden": "MEDIUM_HIGH", "intergenerational_suitability": "HIGH", "assessment": "Safest next architecture: preserve mature demography, declare economic observation initial conditions, then stabilize briefly."},
        {"option": "D_RETROSPECTIVE_SYNTHETIC_GENEALOGY", "demographic_coherence": "LOW_MEDIUM", "economic_coherence": "MEDIUM", "implementation_complexity": "HIGH", "runtime_cost": "LOW", "calibration_burden": "VERY_HIGH", "intergenerational_suitability": "LOW", "assessment": "Should not be used merely to activate support quickly."},
        {"option": "E_DATA_DRIVEN_MATURE_INITIAL_POPULATION", "demographic_coherence": "POTENTIALLY_HIGHEST", "economic_coherence": "DATA_DEPENDENT", "implementation_complexity": "VERY_HIGH", "runtime_cost": "LOW_MEDIUM", "calibration_burden": "VERY_HIGH", "intergenerational_suitability": "HIGH", "assessment": "Longer-term ideal if validated data become available."},
    ]
    sequence = [
        {"order": 1, "stage": "freeze_current_canonical", "action": "Keep current N=5000 / seed42 / 520-week canonical as engineering baseline.", "do_now": True},
        {"order": 2, "stage": "define_mature_checkpoint_schema", "action": "Specify demographic state, genealogy, Household membership and RNG manifest without implementing burn-in.", "do_now": False},
        {"order": 3, "stage": "design_demographic_warmup_screen", "action": "Later run a separate demographic-only or staged prototype at predeclared 1560/2080/2600-week screens.", "do_now": False},
        {"order": 4, "stage": "economic_initialization_contract", "action": "Before any mature-population economics, decide which cash, Firm, debt, capital and ownership state is initialized/reset.", "do_now": False},
        {"order": 5, "stage": "short_stabilization_validation", "action": "Validate startup transients, conservation and analysis week labels after initialization.", "do_now": False},
        {"order": 6, "stage": "research_baseline_freeze", "action": "Create a separate mature-population research baseline; never overwrite the engineering canonical.", "do_now": False},
    ]

    flags = {
        "verdict": "C. STAGED_DEMOGRAPHIC_BURNIN_RECOMMENDED",
        "theoretical_adult_child_horizon_weeks": 1040,
        "theoretical_independent_household_horizon_weeks": 1092,
        "conservative_maternal_elderly_parent_horizon_weeks": 1300,
        "practical_meaningful_coverage_horizon_weeks": "2080-2600",
        "full_economy_burnin_safe_20_years": False,
        "full_economy_burnin_safe_30_years": False,
        "full_economy_burnin_safe_40_years": False,
        "demographic_only_burnin_requires_economic_reset_contract": True,
        "staged_burnin_preferred": True,
        "current_canonical_remains_engineering_baseline": True,
        "separate_mature_research_baseline_required": True,
        "long_behavioral_burnin_run": False,
        "demographics_modified": False,
        "family_support_activated": False,
        "inheritance_changed": False,
        "ownership_activated": False,
        "economic_parameters_changed": False,
        "new_rng_draws": 0,
        "source_canonical_population_tendency": tendency,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUTPUT / "mature_population_target_contract.csv", [
        {"condition": "nonzero_elderly_with_living_adult_children", "required": True, "minimum": ">0", "mature_target": ">=1% screen only"},
        {"condition": "adult_children_in_separate_social_households", "required": True, "minimum": ">0", "mature_target": ">=1% screen only"},
        {"condition": "inheritance_reachability", "required": True, "minimum": ">0 identifiable-heir events", "mature_target": "multiple events"},
        {"condition": "multiple_birth_cohorts", "required": True, "minimum": ">=3", "mature_target": ">=4"},
        {"condition": "stable_age_and_household_distribution", "required": True, "minimum": "predeclared stability test", "mature_target": "two consecutive windows"},
    ])
    write_rows(OUTPUT / "burnin_minimum_genealogy_horizon.csv", horizon)
    write_rows(OUTPUT / "mature_genealogy_horizon_screen.csv", mature_horizon)
    write_rows(OUTPUT / "demographic_only_burnin_risk.csv", demographic_risks)
    write_rows(OUTPUT / "full_economy_burnin_risk.csv", full_economy_risks)
    write_rows(OUTPUT / "staged_burnin_design.csv", staged)
    write_rows(OUTPUT / "economic_reset_contract.csv", reset)
    write_rows(OUTPUT / "genealogy_checkpoint_contract.csv", checkpoint)
    write_rows(OUTPUT / "support_reachability_target.csv", support_target)
    write_rows(OUTPUT / "inheritance_reachability_target.csv", inheritance_target)
    write_rows(OUTPUT / "ownership_burnin_interaction.csv", ownership)
    write_rows(OUTPUT / "observation_window_semantics.csv", observation)
    write_rows(OUTPUT / "burnin_design_option_comparison.csv", options)
    write_rows(OUTPUT / "implementation_sequence_recommendation.csv", sequence)
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = [
        "# Step 15 Demographic Burn-in / Mature Initial Population Design Audit",
        "",
        f"Verdict: **{flags['verdict']}**",
        "",
        "## Required horizons",
        "- Theoretical biological adulthood: 936 weeks (18 years).",
        "- Current labor/adult boundary: 1040 weeks (20 years).",
        "- Earliest plausible separate Social Household: approximately 1092 weeks, allowing the next annual marriage review.",
        "- Elderly parent plus adult child: 1040 weeks is a conditional older-parent lower bound; 1300 weeks is the conservative maternal lower bound under the current fertility age ceiling.",
        "- Practical meaningful multi-generation screen: approximately 2080-2600 weeks (40-50 years), subject to stochastic demographic stability tests.",
        "",
        "## Economic safety",
        "A 20-, 30- or 40-year full-economy burn-in is not safe as the next step. Known Firm cash/profitability feedback, debt/capital lifecycle drift and the absence of a mature public-income counterpart would make the resulting economic state an uncontrolled historical artifact.",
        "",
        "A demographic-only burn-in is more coherent for genealogy, but it cannot simply hand its mature population to the current economy. Household cash, employment, settlement, Firm balances, debt, inventory, capital and ownership would need an explicit observation-window initialization contract. Historical pre-window economic information must be labeled unavailable if it was not modeled.",
        "",
        "## Recommended architecture",
        "Use a staged demographic warm-up, explicit economic initialization, short economic stabilization, then a separate research observation window. Preserve Person IDs, parent/child links, Social Household membership, spouse links, age, alive status and RNG state. Keep the current N=5000 / seed42 / 520-week canonical frozen as the engineering baseline; create a separate mature-population research baseline later.",
        "",
        "No long burn-in was run. No demographic, family-support, inheritance, ownership, retirement, pension, Government, Firm or economic parameter was changed.",
    ]
    (OUTPUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(flags, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    generate()
