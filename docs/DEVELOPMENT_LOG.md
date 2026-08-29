# SocialSim Development Log

This log records architectural decisions that should remain understandable
after the original implementation work has faded from view. Detailed numeric
evidence remains in the referenced `test/output/` artifacts; this document
records the reasoning, retained contracts, and deferred paths.

## Step 16: Performance Architecture

**Date / Step:** 2026-08, Step 16

**Objective.** Reduce the time and memory cost of simulation development and
repeated experiment validation without changing the economic model, RNG
semantics, or authoritative state.

**Problem / Motivation.** Full-model validation had become slow enough that
developer feedback was being delayed by repeated long runs. The architectural
questions were whether to remain Python-only, where a future compiled or GPU
path could fit, whether multi-core CPU execution was useful, and whether a
game engine had a role. The accepted direction is to keep Python as the
orchestration and research/simulation-development layer, compile only measured
numeric hot paths when their data boundary is ready, and keep a future game
client separate from the headless simulation. A game engine is not a
simulation accelerator.

### Baseline profiling

**Date / Step:** 2026-08, Step 16 baseline profiling  
**Objective:** Establish measured cost before changing architecture.  
**Investigation:** The accepted profile found high call counts around
household/person settlement lookup, labor eligibility and allocation, Firm
capacity, wage allocation, diagnostic persistence, and Household diagnostics.
Original canonical Household diagnostics alone were about 493.7 MB and 34.7 s
of write cost.  
**Validation:** For `N=5000`, 520 weeks, diagnostics OFF took about 98.908 s
and ON about 151.261 s, or roughly 52.93% persistence overhead. A mature
workload measured about 84.457 s OFF and 102.399 s ON. Peak memory was about
1783.7 MiB canonical and 1977.3 MiB mature.  
**Result / Verdict:** **A. PERFORMANCE_BASELINE_PROFILE_ACCEPTED**. Profiling
changed no economics and drew no new RNG values. See
`test/output/step16_performance_baseline_profile/`.

### P0.1: authoritative ID indexing

**Date / Step:** 2026-08, P0.1  
**Objective:** Make Person, Household, and Firm identity lookup explicit and
authoritative through `person_by_id`, `household_by_id`, and `firm_by_id`.
**Implementation:** Runtime indexes were retained as a stable lookup contract.
**Validation:** Representative microbenchmarks improved by about 752x for
Person, 236x for Household, and 7.3x for Firm relative to representative
linear scans. The global runtime did not improve consistently because the
legacy Person and Household paths were already substantially dictionary-backed.
One canonical OFF measurement was contaminated by unrelated system load and is
not clean causal evidence.  
**Result / Verdict:** **C. ID_INDEXING_CORRECT_BUT_NOT_PERFORMANCE_RELEVANT**.
The indexes remain permanently enabled because the contract is correct and
future-proof. See `test/output/step16_p0_1_id_indexing/`.

**Lessons Learned.** A high call count does not prove that a linear lookup is
the dominant cost. Performance conclusions require repeated, controlled
measurements rather than one wall-clock run. Future checks use idle-machine
conditions, short deterministic gates, medians where useful, and early stop.
Developer validation time is itself an optimization target.

### P0.2: observability modes

**Date / Step:** 2026-08, P0.2  
**Objective:** Reduce diagnostic memory and output cost while preserving the
fields needed for research and accounting review.  
**Implementation:** Added the explicit `FULL_DIAGNOSTIC`, `RESEARCH_FAST`, and
`DEBUG_DETAILED` modes. `RESEARCH_FAST` retains weekly macro, Firm,
accounting, money/goods reconciliation, and assignment diagnostics; Household
and Person micro histories are sampled at initial, every 13 weeks, and final
snapshots. Values not computed between snapshots are `NaN`, never fabricated
zeros.  
**Validation:** Controlled medians were 79.247 s OFF, 117.718 s FULL, and
112.703 s RESEARCH_FAST. FAST was about 4.45% faster than FULL, while peak RSS
fell from about 1.65 GiB to 0.34 GiB and output size from about 43.37 MB to
37.31 MB.  
**Result / Verdict:** **C. OBSERVABILITY_OPTIMIZATION_CORRECT_BUT_MODEST**.
The memory reduction made independent branch experiments more practical. A
short-lived transition between sparse snapshots may not be reconstructable;
use FULL or DEBUG_DETAILED for forensic timing. See
`test/output/step16_p0_2_observability_optimization/`.

### Staged validation policy

The project moved away from automatically running a full 520-week model for
every small optimization. The accepted development gates are:

1. Microbenchmark.
2. Short 26-week deterministic parity check.
3. Targeted lifecycle or behavior fixtures.
4. 104-week performance signal.
5. 520-week confirmation only after a meaningful signal exists.

This ordering maximizes engineering information per unit developer time.

### P1.1: same-week Firm labor-capacity cache

**Date / Step:** 2026-08, P1.1  
**Objective:** Reuse same-week Firm effective-labor/capacity calculations.
**Implementation:** The cache is valid only for the current simulation step and
the full ordered `employee_ids` tuple. Two invalidation problems were found and
fixed: equal-length roster reordering could otherwise reuse stale state, and a
step-0 initialization cache could survive the post-aging week-0 refresh
boundary.  
**Validation:** 26-week diagnostics and Firm diagnostics had exact hash parity,
with no RNG or economic changes. The 104-week median improved from 19.154 s to
16.3765 s, about 14.5%. A 520-week RESEARCH_FAST validation passed with zero
invariant violations.  
**Result / Verdict:** **A. WEEKLY_LABOR_CACHE_ACCEPTED_MATERIAL_GAIN**.
Retained permanently. See `test/output/step16_p1_1_weekly_labor_cache/`.

### P1.2: heap labor-allocation attempt

**Date / Step:** 2026-08, P1.2  
**Objective:** Replace repeated minimum-capacity Firm scans with a deterministic
heap. Current complexity is approximately `O(N) + O(U*F) + O(U*K)`; the
candidate targeted the `U*F` portion.  
**Implementation:** The temporary heap preserved Firm order, tie behavior, and
assignments in isolation.  
**Validation:** The local microbenchmark was about 1.576x faster and the
candidate Firm checks fell from about 18,255 to zero. In the whole model,
however, the 104-week median regressed from 16.3765 s to 18.0345 s, about
10.13%, because with roughly five Firms heap maintenance and Python object
overhead cost more than the saved scans.  
**Result / Verdict:** **D. LABOR_ALLOCATION_OPTIMIZATION_REGRESSION**.
The candidate was fully rolled back. See
`test/output/step16_p1_2_labor_allocation_optimization/`.

**Lessons Learned.** Better asymptotic complexity is not automatically faster
at the actual problem size. Whole-model measurements outrank isolated
microbenchmarks.

### P1.3: array and compiled-kernel feasibility

**Date / Step:** 2026-08, P1.3  
**Objective:** Determine whether labor hot paths are ready for NumPy, Numba,
GPU, or another compiled representation.  
**Investigation / Validation:** Labor allocation represented about 15.724% in
the current 26-week sample and about 17.428% in the accepted P1.1 profile. A
five-Firm repeated-selection proxy was about 18% of isolated allocation time,
but P1.2 showed that a heap replacement regresses the whole model. NumPy
eligibility was about 1.44x faster in an isolated `N=5000` prototype; age
productivity vectorization was about 14x faster with maximum numerical error
near `1e-16`. Object extraction, settlement-derived state, object write-back,
and the mutable authoritative object graph limit end-to-end gains. An
Amdahl-style aggressive labor compilation estimate was only about 1.10x whole
model speedup under current assumptions.  
**Result / Verdict:** **D. DATA_LAYOUT_RESTRUCTURE_REQUIRED_BEFORE_COMPILED_ACCELERATION**.
No production behavior changed and no compiled kernel was introduced. See
`test/output/step16_p1_3_labor_compiled_feasibility/`.

### Platform and language policy

Python remains the primary implementation and orchestration layer. GPU use is
not justified for one mutable World at approximately `N=5000-300`; revisit it
when retained numeric arrays, much larger populations, or many-world numeric
batches make the workload suitable. Reconsider Numba after a reusable numeric
side-car exists. Reconsider C++ or Rust only after simulation interfaces and
behavior stabilize and profiling shows interpreter overhead dominates. A game
engine should support a separate interactive client, not accelerate the
simulation core.

### P2.1: independent branch multiprocessing

**Date / Step:** 2026-08, P2.1  
**Objective:** Accelerate independent Control/Treatment worlds while preserving
single-World weekly semantics.  
**Implementation:** `experiment_runner.py` provides serializable `BranchTask`,
Windows-safe `spawn` process isolation, a read-only common checkpoint,
independent World and RNG state per worker, unique output directories, and
explicit failure capture. No single mutable World is parallelized.  
**Validation:** In the representative four-branch, 52-week test, serial time
was 25.670 s and two-worker time 14.808 s: 1.734x speedup and 0.867 parallel
efficiency. The 13-week signal was about 1.687x and the 104-week signal about
1.753x. Serial and parallel state fingerprints and RNG fingerprints matched;
failure isolation passed. The current machine policy keeps the worker limit at
two; four-worker stress was intentionally not run because of memory risk.  
**Result / Verdict:** **B. BRANCH_MULTIPROCESSING_ACCEPTED_MATERIAL_SPEEDUP**.
See `test/output/step16_p2_1_branch_multiprocessing/`.

### P2.2: standard experiment workflow

**Date / Step:** 2026-08, P2.2  
**Objective:** Turn independent branch execution into a reusable experiment
contract rather than ad hoc scripts.  
**Implementation:** `experiment_workflow.py` adds serializable
`ExperimentSpec` and `ExperimentBranch` definitions, deterministic conversion
to `BranchTask`, common and branch override merging, namespace/key validation
before worker launch, explicit Control support, unique output directories,
checkpoint and task fingerprints, a portable manifest, generic
`experiment_summary.csv`, effective-config reporting, failure manifests, and
safe resume/skip. JSON input accepts a UTF-8 BOM for Windows compatibility.
`experiment_runner.py` exposes the standard CLI dispatch.
  
**Validation:** The short integration used four behavior-neutral branches, a
common checkpoint, 13 weeks, and two workers. The first invocation completed
4/4. The second invocation skipped 4/4 only after matching the task
fingerprint, checkpoint fingerprint, and completed branch result. Focused tests
passed 13/13. No 520-week multibranch campaign or four-worker stress run was
performed.  
**Result / Verdict:** **A. STEP16_PERFORMANCE_ARCHITECTURE_ACCEPTED**.
Standard invocation:

```powershell
python experiment_runner.py --spec test/examples/step16_p2_2_example.json
```

Evidence is in `test/output/step16_p2_2_experiment_workflow_closeout/`.

## Final Step 16 Architecture

| Status | Retained architecture |
| --- | --- |
| Retained | Authoritative Person/Household/Firm ID indexes |
| Retained | `FULL_DIAGNOSTIC`, `RESEARCH_FAST`, `DEBUG_DETAILED` |
| Retained | Same-week Firm labor-capacity cache with exact invalidation |
| Retained | Independent branch multiprocessing with Windows `spawn` |
| Retained | `ExperimentSpec` workflow, task/checkpoint fingerprints, safe resume |
| Retained | Conservative memory-aware worker policy; two workers on the current machine |
| Rejected / rolled back | Heap-based labor-allocation optimization |
| Deferred | Numeric side-car, Numba, Cython, C++, Rust, GPU |
| Deferred | Single-World parallelization and game-engine integration |

## Engineering Lessons

1. Profile before optimizing.
2. High call count does not imply expensive linear lookup.
3. Microbenchmark gains must survive whole-model measurement.
4. Asymptotically better algorithms can regress at small Firm counts.
5. Cache invalidation must follow simulation timing exactly.
6. Observability can dominate memory even when its runtime gain is modest.
7. Development-validation cost should be optimized deliberately.
8. Parallel independent Worlds are safer and more valuable than parallelizing
   one mutable World.
9. GPU work should follow data architecture, not precede it.
10. Failed optimization results belong in the record so they are not repeated.

## Evidence Index

The log is an architectural narrative, not a replacement for acceptance data:

- `test/output/step16_performance_baseline_profile/`
- `test/output/step16_p0_1_id_indexing/`
- `test/output/step16_p0_2_observability_optimization/`
- `test/output/step16_p1_1_weekly_labor_cache/`
- `test/output/step16_p1_2_labor_allocation_optimization/`
- `test/output/step16_p1_3_labor_compiled_feasibility/`
- `test/output/step16_p2_1_branch_multiprocessing/`
- `test/output/step16_p2_2_experiment_workflow_closeout/`

### Step 17.A - Household Liquidity and Family Transfer Capacity

**Goal**  Measure relative household liquidity and the theoretical capacity of authoritative parent/child household links before designing any transfer policy.

**Result**

- High liquidity is concentrated mainly in working-age household groups.
- Global pooled cash is materially larger than the capacity available through observed family edges; adult-child-to-parent capacity is the stronger direction.
- Initial-population retrospective genealogy remains absent, so network capacity is a lower bound.

**Decision**  Keep this stage shadow-only; no Ledger transfer or policy rule is activated.

**Artifacts**  `test/output/step17_a_household_transfer_capacity/`
### Step 17.B - PAYG Pension Fiscal Capacity

**Goal**  Measure static pay-as-you-go pension capacity from realized runtime-paid wages and Person-level elderly eligibility.

**Result**

- The current one-week contributable wage base is 159,294.56 across 5,107 contributors; 1,109 Persons are pension-eligible.
- A 0.25x minimum-consumption target requires a 6.55% break-even contribution rate; excluding 65+ workers removes 3.14% of the current base.
- 331 of 618 elderly Social Households lack a >=13-week wealthy genealogy connection in the Step17.A lower-bound network.

**Decision**  Keep PAYG shadow-only; the current fiscal classification is `PAYG_REQUIRES_HIGH_CONTRIBUTION_RATE`, with no pension, contribution, retirement, or Ledger behavior activated.

**Artifacts**  `test/output/step17_b_payg_pension_capacity/`

### Step 17.C - Private Family Transfer and PAYG Combined Design Frontier

**Goal**  Identify a compact shadow frontier combining genealogy-dependent private transfers with fiscally balanced PAYG pension support.

**Result**

- Exact balanced PAYG candidates are 3% -> 0.114521x, 5% -> 0.190868x, and 6.5490% -> 0.25x; no contribution or pension was executed.
- R13 child-to-parent transfer capacity is 17,945.46 at the 25% excess-cash screen versus 3,654.96 parent-to-child; the combined screen reduces private flow as pension support reaches overlapping recipients.
- The observed genealogy gap remains institutionally reachable through Person-level PAYG eligibility, while private transfer capacity remains network constrained.

**Decision**  Keep the future design asymmetric child-to-parent first, with one private-only, one balanced PAYG-only, and one combined active-smoke candidate; all remain non-canonical research contracts.

**Artifacts**  `test/output/step17_c_combined_policy_frontier/`

### Step 17.D - Active PAYG and Intergenerational Wealth Transfer Smoke

**Goal**  Activate default-off Ledger-settled PAYG pension and wealth-based adult-child to parent transfers as controlled research institutions.

**Result**

- Gate 1 fixtures, 13-week four-branch smoke, 52-week four-branch smoke, and disabled-control parity passed.
- PAYG used actual realized positive wages, a cash-constrained SocialInsuranceFund, and no government or central-bank backstop; the 52-week PAYG branch was underfunded while the combined branch retained a positive Fund balance.
- Wealth transfers respected the 13-week donor reserve and one-week recipient target; monetary, accounting, goods, assignment, and RNG checks passed.

**Decision**  Keep both institutions research-only and default OFF; do not select a canonical policy or proceed to Step17.E.

**Artifacts**  `test/output/step17_d_active_institutional_smoke/`

## Step17.D.1 �� Transfer absorption and liquidity persistence audit
- Goal: diagnose active PAYG/private-transfer cash absorption using existing Step17.D artifacts only.
- Key diagnosis: recipient-level retention is unavailable because event-window cash/consumption history was not persisted.
- Donor exhaustion is not supported; eligible donor pools expand and settlement reserve checks are zero.
- PAYG cash collection is constrained, especially in PAYG_ONLY; COMBINED has a smaller shortfall and no backstop.
- Decision: E. MULTIPLE_DYNAMIC_MECHANISMS_EXPLAIN_THE_GAP; no policy parameters or runtime behavior changed.
- Artifacts: test/output/step17_d1_transfer_absorption_audit/

## Step17.E - Recipient liquidity persistence instrumentation
- Added compact recipient event instrumentation at the accepted weekly policy boundary.
- Captured pension-only, private-only, and both-receipt Household-weeks with explicit follow-up availability.
- Added threshold persistence, elderly/genealogy coverage, consumption absorption, and policy-role summaries.
- Gate 1 four-week disabled-control parity, Gate 2 13-week smoke, and Gate 3 52-week smoke passed.
- PAYG 3% collection remained operationally executable; private donor reserve and recipient-target violations were zero.
- Four-week descriptive liquidity-buffer persistence was not established; negative Household cash remained an observed runtime signal.
- Decision: D, both active branches support income but neither is judged a liquidity-buffer mechanism.
- No economic behavior, policy parameter, RNG, canonical enablement, or Step17.F work was performed.
- Artifacts: `test/output/step17_e_recipient_persistence_smoke/`
## Step17.E.1 - PAYG / Consumption Cash Safety Audit
- Goal: locate the first negative Household cash transition in PAYG_3_ONLY and COMBINED_3.
- Two targeted 52-week replays found the first negative week at week 15; exact Household IDs and phase cash trace are persisted.
- PAYG contribution collection did not push nonnegative cash below zero and never exceeded available cash.
- Pension cash was non-decreasing; combined private-transfer donor reserve checks remained valid.
- The later Food consumption debit crossed cash below zero because its budget used pre-policy starting cash and gross income.
- Step17.E distributions remain valid with an explicit negative-cash caveat and require post-fix rerun for policy evaluation.
- Recommended minimum contract: re-constrain consumption after PAYG; no fix was implemented in E.1.
- No PAYG/private-support parameter, benefit, ordering, RNG, canonical enablement, or Step17.F work was performed.
- Artifacts: `test/output/step17_e1_payg_consumption_cash_safety/`## Step17.E.2 - Household Consumption Current-Cash Safety Fix
- Root cause: canonical Food planning used pre-policy starting cash/gross income while mandatory PAYG cash had already settled.
- Fix: preserve the behavioral budget, then cap Food settlement at max(current Household cash, 0) and normalize only tiny negative floating residuals.
- Realized Food quantity, consumption, saving, and cash-safety diagnostics now use the same cash-constrained amount.
- Fixtures, 26-week Gate 2, and 52-week four-branch Gate 3 passed; material negative cash fell to zero.
- Control diagnostics-on/off state and RNG parity passed; PAYG collection and Fund stability remained valid.
- Corrected policy outputs changed materially for PAYG/Combined through legitimate consumption suppression; no policy parameters were changed.
- Decision: accept the cash-safety fix, treat corrected Step17 results as authoritative, and stop before Step17.F.
- Artifacts: `test/output/step17_e2_consumption_cash_safety_fix/`
## Step17.F - Household Liquidity Distribution Explorer & Dynamic Low-Tail Audit
- Added opt-in authoritative weekly Social-Household liquidity snapshots at the post-consumption settlement point; default runtime remains unchanged.
- Added fixed low-tail buckets, distribution metrics, longitudinal spells, persistence classes, and matched-household transition tables for CONTROL, PRIVATE_ONLY, PAYG_3_ONLY, and COMBINED_3.
- Embedded an explicit-directory-only Explorer in the existing Household GUI V2 tab with week, branch, group, variable, histogram, ECDF, bucket, inspector, and same-week Control-delta controls.
- The 52-week RESEARCH_FAST run covered 499,654 Social Household-weeks with no duplicate keys, no negative closing cash, and diagnostic ON/OFF state and RNG parity.
- Verdict: LOW_LIQUIDITY_IS_PRIMARILY_PERSISTENT_TRAP; policy branches do not eliminate the persistent low-liquidity tail. No Step17.G work started.
- Artifacts: test/output/step17_f_household_liquidity_explorer/
## Step17.G - Long-Horizon Economic Warm-up & Low-Liquidity Convergence Audit
- Added opt-in compact weekly liquidity diagnostics and authoritative 13-week Household snapshots with append-only checkpoint-safe storage.
- Completed one mature-genealogy CONTROL run: 520-week pilot plus 4,480-week checkpoint continuation, totaling 5,000 weekly observations.
- Added online low-tail transition, anchor-cohort, lifecycle, demographic, distribution-shape, convergence-window, and GUI validation outputs.
- Extended GUI V2 explicit-directory loading to Step17.G with real snapshot-week navigation and no fabricated weekly micro-history.
- Verdict: WARMUP_REDUCES_LOW_LIQUIDITY_BUT_POSITIVE_STRUCTURAL_PLATEAU_REMAINS; artifacts: test/output/step17_g_long_horizon_liquidity_warmup/
## Step17.H - Structural Low-Tail Household Root-Cause Decomposition
- Reused the accepted Step17.G CONTROL artifacts without rerunning the simulation.
- Primary mature window was weeks 3960-4999; Household features were observed at the authoritative 13-week cadence.
- Classified persistent deep-low identities using at least four observations and a 75% deep-low share; disappearance was not treated as recovery.
- Income/need, margin, employment proxy, size, elderly, matched-household, descriptive logistic, polarization, and shadow-capacity outputs were generated.
- Adult/child counts, exact earner counts, age-productivity, opening cash, household formation identity, and full cash-flow outflows remain unavailable and are explicitly marked unavailable.
- Verdict: B, structural low tail is primarily low income relative to need in the observable data.
- No policy, simulation, RNG, or economic behavior changes were made; Step17.I was not started.
- Artifacts: `test/output/step17_h_structural_low_tail_decomposition/`
## Step17.I - Low-Income Mechanism Attribution Audit
- Ran a 26-week passive diagnostic replay from the accepted mature population checkpoint rooted at absolute week 2600; no 5000-week run was performed.
- Sampled 32 DEEP_LOW targets and 6 deterministic non-low controls using same-week structure-first matching with nearest-need fallback.
- Persisted compact household-week member aggregates, labor eligibility, age-productivity, income-source, need, consumption, lifecycle, event-window, and descriptive model outputs.
- Low-tail households generally retained eligible workers, but showed materially lower age productivity and wage per earner than controls.
- Elderly income gaps overlapped strongly; dependent burden and formation evidence were limited to the observed short window.
- Per-member realized wage allocation and unobserved formation history remain explicitly unavailable.
- Verdict: F, low tail has multiple overlapping income mechanisms; labels are descriptive, not causal.
- Observability ON/OFF parity passed for all 26 weeks; no policy, RNG, retirement, wage, or economic behavior change was made.
- Artifacts: `test/output/step17_i_low_income_mechanism_attribution/`
## Step17.J - Old-Age Income & Retirement Joint Contract Design Audit
- Performed a 13-week shadow observation from the accepted mature population checkpoint rooted at absolute week 2600; no long run or policy activation.
- Audited 7,298 authoritative Person rows, including 1,116 persons aged 65+ and runtime payroll provenance.
- Retirement ages 65/67/70 were evaluated algebraically; no labor eligibility or wage rule changed.
- Current old-age productivity and wages are materially low, so a wage-replacement-only pension is inadequate for minimum-need coverage.
- Generated need-based, wage-replacement, hybrid, employee/employer contribution, and mixed-funding fiscal frontiers as shadow obligations.
- Recommended interface candidate: Person-level need-based entitlement, SocialInsuranceFund, active Household settlement, and work allowed after eligibility.
- Numeric ages, benefit multipliers, employee/employer rates, and public funding remain research-configurable; public closure requires an explicit fiscal source.
- Verdict A: simple old-age income plus retirement contract is ready for an active smoke; no pension, retirement, employer debit, RNG, or Step17.K change.
- Artifacts: `test/output/step17_j_old_age_retirement_design/`
## Step17.K - Active Old-Age Pension Institution Smoke
- Reconciled Step17.I low-tail denominators: 3 latest-status nonelderly identities versus 12 identities ever observed nonelderly; nine changed elderly status over the household-week panel.
- Passed deterministic pension fixtures for age eligibility, work-allowed status, aggregation, cash-constrained contribution, proportional underfunding, disabled zero flow, and settlement-only semantics.
- Ran CONTROL and PENSION_65_EMP3_NEED10_WORK_ALLOWED for 26 weeks from the mature population checkpoint; first 13 weeks served as Gate 2.
- Activated only 65+ eligibility, 3% employee contribution, 10% Person need-based benefit, and zero employer/public funding; no hard retirement or age-productivity change.
- Employee collection and pension funding were fully funded in the short run; the SocialInsuranceFund remained solvent with a positive buffer.
- Elderly Household income/need improved dynamically; contributor burden, consumption, Food, Firm, and overlap panels were persisted.
- Control replay and RNG parity passed; accounting, money, goods, assignment, and Fund stock-flow reconciliations passed.
- Interface boundaries are candidates for freeze; numeric ages, benefit and funding parameters remain research-configurable. Step17.L was not started.
- Artifacts: test/output/step17_k_active_old_age_pension_smoke/

## Step17.L - Pre-Retirement Pension Contributor Contract
- Corrected the active PAYG employee-contributor boundary to employed Person + positive authoritative wage + age below the pension eligibility age.
- Persons aged 65+ may continue working and receiving pension, but are excluded from employee contributions; nonworkers remain excluded.
- Passed deterministic contributor fixtures A-J and reconciled the Step17.I low-tail denominator distinction between latest-status and time-varying household identities.
- Ran CONTROL, OLD_CONTRIBUTOR_CONTRACT, and PRE65_CONTRIBUTOR_ONLY for 26 weeks from the accepted mature population checkpoint.
- PRE65 excluded 1,109 elderly wage earners, reduced employee contribution flow by 4,272.476215 over the run, and retained full pension funding with zero shortfall.
- Same-Person contributor/recipient overlap, nonworker contributions, accounting, money, goods, assignment, and Fund stock-flow checks passed; no new RNG draws.
- No retirement, benefit, wage, labor, employer/public funding, or Step13 behavior changed.
- Verdict A: PRE_RETIREMENT_CONTRIBUTOR_CONTRACT_ACCEPTED. Artifacts: test/output/step17_l_pre_retirement_contributor_contract/.


## Step17.M - Active Retirement + Pension Joint Smoke
- Added a single deterministic Person-level retirement state: at the beginning of the weekly labor-state update, after age growth and before labor allocation/payroll, age >= 65 becomes retired.
- Retired Persons are removed from Firm rosters, firm_id is cleared, and the shared labor_formally_eligible interface returns false; age-productivity and wage rules were unchanged.
- Preserved the Step17.L contributor contract, 3% pre-retirement employee contribution, 10% need-based pension, zero employer/public funding, and pension eligibility after retirement.
- Ran CONTROL, WORK_ALLOWED_PENSION, and HARD_RETIREMENT_65_PENSION for 13 Gate-2 and 26 Gate-3 weeks from the accepted mature population checkpoint.
- Hard retirement generated 1,139 events, with 1,109 retired Persons at the final observation and 149.543407 effective labor units removed.
- Pension funding remained complete with zero shortfall; accounting, money, goods, assignment, Fund stock-flow, and control/RNG parity checks passed.
- Elderly income adequacy declined under the unchanged 0.10 benefit, so the smoke supports institutional retirement semantics but not current benefit adequacy.
- Verdict B: RETIREMENT_IS_INSTITUTIONALLY_VALID_BUT_CURRENT_10_PERCENT_BENEFIT_IS_TOO_LOW. Artifacts: test/output/step17_m_active_retirement_pension_joint_smoke/.

## Step17.N - Employer Pension Contribution Affordability & Joint Funding Frontier
- Added a default-off payroll-based Firm employer contribution interface into the existing SocialInsuranceFund settlement boundary.
- Employer scheduling uses actual Firm payroll; actual payment is cash-safe and capped by current Firm cash, with shortfall diagnostic-only and no employer arrears or Step13 credit.
- Passed deterministic zero-payroll, 1%, 2%, sufficient-cash, insufficient-cash, transfer-conservation, and rate-zero fixtures.
- Ran HARD_RETIREMENT_65 with employee 3%, benefit 0.10, and employer rates 0%, 1%, 2% for 13 and 26 weeks from the accepted mature checkpoint.
- Both 1% and 2% employer collections were complete; Firm cash remained nonnegative and no new borrowing occurred.
- Fund stock-flow, accounting, money, goods, assignment, and zero-RNG checks passed; Household cash was not directly debited by employer contributions.
- Shadow benefit frontier estimated maximum fully funded multipliers of 0.120971, 0.159861, and 0.198751 for employer rates 0%, 1%, and 2%; work-allowed benchmark requires about 0.230048.
- Verdict D: EMPLOYER_CONTRIBUTION_INTERFACE_VALID_BUT_FUNDING_STILL_INSUFFICIENT. Artifacts: test/output/step17_n_employer_pension_affordability/.
## Step17.O - Public Pension Fiscal Closure & Benefit-Target Boundary Audit
- Audited accepted Step17.N/M artifacts and existing public-accounting diagnostics only; no World.step(), public funding, tax, benefit, or Step17.P behavior was activated.
- Existing public cash is observable, but receipts are event-driven estate/no-heir and inventory-release flows; no mature recurring Government/PublicBudget revenue source exists.
- At employee 3% plus employer 2%, the 0.20 shadow residual is 1,356.969163, while 0.23 requires 33,958.719163, about 1,306.104583 per reference week and 13.586382% of obligation.
- The 0.20 near-balance is a small payroll/eligibility/timing aggregation edge case; 0.23 shows a continuous recurring co-financing requirement in the 26-week reference window.
- The descriptive WORK_ALLOWED_PENSION replacement benchmark remains 0.230048 for mean income/need 0.206896; it is not a welfare target or frozen benefit.
- Recommended freeze is only the future cash-constrained Government/PublicBudget -> SocialInsuranceFund transfer interface; rates, benefit, taxes, and retirement age remain unfrozen.
- Verdict C: PUBLIC_FUNDING_INTERFACE_IS_REQUIRED_BUT_REVENUE_SOURCE_NOT_YET_MATURE. Artifacts: test/output/step17_o_public_pension_fiscal_closure/.
## Step17.P - Public Revenue Base & Minimal Government Budget Foundation
- Added a default-off GovernmentPublicBudget as a separate real Ledger cash holder.
- Added cash-constrained Firm tax collection after weekly sales and operating results.
- Selected FIRM_SALES_TAX for the initial short smoke; profit, payroll, and household-income bases remain shadow-only.
- Tax expense is separated from Firm cash flow and legacy public wealth; no price pass-through or household tax was added.
- Government-to-SocialInsuranceFund transfer, tax debt, Step13 investment funding, and benefit changes remain disabled.
- Ran deterministic fixtures plus 13/26-week control and active sales-tax branches from the accepted mature checkpoint.
- Accounting, money, goods, assignment, government stock-flow, and zero-RNG checks passed.
- Verdict A: MINIMAL_PUBLIC_FISCAL_ARCHITECTURE_ACCEPTED. Artifacts: test/output/step17_p_public_revenue_foundation/.

## Step17.Q - Active Public-Fiscal Pension Closure Smoke
- Activated the default-off GovernmentPublicBudget to SocialInsuranceFund residual transfer for a controlled fiscal-pension smoke.
- Preserved 3% pre-retirement employee contributions, 2% employer contributions, hard retirement at 65, and sales tax 0.006353357774149682.
- Ran the required benefit-0.10 no-public, benefit-0.23 no-public, and benefit-0.23 public-sales-tax branches for 13/26 weeks.
- Government transfers only the current residual after employee and employer inflows and never uses debt, Central Bank money, legacy public wealth, or Fund overdraft.
- Public closure was partial: 26,498.927283 tax revenue and transfer, with 6,766.566169 residual pension shortfall.
- Government and Fund stock-flow, accounting, money, goods, assignment, cash-safety, and zero-RNG checks passed.
- Investment was not identifiable in this short window; Food tax concentration remains a short-window result.
- Verdict B: ACTIVE_PUBLIC_PENSION_INTERFACE_ACCEPTED_BUT_CURRENT_TAX_RATE_ONLY_PARTIALLY_CLOSES_023. Artifacts: test/output/step17_q_active_public_pension_closure/.

## Step17.R - Progressive Personal & Corporate Profit Tax Architecture Audit
- Audited Person-level progressive tax and positive corporate-profit tax as shadow-only general-revenue candidates; social insurance remains separate.
- Reused accepted Q/P and Step15 artifacts plus a permitted 13-week no-new-tax payroll-provenance observation; no active tax or economic behavior changed.
- Verified marginal bracket, tax-year, positive-profit, employer-contribution deductibility, and no-new-money fixture semantics.
- Recommended PROGRESSIVE_PERSONAL_PLUS_FLAT_CORPORATE_PROFIT_RECOMMENDED; rates, exemptions, thresholds, and corporate progressivity remain unfrozen.
- Artifacts: test/output/step17_r_tax_architecture_audit/.

## Step17.S - Active Progressive Personal + Flat Corporate Tax Foundation
- Activated the accepted Person-level progressive wage-tax and Firm-level flat positive taxable-profit interfaces behind default-off flags.
- Added tax-year Person state, cumulative withholding, cash-constrained Social Household settlement, year-end diagnostic reconciliation, and separate GovernmentPublicBudget tax records.
- Corporate taxable profit uses pre-tax operating profit less employer social contribution exactly once; no tax debt, pension tax, dividend tax, inheritance tax, or investment funding was added.
- Ran Gate 0 denominator/exemption/profit-bridge checks, deterministic personal/corporate fixtures, 13/26-week gates, and 52-week B/C smoke branches; generated 26 required artifacts in test/output/step17_s_active_mixed_tax_foundation/.
- Verdict A: ACTIVE_PROGRESSIVE_PERSONAL_AND_FLAT_CORPORATE_TAX_INTERFACES_ACCEPTED. Step17.T not started.
## Step17.T - Personal Tax Base Completion and General Fiscal Capacity Audit

Completed the diagnostic tax-base audit without activating dividend tax, changing exemptions/rates, or changing pension/fiscal behavior. Reused Step17.S outputs and one 52-week no-tax mature replay. Wage income remains authoritative ordinary income; dividend routing provenance is authoritative, but the mature Legacy-only ownership state produced no Person dividend receipts. WAGE_PLUS_DIVIDEND is therefore retained as a shadow candidate, with retained earnings, inheritance, support, pension, settlement-only flows, and unrealized equity kept separate. The accepted minimum-need exemption protects all observed below-need Persons. Corporate taxable profit is positive early and reaches zero after the observed transition while sales remain positive, so profit tax is not treated as a stable standalone pillar. Verdict: A. PERSONAL_TAX_BASE_COMPLETED_WITH_WAGE_AND_DIVIDEND_INCOME. No Step17.U work started.
## Step17.U - Mature Firm Ownership Transition and Person Dividend Settlement Audit

Gate 0 found that both current Food and capital-good Firm construction paths default a missing CapTable to LegacyOwnershipPool. The mature checkpoint also lacks founder, startup equity contributor, or historical cap-table provenance, so Legacy-only ownership is partly historical compatibility but is also the current new-Firm default. No defensible Legacy-to-Person historical recovery exists; the counterfactual research migration contract remains explicitly inactive. Step17.T dividend provenance was preserved: 26 events, 702165.09212 declared, 0 Person paid, 702165.09212 Legacy entitlement, zero reconciliation gap. Person settlement, Estate compatibility, accounting, money, goods, assignment, and RNG boundaries remain valid. Verdict: C. CURRENT_FIRM_CREATION_STILL_DEFAULTS_TO_LEGACY_AND_REQUIRES_REPAIR. No Step17.V work started.
## Step17.V - Person Founder and New-Firm Equity Formation Contract

Implemented an opt-in deterministic founder-ownership formation layer for newly formed Firms. Eligible adult Persons with valid non-settlement Households receive explicit Person-owned CapTables; no eligible founder produces an explicit Legacy fallback reason. Historical Firms and old checkpoints remain Legacy-owned and are never migrated.

Founder metadata records formation context, provenance, cash/non-cash contribution fields, and bootstrap compatibility separately; bootstrap assignment is non-cash and does not create issuance, debt, dividends, investment, or money. The formation path covers ordinary Food Firms and the canonical capital_goods supplier path, with deterministic Person selection and distinct founders.

Controlled fixtures validated Person and Legacy dividend routing, Household settlement, Estate ownership, checkpoint compatibility, invalid-founder rejection, conservation, and pre-step parity. Verdict: A. NEW_FIRM_PERSON_FOUNDER_OWNERSHIP_CONTRACT_ACCEPTED. No Step17.W work started.
## Step17.W - Active Person Capital-Income Distribution and Shadow Progressive Tax Validation

Ran the required fresh 52-week same-seed control/treatment screen with personal, corporate, and sales taxes OFF. Step17.V founder ownership remained frozen; realized Person dividends were aggregated by unique Person and settled through the authoritative Social Household path.

The Person-founder branch recorded 25 dividend events, one recipient, and 7881.529974 realized Person dividend income. Shadow WAGE_ONLY liability was 0; WAGE_PLUS_REALIZED_DIVIDEND liability was 372.971305, with one dividend-activated taxpayer. The first declaration divergence occurred at week 3 and later differences are treated as endogenous demand feedback.

Dividend and cash concentration is classified as mechanical bootstrap concentration: one active Firm and one founder, not endogenous wealth inequality. Accepted 3026.40 exemption protection, accounting, money, goods, assignment, and RNG checks passed. Verdict: A. PERSON_DIVIDENDS_ACTIVATE_PROGRESSIVE_PERSONAL_TAX_BASE. No Step17.X work started.
## Step17.X - Endogenous Firm Formation Architecture Audit
- Completed a shadow-only endogenous Firm entry audit over deterministic N=100 and N=500, with canonical capital-good plumbing visible and no Firm creation.
- Authoritative demand gap, sales, capacity, utilization, price, cost, operating-profit, labor, founder eligibility, and startup cash inputs were recorded.
- Recommended a causal 13-week unmet-demand plus positive operating-profit signal; 26-week persistence remains a comparison screen.
- Founder selection is deterministic and protected-liquidity aware; founder equity funds the first payroll window, with no silent Step13 startup credit.
- Firm formation, ownership redistribution, exit, tax, pension, grants, and entrepreneurship RNG remain inactive.
- Verdict A: ENDOGENOUS_FIRM_ENTRY_CONTRACT_READY_FOR_ACTIVE_SMOKE. Step17.Y not started.

## Step17.Y - Active Endogenous Capital-Goods Entry & Household Cash Distribution Smoke
- Added a default-off, deterministic capital-goods-only endogenous entry system using the accepted 13-week persistent unmet-demand plus positive operating-profit signal.
- Formation transfers real protected-excess Household cash to a generic capital-good Firm as Person founder equity; no startup credit, free inventory, ownership redistribution, tax/pension change, or new RNG draw was added.
- Ran the primary N=500, seed42, 52-week control/treatment smoke and persisted Household cash distributions at weeks 0/13/26/39/52, low-liquidity transitions, founder/entrant effects, market response, and reconciliation panels.
- One entry was accepted at week 39; technical verdict A. The entrant had no immediate worker or sales because all eligible labor was already assigned, so the North-Star result is descriptive rather than a claim of broad distribution improvement.
- Economic North-Star verdict: ENTRY_MAINLY_RAISES_FOUNDER_UPPER_TAIL. No Step17.Z work started.

## Step17.Z - Household Cash Distribution Polarization Root-Cause Cohort Audit
- Replayed only the accepted Step17.Y CONTROL path at N=500, seed42, 52 weeks; no entry, pension, tax, transfer, wage, consumption, ownership, or behavioral RNG changes were activated.
- Persisted same-Household weekly cash-flow bridges, liquidity cohorts, percentile cohorts, baseline characteristics, labor/age/need/income decomposition, divergence timing, episodes, matched trajectories, log-cash gaps, and overlap diagnostics.
- Week52 DEEP_LOW is 84/183 (45.90%); P10/P25 cash is 0.92945, median 66.43710, and the largest positive-log-cash gap is 1.02003, supporting a descriptive two-regime mixture diagnostic.
- Low income relative to minimum need covers all DEEP_LOW households; age/low-productivity and no-employed-member channels overlap but do not cover a majority alone. Private support and pension are inactive in this audit.
- Verdict: LOW_INCOME_RELATIVE_TO_NEED_DOMINATES_POLARIZATION; next mechanism family is LABOR_INCOME_ARCHITECTURE. Step17.AA not started.
