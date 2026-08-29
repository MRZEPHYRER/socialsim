# Step 12.5A: Marriage / Pairing Timing and RNG Audit

## Scope

This is an audit only. No marriage frequency, matching algorithm, RNG stream,
checkpoint behavior, age threshold, fertility rule, household rule, or economic
mechanism was changed.

## 1. Exact Current Control Flow

For each `World.step()` the current order is:

1. Set the technical step index from the completed population-history length.
2. Begin the ledger step and reset public-wealth audit state.
3. Call `person.grow()` for every person. With the accepted weekly age clock,
   this advances each person by one week.
4. Call `HouseholdManager.step()`:
   - adults with age `>= 20` and no partner are marked
     `is_seeking_partner = True`;
   - dead members from the previous lifecycle state are removed from household
     lists;
   - empty households are cleaned up and their wealth is sent to the public
     sector according to the existing rule.
5. Call `MarriageSystem.process_marriage()`.
6. Refresh active households.
7. Run the firm/economic step: production, wages, household consumption,
   sales, inventory, prices, credit, dividends and public-sector flows.
8. Run fertility once for currently eligible two-parent households.
9. Run mortality once for the current population.
10. Settle inheritance and remove dead relationships/persons.
11. Store population, household, economic, accounting and conservation
    diagnostics.

Marriage therefore occurs **after age progression and household cleanup**, but
**before economic processing, fertility, mortality and same-step inheritance**.
Deaths from the same step cannot affect that step's marriage market because
mortality is processed later. A birth from the same step cannot enter marriage
because fertility is processed later and the newborn is age zero.

## 2. Matching Algorithm

`get_marriage_pool()` includes every alive person who satisfies all of:

- `is_seeking_partner` is true;
- age is at least 20 calendar years;
- `partner_id is None`.

Males and females are collected separately. The full market then:

1. shuffles all eligible males;
2. builds a female pool keyed by integer calendar age;
3. considers every shuffled eligible male;
4. searches female ages from `male_age - 8` through `male_age + 8`;
5. removes already-selected females from consideration;
6. randomly selects one remaining candidate;
7. removes both people from their old households, creates a new household,
   transfers their combined wealth, registers both as parents, sets reciprocal
   partner IDs, and clears both seeking flags.

If a male has no candidate, he remains seeking and can be considered at a later
market. An unmatched female also remains seeking. There is no explicit success
probability: matching success depends on pool composition, age compatibility,
ordering and candidate availability.

## 3. Actual Historical Time Semantics

The current implementation executes the full matching market once per
simulation step. Before the weekly conversion, the model documentation and
behavior treated one step as approximately one year. Consequently the old
model implicitly provided approximately one full matching-market opportunity
per calendar year.

This is an opportunity-frequency statement, not an annual marriage probability.
There is no independent marriage hazard or Bernoulli marriage draw. Some people
can still be skipped or remain unmatched because:

- they are not seeking;
- they are under the age threshold;
- they have a partner;
- no compatible candidate remains;
- another male selected a female earlier in the same market.

## 4. RNG Ownership and Coupling

### Finding

`MarriageSystem` uses the module-level Python `random` stream:

- `random.shuffle(males)`;
- `random.choice(candidates)`.

The stream is seeded globally by `World.__init__` when a seed is supplied.

### Shared consumers

The same global stream is also used by:

- initial age and sex assignment;
- initial household randomization;
- mortality draws in `Person.check_death()`;
- fertility draws;
- newborn sex selection;
- other legacy global-random lifecycle operations.

Firm-specific adaptive pricing uses `firm_rng`, and multi-firm consumer choice
uses `market_rng`, so those newer streams are isolated from marriage. Marriage
itself is not isolated from mortality or fertility.

```text
MARRIAGE_RNG_ISOLATION: SHARED_RNG
```

Changing marriage from every step to every 52 steps will therefore alter the
number and position of global RNG draws. Even if the matching algorithm is
unchanged, subsequent mortality, fertility and newborn-sex random outcomes can
change. This is not a bug in the proposed scheduler; it is a direct RNG-stream
coupling risk that must be accepted, isolated, or separately tested in the next
design stage.

## 5. Existing Schedule State

There is currently no marriage schedule state:

- no `last_marriage_market_step`;
- no `next_marriage_market_step`;
- no modulo-based schedule;
- no active use of `marriage_wait`.

`marriage_wait` is initialized on `Person` but is never read or updated by the
current matching path. Failed matching attempts therefore have no cooldown or
counter side effect beyond people remaining seeking.

## 6. Checkpoint Migration Recommendation

The existing warm checkpoint has no marriage scheduling metadata. Its metadata
reports `global_step = 5000`; after loading, the world has completed 5000
historical steps and the next simulation step is technical step 5000. The
checkpoint's marriage system contains only a reference to the world and no
history of the last market opportunity.

The exact historical annual market anchor is not recoverable. The checkpoint
was produced while the full market ran every old step, and no last-opportunity
field was stored.

For the proposed compatibility migration, the safest documented assumption is:

```text
legacy checkpoint represents a state immediately after a completed market
next_marriage_market_step = checkpoint_global_step + 52
```

For this checkpoint that gives the first new weekly market at approximately
step `5052`. This avoids an immediate repeat at step 5000 and preserves the
requested interpretation that the next opportunity is about one calendar year
after the last known historical opportunity. It is an assumption, not recovered
historical information, and must be recorded in new checkpoint metadata.

Fresh weekly simulations should initialize an explicit schedule deterministically
and persist it. A `next_marriage_market_step` field is preferable to a raw
`global_step % 52 == 0` rule because it survives arbitrary checkpoint origins
and makes the legacy migration assumption explicit.

## 7. Age Eligibility Interaction

Eligibility is evaluated only when the full market runs. A person crossing age
20 one week after a market can wait up to 51 weeks for the next opportunity.
This is consistent with preserving the old model's annual temporal resolution,
but it creates a boundary delay that should be measured in Step 12.5B.

The current age thresholds remain calendar thresholds. No threshold conversion
is needed for this audit.

## 8. Household and Fertility Dependencies

A successful match has direct downstream effects:

- both people leave their current households;
- their wealth is transferred into a newly created household;
- both are registered as household parents;
- reciprocal partner IDs are set;
- seeking flags are cleared;
- the new two-parent household becomes eligible for the existing fertility path
  if the mother satisfies reproductive-age conditions.

The fertility system assumes a two-parent household and reads the two parent IDs.
It does not assume that marriage ran in the current week beyond seeing the final
household state. No downstream code currently reads a marriage-market timestamp.

## 9. Assessment of Full Matching Every 52 Weeks

### Advantages

- preserves the historical full-market algorithm;
- approximates one full matching opportunity per model year;
- avoids inventing a new marriage probability;
- keeps market thickness intact at each opportunity;
- makes age and household timing easier to interpret.

### Risks

- people newly reaching adulthood wait until the next market;
- RNG consumption changes because fewer shuffles and choices occur;
- shared global RNG changes later mortality/fertility outcomes;
- legacy checkpoints lack an exact market anchor;
- the first scheduled market needs an explicit migration convention;
- household and fertility trajectories will not be numerically identical to the
  old every-step model after the intentional behavior change.

Randomly activating approximately `1/52` of singles every week would not be
equivalent. It would thin the market, reduce candidate-pool thickness, make
matching order and sex imbalance more volatile, and produce a different matching
algorithm rather than a lower-frequency full market. The proposed full-market
schedule is the better compatibility direction.

## 10. Required Step 12.5B State

Before implementation, add and persist:

- `next_marriage_market_step`;
- a schedule schema/version marker;
- migration metadata identifying legacy checkpoint anchoring;
- optionally, diagnostic counters for market executions, eligible singles,
  matches, unmatched males and unmatched females.

Do not use `marriage_wait` unless a separate design explicitly assigns it a
meaning; its current state is unused.

## Final Assessment

The current behavior is well understood and the proposed “full matching market
once every 52 weeks” design is semantically coherent. However, the shared RNG
stream and irrecoverable legacy market anchor are material implementation risks
that should be handled explicitly in Step 12.5B tests and metadata.

**A. Safe to implement Step 12.5B annual-equivalent marriage scheduling**

This approval is for implementation planning only. Step 12.5A itself made no
behavioral changes.
