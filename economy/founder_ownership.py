"""Deterministic Person-founder assignment for newly initialized Worlds.

This is an initialization convention, not historical ownership recovery. It
never charges a Household and never mutates a stored checkpoint implicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from economy.ownership_accounting import CapTable

FOUNDER_SCHEMA = "person_founder_bootstrap_v1"


@dataclass(frozen=True)
class FounderAssignment:
    firm_id: object
    founder_person_id: object = None
    ownership_fraction: float = 0.0
    assignment_source: str = ""
    formation_context: str = ""
    equity_cash_contribution: float = 0.0
    equity_non_cash_contribution: float = 0.0
    bootstrap_compatibility_amount: float = 0.0
    provenance_status: str = ""
    ownership_fallback_reason: str = ""

    def to_dict(self):
        return {"schema": FOUNDER_SCHEMA, **asdict(self)}


def eligible_founders(world):
    """Return neutral legal/settlement-eligible Persons in stable ID order."""
    result = []
    for person in getattr(world, "population", []):
        if not getattr(person, "alive", False):
            continue
        if float(getattr(person, "age", 0.0)) < 20.0:
            continue
        if getattr(person, "settlement_only", False):
            continue
        household = world.get_household(getattr(person, "household_id", None))
        if household is None or getattr(household, "settlement_only", False):
            continue
        if not world.has_valid_settlement_household(person):
            continue
        result.append(person)
    return sorted(result, key=lambda person: person.id)


def validate_founder_person(world, person_id, ownership_fraction=1.0):
    """Validate an explicitly supplied founder without searching implicitly."""
    if not 0.0 < float(ownership_fraction) <= 1.0:
        raise ValueError("founder ownership_fraction must be in (0, 1]")
    person = next(
        (item for item in getattr(world, "population", []) if item.id == person_id),
        None,
    )
    if person is None:
        raise ValueError("founder Person does not exist")
    if not getattr(person, "alive", False):
        raise ValueError("founder Person must be alive")
    if float(getattr(person, "age", 0.0)) < 20.0:
        raise ValueError("founder Person must be an adult")
    if getattr(person, "settlement_only", False):
        raise ValueError("settlement-only Person cannot be founder")
    household = world.get_household(getattr(person, "household_id", None))
    if household is None or getattr(household, "settlement_only", False):
        raise ValueError("founder Person needs a valid Social Household")
    if not world.has_valid_settlement_household(person):
        raise ValueError("founder Person needs a valid Social Household")
    return person

def assign_bootstrap_founders(world):
    """Assign distinct Person owners to newly constructed Firms when enabled."""
    if not getattr(world, "person_founder_bootstrap_enabled", False):
        return []
    if not getattr(world, "new_world_formation_context", False):
        return []

    firms = [*getattr(world, "firms", []), *getattr(world, "capital_good_firms", [])]
    firms = sorted(firms, key=lambda firm: str(getattr(firm, "firm_id", "")))
    eligible = eligible_founders(world)
    assignments = getattr(world, "founder_assignment_rows", [])
    used = {
        row.get("founder_person_id")
        for row in assignments
        if row.get("founder_person_id") is not None
    }
    assigned_ids = getattr(world, "founder_assigned_firm_ids", set())
    for firm in firms:
        firm_id = getattr(firm, "firm_id", None)
        table = getattr(firm, "cap_table", None)
        if table is None:
            table = CapTable.initial_legacy_owned(firm_id)
            firm.cap_table = table
        if table.person_shares > 0.0:
            assigned_ids.add(firm_id)
            continue
        # A fresh Firm split can reuse an existing numeric ID while resetting
        # its cap table. Treat that as a new formation event.
        if firm_id in assigned_ids:
            assigned_ids.discard(firm_id)
            old_rows = [
                row for row in assignments
                if row.get("firm_id") == firm_id
            ]
            assignments[:] = [
                row for row in assignments
                if row.get("firm_id") != firm_id
            ]
            for row in old_rows:
                old_founder = row.get("founder_person_id")
                if old_founder in used:
                    used.remove(old_founder)
        founder = next((person for person in eligible if person.id not in used), None)
        if founder is None:
            assignment = FounderAssignment(
                firm_id=firm_id,
                assignment_source="explicit_legacy_compatibility_fallback",
                formation_context="CANONICAL_BOOTSTRAP_FIRM",
                bootstrap_compatibility_amount=max(0.0, float(getattr(firm, "cash", 0.0))),
                provenance_status="LEGACY_COMPATIBILITY",
                ownership_fallback_reason="NO_VALID_PERSON_FOUNDER",
            )
        else:
            assignment = FounderAssignment(
                firm_id=firm_id,
                founder_person_id=founder.id,
                ownership_fraction=1.0,
                assignment_source="deterministic_sorted_eligible_person_id",
                formation_context="CANONICAL_BOOTSTRAP_FIRM",
                equity_cash_contribution=0.0,
                equity_non_cash_contribution=0.0,
                bootstrap_compatibility_amount=max(0.0, float(getattr(firm, "cash", 0.0))),
                provenance_status="BOOTSTRAP_PERSON_FOUNDER_ASSIGNMENT",
            )
            firm.cap_table = CapTable.initial_person_owned(
                firm_id=firm_id,
                person_id=founder.id,
                issued_shares=100.0,
            )
            used.add(founder.id)
        firm.founder_assignment = assignment.to_dict()
        firm.ownership_fallback_reason = assignment.ownership_fallback_reason
        assignments.append(assignment.to_dict())
        assigned_ids.add(firm_id)
    world.founder_assignment_rows = assignments
    world.founder_assigned_firm_ids = assigned_ids
    ownership_system = getattr(world, "equity_ownership_system", None)
    if ownership_system is not None:
        for firm in firms:
            table = getattr(firm, "cap_table", None)
            if table is not None:
                ownership_system.cap_tables[table.firm_id] = table
    return assignments


__all__ = ["FOUNDER_SCHEMA", "FounderAssignment", "eligible_founders", "validate_founder_person", "assign_bootstrap_founders"]