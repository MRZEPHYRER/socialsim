"""Recover the final network-event CSV without rerunning control/weekly runs."""

from __future__ import annotations

import copy
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from step15_mature_genealogy_private_support_experiment import restore_world, STABILIZATION_WEEKS
from economy.private_family_support import PrivateFamilySupportSystem
from step15_periodic_batched_family_support import PeriodicBatchedSupportSystem

SOURCE = ROOT / "test/output/step15_mature_genealogy_private_support_experiment/child_payer_burden.csv"
OUT = ROOT / "test/output/step15_periodic_batched_family_support/family_network_settlement.csv"
OBSERVATION_WEEKS = 520


def row_for(branch, event):
    need = float(event.get("parent_need_before_settlement", event.get("parent_need_gap", 0.0)) or 0.0)
    amount = float(event.get("amount", event.get("transfer_amount", 0.0)) or 0.0)
    return {
        "branch": branch,
        "family_network_id": event.get("family_network_id", "weekly_pairwise"),
        "batch_id": event.get("batch_id", "weekly"),
        "payer_household_id": event.get("payer_household_id", ""),
        "recipient_household_id": event.get("recipient_household_id", ""),
        "parent_need_before_settlement": need,
        "child_capacity_before_settlement": event.get("child_capacity_before_settlement", event.get("payer_available_support_capacity", "")),
        "total_support": amount,
        "unmet_residual_need": max(0.0, need - amount),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["branch", "family_network_id", "batch_id", "payer_household_id", "recipient_household_id", "parent_need_before_settlement", "child_capacity_before_settlement", "total_support", "unmet_residual_need"]
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with SOURCE.open("r", newline="", encoding="utf-8") as source:
            for raw in csv.DictReader(source):
                if raw.get("branch") == "treatment":
                    writer.writerow(row_for("WEEKLY_PAIRWISE", raw))

        world, _ = restore_world()
        world.private_family_support_enabled = False
        for _ in range(STABILIZATION_WEEKS):
            world.step()
        common = copy.deepcopy(world)
        control = copy.deepcopy(common)
        weekly = copy.deepcopy(common)
        batched = copy.deepcopy(common)
        control.private_family_support_enabled = False
        control.private_family_support_system = PrivateFamilySupportSystem(control)
        weekly.private_family_support_enabled = True
        weekly.private_family_support_system = PrivateFamilySupportSystem(weekly)
        batched.private_family_support_enabled = True
        batched.private_family_support_system = PeriodicBatchedSupportSystem(batched, interval=4)
        for _ in range(OBSERVATION_WEEKS):
            control.step()
        for _ in range(OBSERVATION_WEEKS):
            weekly.step()
        system = batched.private_family_support_system
        for _ in range(OBSERVATION_WEEKS):
            batched.step()
            for event in system.last_result.transfers:
                writer.writerow(row_for("BATCHED_4W", event))


if __name__ == "__main__":
    main()
