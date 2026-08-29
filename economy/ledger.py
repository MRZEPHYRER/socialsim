from dataclasses import dataclass
import csv
import os


@dataclass(frozen=True)
class AccountRef:
    name: str
    getter: object
    setter: object

    def get(self):
        return self.getter()

    def set(self, value):
        self.setter(value)


def object_account(obj, attr, name):
    return AccountRef(
        name=name,
        getter=lambda: getattr(obj, attr),
        setter=lambda value: setattr(obj, attr, value),
    )


class Ledger:

    def __init__(self, world=None, record_details=True):
        self.world = world
        self.record_details = record_details
        self.records = []
        self.step_summaries = {}
        self.current_step_index = None

    def current_step(self):
        if self.world is None:
            return None

        if self.current_step_index is not None:
            return self.current_step_index

        return len(getattr(self.world, "population_history", []))

    def begin_step(self, step):
        self.current_step_index = step

    def end_step(self):
        self.current_step_index = None

    def empty_summary(self):
        return {
            "ledger_transaction_count": 0,
            "ledger_transfer_amount": 0.0,
            "ledger_money_created": 0.0,
            "ledger_money_destroyed": 0.0,
            "ledger_loans_created": 0.0,
            "ledger_loans_repaid": 0.0,
        }

    def step_summary(self, step):
        if step not in self.step_summaries:
            self.step_summaries[step] = self.empty_summary()

        return self.step_summaries[step]

    def record_summary(
        self,
        amount,
        creates_money=False,
        destroys_money=False,
        creates_loan=False,
        repays_loan=False,
    ):
        summary = self.step_summary(self.current_step())
        summary["ledger_transaction_count"] += 1

        if creates_money:
            summary["ledger_money_created"] += amount
        elif destroys_money:
            summary["ledger_money_destroyed"] += amount
        else:
            summary["ledger_transfer_amount"] += amount

        if creates_loan:
            summary["ledger_loans_created"] += amount

        if repays_loan:
            summary["ledger_loans_repaid"] += amount

    def record(
        self,
        payer,
        receiver,
        amount,
        reason,
        creates_money=False,
        destroys_money=False,
        creates_loan=False,
        repays_loan=False,
        interest=0.0,
    ):
        step = self.current_step()
        self.record_summary(
            amount=amount,
            creates_money=creates_money,
            destroys_money=destroys_money,
            creates_loan=creates_loan,
            repays_loan=repays_loan,
        )

        if self.record_details:
            self.records.append(
                {
                    "step": step,
                    "payer": payer,
                    "receiver": receiver,
                    "amount": amount,
                    "reason": reason,
                    "creates_money": creates_money,
                    "destroys_money": destroys_money,
                    "creates_loan": creates_loan,
                    "repays_loan": repays_loan,
                    "interest": interest,
                }
            )

    def transfer(self, payer, receiver, amount, reason):
        amount = max(0.0, amount)

        if amount <= 0:
            return 0.0

        payer.set(payer.get() - amount)
        receiver.set(receiver.get() + amount)
        self.record(
            payer=payer.name,
            receiver=receiver.name,
            amount=amount,
            reason=reason,
        )

        return amount

    def transfer_attrs(
        self,
        payer_obj,
        payer_attr,
        payer_name,
        receiver_obj,
        receiver_attr,
        receiver_name,
        amount,
        reason,
    ):
        amount = max(0.0, amount)

        if amount <= 0:
            return 0.0

        setattr(
            payer_obj,
            payer_attr,
            getattr(payer_obj, payer_attr) - amount,
        )
        setattr(
            receiver_obj,
            receiver_attr,
            getattr(receiver_obj, receiver_attr) + amount,
        )
        if self.record_details:
            self.record(
                payer=payer_name,
                receiver=receiver_name,
                amount=amount,
                reason=reason,
            )
        else:
            self.record_summary(amount)

        return amount

    def create_money_attr(
        self,
        receiver_obj,
        receiver_attr,
        receiver_name,
        amount,
        reason,
        money_supply_obj=None,
        money_supply_attr=None,
    ):
        amount = max(0.0, amount)

        if amount <= 0:
            return 0.0

        setattr(
            receiver_obj,
            receiver_attr,
            getattr(receiver_obj, receiver_attr) + amount,
        )

        if money_supply_obj is not None and money_supply_attr is not None:
            setattr(
                money_supply_obj,
                money_supply_attr,
                getattr(money_supply_obj, money_supply_attr) + amount,
            )

        if self.record_details:
            self.record(
                payer="money_creation",
                receiver=receiver_name,
                amount=amount,
                reason=reason,
                creates_money=True,
            )
        else:
            self.record_summary(
                amount,
                creates_money=True,
            )

        return amount

    def destroy_money_attr(
        self,
        payer_obj,
        payer_attr,
        payer_name,
        amount,
        reason,
        money_supply_obj=None,
        money_supply_attr=None,
    ):
        amount = max(0.0, amount)

        if amount <= 0:
            return 0.0

        setattr(
            payer_obj,
            payer_attr,
            getattr(payer_obj, payer_attr) - amount,
        )

        if money_supply_obj is not None and money_supply_attr is not None:
            setattr(
                money_supply_obj,
                money_supply_attr,
                getattr(money_supply_obj, money_supply_attr) - amount,
            )

        if self.record_details:
            self.record(
                payer=payer_name,
                receiver="money_destruction",
                amount=amount,
                reason=reason,
                destroys_money=True,
            )
        else:
            self.record_summary(
                amount,
                destroys_money=True,
            )

        return amount

    def create_money(self, receiver, amount, reason, money_supply=None):
        amount = max(0.0, amount)

        if amount <= 0:
            return 0.0

        receiver.set(receiver.get() + amount)

        if money_supply is not None:
            money_supply.set(money_supply.get() + amount)

        self.record(
            payer="money_creation",
            receiver=receiver.name,
            amount=amount,
            reason=reason,
            creates_money=True,
        )

        return amount

    def destroy_money(self, payer, amount, reason, money_supply=None):
        amount = max(0.0, amount)

        if amount <= 0:
            return 0.0

        payer.set(payer.get() - amount)

        if money_supply is not None:
            money_supply.set(money_supply.get() - amount)

        self.record(
            payer=payer.name,
            receiver="money_destruction",
            amount=amount,
            reason=reason,
            destroys_money=True,
        )

        return amount

    def create_loan(self, bank, borrower, amount, reason, money_supply=None):
        amount = self.create_money(
            receiver=borrower,
            amount=amount,
            reason=reason,
            money_supply=money_supply,
        )

        if amount > 0:
            step = self.current_step()
            self.step_summary(step)["ledger_loans_created"] += amount

            if self.record_details:
                self.records[-1]["payer"] = bank.name
                self.records[-1]["creates_loan"] = True

        return amount

    def create_loan_attr(
        self,
        bank_name,
        borrower_obj,
        borrower_attr,
        borrower_name,
        amount,
        reason,
        money_supply_obj=None,
        money_supply_attr=None,
        loan_obj=None,
        loan_attr=None,
    ):
        amount = self.create_money_attr(
            receiver_obj=borrower_obj,
            receiver_attr=borrower_attr,
            receiver_name=borrower_name,
            amount=amount,
            reason=reason,
            money_supply_obj=money_supply_obj,
            money_supply_attr=money_supply_attr,
        )

        if amount > 0:
            if loan_obj is not None and loan_attr is not None:
                setattr(
                    loan_obj,
                    loan_attr,
                    getattr(loan_obj, loan_attr) + amount,
                )

            step = self.current_step()
            self.step_summary(step)["ledger_loans_created"] += amount

            if self.record_details:
                self.records[-1]["payer"] = bank_name
                self.records[-1]["creates_loan"] = True

        return amount

    def repay_loan(
        self,
        borrower,
        bank,
        principal,
        interest=0.0,
        reason="loan_repayment",
        money_supply=None,
    ):
        principal = max(0.0, principal)
        interest = max(0.0, interest)
        total = principal + interest

        if total <= 0:
            return 0.0

        if interest > 0:
            self.transfer(
                payer=borrower,
                receiver=bank,
                amount=interest,
                reason=f"{reason}: interest",
            )

        if principal > 0:
            self.destroy_money(
                payer=borrower,
                amount=principal,
                reason=f"{reason}: principal",
                money_supply=money_supply,
            )
            step = self.current_step()
            self.step_summary(step)["ledger_loans_repaid"] += principal

            if self.record_details:
                self.records[-1]["receiver"] = bank.name
                self.records[-1]["repays_loan"] = True
                self.records[-1]["interest"] = interest

        return total

    def repay_loan_attrs(
        self,
        borrower_obj,
        borrower_attr,
        borrower_name,
        principal,
        reason="loan_repayment",
        money_supply_obj=None,
        money_supply_attr=None,
        loan_obj=None,
        loan_attr=None,
        bank_obj=None,
        bank_attr=None,
        bank_name=None,
        interest=0.0,
    ):
        principal = max(0.0, principal)
        interest = max(0.0, interest)

        if interest > 0 and bank_obj is not None and bank_attr is not None:
            self.transfer_attrs(
                payer_obj=borrower_obj,
                payer_attr=borrower_attr,
                payer_name=borrower_name,
                receiver_obj=bank_obj,
                receiver_attr=bank_attr,
                receiver_name=bank_name,
                amount=interest,
                reason=f"{reason}: interest",
            )

        if principal > 0:
            self.destroy_money_attr(
                payer_obj=borrower_obj,
                payer_attr=borrower_attr,
                payer_name=borrower_name,
                amount=principal,
                reason=f"{reason}: principal",
                money_supply_obj=money_supply_obj,
                money_supply_attr=money_supply_attr,
            )

            if loan_obj is not None and loan_attr is not None:
                setattr(
                    loan_obj,
                    loan_attr,
                    max(0.0, getattr(loan_obj, loan_attr) - principal),
                )

            step = self.current_step()
            self.step_summary(step)["ledger_loans_repaid"] += principal

            if self.record_details:
                self.records[-1]["receiver"] = bank_name
                self.records[-1]["repays_loan"] = True
                self.records[-1]["interest"] = interest

        return principal + interest

    def records_for_step(self, step):
        return [
            record
            for record in self.records
            if record["step"] == step
        ]

    def export_csv(self, path):
        if not path:
            return

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        fieldnames = [
            "step",
            "payer",
            "receiver",
            "amount",
            "reason",
            "creates_money",
            "destroys_money",
            "creates_loan",
            "repays_loan",
            "interest",
        ]

        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    def summary_for_step(self, step):
        summary = self.step_summaries.get(step)

        if summary is None:
            return self.empty_summary()

        return dict(summary)
