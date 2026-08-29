from productivity import age_productivity
from economy.config import (
    CHILD_EXTRA_COST,
    ELDERLY_EXTRA_COST,
    FIRST_ADULT_WEIGHT,
    NEW_BASE_CONSUMPTION,
    NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER,
    NEW_CHILD_WEIGHT,
    NEW_ELDERLY_WEIGHT,
    PRODUCTIVITY_TO_INCOME,
    PARENT_SUPPORT_RATIO,
    SECOND_ADULT_WEIGHT,
)


class IncomeSystem:

    def __init__(self, world):

        self.world = world
        self.household_need_cache = {}

    def household_basic_consumption_need(self, household):

        members = []

        for pid in household.parents:

            person = self.world.get_person_by_id(pid)

            if person is not None:

                members.append(person)

        for cid in household.children:

            person = self.world.get_person_by_id(cid)

            if person is not None:

                members.append(person)

        if len(members) == 0:

            return 0.0

        adults = 0
        children = 0
        elderly = 0

        for person in members:

            if person.age < 20:

                children += 1

            elif person.age < 65:

                adults += 1

            else:

                elderly += 1

        equivalent_size = 0.0

        if adults > 0:
            equivalent_size += FIRST_ADULT_WEIGHT
            equivalent_size += (
                max(0, adults - 1)
                *
                SECOND_ADULT_WEIGHT
            )

        equivalent_size += children * NEW_CHILD_WEIGHT
        equivalent_size += elderly * NEW_ELDERLY_WEIGHT

        if equivalent_size == 0:
            equivalent_size = len(members) * NEW_CHILD_WEIGHT

        return (
            NEW_BASE_CONSUMPTION
            +
            NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER
            *
            equivalent_size
            +
            CHILD_EXTRA_COST
            *
            children
            +
            ELDERLY_EXTRA_COST
            *
            elderly
        )

    def parent_needs_support(self, parent):

        household = self.world.get_household(
            parent.household_id
        )

        if household is None:

            return False

        if household.id not in self.household_need_cache:

            self.household_need_cache[household.id] = (
                self.household_basic_consumption_need(
                    household
                )
            )

        need = self.household_need_cache[household.id]

        return household.wealth < need


    # LEGACY / NON-AUTHORITATIVE: this pre-Ledger parent-support branch
    # is retained for compatibility only. Do not reconnect it to runtime.
    def distribute_income(self):
        """
        每一步收入分配

        1.
        个体根据生产力产生收入

        2.
        根据存活父母数量支付赡养

        3.
        剩余收入进入自己的家庭财富
        """

        self.household_need_cache = {}

        # =============================
        # 清空本回合收入
        # =============================

        for household in self.world.households:

            household.income_this_step = 0



        # =============================
        # 个体生产收入
        # =============================

        for person in self.world.population:


            if not person.alive:

                continue



            productivity = age_productivity(
                person.age
            )


            income = (
                productivity
                *
                PRODUCTIVITY_TO_INCOME
            )



            # =============================
            # 父母赡养计算
            # =============================


            alive_parents = []


            for parent_id in person.parent_ids:


                parent = self.world.get_person_by_id(
                    parent_id
                )


                if parent is not None and parent.alive:

                    alive_parents.append(parent)



            parents_needing_support = [
                parent
                for parent in alive_parents
                if self.parent_needs_support(parent)
            ]



            parent_number = len(parents_needing_support)



            if parent_number == 2:

                support_ratio = (
                    PARENT_SUPPORT_RATIO
                )


            elif parent_number == 1:

                support_ratio = (
                    PARENT_SUPPORT_RATIO
                    *
                    0.5
                )


            else:

                support_ratio = 0



            support = (
                income
                *
                support_ratio
            )


            remain = (
                income
                -
                support
            )



            # =============================
            # 给存活父母家庭
            # =============================


            if parent_number > 0:


                share = (
                    support
                    /
                    parent_number
                )


                for parent in parents_needing_support:


                    parent_household = (
                        self.world.get_household(
                            parent.household_id
                        )
                    )


                    if parent_household is not None:


                        parent_household.income_this_step += share



            # =============================
            # 自己家庭财富
            # =============================


            own_household = (
                self.world.get_household(
                    person.household_id
                )
            )


            if own_household is not None:


                own_household.income_this_step += remain



        # =============================
        # 收入进入家庭财富
        # =============================


