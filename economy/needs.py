from economy import config


class NeedsSystem:

    def __init__(self, world):
        self.world = world

    def get_members(self, household):
        members = []

        for pid in household.parents:
            person = self.world.get_person_by_id(pid)

            if person is not None:
                members.append(person)

        for cid in household.children:
            person = self.world.get_person_by_id(cid)

            if person is not None:
                members.append(person)

        return members

    def household_minimum_need_units(self, household):
        person_dict = self.world.person_dict
        member_count = 0
        adults = 0
        children = 0
        elderly = 0

        for person_id in household.parents:
            person = person_dict.get(person_id)

            if person is None:
                continue

            member_count += 1

            if person.age < 20:
                children += 1
            elif person.age < 65:
                adults += 1
            else:
                elderly += 1

        for person_id in household.children:
            person = person_dict.get(person_id)

            if person is None:
                continue

            member_count += 1

            if person.age < 20:
                children += 1
            elif person.age < 65:
                adults += 1
            else:
                elderly += 1

        if member_count == 0:
            return 0.0

        equivalent_size = 0.0

        if adults > 0:
            equivalent_size += config.FIRST_ADULT_WEIGHT
            equivalent_size += (
                max(0, adults - 1)
                *
                config.SECOND_ADULT_WEIGHT
            )

        equivalent_size += (
            children
            *
            config.NEW_CHILD_WEIGHT
        )
        equivalent_size += (
            elderly
            *
            config.NEW_ELDERLY_WEIGHT
        )

        if equivalent_size == 0:
            equivalent_size = (
                member_count
                *
                config.NEW_CHILD_WEIGHT
            )

        return (
            config.NEW_BASE_CONSUMPTION
            +
            config.NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER
            *
            equivalent_size
            +
            config.CHILD_EXTRA_COST
            *
            children
            +
            config.ELDERLY_EXTRA_COST
            *
            elderly
        )

    def household_minimum_cost(self, household, price_index=1.0):
        return (
            self.household_minimum_need_units(household)
            *
            price_index
        )
