from economy.ledger import object_account


class InheritanceSystem:


    def __init__(self, world):

        self.world=world



    def process_inheritance(self, dead_people):

        # Person-level equity is settled before demographic relationship
        # cleanup removes the deceased Person from ``person_dict``.  The
        # Estate holder remains valid even when no Household heir exists.
        self.world.ensure_shareholder_estate_state()
        for dead_person in dead_people:
            self.world.shareholder_estate_system.open_for_death(
                dead_person,
                step=getattr(self.world, "current_step_index", 0),
            )


        processed_households=set()
        dead_ids_by_household = {}


        for dead_person in dead_people:


            if dead_person.household_id is None:
                continue


            dead_ids_by_household.setdefault(
                dead_person.household_id,
                set(),
            ).add(dead_person.id)


        for dead_person in dead_people:


            household_id=dead_person.household_id


            if household_id is None:
                continue



            household=self.world.get_household(
                household_id
            )


            if household is None:
                continue



            if household.id in processed_households:
                continue


            processed_households.add(
                household.id
            )


            # 如果家庭还有活人，不继承

            alive_members = []


            for pid in household.parents + household.children:

                member = self.world.get_person_by_id(pid)

                if member is not None and member.alive:

                    alive_members.append(member)



            if len(alive_members)>0:
                if household.wealth > 0:
                    household_member_ids = household.parents + household.children
                    dead_member_count = len(
                        [
                            pid
                            for pid in household_member_ids
                            if pid in dead_ids_by_household.get(
                                household.id,
                                set(),
                            )
                        ]
                    )
                    total_member_count = len(alive_members) + dead_member_count
                    implied_inherited_wealth = (
                        household.wealth
                        *
                        dead_member_count
                        /
                        total_member_count
                        if total_member_count > 0
                        else 0.0
                    )
                    self.world.record_inherited_wealth_to_spouse_or_survivors(
                        implied_inherited_wealth
                    )

                continue



            wealth=household.wealth


            if wealth<=0:
                continue



            heirs=[]



            # 找孩子

            for child_id in dead_person.children_ids:


                child=self.world.get_person_by_id(
                    child_id
                )


                if child is not None and child.alive:

                    if child not in heirs:

                        heirs.append(child)



            if len(heirs)>0:


                share=wealth/len(heirs)


                for child in heirs:


                    child_household=self.world.get_household(
                        child.household_id
                    )


                    if child_household is not None:

                        wealth_before = household.wealth
                        destination_before = child_household.wealth

                        self.world.ledger.transfer(
                            payer=object_account(
                                household,
                                "wealth",
                                f"household.{household.id}.wealth",
                            ),
                            receiver=object_account(
                                child_household,
                                "wealth",
                                f"household.{child_household.id}.wealth",
                            ),
                            amount=share,
                            reason="inheritance_to_heir",
                        )
                        self.world.record_household_lifecycle_event(
                            household_id=household.id,
                            event_type="inheritance_to_child",
                            wealth_before=wealth_before,
                            wealth_after=household.wealth,
                            wealth_to_heirs=share,
                        )
                        self.world.record_lifecycle_transfer_event(
                            event_type="inheritance_to_child",
                            source_household_id=household.id,
                            destination_household_id=child_household.id,
                            person_id=child.id,
                            amount=share,
                            source_wealth_before=wealth_before,
                            source_wealth_after=household.wealth,
                            destination_wealth_before=destination_before,
                            destination_wealth_after=child_household.wealth,
                            source_account=f"household.{household.id}.wealth",
                            destination_account=(
                                f"household.{child_household.id}.wealth"
                            ),
                        )
                        self.world.record_inherited_wealth_to_children(
                            share
                        )



            else:

                self.world.record_public_wealth_source(
                    "no_heir",
                    wealth,
                )
                self.world.ledger.transfer(
                    payer=object_account(
                        household,
                        "wealth",
                        f"household.{household.id}.wealth",
                    ),
                    receiver=object_account(
                        self.world,
                        "public_wealth",
                        "world.public_wealth",
                    ),
                    amount=wealth,
                    reason="inheritance_to_public_wealth",
                )
                self.world.record_lifecycle_transfer_event(
                    event_type="inheritance_to_public",
                    source_household_id=household.id,
                    amount=wealth,
                    source_wealth_before=wealth,
                    source_wealth_after=household.wealth,
                    destination_wealth_before=0.0,
                    destination_wealth_after=0.0,
                    public_sector_amount=wealth,
                    source_account=f"household.{household.id}.wealth",
                    destination_account="world.public_wealth",
                )
                self.world.record_household_lifecycle_event(
                    household_id=household.id,
                    event_type="inheritance_to_public",
                    wealth_before=wealth,
                    wealth_after=household.wealth,
                    wealth_to_public=wealth,
                )
