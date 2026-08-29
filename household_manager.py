class HouseholdManager:


    def __init__(self, world):

        self.world = world



    # =================================================
    # Adult children leave household
    # =================================================

    def process_adulthood(self):


        for person in self.world.population:


            if person.age >= 20:


                if person.partner_id is None:

                    person.is_seeking_partner = True
                    
    # =================================================
    # Remove dead members
    # =================================================

    def remove_dead_members(self):

        person_dict = self.world.person_dict

        for household in self.world.households:


            household.parents=[

                pid

                for pid in household.parents

                if person_dict.get(pid)

            ]


            household.children=[

                cid

                for cid in household.children

                if person_dict.get(cid)

            ]



    # =================================================
    # Remove empty households
    # =================================================

    def remove_empty_households(self):


        alive_households=[]


        for household in self.world.households:

            if household.size()>0:

                alive_households.append(
                    household
                )

            else:

                wealth_before = household.wealth

                tolerance = 1e-9

                if abs(household.wealth) > tolerance and household.wealth > 0:
                    self.world.record_public_wealth_source(
                        "household_dissolution",
                        household.wealth,
                    )

                    self.world.ledger.transfer_attrs(
                        payer_obj=household,
                        payer_attr="wealth",
                        payer_name=f"household.{household.id}.wealth",
                        receiver_obj=self.world,
                        receiver_attr="public_wealth",
                        receiver_name="world.public_wealth",
                        amount=household.wealth,
                        reason="empty_household_to_public_wealth",
                    )

                    self.world.record_household_lifecycle_event(
                        household_id=household.id,
                        event_type="empty_household_cleanup",
                        wealth_before=wealth_before,
                        wealth_after=0.0,
                        wealth_to_public=wealth_before,
                    )
                    self.world.record_lifecycle_transfer_event(
                        event_type="empty_household_to_public",
                        source_household_id=household.id,
                        amount=wealth_before,
                        source_wealth_before=wealth_before,
                        source_wealth_after=household.wealth,
                        public_sector_amount=wealth_before,
                        source_account=f"household.{household.id}.wealth",
                        destination_account="world.public_wealth",
                    )
                elif abs(household.wealth) <= tolerance:
                    self.world.record_household_lifecycle_event(
                        household_id=household.id,
                        event_type="empty_household_cleanup",
                        wealth_before=wealth_before,
                        wealth_after=0.0,
                    )
                else:
                    # Retain the empty household as the owner of its signed
                    # balance. It is excluded from active households, but its
                    # balance must remain visible until a real settlement path
                    # exists; deletion here would change located money.
                    alive_households.append(household)
                    self.world.record_household_lifecycle_event(
                        household_id=household.id,
                        event_type="empty_household_retained_signed_balance",
                        wealth_before=wealth_before,
                        wealth_after=wealth_before,
                    )
                    continue

                if household.id in self.world.household_dict:

                    del self.world.household_dict[
                        household.id
                    ]


        self.world.households = alive_households



    # =================================================
    # Main update
    # =================================================

    def step(self):


        self.process_adulthood()

        self.remove_dead_members()

        self.remove_empty_households()
