import random
from economy.ledger import object_account


class MarriageSystem:


    def __init__(self, world):

        self.world = world
        self.marriage_rng = random.Random(
            f"{getattr(world, 'seed', None)}:marriage"
            if getattr(world, "seed", None) is not None
            else None
        )
        self.marriage_rng_schema = "dedicated_marriage_rng_v1"
        self.legacy_marriage_rng_migration = False

    def __setstate__(self, state):
        state = dict(state)
        self.__dict__.update(state)
        if not hasattr(self, "marriage_rng"):
            world = self.world
            self.marriage_rng = random.Random(
                f"{getattr(world, 'seed', None)}:marriage"
                if getattr(world, "seed", None) is not None
                else None
            )
            self.legacy_marriage_rng_migration = True
        if not hasattr(self, "marriage_rng_schema"):
            self.marriage_rng_schema = "dedicated_marriage_rng_v1"



    # =========================
    # Get marriage pool
    # =========================

    def get_marriage_pool(self):


        males=[]

        females=[]


        for person in self.world.population:


            if (

                person.is_seeking_partner

                and

                person.age >= 20

                and

                person.partner_id is None

                and

                person.alive

            ):


                if person.sex=="M":

                    males.append(person)


                else:

                    females.append(person)



        return males,females



    # =========================
    # Process marriage
    # =========================

    def process_marriage(self):


        males,females = self.get_marriage_pool()


        self.marriage_rng.shuffle(males)


        female_pool = self.build_female_age_pool(
            females
        )


        used_female=set()



        marriages=0



        for male in males:


            candidates=[]


            for age in range(
                int(male.age)-8,
                int(male.age)+9
            ):


                candidates.extend(

                    female_pool.get(
                        age,
                        []
                    )

                )



            candidates=[

                f for f in candidates

                if f.id not in used_female

            ]



            if len(candidates)==0:

                continue



            female=self.marriage_rng.choice(
                candidates
            )

            old_household_1 = male.household_id
            old_household_2 = female.household_id
            # These amounts are the only balances held in the pending
            # household-formation account: cash released from prior Social
            # Households.  Temporary settlement-account cash follows its own
            # direct source-to-destination transfer below.
            male_pending_wealth = self.world.leave_household(male)

            female_pending_wealth = self.world.leave_household(female)



            household = self.world.create_household()

            male_settlement_cash = self.world.merge_settlement_household_into_social_household(
                male,
                household,
            )
            female_settlement_cash = self.world.merge_settlement_household_into_social_household(
                female,
                household,
            )

            pending_formation_transfer = male_pending_wealth + female_pending_wealth
            settlement_transfer = male_settlement_cash + female_settlement_cash
            total_formation_wealth = pending_formation_transfer + settlement_transfer

            pending_before = self.world.pending_household_formation_wealth
            destination_before = household.wealth

            self.world.transfer_signed_balance(
                source_obj=self.world,
                source_attr="pending_household_formation_wealth",
                source_name="world.pending_household_formation_wealth",
                destination_obj=household,
                destination_attr="wealth",
                destination_name=f"household.{household.id}.wealth",
                amount=pending_formation_transfer,
                reason="new_household_formation_wealth_transfer",
            )
            self.world.record_lifecycle_transfer_event(
                event_type="pending_formation_to_new_household",
                destination_household_id=household.id,
                amount=pending_formation_transfer,
                source_wealth_before=pending_before,
                source_wealth_after=self.world.pending_household_formation_wealth,
                destination_wealth_before=destination_before,
                destination_wealth_after=household.wealth,
                source_account="world.pending_household_formation_wealth",
                destination_account=f"household.{household.id}.wealth",
            )
            self.world.record_household_lifecycle_event(
                household_id=household.id,
                event_type="new_household_formation",
                wealth_before=0.0,
                wealth_after=household.wealth,
                wealth_transferred_in=total_formation_wealth,
            )



            self.world.add_person_to_household(
                male,
                household,
                "parent"
            )


            self.world.add_person_to_household(
                female,
                household,
                "parent"
            )



            male.partner_id=female.id

            female.partner_id=male.id


            male.is_seeking_partner=False

            female.is_seeking_partner=False

            self.world.record_demographic_event(
                "marriage",
                person_1_id=male.id,
                person_2_id=female.id,
                person_1_age_years=male.age_years,
                person_2_age_years=female.age_years,
                old_household_1=old_household_1,
                old_household_2=old_household_2,
                new_household_id=household.id,
                wealth_transferred=total_formation_wealth,
            )


            used_female.add(
                female.id
            )


            marriages+=1



        return marriages



    # =========================
    # Female age index
    # =========================

    def build_female_age_pool(
        self,
        females
    ):


        pool={}


        for female in females:


            age = int(female.age)

            if age not in pool:

                pool[age]=[]


            pool[age].append(
                female
            )


        return pool
