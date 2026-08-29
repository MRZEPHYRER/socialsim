class Household:


    def __init__(self, household_id, world):

        self.world = world

        self.id = household_id

        self.wealth = 0.0
        # Passive ownership accounting view.  This does not alter the legacy
        # meaning of ``wealth`` as household cash wealth.
        self.equity_asset_value = 0.0
        self.equity_purchase_cash_outflow_this_step = 0.0

        self.income_this_step = 0.0

        self.consumption_this_step = 0.0
        
        self.saving_this_step = 0.0

        # Step17.D active-policy accounting fields.
        self.payg_contribution_this_step = 0.0
        self.pension_income_this_step = 0.0
        self.wealth_transfer_paid_this_step = 0.0
        self.wealth_transfer_received_this_step = 0.0
        self.intergenerational_transfer_income_this_step = 0.0
        self.personal_income_tax_this_step = 0.0
        self.disposable_income_this_step = 0.0

        # 鐖舵瘝
        self.parents = []

        # 瀛愬コ
        self.children = []

    # -------------------------
    # add parent
    # -------------------------

    def add_parent(self, person_id):

        if person_id not in self.parents:

            self.parents.append(person_id)

    # -------------------------
    # add child
    # -------------------------

    def add_child(self, person_id):

        if person_id not in self.children:

            self.children.append(person_id)

    # -------------------------
    # remove child
    # -------------------------

    def remove_child(self, person_id):

        if person_id in self.children:

            self.children.remove(person_id)

    # -------------------------
    # household size
    # -------------------------

    def size(self):

        return len(self.parents) + len(self.children)
