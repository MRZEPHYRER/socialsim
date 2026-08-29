from dataclasses import dataclass


@dataclass(frozen=True)
class GoodSpec:
    id: str
    name: str
    is_essential: bool
    is_storable: bool
    perish_rate: float
    unit: str = "unit"
    physical: bool = True
    service: bool = False
    durability: object = None
    inventory_policy_compatibility: str = "storable"

    @property
    def good_id(self):
        return self.id

    @property
    def storable(self):
        return self.is_storable

    @property
    def perishability(self):
        return self.perish_rate


class GoodsCatalog:

    def __init__(self, goods):
        self._goods = {
            good.id: good
            for good in goods
        }

    def get(self, good_id):
        return self._goods[good_id]

    def all(self):
        return list(self._goods.values())

    def ids(self):
        return list(self._goods.keys())

    def has(self, good_id):
        return good_id in self._goods

    def by_good_id(self):
        return dict(self._goods)
