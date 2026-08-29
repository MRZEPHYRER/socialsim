# productivity.py

import math
from functools import lru_cache


@lru_cache(maxsize=None)
def age_productivity(age):

    if age < 18:

        return 0


    A = 1.5

    mu = 40

    sigma = 15


    value = A * math.exp(
        -((age-mu)**2)
        /
        (2*sigma**2)
    )


    return value
