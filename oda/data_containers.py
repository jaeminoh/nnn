from dataclasses import dataclass

import numpy as np
from jaxtyping import ArrayLike


@dataclass
class Solution:
    tt: ArrayLike
    reference: ArrayLike
    baseline: ArrayLike
    forecast: ArrayLike
    analysis: ArrayLike
    observation: ArrayLike = None

    def save(self, fname: str):
        np.savez(
            f"results/{fname}",
            tt=self.tt,
            uu=self.reference,
            uu_f=self.baseline,
            uu_a=self.forecast,
            yy=self.observation,
        )
