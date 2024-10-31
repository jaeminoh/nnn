from dataclasses import dataclass

from jaxtyping import ArrayLike


@dataclass
class Solution:
    tt: ArrayLike
    reference: ArrayLike
    baseline: ArrayLike
    forecast: ArrayLike
    analysis: ArrayLike
