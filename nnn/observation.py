from beartype import beartype as typechecker
from jaxtyping import ArrayLike, Float, jaxtyped

from nnn._cores import ObservationOperator


class UniformSubsample(ObservationOperator):
    def __init__(self, num_spatial_dims: int = 1, sensor_every: int = 1):
        self.num_spatial_dims = num_spatial_dims
        self.sensor_every = sensor_every

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[ArrayLike, "*Nx"]) -> Float[ArrayLike, "*No"]:
        if self.num_spatial_dims == 1:
            return x[:: self.sensor_every]
        elif self.num_spatial_dims == 2:
            return x[:: self.sensor_every, :: self.sensor_every]
