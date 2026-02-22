import jax, jax.numpy as jnp
from tensorneat.problem.base import BaseProblem

class TradingProblemBase(BaseProblem):
  jitable = True

  def __init__(
      self,
      epoch_len,
      pop_size,
  ):
    super().__init__()
    self.epoch_len = epoch_len
    self.pop_size = pop_size

  @property
  def input_data_len(self):
    raise NotImplementedError()