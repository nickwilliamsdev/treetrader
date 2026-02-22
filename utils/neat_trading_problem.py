import jax, jax.numpy as jnp
from tensorneat.problem.base import BaseProblem

class TradingProblemBase(BaseProblem):
  
  def __init__(self):
    super.__init__()

  @property
  def input_data_len(self):
    raise NotImplementedError()