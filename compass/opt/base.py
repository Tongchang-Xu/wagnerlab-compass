"""
Abstract class for the optimization model.
Let's list the operations we need for the model/solver:
    1. Configure the solver's parameters
    2. Add variables for each reaction
        2.a - block certain uptake/secretion reactions
        2.b - add secretion/uptake reactions for missing reactions
        2.c - set certain reactions to v_r^opt
    3. Add constraints for each metabolite
    4. Define objective coefficients and 

Alternatively:
    1. Maximize reaction
    2. Maximize metabolite
    3. Minimize penalty subject to v_r^opt

Alternatively:
    1. Add variable
    2. Add constraint
    3. Change constraint
    4. Change upper/lower bound
    5. Update objective
"""
from dataclasses import dataclass, field
import abc
import numpy as np

from compass.models.MetabolicModel import MetabolicModel

@dataclass
class LinearProgramDelta:
    """
    Track deltas to the GSMM-derived linear program. 
    The base problem is the simple Sv = 0, indicating stoichiometric balance.
    """
    # Reaction ID -> coefficient
    objective: dict
    # Sense of optimization: max or min
    sense: str
    # Metabolite ID -> reaction ID
    added_secretion: dict = field(default_factory=dict)
    # Metabolite ID -> reaction ID
    added_uptake: dict = field(default_factory=dict)
    # Reaction IDs
    blocked_reactions: set = field(default_factory=set)

@dataclass
class Solution:
    """Solution from a linear program"""
    obj_status: str
    obj_value: np.float64

class Optimizer(abc.ABC):
    """Abstract base class for an optimization solver."""

    def __init__(self, model: MetabolicModel, config: dict):
        self.model = model

    @abc.abstractmethod
    def solve(self, delta: LinearProgramDelta) -> Solution:
        """Solve the linear program with the delta applied."""
        pass