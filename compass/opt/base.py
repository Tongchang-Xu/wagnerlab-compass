r"""
Abstract base class for flux balance analysis optimization engines.
In this case, the usage is intended that the solver will be initialized with
a base metabolic model and then solve a problem with some delta of changes.

This can be useful for linear program solvers with limited interfaces for updating
an existing problem (e.g. adding an extra term to a constraint).
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
    """
    Abstract base class for an optimization solver.
    """

    def __init__(self, model: MetabolicModel):
        self.model = model

    @abc.abstractmethod
    def solve(self, delta: LinearProgramDelta) -> Solution:
        """
        Solve the linear program with the delta applied.
        If possible, the problem should be restored to the base state after.
        """
        pass
