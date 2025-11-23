from enum import Enum
from typing import Any
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from compass.models.MetabolicModel import MetabolicModel
from .base import Optimizer, LinearProgramDelta, Solution


class GurobiOptimizer(Optimizer):
    """
    Gurobi-based implementation of the Optimizer.
    """

    def __init__(self, model: MetabolicModel, credentials: dict[str, str], logger, config: dict[str, Any] = None):
        super().__init__(model)

        GRB.LOG

        # Create the Gurobi model

        # Gurobi WLS License
        if "WLSACCESSID" in credentials and "WLSSECRET" in credentials and "LICENSEID" in credentials:
            env = gp.Env(params=credentials)
        # Gurobi Named-User License
        else:
            env = gp.Env()

        gp_model = gp.Model(env=env)

        # Set config parameters
        if config:
            for k, v in config.items():
                self.solver_model.setParam(k, v)

        # Default to no output if not specified
        if not config or "OutputFlag" not in config:
            self.solver_model.setParam("OutputFlag", 0)

        self.vars = {}
        self._build_base_model()

    def _build_base_model(self):
        """
        Builds the initial Gurobi model from the provided metabolic model.
        """
        # Add variables for each reaction
        for reaction in self.model.reactions:
            v = self.solver_model.addVar(lb=reaction.lower_bound, ub=reaction.upper_bound, name=reaction.id)
            self.vars[reaction.id] = v

        self.solver_model.update()

        # Build constraints for each metabolite
        # We accumulate terms for each metabolite
        met_exprs = {met.id: gp.LinExpr() for met in self.model.metabolites}

        for reaction in self.model.reactions:
            v = self.vars[reaction.id]
            for met, coeff in reaction.metabolites.items():
                if met.id in met_exprs:
                    met_exprs[met.id].addTerms(coeff, v)

        # Add constraints to model
        for met_id, expr in met_exprs.items():
            self.solver_model.addConstr(expr == 0, name=met_id)

        self.solver_model.update()

    def solve(self, delta: LinearProgramDelta) -> Solution:
        """
        Applies the delta, solves the model, and returns the solution.
        Reverts changes to the model after solving to maintain state.
        """
        # Store original state to revert later
        original_bounds = {}
        added_vars = []

        try:
            # 1. Apply Objective
            obj_expr = gp.LinExpr()
            for rxn_id, coeff in delta.objective.items():
                if rxn_id in self.vars:
                    obj_expr.addTerms(coeff, self.vars[rxn_id])

            sense = GRB.MAXIMIZE if delta.sense == "maximize" else GRB.MINIMIZE
            self.solver_model.setObjective(obj_expr, sense)

            # 2. Block reactions
            for rxn_id in delta.blocked_reactions:
                if rxn_id in self.vars:
                    v = self.vars[rxn_id]
                    original_bounds[rxn_id] = (v.LB, v.UB)
                    v.LB = 0.0
                    v.UB = 0.0

            # 3. Add secretion reactions (Metabolite -> Reaction ID)
            # Secretion: -> M (coeff +1)
            for met_id, rxn_id in delta.added_secretion.items():
                v = self.solver_model.addVar(lb=0, ub=1000, name=rxn_id)
                added_vars.append(v)

                constr = self.solver_model.getConstrByName(met_id)
                if constr:
                    self.solver_model.chgCoeff(constr, v, 1.0)

            # 4. Add uptake reactions (Metabolite -> Reaction ID)
            # Uptake: M -> (coeff -1)
            for met_id, rxn_id in delta.added_uptake.items():
                v = self.solver_model.addVar(lb=0, ub=1000, name=rxn_id)
                added_vars.append(v)

                constr = self.solver_model.getConstrByName(met_id)
                if constr:
                    self.solver_model.chgCoeff(constr, v, -1.0)

            self.solver_model.update()

            # Solve
            self.solver_model.optimize()

            # Extract solution
            status = self.solver_model.Status
            if status == GRB.OPTIMAL:
                obj_status = "optimal"
                obj_value = self.solver_model.ObjVal
            elif status == GRB.INFEASIBLE:
                obj_status = "infeasible"
                obj_value = np.nan
            elif status == GRB.UNBOUNDED:
                obj_status = "unbounded"
                obj_value = np.inf if delta.sense == "maximize" else -np.inf
            else:
                obj_status = "error"
                obj_value = np.nan

            return Solution(obj_status=obj_status, obj_value=obj_value)

        finally:
            # Revert changes

            # Restore bounds
            for rxn_id, (lb, ub) in original_bounds.items():
                v = self.vars[rxn_id]
                v.LB = lb
                v.UB = ub

            # Remove added variables
            # Removing variables automatically removes them from constraints
            if added_vars:
                self.solver_model.remove(added_vars)

            self.solver_model.update()
