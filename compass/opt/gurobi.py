from enum import Enum
from typing import Any
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from compass import utils
from compass.globals import EXCHANGE_LIMIT
from compass.models.MetabolicModel import MetabolicModel
from .base import Optimizer, LinearProgramDelta, Solution

def default_gurobi_config() -> dict[str, Any]:
    """
    Returns the default Gurobi configuration parameters for Compass.
    These defaults are chosen for numerical stability and performance.
    """
    return {
        GRB.Param.OutputFlag: 0,           # Disable all output
        GRB.Param.LogToConsole: 0,         # Disable console output
        GRB.Param.NumericFocus: 3,         # Equivalent to numerical emphasis in CPLEX
        GRB.Param.Presolve: 2,             # 2 means aggressive presolve, 1 for conservative, 0 for off
        GRB.Param.OptimalityTol: 1e-9,     # Default is 1e-6, minimum is 1e-9
        GRB.Param.BarConvTol: 1e-12,       # Default is 1e-8, minimum is 1e-12
        GRB.Param.Threads: 1,              # Set the number of threads to use
        GRB.Param.Method: -1,              # 0: Automatic, 1: Primal Simplex, 2: Dual Simplex, etc.
    }

class GurobiOptimizer(Optimizer):
    """
    Gurobi-based implementation of the Optimizer.
    """

    def __init__(self, model: MetabolicModel, credentials: dict[str, str] = None, logger = None, config: dict[str, Any] = None):
        super().__init__(model)
        self.credentials = credentials

        final_config = default_gurobi_config()
        if config:
            final_config.update(config)
        self.config = final_config

        self.gp_model = self._build_base_model()

    def _build_base_model(self) -> gp.Model:
        """
        Builds the initial Gurobi model from the provided metabolic model.
        """
        # Gurobi WLS License
        if 'WLSACCESSID' in self.credentials and 'WLSSECRET' in self.credentials and 'LICENSEID' in self.credentials:
            env = gp.Env(params=self.credentials)
        # Gurobi Named-User License
        else:
            env = gp.Env()
        
        gp_model = gp.Model(env=env)

        # Set Parameters for the Gurobi model
        for k, v in self.config.items():
            gp_model.setParam(k, v)

        # Add variables

        # Define minimum and maximum flux for each reaction
        for x in self.model.reactions.values():
            gp_model.addVar(lb=x.lower_bound, ub=x.upper_bound, name=x.id, vtype=GRB.CONTINUOUS)
        gp_model.update()
        
        # Add constraints

        # Add stoichiometry constraints
        s_mat = self.model.getSMAT()
        for (metab_id, stoichiometry) in s_mat.items():
            # If there is no reaction associated with the given metabolite, then skip
            if len(stoichiometry) == 0:
                continue

            # x[0] is name of reaction
            # x[1] is stoichiometric coefficient of metabolite in reaction x[0]
            expr = gp.LinExpr()
            for [coeff, rxn_id] in stoichiometry:
                expr += coeff * gp_model.getVarByName(rxn_id)
            
            # Each metabolite must obey mass conservation
            gp_model.addConstr(expr == 0, name=metab_id)

        return gp_model

    def solve(self, delta: LinearProgramDelta) -> Solution:
        """
        Applies the delta, solves the model, and returns the solution.
        Reverts changes to the model after solving to maintain state.
        """
        # Store original state to revert later
        # With gurobi, we should always be able to revert the delta
        original_bounds = {}
        added_vars = []

        # TODO: Double check EXCHANGE_LIMIT vs maximum_flux asymmetry
        # Probably due to limited physical uptake rates vs arbitrary secretion
        for (met_id, rxn_id) in delta.added_secretion.items():
            met_id_constr = self.gp_model.getConstrByName(met_id)
            rxn_var = self.gp_model.addVar(lb=0.0, ub=self.model.maximum_flux, name=rxn_id, vtype=GRB.CONTINUOUS)
            self.gp_model.chgCoeff(met_id_constr, rxn_var, -1.0)
            self.gp_model.update()

        for (met_id, rxn_id) in delta.added_secretion.items():
            met_id_constr = self.gp_model.getConstrByName(met_id)
            rxn_var = self.gp_model.addVar(lb=0.0, ub=EXCHANGE_LIMIT, name=rxn_id, vtype=GRB.CONTINUOUS)
            self.gp_model.chgCoeff(met_id_constr, rxn_var, 1.0)
            self.gp_model.update()

        # Close all blocked reactions by setting upper bound to lower bound
        # and store previous bounds so they can be restored later
        for rxn_id in delta.blocked_reactions:
            # TODO: Note that getVarByName is inefficient.
            var = self.gp_model.getVarByName(rxn_id)
            old_ub = var.getAttr(GRB.Attr.UB)
            old_lb = var.getAttr(GRB.Attr.LB)
            original_bounds[rxn_id] = old_ub
            var.setAttr(GRB.Attr.UB, old_lb)
            
        self.gp_model.update()

        # Construct objective linear expression by summing all coefficient * reaction pairs.
        obj_expr = gp.LinExpr()
        for (rxn_id, coeff) in id, coeff in delta.objective:
            rxn_var = self.gp_model.getVarByName(rxn_id)
            obj_expr += coeff * rxn_var

        if delta.sense == "max":
            sense = GRB.MAXIMIZE
        else:
            sense = GRB.MINIMIZE
        self.gp_model.setObjective(obj_expr, sense)

        self.gp_model.update()
        self.gp_model.optimize()

        status = self.gp_model.Status
        obj_value = self.gp_model.ObjVal
        if self.gp_model.Status == GRB.OPTIMAL:
            success = True
        else:
            success = False
            
        # Revert the delta

        # Restore bounds
        for rxn_id, ub in original_bounds.items():
            # TODO: Note that getVarByName is inefficient.
            var = self.gp_model.getVarByName(rxn_id)
            var.setAttr(GRB.Attr.UB, ub)

        # Remove added variables
        # It appears that removing variables also removes them from constraints
        for var in added_vars:
            self.gp_model.remove(var)
        self.gp_model.update()

        return Solution(success=success, status=status, obj_value=obj_value)
