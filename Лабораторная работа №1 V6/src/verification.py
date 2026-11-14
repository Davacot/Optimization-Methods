from __future__ import annotations

from typing import Any

import numpy as np

from .lp_model import LPArtifacts


def independent_check(
    x: np.ndarray, art: LPArtifacts, tol: float = 1e-7
) -> dict[str, Any]:
    lhs = art.a_ub @ x
    violation = lhs - art.b_ub
    max_violation = float(np.max(violation))
    feasible = bool(np.all(violation <= tol) and np.all(x >= -tol))
    objective = float(art.c @ x)
    active_count = int(np.sum(np.isclose(violation, 0.0, atol=1e-7)))

    return {
        "feasible": feasible,
        "max_constraint_violation": max_violation,
        "recomputed_objective": objective,
        "active_constraints_count": active_count,
    }
