from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linprog


@dataclass
class LPArtifacts:
    c: np.ndarray
    a_ub: np.ndarray
    b_ub: np.ndarray
    bounds: list[tuple[float, float]]
    var_index: list[tuple[int, int]]
    demand_rows: list[int]
    capacity_rows: list[int]
    daily_rows: list[int]


def build_lp(data: dict[str, Any]) -> LPArtifacts:
    stations = data["stations"]
    demand = np.array(data["demand_mwh_by_hour"], dtype=float)
    hours = data["hours"]
    n_s = len(stations)
    n_t = len(hours)
    n_vars = n_s * n_t

    c = np.zeros(n_vars, dtype=float)
    var_index: list[tuple[int, int]] = []
    for i, st in enumerate(stations):
        for t, _ in enumerate(hours):
            idx = i * n_t + t
            c[idx] = float(st["cost_per_mwh"])
            var_index.append((i, t))

    a_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    demand_rows: list[int] = []
    capacity_rows: list[int] = []
    daily_rows: list[int] = []

    for t in range(n_t):
        row = np.zeros(n_vars, dtype=float)
        for i in range(n_s):
            row[i * n_t + t] = -1.0
        demand_rows.append(len(a_rows))
        a_rows.append(row)
        b_rows.append(-demand[t])

    for i, st in enumerate(stations):
        cap = float(st["hourly_capacity_mw"])
        for t in range(n_t):
            row = np.zeros(n_vars, dtype=float)
            row[i * n_t + t] = 1.0
            capacity_rows.append(len(a_rows))
            a_rows.append(row)
            b_rows.append(cap)

    for i, st in enumerate(stations):
        row = np.zeros(n_vars, dtype=float)
        for t in range(n_t):
            row[i * n_t + t] = 1.0
        daily_rows.append(len(a_rows))
        a_rows.append(row)
        b_rows.append(float(data["daily_energy_limit_mwh"][st["name"]]))

    bounds = [(0.0, None)] * n_vars
    a_ub = np.vstack(a_rows)
    b_ub = np.array(b_rows, dtype=float)

    return LPArtifacts(
        c=c,
        a_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        var_index=var_index,
        demand_rows=demand_rows,
        capacity_rows=capacity_rows,
        daily_rows=daily_rows,
    )


def solve_lp(
    art: LPArtifacts,
    method: str = "highs-ds",
    presolve: bool = True,
) -> Any:
    return linprog(
        c=art.c,
        A_ub=art.a_ub,
        b_ub=art.b_ub,
        bounds=art.bounds,
        method=method,
        options={"presolve": presolve},
    )
