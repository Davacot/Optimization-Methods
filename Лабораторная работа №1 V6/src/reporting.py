from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lp_model import LPArtifacts


def solution_table(
    data: dict[str, Any],
    x: np.ndarray,
) -> pd.DataFrame:
    stations = [s["name"] for s in data["stations"]]
    n_t = len(data["hours"])
    rows = []
    for i, st in enumerate(stations):
        for t, hour in enumerate(data["hours"]):
            rows.append(
                {
                    "hour": hour,
                    "station": st,
                    "generation_mwh": round(float(x[i * n_t + t]), 6),
                }
            )
    return pd.DataFrame(rows)


def hourly_balance_table(data: dict[str, Any], sol_df: pd.DataFrame) -> pd.DataFrame:
    gen_hourly = sol_df.groupby("hour", as_index=False)["generation_mwh"].sum()
    demand = pd.DataFrame(
        {"hour": data["hours"], "demand_mwh": data["demand_mwh_by_hour"]}
    )
    merged = demand.merge(gen_hourly, on="hour", how="left").fillna(0.0)
    merged["balance_mwh"] = merged["generation_mwh"] - merged["demand_mwh"]
    return merged


def shadow_price_table(
    data: dict[str, Any],
    art: LPArtifacts,
    result: Any,
    tol: float = 1e-7,
) -> pd.DataFrame:
    marginals = np.array(result.ineqlin.marginals, dtype=float)
    slacks = np.array(result.ineqlin.residual, dtype=float)

    out = []
    n_t = len(data["hours"])

    for local_t, row_idx in enumerate(art.demand_rows):
        out.append(
            {
                "constraint_type": "demand",
                "name": f"hour_{local_t+1}",
                "is_active": bool(abs(slacks[row_idx]) <= tol),
                "shadow_price_obj_per_unit": round(-float(marginals[row_idx]), 6),
            }
        )

    station_names = [s["name"] for s in data["stations"]]
    cap_rows = art.capacity_rows
    for i, st in enumerate(station_names):
        for t in range(n_t):
            row_idx = cap_rows[i * n_t + t]
            out.append(
                {
                    "constraint_type": "capacity",
                    "name": f"{st}_h{t+1}",
                    "is_active": bool(abs(slacks[row_idx]) <= tol),
                    "shadow_price_obj_per_unit": round(-float(marginals[row_idx]), 6),
                }
            )

    for i, st in enumerate(station_names):
        row_idx = art.daily_rows[i]
        out.append(
            {
                "constraint_type": "daily_limit",
                "name": st,
                "is_active": bool(abs(slacks[row_idx]) <= tol),
                "shadow_price_obj_per_unit": round(-float(marginals[row_idx]), 6),
            }
        )

    return pd.DataFrame(out)


def save_dispatch_plot(data: dict[str, Any], sol_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pivot = sol_df.pivot_table(
        index="hour", columns="station", values="generation_mwh", aggfunc="sum"
    ).sort_index()
    demand = np.array(data["demand_mwh_by_hour"], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(
        pivot.index.to_numpy(),
        [pivot[c].to_numpy() for c in pivot.columns],
        labels=list(pivot.columns),
        alpha=0.85,
    )
    ax.plot(pivot.index.to_numpy(), demand, color="black", linewidth=1.8, label="Demand")
    ax.set_title("Variant 6: Optimal Dispatch by Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MWh")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
