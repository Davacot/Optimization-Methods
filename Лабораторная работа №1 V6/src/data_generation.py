from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def generate_variant6_data(seed: int = 10001) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    stations = [
        {"name": "NPP", "cost_per_mwh": 18.0, "hourly_capacity_mw": 300.0},
        {"name": "Coal", "cost_per_mwh": 42.0, "hourly_capacity_mw": 260.0},
        {"name": "CCGT", "cost_per_mwh": 36.0, "hourly_capacity_mw": 220.0},
        {"name": "Hydro", "cost_per_mwh": 24.0, "hourly_capacity_mw": 120.0},
        {"name": "Peaker", "cost_per_mwh": 78.0, "hourly_capacity_mw": 180.0},
    ]

    hours = list(range(1, 25))
    base_profile = np.array(
        [
            430, 410, 395, 385, 380, 395, 430, 470, 520, 560, 590, 610,
            620, 615, 600, 590, 605, 640, 700, 735, 710, 655, 570, 500,
        ],
        dtype=float,
    )
    noise = rng.normal(loc=0.0, scale=8.0, size=24)
    demand = np.maximum(base_profile + noise, 320.0)
    demand = np.round(demand, 2).tolist()

    daily_limits = {
        "NPP": 24.0 * 300.0,
        "Coal": 24.0 * 210.0,
        "CCGT": 24.0 * 165.0,
        "Hydro": 24.0 * 80.0,
        "Peaker": 24.0 * 65.0,
    }

    return {
        "variant": 6,
        "seed": seed,
        "units": {
            "power": "MW",
            "energy": "MWh",
            "cost": "currency_units",
        },
        "hours": hours,
        "stations": stations,
        "demand_mwh_by_hour": demand,
        "daily_energy_limit_mwh": daily_limits,
    }


def save_data(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_data(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
