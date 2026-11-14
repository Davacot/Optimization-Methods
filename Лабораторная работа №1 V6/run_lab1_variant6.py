from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_generation import generate_variant6_data, save_data
from src.lp_model import build_lp, solve_lp
from src.reporting import (
    hourly_balance_table,
    save_dispatch_plot,
    shadow_price_table,
    solution_table,
)
from src.verification import independent_check


@dataclass
class ScenarioResult:
    scenario: str
    status: str
    objective: float | None
    feasible_check: bool | None
    max_violation: float | None
    note: str


def status_name(status_code: int) -> str:
    mapping = {
        0: "optimal",
        1: "iteration_or_time_limit",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }
    return mapping.get(status_code, f"unknown_status_{status_code}")


def compute_seed_from_n_q(n: int, q: int) -> int:
    return 10000 + 100 * q + n


def apply_scenario(data: dict[str, Any], name: str) -> dict[str, Any]:
    d = json.loads(json.dumps(data))

    if name == "base":
        return d

    if name == "deficit":
        d["daily_energy_limit_mwh"]["Hydro"] *= 0.8
        return d

    if name == "stress":
        d["demand_mwh_by_hour"] = [round(v * 1.12, 3) for v in d["demand_mwh_by_hour"]]
        d["daily_energy_limit_mwh"]["Coal"] *= 0.85
        d["daily_energy_limit_mwh"]["CCGT"] *= 0.85
        d["daily_energy_limit_mwh"]["Hydro"] *= 0.7
        return d

    raise ValueError(f"Unknown scenario: {name}")


def solve_one_scenario(
    scenario_name: str,
    data: dict[str, Any],
    out_dir: Path,
    method: str,
    presolve: bool,
) -> tuple[ScenarioResult, dict[str, Any]]:
    art = build_lp(data)
    res = solve_lp(art, method=method, presolve=presolve)
    named_status = status_name(int(res.status))

    solver_payload: dict[str, Any] = {
        "scenario": scenario_name,
        "status_code": int(res.status),
        "status_name": named_status,
        "status_text": str(res.message),
        "success": bool(res.success),
        "nit": int(res.nit) if hasattr(res, "nit") else None,
        "fun": float(res.fun) if res.success else None,
        "algorithm": method,
        "presolve": presolve,
    }

    if not res.success:
        solver_payload["note"] = "No primal solution returned."
        (out_dir / f"solver_log_{scenario_name}.json").write_text(
            json.dumps(solver_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return (
            ScenarioResult(
                scenario=scenario_name,
                status=named_status,
                objective=None,
                feasible_check=None,
                max_violation=None,
                note=str(res.message),
            ),
            solver_payload,
        )

    x = np.array(res.x, dtype=float)
    check = independent_check(x, art)
    sol_df = solution_table(data, x)
    bal_df = hourly_balance_table(data, sol_df)
    shadow_df = shadow_price_table(data, art, res)
    station_totals = (
        sol_df.groupby("station", as_index=False)["generation_mwh"]
        .sum()
        .rename(columns={"generation_mwh": "total_generation_mwh"})
    )

    sol_df.to_csv(out_dir / f"solution_{scenario_name}.csv", index=False)
    bal_df.to_csv(out_dir / f"hourly_balance_{scenario_name}.csv", index=False)
    shadow_df.to_csv(out_dir / f"shadow_prices_{scenario_name}.csv", index=False)
    station_totals.to_csv(out_dir / f"station_totals_{scenario_name}.csv", index=False)

    solver_payload["independent_check"] = check
    (out_dir / f"solver_log_{scenario_name}.json").write_text(
        json.dumps(solver_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if scenario_name == "base":
        save_dispatch_plot(data, sol_df, out_dir / "dispatch_base.png")

    scenario_result = ScenarioResult(
        scenario=scenario_name,
        status=named_status,
        objective=float(res.fun),
        feasible_check=bool(check["feasible"]),
        max_violation=float(check["max_constraint_violation"]),
        note="ok",
    )
    return scenario_result, solver_payload


def write_summary(
    out_dir: Path,
    seed: int,
    variant: int,
    rows: list[ScenarioResult],
    verification_rows: list[dict[str, Any]],
) -> None:
    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(out_dir / "scenario_results.csv", index=False)
    pd.DataFrame(verification_rows).to_csv(out_dir / "verification_results.csv", index=False)
    summary = {
        "variant": variant,
        "seed": seed,
        "scenarios": [asdict(r) for r in rows],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 1, Variant 6 (LP dispatch).")
    parser.add_argument("--seed", type=int, default=None, help="Explicit deterministic seed.")
    parser.add_argument("--n", type=int, default=1, help="Student index N (starts from 1).")
    parser.add_argument(
        "--q",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Group index Q: 0 (507), 1 (517), 2 (527).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="highs-ds",
        choices=["highs", "highs-ds", "highs-ipm"],
        help="HiGHS algorithm: auto, dual simplex, or interior point.",
    )
    parser.add_argument(
        "--no-presolve",
        action="store_true",
        help="Disable solver presolve phase.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory."
    )
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else compute_seed_from_n_q(args.n, args.q)
    presolve = not args.no_presolve

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    base_data = generate_variant6_data(seed=seed)
    save_data(base_data, data_dir / "variant6_base_data.json")

    results: list[ScenarioResult] = []
    verification_rows: list[dict[str, Any]] = []
    for scenario in ["base", "deficit", "stress"]:
        scenario_data = apply_scenario(base_data, scenario)
        res, payload = solve_one_scenario(
            scenario,
            scenario_data,
            out_dir,
            method=args.method,
            presolve=presolve,
        )
        results.append(res)
        check = payload.get("independent_check")
        if check is not None:
            verification_rows.append(
                {
                    "scenario": scenario,
                    "feasible": check["feasible"],
                    "max_constraint_violation": check["max_constraint_violation"],
                    "objective_recomputed": check["recomputed_objective"],
                    "active_constraints_count": check["active_constraints_count"],
                }
            )

    write_summary(
        out_dir,
        seed=seed,
        variant=6,
        rows=results,
        verification_rows=verification_rows,
    )
    print("Done. Outputs saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
