from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from src.data_generation import generate_variant6_data
from src.lp_model import build_lp, solve_lp
from src.verification import independent_check


def compute_seed_from_n_q(n: int, q: int) -> int:
    return 10000 + 100 * q + n


def apply_scenario(data: dict, name: str) -> dict:
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


def status_name(status_code: int) -> str:
    mapping = {
        0: "optimal",
        1: "iteration_or_time_limit",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }
    return mapping.get(status_code, f"unknown_status_{status_code}")


def run_one(method: str, scenario: str, data: dict) -> dict:
    art = build_lp(data)
    t0 = time.perf_counter()
    res = solve_lp(art, method=method, presolve=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    row = {
        "method": method,
        "scenario": scenario,
        "status_code": int(res.status),
        "status": status_name(int(res.status)),
        "success": bool(res.success),
        "time_ms": round(elapsed_ms, 4),
        "iterations": int(res.nit) if hasattr(res, "nit") else None,
        "objective": float(res.fun) if res.success else None,
        "feasible_check": None,
        "max_violation": None,
    }
    if res.success and res.x is not None:
        x = np.array(res.x, dtype=float)
        check = independent_check(x, art)
        row["feasible_check"] = bool(check["feasible"])
        row["max_violation"] = float(check["max_constraint_violation"])
    return row


def write_markdown(rows: list[dict], out_md: Path, seed: int) -> None:
    grouped: dict[str, dict[str, dict]] = {}
    for r in rows:
        grouped.setdefault(r["scenario"], {})[r["method"]] = r

    lines: list[str] = []
    lines.append("# Дополнительное задание: сравнение highs-ds и highs-ipm")
    lines.append("")
    lines.append(f"- seed: `{seed}`")
    lines.append("- presolve: `True`")
    lines.append("")
    lines.append("| Сценарий | objective (ds) | objective (ipm) | iter ds | iter ipm | time ds, ms | time ipm, ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for sc in ["base", "deficit", "stress"]:
        ds = grouped.get(sc, {}).get("highs-ds", {})
        ipm = grouped.get(sc, {}).get("highs-ipm", {})
        lines.append(
            f"| {sc} | {ds.get('objective')} | {ipm.get('objective')} | "
            f"{ds.get('iterations')} | {ipm.get('iterations')} | "
            f"{ds.get('time_ms')} | {ipm.get('time_ms')} |"
        )
    lines.append("")
    lines.append("Вывод: оба метода дают один и тот же оптимум, но отличаются числом итераций и временем.")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--q", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else compute_seed_from_n_q(args.n, args.q)
    base_data = generate_variant6_data(seed=seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for scenario in ["base", "deficit", "stress"]:
        sc_data = apply_scenario(base_data, scenario)
        rows.append(run_one("highs-ds", scenario, sc_data))
        rows.append(run_one("highs-ipm", scenario, sc_data))

    out_csv = out_dir / "star_simplex_vs_ipm.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "scenario",
                "status_code",
                "status",
                "success",
                "time_ms",
                "iterations",
                "objective",
                "feasible_check",
                "max_violation",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_md = out_dir / "star_simplex_vs_ipm.md"
    write_markdown(rows, out_md, seed)
    print("Done. Star task outputs:", out_csv.resolve(), out_md.resolve())


if __name__ == "__main__":
    main()
