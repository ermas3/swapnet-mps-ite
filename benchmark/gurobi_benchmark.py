import csv
import json
from pathlib import Path
from typing import Dict, Tuple
import gurobipy as gp


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent / "data"
RESULT_ROOT = BASE_DIR / "gurobi_results"


def load_qubo_instance(instance_path: Path) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Load the QUBO data from disk."""
    with instance_path.open("r") as f:
        data = json.load(f)
    return data["h"], data["J"], data.get("c", 0.0)


def parse_edge_key(key: str) -> Tuple[str, str]:
    """Split a JSON key of the form 'i,j' into its endpoints."""
    i, j = key.split(",")
    return i.strip(), j.strip()


def build_qubo_model(
    h: Dict[str, float],
    J: Dict[str, float],
    time_limit: float,
):
    """Create a Gurobi model for the provided QUBO data."""
    model = gp.Model("QUBO_Model")
    model.Params.TimeLimit = time_limit

    variables = model.addVars(list(h.keys()), vtype=gp.GRB.BINARY, name="x")

    linear_term = gp.quicksum(h[idx] * variables[idx] for idx in h)
    quadratic_term = gp.quicksum(
        weight * variables[i] * variables[j]
        for key, weight in J.items()
        for i, j in (parse_edge_key(key),)
    )
    model.setObjective(linear_term + quadratic_term, gp.GRB.MINIMIZE)

    return model, variables


def solve_qubo_instance(
    h: Dict[str, float],
    J: Dict[str, float],
    offset: float,
    time_limit: float,
):
    """Solve a single QUBO instance and return metadata about the run."""
    model, variables = build_qubo_model(h, J, time_limit)
    model.optimize()

    status = model.Status
    runtime = model.Runtime
    cost = None
    solution = None

    if model.SolCount:
        solution = [int(round(variables[idx].X)) for idx in h]
        cost = model.ObjVal + offset

    mip_gap = None
    best_bound = None
    try:
        mip_gap = model.MIPGap
    except gp.GurobiError:
        pass
    try:
        best_bound = model.ObjBound
    except gp.GurobiError:
        pass

    return {
        "status": status,
        "runtime": runtime,
        "cost": cost,
        "solution": solution,
        "mip_gap": mip_gap,
        "best_bound": best_bound,
    }


def main():
    graph_type = "3Reg"
    num_vertices = 100
    time_limit = 60  # seconds
    max_instances = 50

    instance_dir = DATA_ROOT / "MaxCut" / graph_type / f"{num_vertices}v" / "QUBO"
    output_dir = RESULT_ROOT / "MaxCut" / graph_type / f"{num_vertices}v"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(max_instances):
        instance_path = instance_dir / f"qubo_graph{idx}.json"
        if not instance_path.exists():
            print(f"Missing instance: {instance_path}")
            continue

        h, J, offset = load_qubo_instance(instance_path)
        result = solve_qubo_instance(h, J, offset, time_limit)

        cost = result["cost"]
        status = result["status"]
        runtime = result["runtime"]

        if cost is not None:
            print(f"{instance_path.name}: cost={cost:.6f}, status={status}, runtime={runtime:.2f}s")
        else:
            print(f"{instance_path.name}: no feasible solution, status={status}, runtime={runtime:.2f}s")

        rows.append(
            {
                "file": instance_path.name,
                "status": status,
                "runtime_seconds": runtime,
                "best_bound": result["best_bound"] if result["best_bound"] is not None else "",
                "mip_gap": result["mip_gap"] if result["mip_gap"] is not None else "",
                "cost": cost if cost is not None else "",
            }
        )

        print(rows)

    csv_path = output_dir / "gurobi_costs.csv"
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["file", "status", "runtime_seconds", "best_bound", "mip_gap", "cost"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()
