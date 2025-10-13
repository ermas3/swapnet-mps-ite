import json
import os
import subprocess
import re
import csv

# ---- PATH CONFIGURATION ----
graph_type = "ER"
n_vertices = 100

biq_path = os.path.join("..", "BiqCrunch", "problems", "max-cut", "biqcrunch")
param_path = os.path.join("..", "BiqCrunch", "problems", "max-cut", "biq_crunch.param")
converter_path = os.path.join("..", "BiqCrunch", "tools", "mc2bc.py")

dir_path = f"../data/MaxCut/{graph_type}/{n_vertices}v/graphs"
save_path = f"../benchmark/BiqCrunch/results/MaxCut/{graph_type}/graphs"
os.makedirs(save_path, exist_ok=True)

# ---- CONVERT JSON → .mc ----
def json_to_mc(json_path, mc_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    N = data["N"]
    edges = data["edges"]
    with open(mc_path, "w") as f:
        f.write(f"{N} {len(edges)}\n")
        for k, w in edges.items():
            i, j = map(int, k.split(","))
            f.write(f"{i+1} {j+1} {w}\n")

# ---- RUN BIQCRUNCH ----
def convert_mc_to_bc(mc_path):
    """Convert .mc → .bc"""
    bc_path = mc_path.replace(".mc", ".bc")

    # Convert to .bc format
    with open(bc_path, "w") as f:
        subprocess.run(["python3", converter_path, mc_path], stdout=f, check=True)

    return

def run_biqcrunch(bc_path):
    """Run BiqCrunch on .bc file and return the output."""
    result = subprocess.run([biq_path, bc_path, param_path], capture_output=True, text=True)
    return result.stdout

def parse_output(path):
    with open(path, "r") as f:
        text = f.read()

    # Try to match both numbers
    match_val = re.search(r"Maximum value\s*=\s*([-\d.]+)", text)
    match_bound = re.search(r"Root node bound\s*=\s*([-\d.]+)", text)

    value = float(match_val.group(1)) if match_val else None
    bound = float(match_bound.group(1)) if match_bound else None

    gap = None
    if value is not None and bound is not None:
        gap = 100 * (bound - value) / abs(value)

    return value, bound, gap

# ---- MAIN LOOP ----
max_idx = 20
for idx in range(max_idx):
    json_path = os.path.join(dir_path, f"graph{idx}.json")
    mc_path = os.path.join(save_path, f"graph{idx}.mc")

    if not os.path.exists(json_path):
        print(f"Missing: {json_path}")
        continue

    json_to_mc(json_path, mc_path)
    print(f"Running BiqCrunch on {mc_path}...")
    convert_mc_to_bc(mc_path)

    bc_path = mc_path.replace(".mc", ".bc")
    run_biqcrunch(bc_path)

    output_path = bc_path.replace(".bc", ".bc.output")
    value, bound, gap = parse_output(output_path)
    print(f"Value: {value}, Bound: {bound}, Gap: {gap}%")

# ---- COLLECT RESULTS ----
csv_path = os.path.join(save_path, "biqcrunch_results.csv")
rows = []
for idx in range(max_idx):
    fname = f"graph{idx}.bc.output"
    fpath = os.path.join(save_path, fname)
    if not os.path.exists(fpath):
        print(f"Missing: {fpath}")
        continue

    value, bound, gap = parse_output(fpath)
    rows.append({
            "file": fname,
            "cost": value,
            "bound": bound,
            "gap_percent": gap
        })
    print(f"Parsed {fname}: cost={value}, bound={bound}, gap={gap:.3f}%")

# Write CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "cost", "bound", "gap_percent"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved results to {csv_path}")