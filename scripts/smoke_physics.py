"""Small geometry perturbation check, not a statistical policy benchmark."""
import contextlib
import copy
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.simulate import run

task = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/physics_smoke")
out.mkdir(parents=True, exist_ok=True)
rows = []
for seed in range(3):
    sample = copy.deepcopy(task)
    rng = np.random.default_rng(seed)
    for obj in sample["objects"]:
        for key in ["start_m", "goal_m"]:
            obj[key] = (np.array(obj[key]) + rng.uniform(-.01, .01, 2)).tolist()
    sample["augmentation"] = {"kind": "simulation_geometry_perturbation", "seed": seed, "range_m": .01}
    for arms in (1, 2):
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(sample, arms, out / f"seed{seed}_arms{arms}", render=False)
        rows.append({"seed": seed, "arms": arms, "success": result["success"],
                     "max_goal_error_m": max(v["goal_error_m"] for v in result["objects"].values()),
                     "interarm_contacts": len(result["interarm_contact_pairs"]),
                     "table_contacts": len(result["nonbase_robot_table_contact_pairs"])})
print(json.dumps(rows, indent=2))
(out / "summary.json").write_text(json.dumps(rows, indent=2)+"\n")
if not all(r["success"] for r in rows):
    raise SystemExit(1)
