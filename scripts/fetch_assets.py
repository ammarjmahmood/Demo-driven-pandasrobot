"""Fetch the pinned, licensed Menagerie Panda model (git and network required)."""
import subprocess
from pathlib import Path

REV = "f054586a8e90465d49ee5be15335c4a0c7f57caf"
dest = Path(__file__).resolve().parents[1] / "assets/menagerie"
dest.parent.mkdir(parents=True, exist_ok=True)
if not dest.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--sparse",
                    "https://github.com/google-deepmind/mujoco_menagerie.git", str(dest)], check=True)
subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "franka_emika_panda"], check=True)
subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", REV], check=True)
subprocess.run(["git", "-C", str(dest), "checkout", "--detach", REV], check=True)
print(f"Panda assets ready at {dest}; revision {REV}. See franka_emika_panda/LICENSE.")
