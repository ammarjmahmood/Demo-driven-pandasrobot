"""Physics-based Panda pick/place using Cartesian waypoints and joint servos.

No object teleportation or grasp welds: blocks move through gripper contact.
Uses privileged simulation state for control and evaluation; not an RGB policy.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np

from .scene import HOME, make_scene, workspace_to_world
from .plan import PHASE_SECONDS, plan, coordination_text


class Robot:
    def __init__(self, model, data, index):
        self.model, self.data = model, data
        self.prefix = f"arm{index+1}_"
        self.joints = np.array([model.joint(self.prefix+f"joint{i}").id for i in range(1, 8)])
        self.qadr = model.jnt_qposadr[self.joints]
        self.dadr = model.jnt_dofadr[self.joints]
        self.actuators = np.array([model.actuator(self.prefix+f"actuator{i}").id for i in range(1, 8)])
        self.gripper = model.actuator(self.prefix+"actuator8").id
        self.site = model.site(self.prefix+"grip").id
        self.scratch = mujoco.MjData(model)
        self.q = np.array(HOME)
        # Orient the finger opening along table x so adjacent hands fit in y.
        self.target_rotation = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def inverse_kinematics(self, position):
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qpos[self.qadr] = self.q
        position = np.array(position)
        for _ in range(150):
            mujoco.mj_forward(self.model, self.scratch)
            current = self.scratch.site_xmat[self.site].reshape(3, 3)
            rot_error = sum(np.cross(current[:, i], self.target_rotation[:, i]) for i in range(3)) / 2
            pos_error = position - self.scratch.site_xpos[self.site]
            error = np.r_[pos_error, rot_error]
            if np.linalg.norm(pos_error) < .0004 and np.linalg.norm(rot_error) < .002:
                break
            mujoco.mj_jacSite(self.model, self.scratch, self.jacp, self.jacr, self.site)
            jac = np.vstack([self.jacp[:, self.dadr], self.jacr[:, self.dadr]])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + .002 * np.eye(6), error)
            self.scratch.qpos[self.qadr] += np.clip(delta, -.15, .15)
            self.scratch.qpos[self.qadr] = np.clip(self.scratch.qpos[self.qadr], self.model.jnt_range[self.joints, 0]+.001, self.model.jnt_range[self.joints, 1]-.001)
        mujoco.mj_forward(self.model, self.scratch)
        residual = np.linalg.norm(position-self.scratch.site_xpos[self.site])
        if residual > .005:
            raise RuntimeError(f"{self.prefix} unreachable waypoint {position.tolist()}, residual={residual:.4f}m")
        self.q = self.scratch.qpos[self.qadr].copy()
        return self.q.copy()


def run(task, arms=1, out=Path("outputs/run"), render=True):
    schedule = plan(task, arms)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    xml = make_scene(task, arms)
    (out / "scene.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    robots = [Robot(model, data, i) for i in range(arms)]
    for robot in robots:
        data.qpos[robot.qadr] = HOME
        for j in (1, 2):
            data.qpos[model.joint(robot.prefix+f"finger_joint{j}").qposadr] = .04
        data.ctrl[robot.actuators] = HOME
        data.ctrl[robot.gripper] = 255
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 640, 960) if render else None
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.35, 0, .18]
    camera.distance, camera.azimuth, camera.elevation = 1.75, 145, -42
    writer = cv2.VideoWriter(str(out / "rollout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (960, 640)) if render else None
    if writer and not writer.isOpened():
        raise RuntimeError("Video writer unavailable")
    phase = "initialize"
    events = [{"time_s": 0., "phase": "coordination", "message": coordination_text(schedule)}]
    steps = 0
    max_heights = {o["id"]: 0. for o in task["objects"]}
    interarm_contacts, table_contacts = set(), set()
    joint_dofs = np.concatenate([r.dadr for r in robots])

    def step():
        nonlocal steps
        data.qfrc_applied[joint_dofs] = data.qfrc_bias[joint_dofs]
        mujoco.mj_step(model, data)
        for name in max_heights:
            max_heights[name] = max(max_heights[name], float(data.body("block_"+name).xpos[2]))
        for c in data.contact:
            a, b = model.geom(int(c.geom1)).name, model.geom(int(c.geom2)).name
            if (a.startswith("arm1_") and b.startswith("arm2_")) or (a.startswith("arm2_") and b.startswith("arm1_")):
                interarm_contacts.add((a, b))
            if (a == "table" and "arm" in b) or (b == "table" and "arm" in a):
                body = model.geom(int(c.geom2 if a == "table" else c.geom1)).bodyid[0]
                # Robot bases rest on the table by construction.
                if "link0" not in model.body(int(body)).name:
                    table_contacts.add((a, b))
        if render and steps % 20 == 0:
            renderer.update_scene(data, camera=camera)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (0, 0), (960, 68), (20, 24, 30), -1)
            source = task.get("source", {})
            kind = source.get("kind", "synthetic") if isinstance(source, dict) else "synthetic"
            cv2.putText(frame, f"{arms} PANDA ARM{'S' if arms > 1 else ''} | {phase} | t={data.time:.1f}s", (22, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 245, 250), 1)
            cv2.putText(frame, f"{kind.upper()} VIDEO INPUT | scripted controller / physical grasps", (22, 53), cv2.FONT_HERSHEY_SIMPLEX, .48, (170, 190, 205), 1)
            writer.write(frame)
        steps += 1

    def hold(seconds):
        for _ in range(round(seconds/model.opt.timestep)):
            step()

    def move(targets, label, seconds=1.2):
        nonlocal phase
        phase = label
        events.append({"time_s": round(float(data.time), 3), "phase": label,
                       "arms": [r.prefix for r, _ in targets]})
        initial = [data.ctrl[r.actuators].copy() for r, _ in targets]
        goals = [r.inverse_kinematics(p) for r, p in targets]
        count = round(seconds/model.opt.timestep)
        for tick in range(count):
            alpha = (tick+1)/count
            alpha = alpha*alpha*(3-2*alpha)
            for (robot, _), start, goal in zip(targets, initial, goals):
                data.ctrl[robot.actuators] = start + alpha*(goal-start)
            step()

    def execute(assignments):
        nonlocal phase
        starts = {name: data.body("block_"+name).xpos.copy() for _, name in assignments}
        objects = {o["id"]: o for o in task["objects"]}
        goals = {name: np.array(workspace_to_world(objects[name]["goal_m"], task["workspace_m"])) for _, name in assignments}
        def targets(points, height):
            return [(r, [points[n][0], points[n][1], height]) for r, n in assignments]
        move(targets(starts, .20), "approach", PHASE_SECONDS["approach"])
        move(targets(starts, .023), "descend", PHASE_SECONDS["descend"])
        phase = "close gripper"
        for r, _ in assignments:
            data.ctrl[r.gripper] = 0
        hold(PHASE_SECONDS["close"])
        move(targets(starts, .19), "lift", PHASE_SECONDS["lift"])
        move(targets(goals, .19), "transport", PHASE_SECONDS["transport"])
        move(targets(goals, .025), "lower", PHASE_SECONDS["lower"])
        phase = "release"
        for r, _ in assignments:
            data.ctrl[r.gripper] = 255
        hold(PHASE_SECONDS["release"])
        move(targets(goals, .20), "retreat", PHASE_SECONDS["retreat"])
        hold(PHASE_SECONDS["settle"])

    try:
        hold(.4)
        assignments = [(robots[int(s["arm"].split("_")[1])-1], s["object"]) for s in schedule["steps"]]
        if not schedule["parallel"]:
            for assignment in assignments:
                execute([assignment])
        else:
            execute(assignments)
        phase = "complete"
        hold(.4)
        errors = {}
        for obj in task["objects"]:
            p = data.body("block_"+obj["id"]).xpos.copy()
            goal = np.array(workspace_to_world(obj["goal_m"], task["workspace_m"]))
            error = float(np.linalg.norm(p[:2]-goal[:2]))
            errors[obj["id"]] = {"position_m": p.tolist(), "goal_error_m": round(error, 5),
                                   "max_height_m": round(max_heights[obj["id"]], 5),
                                   "success": bool(error < .03 and .01 < p[2] < .045 and max_heights[obj["id"]] > .08)}
        result = {"arms": arms, "simulated_seconds": round(float(data.time), 3), "objects": errors,
                  "placement_success": all(o["success"] for o in errors.values()),
                  "success": all(o["success"] for o in errors.values()) and not interarm_contacts and not table_contacts,
                  "interarm_contact_pairs": sorted(interarm_contacts),
                  "nonbase_robot_table_contact_pairs": sorted(table_contacts),
                  "controller": "scripted IK + joint servos; privileged simulation object state",
                  "grasp": "physical friction contacts; no welds or object pose resets",
                  "source": task.get("source"), "schedule": schedule, "events": events}
        (out / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        if render:
            renderer.update_scene(data, camera=camera)
            cv2.imwrite(str(out / "final.jpg"), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        print(json.dumps(result, indent=2))
        return result
    finally:
        if writer:
            writer.release()
        if renderer:
            renderer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", type=Path)
    p.add_argument("--arms", type=int, choices=[1, 2], default=1)
    p.add_argument("--out", type=Path, default=Path("outputs/run"))
    p.add_argument("--no-render", action="store_true")
    a = p.parse_args()
    run(json.loads(a.task.read_text()), a.arms, a.out, not a.no_render)


if __name__ == "__main__":
    main()
