"""Compose one or two Menagerie Panda robots and free rigid blocks."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PANDA = ROOT / "assets/menagerie/franka_emika_panda"
HOME = [0, -0.35, 0, -2.1, 0, 1.75, 0.785398]


def workspace_to_world(p, workspace):
    # Image left/right -> table y; image top/bottom -> reach distance x.
    return [0.30 + p[1], p[0] - workspace[0] / 2, 0.022]


def make_scene(task, arms):
    path = PANDA / "panda.xml"
    if not path.exists():
        raise FileNotFoundError("Panda assets missing. Run python3 scripts/fetch_assets.py")
    source = ET.parse(path).getroot()
    root = ET.Element("mujoco", model="two_hands")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(PANDA / "assets"), autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", cone="elliptic", iterations="60")
    root.append(deepcopy(source.find("default")))
    for geom in root.findall(".//default//geom"):
        if geom.get("class") is None:
            geom.set("friction", "1.5 0.01 0.001")
    root.append(deepcopy(source.find("asset")))
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="640")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0.3 0 2", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="table", type="plane", size="1.3 1.1 0.02", rgba="0.16 0.20 0.25 1", friction="1 0.01 0.001")
    sections = {tag: ET.SubElement(root, tag) for tag in ["tendon", "equality", "actuator", "contact"]}
    references = {"name", "joint", "joint1", "joint2", "body1", "body2", "tendon", "site", "site1", "site2"}
    for index in range(arms):
        prefix = f"arm{index + 1}_"
        body = deepcopy(source.find("worldbody/body"))
        body.set("pos", f"0 {0 if arms == 1 else (-.28 if index == 0 else .28)} 0")
        hand = next(e for e in body.iter("body") if e.get("name") == "hand")
        ET.SubElement(hand, "site", name="grip", pos="0 0 0.1034", size="0.006", rgba="0 1 0 0")
        chunks = [body]
        for tag, section in sections.items():
            if source.find(tag) is not None:
                for element in source.find(tag):
                    copied = deepcopy(element)
                    section.append(copied)
                    chunks.append(copied)
        geom_index = 0
        for chunk in chunks:
            for element in chunk.iter():
                for attr in references:
                    if attr in element.attrib:
                        element.set(attr, prefix + element.get(attr))
                if element.tag == "geom" and "name" not in element.attrib:
                    element.set("name", f"{prefix}geom{geom_index}")
                    geom_index += 1
        world.append(body)
    for obj in task["objects"]:
        name = obj["id"]
        rgba = "0.85 0.12 0.13 1" if name == "red" else "0.08 0.35 0.9 1"
        start = workspace_to_world(obj["start_m"], task["workspace_m"])
        goal = workspace_to_world(obj["goal_m"], task["workspace_m"])
        body = ET.SubElement(world, "body", name=f"block_{name}", pos=" ".join(map(str, start)))
        ET.SubElement(body, "freejoint", name=f"block_{name}_joint")
        ET.SubElement(body, "geom", name=f"block_{name}_geom", type="box", size="0.02 0.02 0.02", mass="0.045", rgba=rgba, friction="1.5 0.01 0.001", condim="4")
        ET.SubElement(world, "geom", name=f"target_{name}", type="cylinder", pos=f"{goal[0]} {goal[1]} 0.0005", size="0.042 0.0004", rgba=rgba[:-1] + "0.35", contype="0", conaffinity="0")
    return ET.tostring(root, encoding="unicode")
