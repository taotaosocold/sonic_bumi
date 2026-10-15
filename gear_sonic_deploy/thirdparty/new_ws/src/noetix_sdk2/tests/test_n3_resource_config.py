#!/usr/bin/env python3
"""Check N3 ROS resource consistency before deploying. Requires PyYAML.

Usage: python3 noetix_sdk2/tests/test_n3_resource_config.py noetix_robots
"""
import argparse
import math
from pathlib import Path
import yaml


def validate(resource_root: Path):
    config = resource_root / "n3" / "config"
    robot = yaml.safe_load((config / "robot.yaml").read_text(encoding="utf-8"))
    hw = yaml.safe_load((config / "hardware.yaml").read_text(encoding="utf-8"))
    order = robot["robot"]["joint_order"]
    joints = robot["joints"]
    assert robot["robot"]["dof"] == len(order) == len(joints) == 29
    assert len(set(order)) == 29, "duplicate names in robot joint_order"
    cal = {}
    for joint in joints:
        name = joint["actuator"]
        assert joint["name"] == name
        assert name not in cal, f"duplicate calibration {name}"
        assert type(joint["direction"]) is int and joint["direction"] in (-1, 1)
        assert isinstance(joint["offset"], (int, float)) and math.isfinite(joint["offset"])
        cal[name] = (joint["direction"], joint["offset"])
    assert set(cal) == set(order), "robot actuator calibration does not match joint_order"

    canus = hw["canu_network"]
    assert len(canus) == 2, "unexpected N3 CANU topology"
    names = []
    ecat_ids = set()
    for canu in canus:
        assert canu["ecat_id"] not in ecat_ids
        ecat_ids.add(canu["ecat_id"])
        for channel in ("channel_1", "channel_2", "channel_3"):
            entries = canu.get(channel, [])
            ids = [entry["can_id"] for entry in entries]
            assert len(ids) == len(set(ids)), f"duplicate CAN ID on {canu['name']}/{channel}"
            names.extend(entry["name"] for entry in entries)
    assert len(names) == 29 and len(set(names)) == 29, "wrong/duplicate topology actuator"
    assert set(names) == set(order), "hardware topology and robot actuator names differ"
    assert "actuator_list" not in hw, "calibration should only be in robot.yaml"
    assert type(hw["hardware"]["diagnostic_read_only"]) is bool
    assert hw["ethercat"]["ifname"], "EtherCAT interface missing"
    print(f"N3 resource validated: {len(joints)} calibrated joints, {len(canus)} CANUs, "
          f"interface={hw['ethercat']['ifname']}, "
          f"read_only={hw['hardware']['diagnostic_read_only']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("resource", type=Path, help="path to noetix_robots package root")
    args = parser.parse_args()
    validate(args.resource)


if __name__ == "__main__":
    main()
