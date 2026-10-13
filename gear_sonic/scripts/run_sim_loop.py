"""Entry point for MuJoCo: G1 by default, or BUMI3 via --robot-type bumi3.

Selects the robot, parses its configuration via tyro, and launches its simulator.
Robot-specific dependencies are imported only for the selected robot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from gear_sonic.data.robot_model.robot_model import RobotModel
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig


def _parse_robot_type(argv: list[str]) -> tuple[str, list[str]]:
    # Leave --help and robot-specific arguments for that robot's tyro parser.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--robot-type", choices=("g1", "bumi3"), default="g1")
    selection, remaining = parser.parse_known_args(argv)
    return selection.robot_type, remaining


class SimWrapper:
    def __init__(self, robot_model: RobotModel, env_name: str, config: dict[str, Any], **kwargs):
        from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel

        self.robot_model = robot_model
        self.config = config

        init_channel(config=self.config)

        # Create simulator using factory
        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name=env_name,
            **kwargs,
        )


def _run_g1(config: SimLoopConfig):
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory

    wbc_config = config.load_wbc_yaml()
    # NOTE: we will override the interface to local if it is not specified
    wbc_config["ENV_NAME"] = config.env_name

    if config.enable_image_publish:
        assert (
            config.enable_offscreen
        ), "enable_offscreen must be True when enable_image_publish is True"

    robot_model = instantiate_g1_robot_model()

    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=config.env_name,
        config=wbc_config,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=config.enable_image_publish,
    )
    # Start simulator as independent process
    SimulatorFactory.start_simulator(
        sim_wrapper.sim,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )


def _run_bumi3(config):
    from gear_sonic.utils.mujoco_sim.bumi3_ros2_bridge import main as run_bumi3

    run_bumi3(config)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    robot_type, remaining = _parse_robot_type(sys.argv[1:])

    import tyro

    if robot_type == "bumi3":
        from gear_sonic.utils.mujoco_sim.bumi3_ros2_bridge import Args

        _run_bumi3(tyro.cli(Args, args=remaining))
    else:
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

        _run_g1(tyro.cli(SimLoopConfig, args=remaining))


if __name__ == "__main__":
    main()
