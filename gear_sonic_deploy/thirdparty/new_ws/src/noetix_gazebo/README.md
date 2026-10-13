# noetix_gazebo (ROS 2 Humble + Gazebo Classic 11)

This package adds a Gazebo Classic backend without changing the real-hardware
Bumi backend.  The upper-layer API is intentionally identical:

- command: `/noetix/low_cmd`
- state: `/noetix/low_state`
- standard joints: `/joint_states`
- standard IMU: `/noetix/imu`
- mode request: `/noetix/control_request`

The simulator plugin computes actuator effort using the same MIT-style command
semantics used by the real deployment:

`tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff`

Effort is saturated by each joint's actual URDF `limit.effort`.  Robot joint
coordinates are already logical coordinates in the URDF, so the physical motor
`direction` values from `hardware.yaml` are deliberately NOT applied again.

## Virtual hoist / real-world startup emulation

The robot is spawned with its feet only `spawn_clearance_m` above the floor
(default 0.03 m).  `base_link` is held at the spawn pose while the existing
`noetix_pose_mode_node` executes the configured STAND trajectory.  After STAND
and the short settle period, the hoist is released immediately.  There is no
slow lowering: Gazebo gravity produces the short free fall and contact.

The standing base height is not hard-coded.  `sim.launch.py` computes it from:

- the installed Bumi2 URDF joint origins;
- `stand_joint_angles` from `robot.yaml`;
- the existing spherical foot collision points.

For the supplied Bumi2 model this evaluates to 0.482138 m, therefore the
default 3 cm clearance spawns / holds `base_link` at 0.512138 m.

## Install dependencies (Ubuntu 22.04 / ROS 2 Humble)

```bash
sudo apt update
sudo apt install \
  ros-humble-gazebo-dev \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-joy
```

Gazebo Classic is intentionally used here; this package does not depend on the
new Gazebo / gz-sim stack.

## Build

```bash
cd ~/your_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to noetix_gazebo noetix_policy
source install/setup.bash
```

## Start Gazebo

```bash
ros2 launch noetix_gazebo sim.launch.py robot_name:=bumi2
```

Headless:

```bash
ros2 launch noetix_gazebo sim.launch.py robot_name:=bumi2 gui:=false
```

Expected startup sequence:

1. Gazebo spawns the robot a few centimetres above the floor.
2. Gazebo backend starts `DISABLED`, hoist state is `hold`.
3. startup manager requests `enable`.
4. startup manager requests `stand`.
5. the existing pose-mode node publishes the real STAND LowCmd at its configured rate.
6. after `stand_control.duration_s + stand_settle_s`, startup manager sends `release`.
7. the base is no longer constrained and falls freely under Gazebo gravity.
8. STAND continues holding the final joint targets until another policy takes over.

## Start a policy in simulation

Use the same policy package, but enable simulation time for entry-pose timing:

```bash
ros2 launch noetix_policy policy.launch.py \
  robot_name:=bumi2 \
  use_sim_time:=true \
  trajectory_use_ros_time:=true
```

## Manual hoist control

Release immediately:

```bash
ros2 topic pub --once /noetix/sim/hoist_request \
  std_msgs/msg/String "{data: release}"
```

Freeze the base at its current pose:

```bash
ros2 topic pub --once /noetix/sim/hoist_request \
  std_msgs/msg/String "{data: hold}"
```

Return to the original suspended spawn pose:

```bash
ros2 topic pub --once /noetix/sim/hoist_request \
  std_msgs/msg/String "{data: reset}"
```

## Useful checks

```bash
ros2 topic echo /noetix/sim/hoist_state
ros2 topic echo /noetix/control_state
ros2 topic hz /noetix/low_state
ros2 topic info /noetix/low_cmd --verbose
```

At startup the upright simulated IMU should be close to:

- quaternion: `[1, 0, 0, 0]` in LowState `[w,x,y,z]` order;
- accelerometer: `[0, 0, +9.81]` m/s^2 while stationary;
- policy projected gravity: `[0, 0, -1]`.

## Calibration knobs

`config/bumi2.yaml` contains only simulation/environment values:

- floor friction (`mu1`, `mu2`);
- contact stiffness/damping;
- spawn clearance;
- optional passive joint damping/friction.

Robot geometry, masses, inertia, joint limits and collision geometry continue
to come from the SAME `noetix_robots/bumi2/urdf/bumi2_21dof.urdf` used by the
real deployment.  Do not copy those values into the Gazebo config.
