# N3 SDK2 and unified noetix_robots resource integration

## Runtime data locations

The old error `Missing actuator_list calibration in hardware.yaml` was caused
by storing a copy of N3 calibration in SDK2, while the ROS launch used the
`noetix_robots` package installed resource. That duplicate is now removed.

- `noetix_robots/n3/config/robot.yaml` contains the canonical 29 N3
  `joints` entries with `name`, `actuator`, `direction` and `offset`.
- `noetix_robots/n3/config/hardware.yaml` contains `hardware.diagnostic_read_only`,
  the EtherCAT interface/cycle time and the two-CANU topology.
- The SDK2 backend looks up both files under the `RobotConfig.config_root`
  selected by the ROS launch; it does not use build-tree references.
- The existing `robot.yaml` values were checked against the original N3HW
  `actuator_list`; all 29 directions and zero biases matched exactly.

N3Backend resolves all records by actuator name and rejects missing, duplicate,
non-finite, invalid or conflicting calibration. If an old hardware.yaml contains
`actuator_list`, it must agree with `robot.yaml`; never hide a discrepancy by
silently taking one set of values over the other. If robot.yaml does not have a
`joints` mapping, the backend may use legacy hardware actuator_list.

## Deployment on the robot

From `/home/nano/Desktop/new_ws`, back up and replace the two packages.
The combined release ZIP has `noetix_sdk2/` and `noetix_robots/` at its root.
Save the ZIP at a convenient absolute path before using these commands:

```bash
cd /home/nano/Desktop/new_ws
mkdir -p ../noetix_source_backup
# Keep backups outside src/, otherwise colcon can see duplicate packages.
mv src/noetix_sdk2 ../noetix_source_backup/noetix_sdk2
mv src/noetix_robots ../noetix_source_backup/noetix_robots
unzip /path/to/noetix_n3_resource_integrated.zip -d src/
colcon build --packages-up-to noetix_bridge noetix_bringup \
  --cmake-force-configure --cmake-args -DNOETIX_REQUIRE_N3_HW=ON
source install/setup.bash
```

If your actual source packages live deeper under `src/`, adjust the package
paths. Do not edit `install/` YAML directly; those are build outputs. If build
fails, restore the two backups and rebuild.

The `noetix_robots` CMakeLists installs the entire `n3/` directory under
`share/noetix_robots/n3`, so both YAML files are available to the real launch.
Check the installed files instead of editing them manually:

```bash
python3 - <<'PY'
import os, yaml
p='install/noetix_robots/share/noetix_robots/n3/config'
r=yaml.safe_load(open(f'{p}/robot.yaml'))
h=yaml.safe_load(open(f'{p}/hardware.yaml'))
print('joint calibrations:', len(r['joints']))
print('CANU devices:', len(h['canu_network']))
print('read only:', h['hardware']['diagnostic_read_only'])
print('EtherCAT interface:', h['ethercat']['ifname'])
PY
```

The supplied N3 hardware.yaml enables `hardware.diagnostic_read_only: true`
for initial verification. This mode does **not** enable motors and deliberately
ignores motion commands. The robot must be physically supported and E-stop
available even for read-only checks. After confirming IMU, joint feedback,
all signs, ankle kinematics, and emergency stop, explicitly set the source
resource `hardware.diagnostic_read_only: false` and rebuild `noetix_robots`
before attempting a controlled enable. Never toggle it merely to suppress a
startup error.

## What this does not solve

The supplied N3 `robot.yaml` does not define validated `lie_control`,
`stand_control`, `lie_joint_angles` or `stand_joint_angles`, so the separate
`lie=missing stand=missing` message from the pose-mode node can remain.
Do not invent or reuse angles from other robot families on the N3.

The three SDK2 DSOs keep their original names:
`libnoetix_sdk2.so`, `libnoetix_n3_hardware.so`, and
`libnoetix_bumi_hardware.so` (the last two depend on matching vendor archives).
The N3 static vendor archive `libhw_controller.a` is linked into the N3 DSO;
it does not produce `libhw_controller.so`.

## Scope

No N3 parallel-ankle numerical algorithm, CANU topology, Bumi vendor library,
 or public RobotInterface/LowCmd/LowState API was changed. This revision
changes which existing robot YAML supplies the N3 calibration. A complete
ROS 2 build and EtherCAT physical commissioning require the target environment.
