# N3 hw integration with noetix_sdk2

## Source of truth

- `n3_hw/legged_n3_hw/src/N3HW.cpp`
- `n3_hw/legged_n3_hw/config/hardware.yaml`
- `n3_hw/legged_n3_hw/src/{FK,IK,Jacobian,coeffs,parallel_ankle}.cpp`
- `n3_hw/legged_n3_hw/hw_controller/lib/libhw_controller.a` (x86_64)

Old N3HW published calibrated joint-state and task-space ankle state. Prior
SDK2 N3 returned raw motor state and sent unmodified joint commands. That was
not compatible with the old N3HW behaviour.

## Changes to the new SDK2 N3 only

1. Check `noetix_robots/n3/config/robot.yaml` `joints` calibration against `RobotConfig.actuator_order` by **name**.
   Every motor MUST provide direction +/-1 and a finite bias, or startup fails.
   Receive: `q_joint = direction*q_motor + bias`, `dq/tau_joint = direction * raw`.
   Send: `q_motor = direction*(q_joint - bias)`, `dq/tau_motor = direction * joint`.
   These changes are done ONCE in the N3 backend. Do not repeat calibration
   in a bridge, policy or `robot.yaml` conversion.
2. Use the original N3HW's Chebyshev FK, polished IK, Jacobian and two parallel
   ankle pairs (left/right), and the original pitch/roll signs. Match joints
   by name, not indexes 21/22/27/28, to support different robot orderings.
   Task-space ankle q/dq/tau is returned through LowState. In LowCmd, the
   ankle task-space kp/kd position error becomes actuator-space torque, while
   the actual ankle MIT q/dq/kp/kd are set to 0. The original code ignores
   ankle task-space `tau_ff` in this conversion, and so does this port.
   If you need torque-only RL control, you must explicitly design/test
   an ankle torque feedforward rule before using it on the robot.
3. The old N3HW clamps parallel ankle torque to +/-50 Nm after 2 seconds.
   This version applies the same +/-50 clamp **from the first enabled frame**
   instead of retaining a 2-second unclamped phase, for safety. Invalid
   ankle FK causes the solver to clear that motor pair's torque.
4. Preserve old N3HW EtherCAT operation order: heartbeat -> decode previous
   PDO -> read joints/IMU -> write MIT (only after enable) -> stage PDO -> run.
5. The vendor `SetMitCmd` may implicitly enable motors even if every MIT
   field equals zero. In disabled/diagnostic mode, NO `SetMitCmd` is called.
   A DISABLE request is staged on startup and every passive cycle. Explicit
   runtime ENABLE is required and checks normal motor feedback and ankle FK.
   A partial feedback loss or malformed command schedules DISABLE.
6. Replaced SDK2 N3 x86_64 static library with the x86_64 archive from the
   old N3HW. The old `hw_controller.h` public method list and N3 SDK2
   header are compatible at source/API level. The aarch64 library stays
   unchanged from SDK2. This archive does not contain robot configs or a
   runtime; it is a library for the entire ROS workspace.

## How to apply in your workspace

1. Unpack `noetix_sdk2/` into the source workspace where this package lives.
2. Install the supplied `noetix_robots` resource package. N3 motor
   direction/offset is read from `noetix_robots/n3/config/robot.yaml` under
   `joints`. EtherCAT, CANU and diagnostics are read from
   `noetix_robots/n3/config/hardware.yaml`. Do not maintain a second N3
   calibration copy under SDK2. Keep the existing `eno2` network interface
   name unless the actual PC wiring differs.
3. Set `diagnostic_read_only: true` for the initial dry run. Use the existing
   bridge/bringup from your full workspace. When feedback, IMU quaternion,
   calibrated motor directions, and both ankle positions are correct, change
   that parameter to `false`, restart and use your normal operator-controlled
   `enableActuators()` switch. Startup remains DISABLED in either mode.
4. Build in the complete ROS 2 workspace on the intended platform:
   `colcon build --packages-up-to noetix_bridge noetix_bringup --cmake-force-configure --cmake-args -DNOETIX_REQUIRE_N3_HW=ON`.
   SDK2 itself is only a package. This upload does **not** include the N3
   robot package, `noetix_bridge` or `noetix_bringup`.
5. Test with the robot securely supported and a physical emergency stop;
   check all 29 directions, limit handling, ankle FK, `ENABLE/DISABLE`,
   changing to read-only mode, and feedback dropout, before ground testing.

## Checks performed here

- Compared the two available N3 hw-controller headers and archive symbol names.
- Verified the original supplied archive is x86_64, and the SDK2 aarch64
  archive is really ARM aarch64.
- Compiled the copied ankle math independently of ROS and Eigen; ran a CPU
  regression for index permutation, left/right roll sign, and invalid angles.
- Syntax-checked the N3 controller/backend with an available compatibility
  YAML header; the deployed distro's `yaml-cpp` and complete ament/ROS build
  were not available in this environment.
- **No real EtherCAT device or N3 robot was used.** Real motion correctness,
  firmware ABI and drive state behaviour must still be verified on hardware.

## Source layout (unified)

```text
noetix_sdk2/
  # Runtime robot configurations are under noetix_robots/, not SDK2.
  src/n3/
    n3_backend.hpp / n3_backend.cpp # N3 hardware backend (private)
    n3_controller.hpp / n3_controller.cpp
  third-party/vendor/parallel_ankle/
    n3/                           # N3 source algorithm, paired .hpp/.cpp
      Types.hpp                   # MotorData (header-only)
      FK.hpp / FK.cpp
      IK.hpp / IK.cpp
      Jacobian.hpp / Jacobian.cpp
      coeffs.hpp / coeffs.cpp
      parallel_ankle.hpp / parallel_ankle.cpp
    bumi/                         # Original Bumi binary vendor package
      include/                    # Bumi vendor headers
      lib/x86_64/libparallel_ankle_cpp.a
      lib/aarch64/libparallel_ankle_cpp.a
```

`legacy_ankle` was a misleading directory name: its code is not optional or
fallback logic. The N3 algorithm is now located under `third-party/vendor/parallel_ankle/n3/`,
independent of Bumi's prebuilt `parallel_ankle/bumi/` library. N3-private headers moved from the exported `include/` directory to
`src/n3/`. The public user API (`RobotInterface`, `LowCmd`, `LowState`, etc.) is
not changed; clients that directly included the former implementation-only
`noetix/hardware/n3/n3_backend.hpp` must stop using that internal header.

This revision only reorganizes vendor ankle source/binary directories and
updates include paths, CMake source lists, regression test build instructions,
and documentation. The actual ankle math and libraries are unchanged. The reference YAML
was simplified by removing old N3HW runtime keys (`publish_frequecy`,
`enable_actuator`, `actuator_debug`, `hw_log`) that SDK2 does not read. Its
EtherCAT topology and calibrated actuator records are unchanged.
