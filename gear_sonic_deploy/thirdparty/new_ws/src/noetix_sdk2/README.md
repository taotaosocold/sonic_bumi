# noetix_sdk2

Core real-hardware runtime shared by N3 and the Bumi family (`bumi3`, `bumi4340`, `bumi2`).

## Runtime design

```text
upper-layer LowCmd
  -> RobotInterface
  -> real hardware backend
  -> EtherCAT / actuator bus
```

The SDK does not own policy arbitration, command freshness, LIN modes, gain
shaping, or fallback holds. The bottom layer only performs hardware I/O and
actuator ENABLE / DISABLE.

## Bumi unified hardware path

`bumi3`, `bumi4340`, and `bumi2` use the same `BumiBackend`, vendor API and
shared network configuration:

```text
noetix_robots/common/bumi/network.yaml
```

Model-specific `hardware.yaml` files contain the vendor robot model, actuator
calibration, parallel ankle mapping and CANU / actuator topology. EtherCAT
interface settings remain in the shared Bumi network file.

Bumi3 is the known-good real-hardware reference. N3 keeps its independent
hardware backend.

## Vendor layout

```text
third-party/vendor/hw_controller/bumi/
  include/
  x86_64/lib/
  aarch64/lib/
```

The build selects the matching binary directory from the target architecture.

Bumi x86_64 uses `hw_controller/bumi/x86_64/lib/libhw_controller.a`,
restored from the supplied `bumi_hw_x86.tar.gz`. Its three vendor headers are
byte-identical to the existing shared `bumi/include/` headers; aarch64 keeps
its separate archive. This x86_64 package does not ship `libsoem.a`, so the
existing CMake logic links the in-tree SOEM build. Reconfigure CMake after
restoring a previously missing archive so the Bumi backend is enabled.


N3 uses the same directory layout, with its own shared headers and archives:

```text
third-party/vendor/hw_controller/n3/
  include/hw_controller.h
  include/common_type.h
  x86_64/lib/libhw_controller.a
  aarch64/lib/libhw_controller.a
```

The N3 headers in `n3/include/` are shared by x86_64 and aarch64. The
`x86_64/lib/libhw_controller.a` archive was replaced with the **old, known-good**
N3HW x86_64 archive supplied in `n3_hw.zip`. The existing SDK2 aarch64 archive
has been preserved; its exact behaviour has **not** been verified against the
old x86_64 binary. The build selects the target architecture automatically.
These are compiled archives, not their original vendor source.

The Bumi archive cannot substitute for the N3 archive: actuator enums,
`CreateCanu`, temperature/error accessors and IMU APIs differ. Keep the
backends and their vendor symbols isolated.

For an N3 deployment, require its backend explicitly:

```bash
colcon build --packages-up-to noetix_bridge noetix_bringup \
  --cmake-force-configure --cmake-args -DNOETIX_REQUIRE_N3_HW=ON
```

This fails at configuration if the N3 library or headers are missing, instead
of installing an SDK that cannot start an N3 robot. The default remains
optional so Bumi deployments do not require an N3 vendor package.

## N3 hardware alignment (2026-10-08)

The new N3 backend preserves the `RobotInterface` / `LowCmd` / `LowState` APIs;
Bumi implementations are unchanged. It now uses the old N3HW's actual motor
calibration and parallel-ankle mathematics. Detailed changes and commissioning
checklist are in `docs/N3_HW_ALIGNMENT.md`.

N3 robot calibration and hardware topology are maintained by the single ROS 2
resource package `noetix_robots`, not duplicated inside SDK2. The canonical
paths loaded from `RobotConfig.config_root` are:

- `n3/config/robot.yaml` for all 29 joint motor `direction` and `offset` values.
- `n3/config/hardware.yaml` for EtherCAT, CANU and `hardware.diagnostic_read_only`.

The N3 backend validates all names by actuator name, rejects duplicate/invalid
calibration records, and rejects conflicting legacy `hardware.yaml.actuator_list`
values if such a list still exists. A legacy hardware-only actuator list is
supported for deployments without `robot.yaml` joint mapping, but the supplied
`noetix_robots` package does not need that list. It must be rebuilt/installed
along with SDK2. N3 private `.hpp` and `.cpp` files are paired under `src/n3/`;
the N3 vendor ankle solver remains in
`third-party/vendor/parallel_ankle/n3/` and Bumi is unchanged.
See `docs/N3_RESOURCE_INTEGRATION.md` for deployment and read-only checks.

**Important:** Merely providing a zero `SetMitCmd` is not safe for this vendor
API. In passive/read-only mode SDK2 now queues DISABLE, not zero MIT. Enabling
real motors requires an explicit `enableActuators()` request and valid feedback.

CPU-only ankle math check (does not touch real hardware):

```bash
cd noetix_sdk2
c++ -std=c++17 -O2 \
  -Ithird-party/vendor/parallel_ankle/n3 \
  tests/n3_parallel_ankle_test.cpp \
  third-party/vendor/parallel_ankle/n3/{FK,IK,Jacobian,coeffs,parallel_ankle}.cpp \
  -o /tmp/n3_ankle_test && /tmp/n3_ankle_test
```
