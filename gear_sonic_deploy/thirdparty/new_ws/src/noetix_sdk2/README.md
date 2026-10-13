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
