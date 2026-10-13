# noetix_policy

ROS 2 ONNX policy runner for the Noetix real-hardware LowState / LowCmd interface.

## Ownership rule

The policy package never changes the robot drive mode.

```text
DISABLED -> policy output is blocked
anything else -> policy may activate and publish LowCmd
```

LIN / zero-return is an upper-layer LowCmd producer. A new policy activation
publishes `/noetix/policy_state`; the zero-return tool stops and the policy
takes over `/noetix/low_cmd`.

Default gamepad mapping:

- button 7: ENABLE
- button 3: zero return
- button 6: DISABLE
- button 0: selected robot walk policy
- button 1: stop the active policy
- button 4: BUMI4340 TaiChi tracking policy


## BUMI4340 TaiChi tracking policy

BUMI4340 now has a second policy, `bumi4340_taiji`, on gamepad **button 4**.
It keeps the original walk policy on button 0. The TaiChi ONNX uses two model
inputs, `command[240]` and `policy[378]`, and outputs 21 joint actions.

Unlike the walk policy, this is a reference-conditioned MimicLite policy. The
ONNX does **not** contain the TaiChi trajectory. Before pressing button 4, put
the same `bumi4340/score1` reference used during training at:

```text
share/noetix_policy/motion/bumi4340/taiji_reference.npz
```

In the source workspace the corresponding path is
`src/noetix_policy/motion/bumi4340/taiji_reference.npz`. See the README in that
directory for the accepted NPZ fields. If the reference is absent or has the
wrong 50 Hz layout, activation is rejected instead of running the ONNX with an
invalid zero reference.

## Real robot launch

Bumi3 example:

```bash
ros2 launch noetix_bringup real.launch.py robot_name:=bumi3
ros2 launch noetix_policy policy.launch.py robot_name:=bumi3
```

Bumi4340 uses the same command shape with `robot_name:=bumi4340`.

The learned Bumi policy runs at 50 Hz while the real hardware I/O loop runs at
500 Hz and holds the latest command between policy updates.

## GPU inference

The default configuration is GPU-first and does not silently fall back to CPU.
Check the runtime with:

```bash
ros2 run noetix_policy policy_gpu_check
```

## Bumi3 legacy walk compatibility

The `bumi3_walk` profile mirrors the legacy Bumi walk contract from
`legged_bumi_lab_v3_dance` while using the current ROS 2 LowState/LowCmd API.
It keeps the old actuator-name policy order and startup stand interpolation.

## Bumi2 N4 walking policy

Bumi2 uses `policy/bumi2/N4_10500.onnx` with its model-embedded joint order,
PD gains, nominal joint pose and action scales. The model input is `[1, 355]`:
five 71-value history frames. Each frame contains three command values, three
body angular velocities, projected gravity X/Y, 21 relative joint positions,
21 joint velocities and 21 previous actions.

Launch on real hardware with matching robot selection in both processes:

```bash
ros2 launch noetix_bringup real.launch.py robot_name:=bumi2
ros2 launch noetix_policy policy.launch.py robot_name:=bumi2
```
