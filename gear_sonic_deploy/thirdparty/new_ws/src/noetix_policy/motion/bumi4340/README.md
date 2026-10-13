# BUMI4340 TaiChi reference motion

`taiji.onnx` is a MimicLite **reference-conditioned tracking policy**.  The
reference TaiChi motion is not embedded in the ONNX.  Put the `score1` motion
used for training here as:

```
taiji_reference.npz
```

The runtime accepts either of these layouts (all arrays float32 is preferred):

1. Direct root layout:
   - `joint_pos`: `[T, 21]`
   - `joint_names`: optional `[21]` strings; strongly recommended
   - `root_pos`: `[T, 3]`
   - `root_quat`: `[T, 4]`, quaternion order **w, x, y, z**
   - `fps`: scalar `50`

2. MimicLite/Any4HDMI-style body layout:
   - `joint_pos`: `[T, J]`
   - `joint_names`: `[J]`
   - `body_pos_w`: `[T, B, 3]`
   - `body_quat_w`: `[T, B, 4]`, **w, x, y, z**
   - `body_names`: `[B]`, must contain `base_link`
   - `fps`: scalar `50`

The reference must be the same `bumi4340/score1` trajectory (or an exactly
compatible export of it) that the TaiChi policy was trained against. Do not
replace it with a zero/static trajectory: the ONNX command input contains the
future reference root pose/orientation and future 21-joint positions.
