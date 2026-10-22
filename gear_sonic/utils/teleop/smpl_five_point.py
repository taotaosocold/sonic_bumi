"""定义五点人体输入，供新 BUMI3 训练观测与 PICO 直接选点共用。

五点顺序固定为头部、左腕、右腕、左踝、右踝，对应身体骨架索引
15、20、21、7、8；22、23 在当前数据准备器中可能是拇指端点，不能当作手腕。
每帧特征先放五点位置的十五维，再放五点旋转的三十维；六维旋转采用
旋转矩阵前两列按行展开。根部的世界朝向单独保留，送入网络前再相对机器人
根部表达。因此新参考窗口为十帧五点姿态四百五十维加根部姿态六十维。

训练旋转由 SMPL 局部轴角沿骨架累乘得到，不能直接拿腕、踝的局部关节旋转
充当相对骨盆旋转。实时入口直接处理身体跟踪的全局姿态，不调用 SMPL 身体
模型、不反求关节角，也不使用独立头显或手柄设备的原点。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    import torch


FIVE_POINT_NAMES = ("head", "left_wrist", "right_wrist", "left_ankle", "right_ankle")
FIVE_POINT_INDICES = (15, 20, 21, 7, 8)
FIVE_POINT_FRAMES = 10
FIVE_POINT_POSE_DIM = 45
FIVE_POINT_TOKENIZER_DIM = FIVE_POINT_FRAMES * (FIVE_POINT_POSE_DIM + 6)
FIVE_POINT_SCHEMA = 1

# 只使用 SMPL 主体的前二十二个关节；这部分也与数据准备器的 SMPL-X 主体一致，
# 避开末尾两点可能是手掌或拇指端点的差异。父节点严格位于子节点之前。
SMPL_BODY_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
)

# 现有根朝向为 R_zup * R_smpl_root * inverse(B)，其中只有 B 取逆；
# B 对应 xyzw 四元数 [0.5,0.5,0.5,0.5]。因此相对这个根的骨架旋转
# 必须左乘 B，不能省略基准变换，也不能把整条根部旋转链都取逆。
SMPL_BASE_MATRIX = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def smpl_five_point_local_pose(
    joints_z_up: torch.Tensor,
    pose_aa: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
) -> torch.Tensor:
    """把训练参考构造成每帧四十五维五点局部姿态，保留输入的批次和时间维。

    输入分别为 [...,24,3] 的 Z-up 关键点、[...,72] 的 SMPL 局部轴角和
    [...,4] 的已去除 SMPL 基准旋转的世界根四元数。位置先减去骨盆，再用
    当前参考帧根旋转的逆变换到局部系；关节旋转沿父链累乘，并用同一根部
    基准表达。函数不读取机器人重定向点、不计算根部相对机器人朝向，也不加载
    SMPL 网格或身体模型；根部相对朝向由已有观测项负责。
    """
    import torch

    from gear_sonic.trl.utils.torch_transform import (
        angle_axis_to_rotation_matrix,
        quaternion_to_rotation_matrix,
    )

    if joints_z_up.shape[-2:] != (24, 3) or pose_aa.shape[-1] != 72:
        raise ValueError("五点训练输入要求 smpl_joints[...,24,3] 和 pose_aa[...,72]")
    batch_shape = pose_aa.shape[:-1]
    if joints_z_up.shape[:-2] != batch_shape or root_quat_wxyz.shape != batch_shape + (4,):
        raise ValueError("五点位置、轴角与根部四元数的批次及时间维必须一致")

    root_matrix = quaternion_to_rotation_matrix(root_quat_wxyz.reshape(-1, 4)).reshape(
        batch_shape + (3, 3)
    )
    offsets = joints_z_up[..., FIVE_POINT_INDICES, :] - joints_z_up[..., :1, :]
    positions_local = torch.matmul(offsets, root_matrix)

    local_rotations = angle_axis_to_rotation_matrix(
        pose_aa.reshape(batch_shape + (24, 3))[..., :22, :].reshape(-1, 3)
    ).reshape(batch_shape + (22, 3, 3))
    chain = [pose_aa.new_tensor(SMPL_BASE_MATRIX).expand(batch_shape + (3, 3))]
    for index, parent in enumerate(SMPL_BODY_PARENTS[1:], start=1):
        chain.append(torch.matmul(chain[parent], local_rotations[..., index, :, :]))
    rotations_local = torch.stack([chain[index] for index in FIVE_POINT_INDICES], dim=-3)
    return torch.cat(
        (positions_local.reshape(batch_shape + (15,)),
         rotations_local[..., :2].reshape(batch_shape + (30,))),
        dim=-1,
    )


def pico_body_to_five_point(body_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """直接把 PICO 身体跟踪 [...,24,7] 转为五点局部姿态和世界根朝向。

    原始每点为米制 xyz 加 xyzw 世界四元数；返回 [...,5,7]，每点为相对骨盆
    的 xyz 加 wxyz 四元数，以及单独的 [...,4] 世界根四元数。原始人体比例
    被保留，不使用 SMPL 模型重建位置，也不做机器人尺寸标定。

    旋转沿用当前 compute_from_body_poses 的右乘绕 Y 轴半周修正，再采用
    process_smpl_joints 的 Y-up 到 Z-up 及去除根部基准旋转约定；位置应用
    同一个世界转轴后取根部局部值。这里不能复用旧 VR 三点的机器人关节偏置，
    否则训练的人体五点与实时输入会采用不同的坐标系。
    """
    body = np.asarray(body_poses, dtype=np.float64)
    if body.ndim < 2 or body.shape[-2:] != (24, 7) or not np.isfinite(body).all():
        raise ValueError("PICO 五点入口要求有限数值组成的 [...,24,7] 身体姿态")
    batch_shape = body.shape[:-2]
    selected = body[..., (0,) + FIVE_POINT_INDICES, :]
    quaternions = selected[..., 3:]
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("PICO 骨盆或五点含无效的零四元数，不能构造五点参考")

    turn_up = Rotation.from_rotvec([np.pi / 2.0, 0.0, 0.0])
    joint_frame = Rotation.from_rotvec([0.0, np.pi, 0.0])
    base = Rotation.from_matrix(np.asarray(SMPL_BASE_MATRIX))
    raw = Rotation.from_quat((quaternions / norms).reshape(-1, 4))
    world_matrices = (turn_up * raw * joint_frame).as_matrix().reshape(
        batch_shape + (6, 3, 3)
    )
    root_matrix = world_matrices[..., 0, :, :] @ base.as_matrix().T
    offsets_world = (selected[..., 1:, :3] - selected[..., :1, :3]) @ turn_up.as_matrix().T
    positions_local = offsets_world @ root_matrix
    rotations_local = (
        root_matrix[..., None, :, :].swapaxes(-1, -2) @ world_matrices[..., 1:, :, :]
    )
    point_quat_xyzw = Rotation.from_matrix(rotations_local.reshape(-1, 3, 3)).as_quat()
    point_quat_wxyz = point_quat_xyzw[:, [3, 0, 1, 2]].reshape(batch_shape + (5, 4))
    root_xyzw = Rotation.from_matrix(root_matrix.reshape(-1, 3, 3)).as_quat()
    root_wxyz = root_xyzw[:, [3, 0, 1, 2]].reshape(batch_shape + (4,))
    poses_local = np.concatenate((positions_local, point_quat_wxyz), axis=-1)
    return poses_local.astype(np.float32), root_wxyz.astype(np.float32)


def build_five_point_tokenizer(
    poses_local: np.ndarray,
    root_quat_wxyz: np.ndarray,
    robot_anchor_wxyz: np.ndarray,
    heading_delta_wxyz: np.ndarray,
) -> np.ndarray:
    """将十帧五点数据变成联合 ONNX 的五百一十维参考输入，不包含本体历史。

    输入五点姿态为 [10,5,7]，四元数顺序为 wxyz，且已相对每帧的人体根部；
    根部四元数为 [10,4] 世界朝向。输出先展开十帧各四十五维五点特征，再展开
    十帧根部六维旋转，与训练两项观测的顺序一致。根部使用
    robot_root 的逆 * 初始 yaw 补偿 * human_root；五点局部姿态不受世界 yaw
    补偿影响。调用方负责维护启动补偿和连续的五十赫兹窗口，本函数不生成未来帧。
    """
    points = np.asarray(poses_local, dtype=np.float64)
    roots = np.asarray(root_quat_wxyz, dtype=np.float64)
    robot = np.asarray(robot_anchor_wxyz, dtype=np.float64)
    heading = np.asarray(heading_delta_wxyz, dtype=np.float64)
    if points.shape != (FIVE_POINT_FRAMES, 5, 7) or roots.shape != (FIVE_POINT_FRAMES, 4):
        raise ValueError("五点 tokenizer 要求 [10,5,7] 局部姿态和 [10,4] 世界根朝向")
    if robot.shape != (4,) or heading.shape != (4,):
        raise ValueError("机器人根朝向与初始 yaw 补偿均须为单个 wxyz 四元数")
    for values in (points, roots, robot, heading):
        if not np.isfinite(values).all():
            raise ValueError("五点 tokenizer 输入包含非有限数值")
    for quaternions in (points[..., 3:], roots, robot, heading):
        if np.any(np.linalg.norm(quaternions, axis=-1) < 1e-8):
            raise ValueError("五点 tokenizer 输入包含零四元数")

    poses = five_point_pose_features(points)
    root_rotation = Rotation.from_quat(roots[:, [1, 2, 3, 0]])
    robot_rotation = Rotation.from_quat(robot[[1, 2, 3, 0]])
    delta_rotation = Rotation.from_quat(heading[[1, 2, 3, 0]])
    relative_root = (robot_rotation.inv() * delta_rotation * root_rotation).as_matrix()[..., :2]
    return np.concatenate((poses.reshape(-1), relative_root.reshape(-1))).astype(np.float32)


def five_point_pose_features(poses_local: np.ndarray) -> np.ndarray:
    """把任意批次的五点局部 xyz、wxyz 转为每帧十五维位置加三十维旋转。

    输入形状为 [...,5,7]，输出为 [...,45]；PICO 发布端与联合模型输入构造
    共用此函数。拒绝非有限值与零四元数，不把无效追踪点伪装成静止参考。
    """
    points = np.asarray(poses_local, dtype=np.float64)
    if points.ndim < 2 or points.shape[-2:] != (5, 7) or not np.isfinite(points).all():
        raise ValueError("五点姿态必须为有限的 [...,5,7] 数组")
    quaternions = points[..., 3:].reshape(-1, 4)
    if np.any(np.linalg.norm(quaternions, axis=-1) < 1e-8):
        raise ValueError("五点姿态含零四元数")
    batch = points.shape[:-2]
    matrices = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
    return np.concatenate(
        (points[..., :3].reshape(batch + (15,)), matrices[..., :2].reshape(batch + (30,))),
        axis=-1,
    ).astype(np.float32)


def smpl_pose_to_five_point_features(pose_aa: np.ndarray, local_joints: np.ndarray) -> np.ndarray:
    """供离线转换使用的 NumPy 版本，与训练观测采用相同父链和根部基准。

    输入 [...,72] 轴角及已经转到人体根部局部坐标的 [...,24,3] 关键点；
    先明确减去局部骨盆，再选头、双腕、双踝。只累乘主体局部旋转，不重新
    加载或拟合人体模型，输出与训练相同的 [...,45] 数值布局。
    """
    pose = np.asarray(pose_aa, dtype=np.float64)
    joints = np.asarray(local_joints, dtype=np.float64)
    if pose.ndim < 1 or pose.shape[-1] != 72 or joints.shape != pose.shape[:-1] + (24, 3):
        raise ValueError("离线五点转换需要对齐的 [...,72] 轴角和 [...,24,3] 局部关键点")
    if not np.isfinite(pose).all() or not np.isfinite(joints).all():
        raise ValueError("离线五点转换输入含非有限值")
    batch = pose.shape[:-1]
    local = Rotation.from_rotvec(pose.reshape(-1, 24, 3)[:, :22].reshape(-1, 3))
    local = local.as_matrix().reshape(batch + (22, 3, 3))
    chain = [np.broadcast_to(np.asarray(SMPL_BASE_MATRIX), batch + (3, 3))]
    for index, parent in enumerate(SMPL_BODY_PARENTS[1:], start=1):
        chain.append(chain[parent] @ local[..., index, :, :])
    matrices = np.stack([chain[index] for index in FIVE_POINT_INDICES], axis=-3)
    positions = joints[..., FIVE_POINT_INDICES, :] - joints[..., :1, :]
    return np.concatenate(
        (positions.reshape(batch + (15,)), matrices[..., :2].reshape(batch + (30,))), axis=-1,
    ).astype(np.float32)
