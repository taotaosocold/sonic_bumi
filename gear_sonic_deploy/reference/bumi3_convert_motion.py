"""将 BUMI3 训练参考转换为供部署读取的持久化 CSV。

在启动控制端前运行，复用训练数据加载器和参考正运动学，不运行学习策略或
物理步进。完整人体与五点人体分别导出，五点朝向从原始局部轴角沿骨架累乘得到。
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import yaml

from gear_sonic.utils.mujoco_sim.bumi3_deploy_common import (
    DEFAULT_DEPLOY_CONFIG, REPO_ROOT, contract_id, load_deploy_config, repo_path,
)
from gear_sonic.utils.mujoco_sim.bumi3_motion_dataset import load_smpl_reference_motion
from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract, Bumi3SonicSim2Sim, ZeroPolicy, load_reference_motion,
)

DEFAULT_RAW_DATASET = "gear_sonic_deploy/reference/bumi3_example/pkl/dataset_smoke_24.json"
FORMAT = "bumi3_csv_v1"

class MotionPlaylist:
    """校验原始动作清单，并按所选人体布局加载当前动作，避免重复读取全部数据。"""

    def __init__(self, path: Path, contract: Bumi3Contract, encoder: str, smpl_input: str = "full"):
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(content, dict) or content.get("version") != 1 or content.get("robot_type") != "bumi3":
            raise ValueError("dataset requires version: 1, robot_type: bumi3")
        self.entries = content.get("motions")
        if not isinstance(self.entries, list) or not self.entries:
            raise ValueError("dataset motions must be a nonempty list")
        names = []
        required = "robot" if encoder == "robot" else "smpl"
        for entry in self.entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"].strip():
                raise ValueError("each motion needs a nonempty name")
            names.append(entry["name"])
            for field in (required, "robot"):
                value = entry.get(field)
                if value is None and field != required:
                    continue
                if not isinstance(value, str) or not value:
                    raise ValueError(f"motion {entry['name']} requires {field}")
                target = Path(value).expanduser()
                target = target if target.is_absolute() else path.parent / target
                if not target.exists():
                    raise FileNotFoundError(target)
        if len(set(names)) != len(names):
            raise ValueError("duplicate motion names")
        self.path, self.contract, self.encoder = path, contract, encoder
        self.smpl_input = smpl_input
        self.indices = {name: index for index, name in enumerate(names)}
        self.index = 0

    def load(self):
        entry = self.entries[self.index]

        def target(field):
            value = entry.get(field)
            if value is None:
                return None
            path = Path(value).expanduser()
            return path if path.is_absolute() else self.path.parent / path

        # 复用统一加载器的坐标、时序与关节顺序处理，避免转换器维护另一套数据解释。
        options = dict(joint_order=entry.get("joint_order", "auto"),
                       quaternion_order=entry.get("quaternion_order", "auto"))
        if self.encoder == "smpl":
            motion = load_smpl_reference_motion(
                target("smpl"), self.contract,
                motion_key=entry.get("smpl_motion_key", entry.get("motion_key")),
                robot_path=target("robot"), robot_motion_key=entry.get("motion_key"),
                five_point=self.smpl_input == "five_point", **options,
            )
        else:
            motion = load_reference_motion(target("robot"), self.contract,
                                           motion_key=entry.get("motion_key"), **options)
        return replace(motion, name=entry["name"])


def motion_directory(name: str) -> Path:
    parts = PurePosixPath(name)
    if parts.is_absolute() or not parts.parts or any(item in ('.', '..') for item in name.split('/')) or '\\' in name:
        raise ValueError(f'motion name must be a relative name without traversal: {name}')
    return Path(*parts.parts)


class CsvExporter:
    def __init__(self, contract, smpl_input: str = "full"):
        self.contract = contract
        self.smpl_input = smpl_input
        self.runners = {}

    def export(self, motion, encoder: str, directory: Path):
        runner = self.runners.get(encoder)
        if runner is None:
            runner = Bumi3SonicSim2Sim(self.contract, motion, ZeroPolicy(self.contract, encoder=encoder),
                                      encoder=encoder, start_paused=True)
            self.runners[encoder] = runner
        else:
            runner.motion = motion
            runner.motions = (motion,)
            runner.reference_anchor_quat_wxyz = runner._compute_reference_anchor_quaternions()
        arrays = dict(joint_pos=motion.joint_pos_policy, joint_vel=motion.joint_vel_policy,
                      anchor_quat=runner.reference_anchor_quat_wxyz,
                      qpos=np.stack([runner._reference_qpos(i) for i in range(motion.num_frames)]),
                      qvel=np.stack([runner._reference_qvel(i) for i in range(motion.num_frames)]))
        if encoder == 'smpl':
            arrays['smpl_quat'] = motion.smpl_reference.root_quat_wxyz
            if self.smpl_input == 'five_point':
                if motion.smpl_reference.five_point_pose is None:
                    raise ValueError('五点 CSV 需要从原始 SMPL pose_aa 计算，不能仅由旧位置 CSV 推断朝向')
                arrays['smpl_five_point_pose'] = motion.smpl_reference.five_point_pose
            else:
                arrays['smpl_joints'] = motion.smpl_reference.local_joints.reshape(motion.num_frames, 72)
        filenames = dict(joint_pos='joint_pos.csv', joint_vel='joint_vel.csv',
                         anchor_quat='body_quat.csv', qpos='qpos.csv', qvel='qvel.csv',
                         smpl_joints='smpl_joint.csv', smpl_quat='smpl_root_quat.csv',
                         smpl_five_point_pose='smpl_five_point_pose.csv')
        directory.mkdir(parents=True, exist_ok=True)
        fields = {}
        for key, values in arrays.items():
            values = np.asarray(values, dtype=np.float64)
            if values.ndim != 2 or values.shape[0] != motion.num_frames or not np.isfinite(values).all():
                raise ValueError(f'invalid reference array: {key}')
            if key in ('joint_pos', 'joint_vel'):
                columns = self.contract.policy_joint_names
            elif key in ('anchor_quat', 'smpl_quat'):
                columns = ('w', 'x', 'y', 'z')
            else:
                columns = [f'{key}_{i}' for i in range(values.shape[1])]
            np.savetxt(directory / filenames[key], values, delimiter=',', fmt='%.17g',
                       header=','.join(columns), comments='')
            fields[key] = dict(path=filenames[key], shape=list(values.shape))
        metadata = dict(version=1, format=FORMAT, robot_type='bumi3', encoder=encoder,
                        name=motion.name, frames=motion.num_frames, fps=motion.fps,
                        has_robot_reference=motion.has_robot_reference,
                        reference_contract_id=contract_id(self.contract),
                        policy_joint_names=list(self.contract.policy_joint_names), arrays=fields)
        if encoder == 'smpl':
            metadata['smpl_input'] = self.smpl_input
            if self.smpl_input == 'five_point':
                from gear_sonic.utils.teleop.smpl_five_point import FIVE_POINT_INDICES, FIVE_POINT_SCHEMA

                metadata['smpl_five_point_schema'] = FIVE_POINT_SCHEMA
                metadata['smpl_five_point_indices'] = list(FIVE_POINT_INDICES)
        (directory / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_DEPLOY_CONFIG.relative_to(REPO_ROOT)))
    parser.add_argument('--obs-config', help='observation/control YAML; use the same file for simulation and deployment')
    parser.add_argument('--dataset', help='raw PKL manifest; default is the bundled smoke 24 manifest')
    parser.add_argument('--motion', help='single raw PKL/NPZ file; requires --encoder robot or smpl')
    parser.add_argument('--motion-key')
    parser.add_argument('--robot-motion', help='paired Robot motion for a single SMPL input')
    parser.add_argument('--robot-motion-key')
    parser.add_argument('--motion-name', help='select one manifest motion, or name a single-file export')
    parser.add_argument('--encoder', choices=('both', 'robot', 'smpl'), default='both')
    parser.add_argument('--smpl-input', choices=('full', 'five_point'), help='人体输入布局；默认读取部署配置')
    parser.add_argument('--output-dir', help='输出目录；默认使用部署配置 dataset 所在目录')
    parser.add_argument('--overwrite', action='store_true', help='regenerate already converted motions')
    args = parser.parse_args()
    if args.motion and args.dataset:
        parser.error('choose --dataset or --motion')
    if args.motion and args.encoder == 'both':
        parser.error('single-file conversion requires --encoder robot or smpl')
    if not args.motion and (args.motion_key or args.robot_motion or args.robot_motion_key):
        parser.error('motion keys and paired Robot input require --motion')
    if args.robot_motion and args.encoder != 'smpl':
        parser.error('--robot-motion requires --encoder smpl')
    if args.robot_motion_key and not args.robot_motion:
        parser.error('--robot-motion-key requires --robot-motion')
    config = load_deploy_config(args.config)
    contract = Bumi3Contract.from_yaml(repo_path(args.obs_config or config['contract']))
    smpl_input = args.smpl_input or config.get('smpl_input', 'full')
    output = repo_path(args.output_dir) if args.output_dir else repo_path(config['dataset']).parent
    exporter = CsvExporter(contract, smpl_input)
    encoders = ('robot', 'smpl') if args.encoder == 'both' else (args.encoder,)
    playlists = {}
    if args.motion:
        names = [args.motion_name or args.motion_key or Path(args.motion).stem]
    else:
        for encoder in encoders:
            playlist = MotionPlaylist(repo_path(args.dataset or DEFAULT_RAW_DATASET), contract, encoder, smpl_input)
            playlists[encoder] = playlist
        names = [item['name'] for item in next(iter(playlists.values())).entries]
        if args.motion_name:
            if args.motion_name not in names:
                raise ValueError(f'motion not found: {args.motion_name}')
            names = [args.motion_name]
    exported = []
    for position, name in enumerate(names):
        relative = motion_directory(name)
        entry = dict(name=name)
        for encoder in encoders:
            directory = output / relative / encoder
            metadata_file = directory / 'metadata.json'
            if metadata_file.exists() and not args.overwrite:
                metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
                if (metadata.get('format') != FORMAT or metadata.get('reference_contract_id') != contract_id(contract)
                        or metadata.get('encoder') != encoder or metadata.get('name') != name
                        or metadata.get('policy_joint_names') != list(contract.policy_joint_names)):
                    raise ValueError(f'old/incompatible export; use --overwrite: {metadata_file}')
                if encoder == 'smpl' and (metadata.get('smpl_input', 'full') != smpl_input or
                        (smpl_input == 'five_point' and
                         (metadata.get('smpl_five_point_schema') != 1 or
                          metadata.get('smpl_five_point_indices') != [15, 20, 21, 7, 8]))):
                    raise ValueError(f'人体输入布局不一致，请选择独立目录或 --overwrite: {metadata_file}')
                for field in metadata['arrays'].values():
                    if not (directory / field['path']).is_file():
                        raise FileNotFoundError(f"incomplete export; use --overwrite: {directory / field['path']}")
            else:
                if args.motion:
                    if encoder == 'robot':
                        motion = load_reference_motion(repo_path(args.motion), contract, motion_key=args.motion_key)
                    else:
                        motion = load_smpl_reference_motion(repo_path(args.motion), contract,
                            motion_key=args.motion_key, robot_path=repo_path(args.robot_motion) if args.robot_motion else None,
                            robot_motion_key=args.robot_motion_key, five_point=smpl_input == 'five_point')
                    motion = replace(motion, name=name)
                else:
                    playlist = playlists[encoder]
                    playlist.index = playlist.indices[name]
                    motion = playlist.load()
                exporter.export(motion, encoder, directory)
            entry[encoder] = (relative / encoder).as_posix()
        exported.append(entry)
        print(f'BUMI3_CONVERT {position+1}/{len(names)} {name}', flush=True)
    manifest = dict(version=1, robot_type='bumi3', format=FORMAT, motions=exported)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'dataset.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'BUMI3_REFERENCE_READY {output / "dataset.json"}', flush=True)


if __name__ == '__main__':
    main()
