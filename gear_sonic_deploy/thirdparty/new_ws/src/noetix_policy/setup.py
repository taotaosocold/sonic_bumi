import os
from glob import glob
from setuptools import find_packages, setup

package_name = "noetix_policy"

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ("share/" + package_name + "/config", glob("config/*.yaml")),
    ("share/" + package_name + "/config/policies", glob("config/policies/*.yaml")),
]

# policy/<子目录>/* -> share/<package_name>/policy/<子目录>/
# ONNX 旁边的 export metadata YAML 也一起安装，便于部署核对模型接口。
for policy_asset in glob("policy/*/*.onnx") + glob("policy/*/*.yaml"):
    target_dir = os.path.join("share", package_name, os.path.dirname(policy_asset))
    data_files.append((target_dir, [policy_asset]))

# Reference-conditioned tracking policies need their motion assets at runtime.
for motion_asset in glob("motion/*/*.npz") + glob("motion/*/*.md"):
    target_dir = os.path.join("share", package_name, os.path.dirname(motion_asset))
    data_files.append((target_dir, [motion_asset]))

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Noetix",
    maintainer_email="dev@noetix.local",
    description="Generic ONNX policy runner for Noetix LowState/LowCmd.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "policy_node = noetix_policy.policy_node:main",
            "policy_gpu_check = noetix_policy.gpu_check:main",
        ],
    },
)