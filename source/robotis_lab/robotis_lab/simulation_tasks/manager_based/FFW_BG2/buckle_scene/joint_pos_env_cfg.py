# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from robotis_lab.assets.robots.FFW_BG2 import FFW_BG2_CFG  # isort: skip
from robotis_lab.simulation_tasks.manager_based.FFW_BG2.pick_place import mdp
from robotis_lab.simulation_tasks.manager_based.FFW_BG2.pick_place.mdp import ffw_bg2_pick_place_events
from robotis_lab.simulation_tasks.manager_based.FFW_BG2.buckle_scene import mdp as buckle_mdp

from .buckle_scene_env_cfg import BuckleSceneEnvCfg

DEFAULT_GRIPPER_CLOSED_POS = 0.61
_ARM_MIRROR_SIGNS = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)


def _arm_joint_dict(side: str, values: tuple[float, ...]) -> dict[str, float]:
    return {f"arm_{side}_joint{joint_idx}": value for joint_idx, value in enumerate(values, start=1)}


def _mirror_right_arm_to_left(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sign * value for sign, value in zip(_ARM_MIRROR_SIGNS, values, strict=True))


def _make_init_pose(left_arm: tuple[float, ...], right_arm: tuple[float, ...]) -> dict[str, float]:
    return {
        "lift_joint": 0.0,
        **_arm_joint_dict("l", left_arm),
        "gripper_l_joint1": DEFAULT_GRIPPER_CLOSED_POS,
        **_arm_joint_dict("r", right_arm),
        "gripper_r_joint1": DEFAULT_GRIPPER_CLOSED_POS,
        "head_joint1": 0.8,
        "head_joint2": 0.0,
    }


_SYM_FLAT_0_RIGHT_ARM = (-0.5867, -0.7635, 1.1075, -1.1013, -1.0725, -0.5312, 1.2686)
_SYM_UP_30_RIGHT_ARM = (-0.6325, -0.7137, 1.2575, -1.1521, -1.6740, -0.7189, 1.4798)
_CURRENT_DEFAULT_RIGHT_ARM = (-1.1908, -0.7258, 1.1927, -1.8772, -1.0344, 0.6439, 0.6263)
_CURRENT_DEFAULT_LEFT_ARM = (-0.6325, 0.7137, -1.2575, -1.1521, 1.6740, -0.7189, -1.4798)
_DEG30_INSERTED_LEFT_ARM = (-0.6273, 0.7124, -1.2592, -1.1518, 1.6736, -0.7189, -1.4796)
_DEG30_INSERTED_RIGHT_ARM = (-1.1892, -0.7551, 1.2172, -1.8898, -1.0371, 0.613, 0.6241)
_DEG30_NOT_INSERTED_LEFT_ARM = (-0.6273, 0.7124, -1.2592, -1.1518, 1.6736, -0.7189, -1.4796)
_DEG30_NOT_INSERTED_RIGHT_ARM = (-1.2215, -0.9796, 1.1166, -1.9488, -1.0686, 0.6155, 0.8231)
_DEG0_INSERTED_RIGHT_ARM = (-0.6292, -0.6842, 1.1943, -1.1898, -0.9902, -0.4855, 1.0605)
_DEG0_NOT_INSERTED_RIGHT_ARM = (-0.5961, -0.7354, 1.1299, -1.1328, -1.0359, -0.5155, 1.1938)

INIT_POSE_OPTIONS = {
    "current_default": _make_init_pose(_CURRENT_DEFAULT_LEFT_ARM, _CURRENT_DEFAULT_RIGHT_ARM),
    "sym_flat_0": _make_init_pose(_mirror_right_arm_to_left(_SYM_FLAT_0_RIGHT_ARM), _SYM_FLAT_0_RIGHT_ARM),
    "sym_up_30": _make_init_pose(_mirror_right_arm_to_left(_SYM_UP_30_RIGHT_ARM), _SYM_UP_30_RIGHT_ARM),
    "deg30_inserted": _make_init_pose(_DEG30_INSERTED_LEFT_ARM, _DEG30_INSERTED_RIGHT_ARM),
    "deg30_not_inserted": _make_init_pose(_DEG30_NOT_INSERTED_LEFT_ARM, _DEG30_NOT_INSERTED_RIGHT_ARM),
    "deg0_inserted": _make_init_pose(_mirror_right_arm_to_left(_DEG0_INSERTED_RIGHT_ARM), _DEG0_INSERTED_RIGHT_ARM),
    "deg0_not_inserted": _make_init_pose(
        _mirror_right_arm_to_left(_DEG0_NOT_INSERTED_RIGHT_ARM), _DEG0_NOT_INSERTED_RIGHT_ARM
    ),
}

# Available options:
# - current_default
# - sym_flat_0
# - sym_up_30
# - deg30_inserted
# - deg30_not_inserted
# - deg0_inserted
# - deg0_not_inserted
SELECTED_INIT_POSE = "deg0_inserted"
INSERT_GRASP_RANDOM_AXIS = (0.0, 1.0, 0.0)
INSERT_GRASP_RANDOM_RANGE_DEG = (-15.0, 15.0)


@configclass
class EventCfg:
    reset_scene = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    detach_buckles = EventTerm(
        func=buckle_mdp.detach_buckle_fixed_joints,
        mode="reset",
    )

    init_ffw_bg2_pose = EventTerm(
        func=ffw_bg2_pick_place_events.set_default_joint_pose,
        mode="reset",
        params={
            "joint_positions": INIT_POSE_OPTIONS[SELECTED_INIT_POSE],
        },
    )

    reset_buckles_to_nominal_grasp_pose = EventTerm(
        func=buckle_mdp.reset_buckle_objects_to_nominal_grasp_pose,
        mode="reset",
    )

    randomize_insert_grasp_pose = EventTerm(
        func=buckle_mdp.randomize_buckle_grasp_pose,
        mode="reset",
        params={
            "rotation_axis": INSERT_GRASP_RANDOM_AXIS,
            "angle_range_deg": INSERT_GRASP_RANDOM_RANGE_DEG,
        },
    )

    attach_buckles = EventTerm(
        func=buckle_mdp.attach_buckle_objects_with_fixed_joints,
        mode="reset",
    )


@configclass
class BuckleSceneFFWBG2JointPosEnvCfg(BuckleSceneEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()
        self.teleop_default_close_gripper = True

        self.scene.robot = FFW_BG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot.actuators["gripper_master"].effort_limit_sim = 2.0
        self.scene.robot.actuators["gripper_slave"].effort_limit_sim = 2.0

        self.scene.plane.semantic_tags = [("class", "ground")]
        self.scene.insert.spawn.semantic_tags = [("class", "buckle_insert")]
        self.scene.housing.spawn.semantic_tags = [("class", "buckle_housing")]

        arm_joint_names = [
            "arm_r_joint1",
            "arm_r_joint2",
            "arm_r_joint3",
            "arm_r_joint4",
            "arm_r_joint5",
            "arm_r_joint6",
            "arm_r_joint7",
        ]

        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            name="robot",
            joint_names=arm_joint_names,
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            name="robot",
            joint_names=arm_joint_names,
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=arm_joint_names,
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_r_joint1"],
            open_command_expr={"gripper_r_joint1": 0.0},
            close_command_expr={"gripper_r_joint1": DEFAULT_GRIPPER_CLOSED_POS},
        )
        self.actions.gripper_action.class_type = buckle_mdp.DefaultClosedBinaryJointPositionAction

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_bg2_follower/world",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ffw_bg2_follower/right_arm/arm_r_link7",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )
