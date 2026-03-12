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


@configclass
class EventCfg:
    init_ffw_bg2_pose = EventTerm(
        func=ffw_bg2_pick_place_events.set_default_joint_pose,
        mode="reset",
        params={
            "joint_positions": {
                # Copy/paste presets as needed.
                #
                # Symmetric flat-0 preset:
                # "arm_l_joint1": -0.5867,
                # "arm_l_joint2": 0.7635,
                # "arm_l_joint3": -1.1075,
                # "arm_l_joint4": -1.1013,
                # "arm_l_joint5": 1.0725,
                # "arm_l_joint6": -0.5312,
                # "arm_l_joint7": -1.2686,
                # "arm_r_joint1": -0.5867,
                # "arm_r_joint2": -0.7635,
                # "arm_r_joint3": 1.1075,
                # "arm_r_joint4": -1.1013,
                # "arm_r_joint5": -1.0725,
                # "arm_r_joint6": -0.5312,
                # "arm_r_joint7": 1.2686,
                #
                # Symmetric up-30 preset:
                # "arm_l_joint1": -0.6325,
                # "arm_l_joint2": 0.7137,
                # "arm_l_joint3": -1.2575,
                # "arm_l_joint4": -1.1521,
                # "arm_l_joint5": 1.6740,
                # "arm_l_joint6": -0.7189,
                # "arm_l_joint7": -1.4798,
                # "arm_r_joint1": -0.6325,
                # "arm_r_joint2": -0.7137,
                # "arm_r_joint3": 1.2575,
                # "arm_r_joint4": -1.1521,
                # "arm_r_joint5": -1.6740,
                # "arm_r_joint6": -0.7189,
                # "arm_r_joint7": 1.4798,
                #
                "lift_joint": 0.0,
                "arm_l_joint1": -0.6325,
                "arm_l_joint2": 0.7137,
                "arm_l_joint3": -1.2575,
                "arm_l_joint4": -1.1521,
                "arm_l_joint5": 1.6740,
                "arm_l_joint6": -0.7189,
                "arm_l_joint7": -1.4798,
                "gripper_l_joint1": 0.65,
                "arm_r_joint1": -1.1908,
                "arm_r_joint2": -0.7258,
                "arm_r_joint3": 1.1927,
                "arm_r_joint4": -1.8772,
                "arm_r_joint5": -1.0344,
                "arm_r_joint6": 0.6439,
                "arm_r_joint7": 0.6263,
                "gripper_r_joint1": 0.65,
                "head_joint1": 0.8,
                "head_joint2": 0.0,
            },
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
            close_command_expr={"gripper_r_joint1": 0.65},
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
