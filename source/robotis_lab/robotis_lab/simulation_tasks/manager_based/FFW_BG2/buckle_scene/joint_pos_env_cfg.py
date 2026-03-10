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
from robotis_lab.assets.robots.FFW_BG2 import FFW_BG2_WITHOUT_MIMIC_CFG  # isort: skip
from robotis_lab.simulation_tasks.manager_based.FFW_BG2.pick_place import mdp
from robotis_lab.simulation_tasks.manager_based.FFW_BG2.pick_place.mdp import ffw_bg2_pick_place_events

from .buckle_scene_env_cfg import BuckleSceneEnvCfg


def deactivate_packing_table_container(env, env_ids, prim_name: str = "container_h20"):
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for env_id in range(env.num_envs):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/PackingTable/{prim_name}")
        if prim.IsValid():
            prim.SetActive(False)


@configclass
class EventCfg:
    deactivate_container = EventTerm(
        func=deactivate_packing_table_container,
        mode="startup",
    )

    init_ffw_bg2_pose = EventTerm(
        func=ffw_bg2_pick_place_events.set_default_joint_pose,
        mode="reset",
        params={
            "joint_positions": {
                "lift_joint": 0.0,
                "arm_l_joint1": -0.7993,
                "arm_l_joint2": 0.8486,
                "arm_l_joint3": -1.4219,
                "arm_l_joint4": -1.2209,
                "arm_l_joint5": 0.7279,
                "arm_l_joint6": -0.296,
                "arm_l_joint7": -0.9213,
                "gripper_l_joint1": 0.55,
                "gripper_l_joint2": 0.55,
                "gripper_l_joint3": 0.55,
                "gripper_l_joint4": 0.55,
                "arm_r_joint1": -0.7993,
                "arm_r_joint2": -0.8486,
                "arm_r_joint3": 1.4219,
                "arm_r_joint4": -1.2209,
                "arm_r_joint5": -0.7279,
                "arm_r_joint6": -0.296,
                "arm_r_joint7": 0.9213,
                "head_joint1": 0.695,
                "head_joint2": -0.35,
            },
        },
    )


@configclass
class BuckleSceneFFWBG2JointPosEnvCfg(BuckleSceneEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()

        self.scene.robot = FFW_BG2_WITHOUT_MIMIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        self.scene.table.spawn.semantic_tags = [("class", "table")]
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
            joint_names=["gripper_r_joint[1-4]"],
            open_command_expr={"gripper_r_joint.*": 0.0},
            close_command_expr={"gripper_r_joint.*": 1.0},
        )

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
