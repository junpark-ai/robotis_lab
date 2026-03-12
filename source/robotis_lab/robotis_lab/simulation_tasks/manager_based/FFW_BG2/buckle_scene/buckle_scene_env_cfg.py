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

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as base_mdp
from robotis_lab.assets.object import SEATBELT_BUCKLE_HOUSING_CFG, SEATBELT_BUCKLE_INSERT_CFG
from robotis_lab.simulation_tasks.manager_based.FFW_BG2.pick_place import mdp

BUCKLE_YAW_90_QUAT = [0.70710678, 0.0, 0.0, 0.70710678]


@configclass
class BuckleTableSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    insert = SEATBELT_BUCKLE_INSERT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Insert",
        spawn=SEATBELT_BUCKLE_INSERT_CFG.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=1.0,
                angular_damping=1.0,
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, -0.14, 0.995], rot=BUCKLE_YAW_90_QUAT),
    )

    housing = SEATBELT_BUCKLE_HOUSING_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Housing",
        spawn=SEATBELT_BUCKLE_HOUSING_CFG.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=1.0,
                angular_damping=1.0,
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.46, 0.14, 0.995], rot=BUCKLE_YAW_90_QUAT),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)

        insert_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("insert")})
        insert_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("insert")})
        housing_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("housing")})
        housing_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("housing")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    insert_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.85, "asset_cfg": SceneEntityCfg("insert")},
    )
    success = None


@configclass
class BuckleSceneEnvCfg(ManagerBasedRLEnvCfg):
    scene: BuckleTableSceneCfg = BuckleTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 60.0

        self.sim.dt = 0.01
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
