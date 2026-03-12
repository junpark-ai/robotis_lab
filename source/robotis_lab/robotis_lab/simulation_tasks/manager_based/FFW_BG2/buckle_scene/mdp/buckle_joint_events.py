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
# Author: OpenAI Codex

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject


# Fixed transform from arm_*_link7 to end_effector_*_link in ffw_bg2_follower.urdf.xacro.
_EE_OFFSET_POS = torch.tensor([[-0.015, 0.0, -0.23]], dtype=torch.float32)
_EE_OFFSET_QUAT = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
_IDENTITY_QUAT = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
# The buckle USD roots sit near the lower side of the geometry, so we attach a point
# slightly above each root to the end-effector frame instead of the raw root prim.
_HOUSING_ATTACH_OFFSET_POS = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
_INSERT_ATTACH_OFFSET_POS = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
_ATTACH_OFFSET_QUAT = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)


def _resolve_env_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if not torch.is_tensor(env_ids):
        return torch.tensor(env_ids, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def _ensure_joint_cache(env):
    if not hasattr(env, "_buckle_fixed_joints_created"):
        env._buckle_fixed_joints_created = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def _quat_to_gf(quat_tensor):
    from pxr import Gf

    return Gf.Quatf(
        float(quat_tensor[0]),
        Gf.Vec3f(float(quat_tensor[1]), float(quat_tensor[2]), float(quat_tensor[3])),
    )


def _vec3_to_gf(vec_tensor):
    from pxr import Gf

    return Gf.Vec3f(float(vec_tensor[0]), float(vec_tensor[1]), float(vec_tensor[2]))


def _get_robot_link_paths(stage, env_id: int) -> tuple[str, str, str | None, str | None]:
    del stage
    arm_l_link7_path = f"/World/envs/env_{env_id}/Robot/ffw_bg2_follower/left_arm/arm_l_link7"
    arm_r_link7_path = f"/World/envs/env_{env_id}/Robot/ffw_bg2_follower/right_arm/arm_r_link7"
    return arm_l_link7_path, arm_r_link7_path, None, None


def _create_fixed_joint(
    stage,
    joint_path: str,
    actor0_path: str,
    actor1_path: str,
    local_pos0,
    local_rot0,
    local_pos1,
    local_rot1,
):
    from omni.physx.scripts import physicsUtils

    joint = physicsUtils.add_joint_fixed(
        stage=stage,
        jointPath=joint_path,
        actor0=actor0_path,
        actor1=actor1_path,
        localPos0=local_pos0,
        localRot0=local_rot0,
        localPos1=local_pos1,
        localRot1=local_rot1,
        breakForce=1.0e20,
        breakTorque=1.0e20,
    )
    joint.CreateExcludeFromArticulationAttr().Set(True)
    joint.CreateJointEnabledAttr().Set(True)
    return joint


def attach_buckle_objects_with_fixed_joints(env, env_ids):
    import omni.log
    import omni.usd

    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    _ensure_joint_cache(env)

    stage = omni.usd.get_context().get_stage()
    robot: Articulation = env.scene["robot"]
    housing: RigidObject = env.scene["housing"]
    insert: RigidObject = env.scene["insert"]

    left_body_id = robot.find_bodies("arm_l_link7")[0][0]
    right_body_id = robot.find_bodies("arm_r_link7")[0][0]

    ee_offset_pos = _EE_OFFSET_POS.to(device=env.device)
    ee_offset_quat = _EE_OFFSET_QUAT.to(device=env.device)
    housing_attach_offset_pos = _HOUSING_ATTACH_OFFSET_POS.to(device=env.device)
    insert_attach_offset_pos = _INSERT_ATTACH_OFFSET_POS.to(device=env.device)
    attach_offset_quat = _ATTACH_OFFSET_QUAT.to(device=env.device)

    for env_id in env_ids.tolist():
        left_joint_path = f"/World/envs/env_{env_id}/LeftHousingFixedJoint"
        right_joint_path = f"/World/envs/env_{env_id}/RightInsertFixedJoint"
        left_joint_exists = stage.GetPrimAtPath(left_joint_path).IsValid()
        right_joint_exists = stage.GetPrimAtPath(right_joint_path).IsValid()

        arm_l_link7_path, arm_r_link7_path, ee_l_path, ee_r_path = _get_robot_link_paths(stage, env_id)
        if arm_l_link7_path is None or arm_r_link7_path is None:
            omni.log.warn(f"Failed to resolve arm link7 prims for env {env_id}; skipping buckle joint creation.")
            continue

        arm_l_pos = robot.data.body_state_w[env_id : env_id + 1, left_body_id, :3]
        arm_l_quat = robot.data.body_state_w[env_id : env_id + 1, left_body_id, 3:7]
        arm_r_pos = robot.data.body_state_w[env_id : env_id + 1, right_body_id, :3]
        arm_r_quat = robot.data.body_state_w[env_id : env_id + 1, right_body_id, 3:7]

        ee_l_pos, ee_l_quat = math_utils.combine_frame_transforms(arm_l_pos, arm_l_quat, ee_offset_pos, ee_offset_quat)
        ee_r_pos, ee_r_quat = math_utils.combine_frame_transforms(arm_r_pos, arm_r_quat, ee_offset_pos, ee_offset_quat)

        attach_to_root_quat = math_utils.quat_inv(attach_offset_quat)
        housing_attach_to_root_pos = math_utils.quat_apply(attach_to_root_quat, -housing_attach_offset_pos)
        insert_attach_to_root_pos = math_utils.quat_apply(attach_to_root_quat, -insert_attach_offset_pos)

        desired_housing_root_pos, desired_housing_root_quat = math_utils.combine_frame_transforms(
            ee_l_pos,
            ee_l_quat,
            housing_attach_to_root_pos,
            attach_to_root_quat,
        )
        desired_insert_root_pos, desired_insert_root_quat = math_utils.combine_frame_transforms(
            ee_r_pos,
            ee_r_quat,
            insert_attach_to_root_pos,
            attach_to_root_quat,
        )

        target_env_ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
        zero_velocity = torch.zeros((1, 6), device=env.device)
        housing.write_root_pose_to_sim(
            torch.cat([desired_housing_root_pos, desired_housing_root_quat], dim=-1), env_ids=target_env_ids
        )
        housing.write_root_velocity_to_sim(zero_velocity, env_ids=target_env_ids)
        insert.write_root_pose_to_sim(
            torch.cat([desired_insert_root_pos, desired_insert_root_quat], dim=-1), env_ids=target_env_ids
        )
        insert.write_root_velocity_to_sim(zero_velocity, env_ids=target_env_ids)

        left_actor0_path = ee_l_path if ee_l_path is not None else arm_l_link7_path
        right_actor0_path = ee_r_path if ee_r_path is not None else arm_r_link7_path

        if ee_l_path is None:
            left_local_pos0 = _vec3_to_gf(ee_offset_pos[0])
            left_local_rot0 = _quat_to_gf(ee_offset_quat[0])
        else:
            left_local_pos0 = _vec3_to_gf(torch.zeros(3, device=env.device))
            left_local_rot0 = _quat_to_gf(_IDENTITY_QUAT)

        if ee_r_path is None:
            right_local_pos0 = _vec3_to_gf(ee_offset_pos[0])
            right_local_rot0 = _quat_to_gf(ee_offset_quat[0])
        else:
            right_local_pos0 = _vec3_to_gf(torch.zeros(3, device=env.device))
            right_local_rot0 = _quat_to_gf(_IDENTITY_QUAT)

        if not left_joint_exists:
            _create_fixed_joint(
                stage=stage,
                joint_path=left_joint_path,
                actor0_path=left_actor0_path,
                actor1_path=housing.root_physx_view.prim_paths[env_id],
                local_pos0=left_local_pos0,
                local_rot0=left_local_rot0,
                local_pos1=_vec3_to_gf(housing_attach_offset_pos[0]),
                local_rot1=_quat_to_gf(attach_offset_quat[0]),
            )
        if not right_joint_exists:
            _create_fixed_joint(
                stage=stage,
                joint_path=right_joint_path,
                actor0_path=right_actor0_path,
                actor1_path=insert.root_physx_view.prim_paths[env_id],
                local_pos0=right_local_pos0,
                local_rot0=right_local_rot0,
                local_pos1=_vec3_to_gf(insert_attach_offset_pos[0]),
                local_rot1=_quat_to_gf(attach_offset_quat[0]),
            )

        env._buckle_fixed_joints_created[env_id] = stage.GetPrimAtPath(left_joint_path).IsValid() and stage.GetPrimAtPath(
            right_joint_path
        ).IsValid()
