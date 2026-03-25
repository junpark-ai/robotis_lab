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
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg


# Fixed transform from arm_*_link7 to end_effector_*_link in ffw_bg2_follower.urdf.xacro.
_EE_OFFSET_POS = torch.tensor([[-0.015, 0.0, -0.23]], dtype=torch.float32)
_EE_OFFSET_QUAT = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
_IDENTITY_QUAT = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


def _resolve_env_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if not torch.is_tensor(env_ids):
        return torch.tensor(env_ids, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def _ensure_joint_cache(env):
    if not hasattr(env, "_buckle_fixed_joints_created"):
        env._buckle_fixed_joints_created = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if not hasattr(env, "_buckle_collision_filters_created"):
        env._buckle_collision_filters_created = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


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


def _set_joint_local_pose(stage, joint_path: str, local_pos0, local_rot0, local_pos1, local_rot1) -> None:
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        return

    joint_prim.GetAttribute("physics:localPos0").Set(local_pos0)
    joint_prim.GetAttribute("physics:localRot0").Set(local_rot0)
    joint_prim.GetAttribute("physics:localPos1").Set(local_pos1)
    joint_prim.GetAttribute("physics:localRot1").Set(local_rot1)


def _filter_robot_buckle_collisions(stage, robot_prim_path: str, buckle_prim_path: str) -> None:
    """Masks collisions between the robot articulation and a buckle rigid object."""
    from omni.physx.scripts import utils as physx_utils

    physx_utils.addPairFilter(stage, [robot_prim_path, buckle_prim_path])


def detach_buckle_fixed_joints(env, env_ids):
    import omni.usd

    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    _ensure_joint_cache(env)
    stage = omni.usd.get_context().get_stage()

    for env_id in env_ids.tolist():
        left_joint_path = f"/World/envs/env_{env_id}/LeftHousingFixedJoint"
        right_joint_path = f"/World/envs/env_{env_id}/RightInsertFixedJoint"

        if stage.GetPrimAtPath(left_joint_path).IsValid():
            stage.RemovePrim(left_joint_path)
        if stage.GetPrimAtPath(right_joint_path).IsValid():
            stage.RemovePrim(right_joint_path)

        env._buckle_fixed_joints_created[env_id] = False


def reset_buckle_objects_to_nominal_grasp_pose(env, env_ids):
    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    robot: Articulation = env.scene["robot"]
    housing: RigidObject = env.scene["housing"]
    insert: RigidObject = env.scene["insert"]

    left_body_id = robot.find_bodies("arm_l_link7")[0][0]
    right_body_id = robot.find_bodies("arm_r_link7")[0][0]

    ee_offset_pos = _EE_OFFSET_POS.to(device=env.device).repeat(len(env_ids), 1)
    ee_offset_quat = _EE_OFFSET_QUAT.to(device=env.device).repeat(len(env_ids), 1)
    zero_velocity = torch.zeros((len(env_ids), 6), device=env.device)

    arm_l_pos = robot.data.body_state_w[env_ids, left_body_id, :3]
    arm_l_quat = robot.data.body_state_w[env_ids, left_body_id, 3:7]
    arm_r_pos = robot.data.body_state_w[env_ids, right_body_id, :3]
    arm_r_quat = robot.data.body_state_w[env_ids, right_body_id, 3:7]

    ee_l_pos, ee_l_quat = math_utils.combine_frame_transforms(arm_l_pos, arm_l_quat, ee_offset_pos, ee_offset_quat)
    ee_r_pos, ee_r_quat = math_utils.combine_frame_transforms(arm_r_pos, arm_r_quat, ee_offset_pos, ee_offset_quat)

    housing.write_root_pose_to_sim(torch.cat([ee_l_pos, ee_l_quat], dim=-1), env_ids=env_ids)
    housing.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    insert.write_root_pose_to_sim(torch.cat([ee_r_pos, ee_r_quat], dim=-1), env_ids=env_ids)
    insert.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def randomize_buckle_grasp_pose(
    env,
    env_ids,
    rotation_axis: tuple[float, float, float],
    angle_range_deg: float | tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: tuple[str, ...] = (
        "arm_r_joint1",
        "arm_r_joint2",
        "arm_r_joint3",
        "arm_r_joint4",
        "arm_r_joint5",
        "arm_r_joint6",
        "arm_r_joint7",
    ),
    body_name: str = "arm_r_link7",
    num_iterations: int = 16,
):
    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(list(joint_names))
    if len(joint_ids) != len(joint_names):
        raise ValueError(f"Expected {len(joint_names)} joint matches for right arm, found {len(joint_ids)}.")
    body_ids, body_names = asset.find_bodies(body_name)
    if len(body_ids) != 1:
        raise ValueError(f"Expected one match for {body_name}, found {len(body_ids)}: {body_names}.")

    ee_body_idx = body_ids[0]
    if asset.is_fixed_base:
        jacobi_ee_body_idx = ee_body_idx - 1
        jacobi_joint_idx = joint_ids
    else:
        jacobi_ee_body_idx = ee_body_idx
        jacobi_joint_idx = [joint_id + 6 for joint_id in joint_ids]

    if isinstance(angle_range_deg, (int, float)):
        magnitude = abs(float(angle_range_deg))
        min_deg, max_deg = -magnitude, magnitude
    else:
        if len(angle_range_deg) != 2:
            raise ValueError("angle_range_deg must be a float or a (min_deg, max_deg) tuple.")
        min_deg, max_deg = float(angle_range_deg[0]), float(angle_range_deg[1])

    axis = torch.tensor(rotation_axis, device=env.device, dtype=torch.float32)
    axis_norm = torch.linalg.vector_norm(axis)
    if axis_norm <= 0:
        raise ValueError("rotation_axis must be non-zero.")
    axis = (axis / axis_norm).unsqueeze(0).repeat(len(env_ids), 1)

    delta_angle = torch.empty(len(env_ids), device=env.device, dtype=torch.float32).uniform_(
        torch.deg2rad(torch.tensor(min_deg, device=env.device, dtype=torch.float32)),
        torch.deg2rad(torch.tensor(max_deg, device=env.device, dtype=torch.float32)),
    )
    delta_quat = math_utils.quat_from_angle_axis(delta_angle, axis)

    controller = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=len(env_ids),
        device=env.device,
    )

    offset_pos = _EE_OFFSET_POS.to(device=env.device).repeat(len(env_ids), 1)
    offset_rot = _EE_OFFSET_QUAT.to(device=env.device).repeat(len(env_ids), 1)
    joint_vel = asset.data.default_joint_vel[env_ids][:, joint_ids].clone()
    root_pos_w = asset.data.root_pos_w[env_ids]
    root_quat_w = asset.data.root_quat_w[env_ids]
    insert: RigidObject = env.scene["insert"]
    target_ee_pos_w = insert.data.root_pos_w[env_ids]
    target_ee_quat_w = math_utils.quat_mul(insert.data.root_quat_w[env_ids], delta_quat)
    target_ee_pos_b, target_ee_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, target_ee_pos_w, target_ee_quat_w
    )
    controller.set_command(torch.cat([target_ee_pos_b, target_ee_quat_b], dim=-1))

    for _ in range(num_iterations):
        ee_body_pos_w = asset.data.body_pos_w[env_ids, ee_body_idx]
        ee_body_quat_w = asset.data.body_quat_w[env_ids, ee_body_idx]
        ee_pos_w, ee_quat_w = math_utils.combine_frame_transforms(ee_body_pos_w, ee_body_quat_w, offset_pos, offset_rot)
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

        jacobian = asset.root_physx_view.get_jacobians()[env_ids][:, jacobi_ee_body_idx, :, jacobi_joint_idx]
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(offset_rot), jacobian[:, 3:, :])

        current_joint_pos = asset.data.joint_pos[env_ids][:, joint_ids]
        desired_joint_pos = controller.compute(ee_pos_b, ee_quat_b, jacobian, current_joint_pos)

        joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids, :]
        desired_joint_pos = desired_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

        asset.set_joint_position_target(desired_joint_pos, joint_ids=joint_ids, env_ids=env_ids)
        asset.set_joint_velocity_target(joint_vel, joint_ids=joint_ids, env_ids=env_ids)
        asset.write_joint_state_to_sim(desired_joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)


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

    for env_id in env_ids.tolist():
        left_joint_path = f"/World/envs/env_{env_id}/LeftHousingFixedJoint"
        right_joint_path = f"/World/envs/env_{env_id}/RightInsertFixedJoint"
        left_joint_exists = stage.GetPrimAtPath(left_joint_path).IsValid()
        right_joint_exists = stage.GetPrimAtPath(right_joint_path).IsValid()
        robot_root_path = robot.root_physx_view.prim_paths[env_id]
        housing_root_path = housing.root_physx_view.prim_paths[env_id]
        insert_root_path = insert.root_physx_view.prim_paths[env_id]

        if not bool(env._buckle_collision_filters_created[env_id].item()):
            _filter_robot_buckle_collisions(stage, robot_root_path, housing_root_path)
            _filter_robot_buckle_collisions(stage, robot_root_path, insert_root_path)
            env._buckle_collision_filters_created[env_id] = True

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
        housing_root_pos = housing.data.root_pos_w[env_id : env_id + 1]
        housing_root_quat = housing.data.root_quat_w[env_id : env_id + 1]
        insert_root_pos = insert.data.root_pos_w[env_id : env_id + 1]
        insert_root_quat = insert.data.root_quat_w[env_id : env_id + 1]

        housing_local_pos1, housing_local_rot1 = math_utils.subtract_frame_transforms(
            housing_root_pos, housing_root_quat, ee_l_pos, ee_l_quat
        )
        insert_local_pos1, insert_local_rot1 = math_utils.subtract_frame_transforms(
            insert_root_pos, insert_root_quat, ee_r_pos, ee_r_quat
        )

        target_env_ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
        zero_velocity = torch.zeros((1, 6), device=env.device)
        housing.write_root_velocity_to_sim(zero_velocity, env_ids=target_env_ids)
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
                actor1_path=housing_root_path,
                local_pos0=left_local_pos0,
                local_rot0=left_local_rot0,
                local_pos1=_vec3_to_gf(housing_local_pos1[0]),
                local_rot1=_quat_to_gf(housing_local_rot1[0]),
            )
        _set_joint_local_pose(
            stage=stage,
            joint_path=left_joint_path,
            local_pos0=left_local_pos0,
            local_rot0=left_local_rot0,
            local_pos1=_vec3_to_gf(housing_local_pos1[0]),
            local_rot1=_quat_to_gf(housing_local_rot1[0]),
        )
        if not right_joint_exists:
            _create_fixed_joint(
                stage=stage,
                joint_path=right_joint_path,
                actor0_path=right_actor0_path,
                actor1_path=insert_root_path,
                local_pos0=right_local_pos0,
                local_rot0=right_local_rot0,
                local_pos1=_vec3_to_gf(insert_local_pos1[0]),
                local_rot1=_quat_to_gf(insert_local_rot1[0]),
            )
        _set_joint_local_pose(
            stage=stage,
            joint_path=right_joint_path,
            local_pos0=right_local_pos0,
            local_rot0=right_local_rot0,
            local_pos1=_vec3_to_gf(insert_local_pos1[0]),
            local_rot1=_quat_to_gf(insert_local_rot1[0]),
        )

        env._buckle_fixed_joints_created[env_id] = stage.GetPrimAtPath(left_joint_path).IsValid() and stage.GetPrimAtPath(
            right_joint_path
        ).IsValid()
