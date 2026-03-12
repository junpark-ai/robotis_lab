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

from collections.abc import Sequence

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction


class DefaultClosedBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary gripper action that resets to the close command by default."""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids=env_ids)
        self._raw_actions[env_ids] = -1.0
        self._processed_actions[env_ids] = self._close_command
