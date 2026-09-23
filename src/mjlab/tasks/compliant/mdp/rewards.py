"""Phase-aware rewards for compliant humanoid locomotion."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg


def _phase_masks(
  env,
  walking_duration: float,
  recovery_duration: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  time = env.episode_length_buf.to(torch.float32) * env.step_dt
  walking = time < walking_duration
  recovery = (time >= walking_duration) & (
    time < walking_duration + recovery_duration
  )
  post_recovery = time >= walking_duration + recovery_duration
  return walking, recovery, post_recovery


def phase_linear_velocity_tracking(
  env,
  std: float,
  command_name: str,
  walking_duration: float,
  recovery_duration: float,
  recovery_command_scale: float = 0.35,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Track commands normally, then use a conservative recovery target."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  asset: Entity = env.scene[asset_cfg.name]
  actual = asset.data.root_link_lin_vel_b
  _, recovery, _ = _phase_masks(env, walking_duration, recovery_duration)
  target = command[:, :2].clone()
  target[recovery] *= recovery_command_scale
  error = torch.sum(torch.square(target - actual[:, :2]), dim=1)
  error += torch.square(actual[:, 2])
  return torch.exp(-error / std**2)


def phase_angular_velocity_tracking(
  env,
  std: float,
  command_name: str,
  walking_duration: float,
  recovery_duration: float,
  recovery_command_scale: float = 0.35,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Track yaw commands normally and reduce them during recovery."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  asset: Entity = env.scene[asset_cfg.name]
  actual = asset.data.root_link_ang_vel_b
  _, recovery, _ = _phase_masks(env, walking_duration, recovery_duration)
  target_z = command[:, 2].clone()
  target_z[recovery] *= recovery_command_scale
  error = torch.square(target_z - actual[:, 2])
  error += torch.sum(torch.square(actual[:, :2]), dim=1)
  return torch.exp(-error / std**2)


def recovery_stability(
  env,
  walking_duration: float,
  recovery_duration: float,
  asset_cfg: SceneEntityCfg,
  std: float = 0.5,
) -> torch.Tensor:
  """Reward low torso roll/pitch angular velocity during recovery only."""
  _, recovery, _ = _phase_masks(env, walking_duration, recovery_duration)
  asset: Entity = env.scene[asset_cfg.name]
  body_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  body_vel = body_vel.squeeze(1)
  value = torch.exp(-torch.sum(torch.square(body_vel[:, :2]), dim=1) / std**2)
  return value * recovery.float()
