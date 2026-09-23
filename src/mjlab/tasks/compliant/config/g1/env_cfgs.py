"""Unitree G1 compliant recovery environment configurations."""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.compliant import mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)


def _configure_compliant(cfg):
  """Apply phase-aware recovery shaping to an existing G1 velocity config."""
  cfg.episode_length_s = 4.0

  # The fixed-time kick starts the one-second recovery window at t=2 s.
  push = cfg.events.get("push_robot")
  if push is not None:
    push.interval_range_s = (2.0, 2.0)

  phase_params = {
    "command_name": "twist",
    "walking_duration": 2.0,
    "recovery_duration": 1.0,
  }
  cfg.rewards["track_linear_velocity"].func = mdp.phase_linear_velocity_tracking
  cfg.rewards["track_linear_velocity"].params.update(phase_params)
  cfg.rewards["track_angular_velocity"].func = mdp.phase_angular_velocity_tracking
  cfg.rewards["track_angular_velocity"].params.update(phase_params)

  cfg.rewards["recovery_stability"] = RewardTermCfg(
    func=mdp.recovery_stability,
    weight=1.0,
    params={
      "walking_duration": 2.0,
      "recovery_duration": 1.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "std": 0.5,
    },
  )
  return cfg


def unitree_g1_compliant_rough_env_cfg(play: bool = False):
  cfg = _configure_compliant(unitree_g1_rough_env_cfg(play=play))
  if play:
    # Play mode is continuous and should retain the original viewer behavior.
    cfg.episode_length_s = int(1e9)
    cfg.events.pop("push_robot", None)
  return cfg


def unitree_g1_compliant_flat_env_cfg(play: bool = False):
  cfg = _configure_compliant(unitree_g1_flat_env_cfg(play=play))
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.events.pop("push_robot", None)
  return cfg
