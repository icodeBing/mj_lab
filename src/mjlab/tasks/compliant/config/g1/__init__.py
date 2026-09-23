"""Unitree G1 compliant recovery tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_g1_compliant_flat_env_cfg, unitree_g1_compliant_rough_env_cfg
from .rl_cfg import unitree_g1_compliant_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Compliant-Rough-Unitree-G1",
  env_cfg=unitree_g1_compliant_rough_env_cfg(),
  play_env_cfg=unitree_g1_compliant_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_compliant_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Compliant-Flat-Unitree-G1",
  env_cfg=unitree_g1_compliant_flat_env_cfg(),
  play_env_cfg=unitree_g1_compliant_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_compliant_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
