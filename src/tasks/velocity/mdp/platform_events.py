"""Platform-specific MDP functions for the moving platform environment.

Contains:
- set_platform_velocity: Stateful step event for smoothly ramping platform velocity.
- track_linear_velocity_platform_relative: Reward for velocity tracking relative to platform.
- feet_slip_platform_relative: Reward penalizing foot slip relative to platform surface.
- platform_velocity_curriculum: Curriculum for increasing platform speed and ramp sharpness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ---------------------------------------------------------------------------
# Helper: resolve platform joint IDs from SceneEntityCfg
# ---------------------------------------------------------------------------

def _resolve_platform_joint_ids(
  asset_cfg: SceneEntityCfg,
  device: torch.device,
) -> torch.Tensor:
  """Extract already-resolved joint IDs from a SceneEntityCfg as a tensor.

  The SceneEntityCfg has already been resolved by _resolve_common_term_cfg
  by the time class-based event __init__ is called, so joint_ids is populated.
  """
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    raise ValueError(
      "Platform SceneEntityCfg must specify explicit joint_names so that "
      "joint_ids are resolved to a list, not slice(None)."
    )
  return torch.tensor(joint_ids, device=device, dtype=torch.long)


class set_platform_velocity:
  """Smoothly ramp platform velocity toward randomly sampled targets.

  Lifecycle per environment:
    1. Hold current velocity for a random duration (hold_time_s).
    2. Sample a new target velocity from velocity_range.
    3. Ramp toward target at ramp_rate (m/s per second).
    4. Once target is reached (or hold timer expires), go to step 1.

  The ramp_rate controls sharpness of velocity changes:
    - Low ramp_rate (e.g. 0.5): gentle, smooth acceleration.
    - High ramp_rate (e.g. 20.0): near-instantaneous jumps.

  Use with ``mode="step"`` so it runs every environment step.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt

    # Resolve the asset.
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    
    # Resolve the platform local actuator IDs.
    actuator_ids, _ = self._asset.find_actuators(("platform_x_vel", "platform_y_vel"))
    self._platform_ctrl_ids = torch.tensor(actuator_ids, device=self._device, dtype=torch.long)
    self._platform_global_ctrl_ids = self._asset.data.indexing.ctrl_ids[self._platform_ctrl_ids]

    # Resolve the platform joint IDs for reset.
    self._platform_joint_ids = _resolve_platform_joint_ids(asset_cfg, self._device)

    # State tensors (per-environment).
    self._current_vel = torch.zeros(self._num_envs, 2, device=self._device)
    self._target_vel = torch.zeros(self._num_envs, 2, device=self._device)
    self._hold_timer = torch.zeros(self._num_envs, device=self._device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    ramp_rate: float,
    hold_time_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
  ) -> None:
    """Tick platform velocity: ramp current toward target, resample expired."""
    del env_ids, asset_cfg  # Step events always operate on all envs.
    dt = self._step_dt

    # --- Decrement hold timers ---
    self._hold_timer -= dt

    # --- Resample targets for envs whose timer expired ---
    expired = self._hold_timer <= 0
    if expired.any():
      expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
      n = len(expired_ids)

      # Sample new target velocities.
      vx_range = velocity_range.get("x", (0.0, 0.0))
      vy_range = velocity_range.get("y", (0.0, 0.0))
      self._target_vel[expired_ids, 0] = (
        torch.rand(n, device=self._device) * (vx_range[1] - vx_range[0]) + vx_range[0]
      )
      self._target_vel[expired_ids, 1] = (
        torch.rand(n, device=self._device) * (vy_range[1] - vy_range[0]) + vy_range[0]
      )

      # Sample new hold durations.
      self._hold_timer[expired_ids] = (
        torch.rand(n, device=self._device) * (hold_time_s[1] - hold_time_s[0])
        + hold_time_s[0]
      )

    # --- Ramp current velocity toward target ---
    diff = self._target_vel - self._current_vel
    max_step = ramp_rate * dt
    # Clamp each component independently to respect the ramp rate.
    step = torch.clamp(diff, -max_step, max_step)
    self._current_vel += step

    # --- Apply to MuJoCo native ctrl inputs ---
    self._asset.data.data.ctrl[:, self._platform_global_ctrl_ids] = self._current_vel

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Zero platform velocity and timers on episode reset."""
    if env_ids is None:
      env_ids = slice(None)
    self._current_vel[env_ids] = 0.0
    self._target_vel[env_ids] = 0.0
    self._hold_timer[env_ids] = 0.0
    
    # Zero the ctrl target and joint target so platform stops.
    gctrl_ids = self._platform_global_ctrl_ids
    jids = self._platform_joint_ids
    if isinstance(env_ids, slice):
      self._asset.data.data.ctrl[:, gctrl_ids] = 0.0
      self._asset.data.joint_vel_target[:, jids] = 0.0
    else:
      self._asset.data.data.ctrl[env_ids.unsqueeze(1), gctrl_ids.unsqueeze(0)] = 0.0
      self._asset.data.joint_vel_target[env_ids.unsqueeze(1), jids.unsqueeze(0)] = 0.0


# ---------------------------------------------------------------------------
# Reward: Platform-relative velocity tracking
# ---------------------------------------------------------------------------

class track_linear_velocity_platform_relative:
  """Reward for tracking commanded velocity relative to the platform.

  On a moving platform, the robot's world-frame velocity includes the
  platform's velocity. The command is relative to the platform surface,
  so we subtract the platform velocity before computing the tracking error.

  Replaces ``track_linear_velocity`` for the platform environment.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._asset: Entity = env.scene[asset_cfg.name]
    # Look up platform joint IDs by name (these are exact joint names).
    joint_ids, _ = self._asset.find_joints(("platform_x", "platform_y"))
    self._platform_joint_ids = torch.tensor(
      joint_ids, device=env.device, dtype=torch.long
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    # Robot velocity in body frame (includes platform motion).
    robot_vel_b = asset.data.root_link_lin_vel_b

    # Platform velocity in world frame (from slide joint velocities).
    platform_vel_w = torch.zeros(env.num_envs, 3, device=env.device)
    platform_vel_w[:, 0] = asset.data.joint_vel[:, self._platform_joint_ids[0]]
    platform_vel_w[:, 1] = asset.data.joint_vel[:, self._platform_joint_ids[1]]

    # Rotate platform velocity into robot body frame.
    platform_vel_b = quat_apply_inverse(
      asset.data.root_link_quat_w, platform_vel_w
    )

    # Robot velocity relative to platform, in body frame.
    relative_vel_b = robot_vel_b - platform_vel_b

    xy_error = torch.sum(
      torch.square(command[:, :2] - relative_vel_b[:, :2]), dim=1
    )
    z_error = torch.square(relative_vel_b[:, 2])
    lin_vel_error = xy_error + (2 * z_error)
    return torch.exp(-lin_vel_error / std**2)


# ---------------------------------------------------------------------------
# Reward: Platform-relative foot slip penalty
# ---------------------------------------------------------------------------

class feet_slip_platform_relative:
  """Penalize foot sliding relative to the platform surface.

  On a moving platform, feet in contact should move at the platform's
  velocity. This subtracts the platform's world-frame XY velocity from
  the foot velocity before computing the slip penalty.

  Replaces ``feet_slip`` for the platform environment.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._asset: Entity = env.scene[asset_cfg.name]
    # Look up platform joint IDs by name.
    joint_ids, _ = self._asset.find_joints(("platform_x", "platform_y"))
    self._platform_joint_ids = torch.tensor(
      joint_ids, device=env.device, dtype=torch.long
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()

    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()  # [B, N]

    # Foot world-frame velocity.
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]

    # Platform world-frame velocity (broadcast over feet).
    platform_vel_xy = torch.stack([
      asset.data.joint_vel[:, self._platform_joint_ids[0]],
      asset.data.joint_vel[:, self._platform_joint_ids[1]],
    ], dim=-1).unsqueeze(1)  # [B, 1, 2]

    # Foot velocity relative to platform.
    relative_foot_vel_xy = foot_vel_xy - platform_vel_xy  # [B, N, 2]

    vel_xy_norm = torch.norm(relative_foot_vel_xy, dim=-1)  # [B, N]
    vel_xy_norm_sq = torch.square(vel_xy_norm)
    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active

    # Log mean slip velocity for metrics.
    num_in_contact = torch.sum(in_contact)
    mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
      num_in_contact, min=1
    )
    env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
    return cost


# ---------------------------------------------------------------------------
# Curriculum: Platform velocity range and ramp rate
# ---------------------------------------------------------------------------

class PlatformVelocityStage(TypedDict):
  step: int
  x: tuple[float, float]
  y: tuple[float, float]
  ramp_rate: float


def platform_velocity_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  velocity_stages: list[PlatformVelocityStage],
) -> dict[str, torch.Tensor]:
  """Gradually increase platform velocity range and ramp sharpness.

  Modifies the params of the ``set_platform_velocity`` event term in-place
  based on the current training step. Follows the same pattern as
  ``commands_vel``.
  """
  del env_ids  # Unused.
  term_cfg = env.event_manager.get_term_cfg(event_name)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      term_cfg.params["velocity_range"] = {
        "x": stage["x"],
        "y": stage["y"],
      }
      term_cfg.params["ramp_rate"] = stage["ramp_rate"]
  return {}
