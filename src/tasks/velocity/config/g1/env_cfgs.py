"""Unitree G1 velocity environment configurations."""

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
  get_g1_platform_robot_cfg,
)
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_g1_platform_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 moving platform configuration.

  The platform moves with random velocities during training so the robot
  learns to balance on moving surfaces (e.g. bus, train). Velocity tracking
  and foot slip rewards use platform-relative computations.
  """
  cfg = unitree_g1_flat_env_cfg(play=play)

  import src.tasks.velocity.mdp as src_mdp

  # 1. Load robot config with the platform and enable platform collisions.
  robot_cfg = get_g1_platform_robot_cfg()
  from mjlab.utils.spec_config import CollisionCfg
  platform_collision = CollisionCfg(
    geom_names_expr=(".*_collision", "platform"),
    condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1, "platform": 3},
    priority={r"^(left|right)_foot[1-7]_collision$": 1, "platform": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,), "platform": (0.6,)},
  )
  robot_cfg.collisions = (platform_collision,)
  cfg.scene.entities = {"robot": robot_cfg}

  # 2. Disable default terrain.
  cfg.scene.terrain = None

  # 3. Update contact sensor pattern to use prefixed platform name.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "feet_ground_contact":
      sensor.secondary.pattern = "robot/platform"

  # 4. Filter actor and critic joint observation spaces back to 29 joints.
  #    (Platform slide joints must not appear in the observation space.)
  for group_name in ["actor", "critic"]:
    group = cfg.observations[group_name]

    if "joint_pos" in group.terms:
      term = group.terms["joint_pos"]
      group.terms["joint_pos"] = ObservationTermCfg(
        func=term.func,
        noise=term.noise,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_joint",))}
      )

    if "joint_vel" in group.terms:
      term = group.terms["joint_vel"]
      group.terms["joint_vel"] = ObservationTermCfg(
        func=term.func,
        noise=term.noise,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_joint",))}
      )

  # 5. Filter event joint resets to robot joints only.
  if "reset_robot_joints" in cfg.events:
    cfg.events["reset_robot_joints"].params["asset_cfg"].joint_names = (".*_joint",)

  # 6. Filter posture and joint-limit rewards to prevent tracking the platform joints.
  if "pose" in cfg.rewards:
    cfg.rewards["pose"].params["asset_cfg"].joint_names = ".*_joint"
  if "stand_still" in cfg.rewards:
    cfg.rewards["stand_still"].params["asset_cfg"].joint_names = ".*_joint"
  if "joint_pos_limits" in cfg.rewards:
    cfg.rewards["joint_pos_limits"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=".*_joint")
  if "joint_acc_l2" in cfg.rewards:
    cfg.rewards["joint_acc_l2"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=".*_joint")

  # 7. Explicitly filter action space to robot joints only.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (".*_joint",)

  # ------------------------------------------------------------------
  # 8. Moving platform events.
  # ------------------------------------------------------------------

  # 8a. Step event: smoothly ramp platform velocity toward random targets.
  cfg.events["set_platform_velocity"] = EventTermCfg(
    func=src_mdp.set_platform_velocity,
    mode="step",
    params={
      "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
      "ramp_rate": 0.5,
      "hold_time_s": (2.0, 5.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=("platform_x", "platform_y")),
    },
  )

  # 8b. Reset event: zero platform joint position & velocity each episode.
  cfg.events["reset_platform_joints"] = EventTermCfg(
    func=src_mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=("platform_x", "platform_y")),
    },
  )

  # ------------------------------------------------------------------
  # 9. Platform-relative rewards.
  # ------------------------------------------------------------------

  # 9a. Replace velocity tracking with platform-relative version.
  import math
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=src_mdp.track_linear_velocity_platform_relative,
    weight=1.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )

  # 9b. Replace foot slip with platform-relative version.
  site_names = ("left_foot", "right_foot")
  cfg.rewards["foot_slip"] = RewardTermCfg(
    func=src_mdp.feet_slip_platform_relative,
    weight=-0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )

  # 9c. Adjust action rate penalty for platform walking.
  if "action_rate_l2" in cfg.rewards:
    cfg.rewards["action_rate_l2"].weight = -0.03

  # ------------------------------------------------------------------
  # 10. Platform velocity curriculum (speed + sharpness).
  # ------------------------------------------------------------------
  cfg.curriculum["platform_velocity"] = CurriculumTermCfg(
    func=src_mdp.platform_velocity_curriculum,
    params={
      "event_name": "set_platform_velocity",
      "velocity_stages": [
        # Stage 0: Gentle, slow platform motion.
        {"step": 10000 * 24,
         "x": (-0.8, 0.8), "y": (-0.8, 0.8),
         "ramp_rate": 5.0},
        # Stage 1 (~5k iters): Medium speed, moderate ramp.
        {"step": 10500 * 24,
         "x": (-1.4, 1.4), "y": (-1.4, 1.4),
         "ramp_rate": 10.0},
        # Stage 2 (~10k iters): Full speed, sharp changes.
        {"step": 12000 * 24,
         "x": (-2.0, 2.0), "y": (-2.0, 2.0),
         "ramp_rate": 20.0},
      ],
    },
  )

  # Apply play mode overrides for the platform.
  if play:
    # Disable platform velocity randomization during play.
    cfg.events.pop("set_platform_velocity", None)
    cfg.curriculum.pop("platform_velocity", None)

  return cfg
