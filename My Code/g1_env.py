import numpy as np
import math
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from g1_config import *

class G1Env(MujocoEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": int(1/STEP_DT)}

    def __init__(self, render_mode=None):

        # Observation space is 98-dim based on the mjlab policy
        observation_space = Box(low=-np.inf, high=np.inf, shape=(98,), dtype=np.float32)
        # Use frame_skip=10 initially so that self.dt = 0.002 * 10 = 0.02s (matches render_fps=50)
        super().__init__(model_path=MODEL_PATH, frame_skip=10, observation_space=observation_space, render_mode=render_mode)
        self.action_space = Box(low=-1.0, high=1.0, shape=(29,), dtype=np.float32)
        self._step_count = 0
        self._prev_action = np.zeros(29, dtype=np.float32)

        # Now override to match mjlab solver and physics configuration exactly (0.005s timestep * 4 decimation = 0.02s dt)
        self.model.opt.timestep = 0.005
        self.frame_skip = 4
        self.model.opt.iterations = 10
        self.model.opt.ls_iterations = 20

        # Match robot joint armatures and damping with mjlab exactly
        # Joints map to indices 6 to 34 in dof arrays (free joint takes 6 indices)
        self.model.dof_armature[6:35] = [
            0.01017752, 0.02510192, 0.01017752, 0.02510192, 0.00721945, 0.00721945, # left leg
            0.01017752, 0.02510192, 0.01017752, 0.02510192, 0.00721945, 0.00721945, # right leg
            0.01017752, 0.00721945, 0.00721945,                                     # waist
            0.00360973, 0.00360973, 0.00360973, 0.00360973, 0.00360973, 0.00425, 0.00425, # left arm
            0.00360973, 0.00360973, 0.00360973, 0.00360973, 0.00360973, 0.00425, 0.00425  # right arm
        ]
        self.model.dof_damping[6:35] = 0.0

    def _get_obs(self):
        pelvis_xmat = self.data.body("pelvis").xmat.reshape(3, 3)
        base_ang_vel = self.data.qvel[3:6]
        body_ang_vel = base_ang_vel
        projected_gravity = pelvis_xmat.T @ np.array([0.0, 0.0, -1.0])
        
        # Command (vx, vy, yaw_rate)
        command = VELOCITY_COMMAND

        # Gait phase
        global_phase = (self._step_count * STEP_DT) % PHASE_PERIOD / PHASE_PERIOD
        phase = np.array([
            math.sin(global_phase * math.pi * 2.0),
            math.cos(global_phase * math.pi * 2.0)
        ], dtype=np.float32)
        
        # If command norm is very small, phase is 0 (stand mask in mjlab)
        if np.linalg.norm(command) < 0.1:
            phase = np.zeros(2, dtype=np.float32)

        # Joint positions relative to STANDING_POSE
        joint_pos = self.data.qpos[7:36] - STANDING_POSE
        joint_vel = self.data.qvel[6:35]
        
        obs = np.concatenate([
            body_ang_vel,       # 3
            projected_gravity,  # 3
            command,            # 3
            phase,              # 2
            joint_pos,          # 29
            joint_vel,          # 29
            self._prev_action   # 29
        ], dtype=np.float32)    # Total: 98
        
        return obs
       
    def reset_model(self):
        qpos = np.zeros(self.model.nq)
        qpos[2] = STANDING_HEIGHT       # z
        qpos[3] = 1.0                   # quaternion w
        qpos[7:36] = STANDING_POSE.copy()
        
        qvel = np.zeros(self.model.nv) 

        self.set_state(qpos, qvel)
        self._step_count = 0
        self._prev_action = np.zeros(29, dtype=np.float32)

        return self._get_obs()

    def step(self, action):
        self._step_count += 1
        
        # Target position from policy action
        target_q = STANDING_POSE + ACTION_SCALE * action

        for _ in range(self.frame_skip):
            q = self.data.qpos[7:36]      # joint positions
            qd = self.data.qvel[6:35]     # joint velocities
            torque = kp * (target_q - q) - kd * qd
            torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)
            self.data.ctrl[:29] = torque
            self.data.ctrl[29:31] = 0.0
            mujoco.mj_step(self.model, self.data)

        self._prev_action = action
        obs = self._get_obs()
        
        height = self.data.qpos[2]
        upright = float(self.data.body("pelvis").xmat[8])

        terminated = bool(height < 0.3 or upright < 0.5)
        truncated = bool(self._step_count >= MAX_EPISODE_STEPS)
        info = {}
        
        # Dummy reward since we're just doing inference
        reward = 0.0
        
        return (obs, reward, terminated, truncated, info)

def make_env(rank: int):
    def _init():
        env = G1Env(render_mode=None)
        env.reset(seed=1000 + rank)
        return env
    return _init