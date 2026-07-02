import numpy as np 
import os

# Point directly to the platform_29dof.xml in the same folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "platform_29dof.xml")

# Path to the mjlab trained ONNX model
ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "rsl_rl", "g1_velocity", "2026-07-01_18-08-36", "policy.onnx")

# Desired velocity command: [vx, vy, yaw_rate]
VELOCITY_COMMAND = np.array([0.5, 0.0, 0.0], dtype=np.float64)

# Gait phase period in seconds
PHASE_PERIOD = 0.6
# Physics dt * frame_skip = 0.005 * 4 = 0.02s
STEP_DT = 0.02

"""
Variable Structure
Left leg:  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
Waist:     yaw, roll, pitch
Left arm:  shoulder_p, shoulder_r, shoulder_y, elbow, wrist_r, wrist_p, wrist_y
Right arm: shoulder_p, shoulder_r, shoulder_y, elbow, wrist_r, wrist_p, wrist_y
"""

TORQUE_LIMITS = np.array([
    88, 88, 88, 139, 50, 50,        
    88, 88, 88, 139, 50, 50,        
    88, 50, 50,                     
    25, 25, 25, 25, 25, 5, 5,       
    25, 25, 25, 25, 25, 5, 5,       
], dtype=np.float64)

STANDING_HEIGHT = 0.8  # mjlab HOME_KEYFRAME height

# Action scale mapped from mjlab training config
ACTION_SCALE = np.array([
    0.547546, 0.350661, 0.547546, 0.350661, 0.438577, 0.438577, # left leg
    0.547546, 0.350661, 0.547546, 0.350661, 0.438577, 0.438577, # right leg
    0.547546, 0.438577, 0.438577,                               # waist
    0.438577, 0.438577, 0.438577, 0.438577, 0.438577, 0.074501, 0.074501, # left arm
    0.438577, 0.438577, 0.438577, 0.438577, 0.438577, 0.074501, 0.074501, # right arm
], dtype=np.float64)

# mjlab default joint positions (nominal offsets)
STANDING_POSE = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,     # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,     # right leg
    0.0, 0.0, 0.0,                            # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,        # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,       # right arm
], dtype=np.float64)

# PD gains (actual simulator actuator stiffness and damping)
kp = np.array([             
    40.179238, 99.098427, 40.179238, 99.098427, 28.501246, 28.501246, # left leg
    40.179238, 99.098427, 40.179238, 99.098427, 28.501246, 28.501246, # right leg
    40.179238, 28.501246, 28.501246,                                 # waist
    14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327, # left arm
    14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327, # right arm
], dtype=np.float64)

kd = np.array([
    2.557889, 6.308801, 2.557889, 6.308801, 1.814445, 1.814445, # left leg
    2.557889, 6.308801, 2.557889, 6.308801, 1.814445, 1.814445, # right leg
    2.557889, 1.814445, 1.814445,                               # waist
    0.907222, 0.907222, 0.907222, 0.907222, 0.907222, 1.068141, 1.068141, # left arm
    0.907222, 0.907222, 0.907222, 0.907222, 0.907222, 1.068141, 1.068141, # right arm
], dtype=np.float64)

MAX_EPISODE_STEPS = 2000