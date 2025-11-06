
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import homestri_ur5e_rl
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[2]

# Callbacks
from pju_final_callbacks import (
    UR5eTaskWrapper,
    DualCurriculum,
    DualCurriculumCallback,
    StageEvalCallback,
    GoalGMMSampler,
    RandomizedGoalEnv
)

# DoF Sequences
ARC_DOF_SEQUENCE = [1, 3, 4, 5, 6]
DESCENT_DOF_SEQUENCE = [1, 4, 5, 6]

# Arc tolerances
ARC_STAGE_CONFIGS = {
    1: {"angle_tol": 5, "speed_tol": 0.05, "dwell_episodes": 5},
    2: {"distance_tol": 0.05, "speed_tol": 0.15, "dwell_episodes": 5},
    3: {"distance_tol": 0.05, "speed_tol": 0.15, "dwell_episodes": 5},
    4: {"distance_tol": 0.05, "speed_tol": 0.15, "orientation_tol": 10, "dwell_episodes": 5},
    5: {"distance_tol": 0.05, "speed_tol": 0.15, "orientation_tol": 10, "dwell_episodes": 5},
    6: {"distance_tol": 0.05, "speed_tol": 0.15, "orientation_tol": 10, "gripper_align_tol": 10, "dwell_episodes": 5},
}

# Descent tolerances
DESCENT_STAGE_CONFIGS = {
    1: {"angle_tol": 5, "speed_tol": 0.05, "dwell_episodes": 5},
    2: {"distance_tol": 0.05, "speed_tol": 0.1, "dwell_episodes": 5},
    3: {"distance_tol": 0.05, "speed_tol": 0.1, "dwell_episodes": 5},
    4: {"distance_tol": 0.05, "speed_tol": 0.1, "orientation_tol": 10, "dwell_episodes": 5},
    5: {"distance_tol": 0.05, "speed_tol": 0.1, "orientation_tol": 10, "dwell_episodes": 5},
    6: {"distance_tol": 0.03, "speed_tol": 0.1, "orientation_tol": 10, "gripper_align_tol": 10, "dwell_episodes": 5},
}

# Arc parameters
ARC_HEIGHT = 0.328
PROGRESS_RADIUS = 0.1
ORIENTATION_TOL_DEG = 10.0
ARC_CHECKPOINT_RADIUS = 0.03
ARC_CHECKPOINT_SPEED = 0.1

# Training parameters
SEED = 10
SAVE_FREQ = 5_000
TOTAL_TIMESTEPS = 1_000_000

# Goal randomisation parameters
USE_GOAL_RANDOMIZATION = True
NUM_ARC_FINAL_POSES = 10
GOAL_CENTER = np.array([-0.65, 0.1, 0.03])
GOAL_RADIUS = 0.15
GOAL_PEAK_HEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])
GOAL_COVS = [np.diag([0.01, 0.01]) for _ in range(4)]

# Directory definitions
ARC_FOLDER = "".join(str(d) for d in ARC_DOF_SEQUENCE)
DESCENT_FOLDER = "".join(str(d) for d in DESCENT_DOF_SEQUENCE)
LOG_DIR = ROOT / "models" / f"{ARC_FOLDER}_arc_{DESCENT_FOLDER}_descent"
LOG_DIR.mkdir(parents=True, exist_ok=True)
ARC_FINAL_POSES_DIR = LOG_DIR / "arc_final_poses"
ARC_FINAL_POSES_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_PATH = ROOT / "homestri_ur5e_rl" / "envs" / "base_robot" / "pose.json"
DESIRED_GOAL = np.array([-0.65, 0.1, 0.03])

# Load model
LOAD_PREVIOUS_MODEL = False
STARTING_PHASE = "arc"
STARTING_STAGE = 2
ARC_MODEL_PATH = LOG_DIR / "best_arc_dof_3.zip"
DESCENT_MODEL_PATH = LOG_DIR / "best_descent_dof_6.zip"

if LOAD_PREVIOUS_MODEL:
    INITIAL_PHASE = STARTING_PHASE
    INITIAL_STAGE = STARTING_STAGE
else:
    INITIAL_PHASE = "arc"
    INITIAL_STAGE = 0

# Load pose
with open(SNAPSHOT_PATH, "r") as f:
    pose = json.load(f)

reward_config = {
    "arc_height": ARC_HEIGHT,
    "progress_radius": PROGRESS_RADIUS,
    "orientation_tolerance_deg": ORIENTATION_TOL_DEG,
    "arc_checkpoint_radius": ARC_CHECKPOINT_RADIUS,
    "arc_checkpoint_speed": ARC_CHECKPOINT_SPEED,
}

# Frozen joints handling
def _frozen_from_dof(dof: int):
    return {1: [1,2,3,4,5], 3: [3,4,5], 4: [4,5], 5: [5], 6: []}.get(dof, [])

# Initial DoF and frozen joints
start_dof = (ARC_DOF_SEQUENCE if INITIAL_PHASE == "arc" else DESCENT_DOF_SEQUENCE)[INITIAL_STAGE]
initial_frozen = _frozen_from_dof(start_dof)

# Environment making for training env and evaluation env
def make_env(render_mode=None):
    env = gym.make(
        "BaseRobot-v0",
        render_mode=render_mode,
        init_qpos_config=pose["arm_init_qpos"],
        desired_goal=DESIRED_GOAL,
        box_start_xyz=pose["box_start_xyz"],
        box_start_rpy=pose["box_start_rpy"],
        gripper_start=pose["gripper_start"],
        frozen_joints=initial_frozen,
        reward_config=reward_config,
        dof_sequence_arc=ARC_DOF_SEQUENCE,
        dof_sequence_descent=DESCENT_DOF_SEQUENCE,
        training_phase=INITIAL_PHASE,
        starting_stage=INITIAL_STAGE,
        arc_final_poses_dir=ARC_FINAL_POSES_DIR
    )
    env.unwrapped.set_training_phase(INITIAL_PHASE, starting_stage=INITIAL_STAGE)
    env.unwrapped.set_curriculum_stage(INITIAL_STAGE)
    return env

env = make_env("human")
eval_env = make_env(None)

# Goal randomisation wrappers
if USE_GOAL_RANDOMIZATION:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gmm_sampler = GoalGMMSampler(GOAL_CENTER, GOAL_RADIUS, GOAL_PEAK_HEIGHTS, GOAL_COVS, device=device)
    env = RandomizedGoalEnv(env, gmm_sampler)
    eval_env = RandomizedGoalEnv(eval_env, gmm_sampler)

# Environment wrappers
env = Monitor(UR5eTaskWrapper(env, ARC_DOF_SEQUENCE, DESCENT_DOF_SEQUENCE), str(LOG_DIR))
eval_env = Monitor(UR5eTaskWrapper(eval_env, ARC_DOF_SEQUENCE, DESCENT_DOF_SEQUENCE))

# Curriculum initialisation
curriculum = DualCurriculum(
    ARC_DOF_SEQUENCE,
    DESCENT_DOF_SEQUENCE,
    ARC_STAGE_CONFIGS,
    DESCENT_STAGE_CONFIGS,
    INITIAL_PHASE
)

# Setting phase and stage
if INITIAL_PHASE == "arc":
    curriculum.arc_idx = INITIAL_STAGE
else:
    curriculum.descent_idx = INITIAL_STAGE

# Initialising previous or new model
if LOAD_PREVIOUS_MODEL:
    model_path = ARC_MODEL_PATH if INITIAL_PHASE == "arc" else DESCENT_MODEL_PATH
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = PPO.load(model_path, env=env, device="cuda")
    model.learning_rate = 1e-4
    model.gamma = 0.97
else:
    model = PPO(
        "MultiInputPolicy",
        env,
        seed=SEED,
        verbose=1,
        learning_rate=1e-4,
        gamma=0.97,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        policy_kwargs=dict(net_arch=[256, 256], activation_fn=nn.ReLU),
        device="cuda",
    )

# Initialising callbacks
curriculum_cb = DualCurriculumCallback(
    curriculum,
    ARC_STAGE_CONFIGS,
    DESCENT_STAGE_CONFIGS,
    eval_env,
    log_dir=LOG_DIR,
    verbose=1,
    use_goal_randomization=USE_GOAL_RANDOMIZATION,
    num_arc_final_poses=NUM_ARC_FINAL_POSES
)

eval_cb = StageEvalCallback(
    curriculum,
    ARC_DOF_SEQUENCE,
    DESCENT_DOF_SEQUENCE,
    curriculum_cb,
    eval_env=eval_env,
    eval_freq=SAVE_FREQ,
    n_eval_episodes=5,
    deterministic=True,
    best_model_save_path=LOG_DIR,
    log_path=os.path.join(LOG_DIR, "eval_logs"),
    verbose=1,
)

curriculum_cb.eval_cb_ref = eval_cb

checkpoint_cb = CheckpointCallback(save_freq=SAVE_FREQ, save_path=LOG_DIR, name_prefix="checkpoint")

model.env.envs[0].unwrapped.arc_final_poses_dir = ARC_FINAL_POSES_DIR

# Begin training
print(f"Starting {INITIAL_PHASE.upper()} phase at stage {INITIAL_STAGE} (DoF {start_dof})")

# Begin training
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=CallbackList([curriculum_cb, eval_cb, checkpoint_cb]),
    progress_bar=True,
)

print("Training complete")

