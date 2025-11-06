import os
import json
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import mujoco
import homestri_ur5e_rl
from stable_baselines3 import PPO
from pju_final_callbacks import UR5eTaskWrapper, GoalGMMSampler

ROOT = Path(__file__).resolve().parents[2]

TEST_PHASE = "arc"  # "arc", "descent", or "cascade"
TEST_STAGE = 0  
CASCADE_ARC_DISTANCE_TOL = 0.1  

ARC_DOF_SEQUENCE = [1, 3, 4, 5, 6]
DESCENT_DOF_SEQUENCE = [1, 4, 5, 6]

ARC_STAGE_CONFIGS = {
    1: {"angle_tol": 5, "speed_tol": 0.05},
    2: {"distance_tol": 0.05, "speed_tol": 0.15},
    3: {"distance_tol": 0.05, "speed_tol": 0.15},
    4: {"distance_tol": 0.05, "speed_tol": 0.15, "orientation_tol": 10},
    5: {"distance_tol": 0.05, "speed_tol": 0.15, "orientation_tol": 10},
    6: {"distance_tol": 0.05, "speed_tol": 0.15, "orientation_tol": 10, "gripper_align_tol": 10},
}

DESCENT_STAGE_CONFIGS = {
    1: {"angle_tol": 5, "speed_tol": 0.05},
    2: {"distance_tol": 0.05, "speed_tol": 0.1},
    3: {"distance_tol": 0.05, "speed_tol": 0.1},
    4: {"distance_tol": 0.05, "speed_tol": 0.1, "orientation_tol": 10},
    5: {"distance_tol": 0.05, "speed_tol": 0.1, "orientation_tol": 10},
    6: {"distance_tol": 0.03, "speed_tol": 0.1, "orientation_tol": 10, "gripper_align_tol": 10},
}

# Arc/reward parameters
ARC_HEIGHT = 0.328
PROGRESS_RADIUS = 0.1
ORIENTATION_TOL_DEG = 10.0
ARC_CHECKPOINT_RADIUS = 0.03
ARC_CHECKPOINT_SPEED = 0.1

# Goal randomization
USE_GOAL_RANDOMIZATION = False
GOAL_CENTER = np.array([-0.65, 0.1, 0.03])
GOAL_RADIUS = 0.15
GOAL_PEAK_HEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])
GOAL_COVS = [np.diag([0.01, 0.01]) for _ in range(4)]

# Paths
ARC_FOLDER = "".join(map(str, ARC_DOF_SEQUENCE))
DESCENT_FOLDER = "".join(map(str, DESCENT_DOF_SEQUENCE))

LOG_DIR = ROOT / "models" / f"{ARC_FOLDER}_arc_{DESCENT_FOLDER}_descent"
ARC_FINAL_POSES_DIR = LOG_DIR / "arc_final_poses"
SNAPSHOT_PATH = ROOT / "homestri_ur5e_rl" / "envs" / "base_robot" / "pose.json"

DESIRED_GOAL = np.array([-0.65, 0.1, 0.03])


def frozen_joints_for_dof(dof):
    return {1: [1,2,3,4,5], 3: [3,4,5], 4: [4,5], 5: [5], 6: []}.get(dof, [])

def make_env(phase, stage, render_mode="human"):
    pose = json.load(open(SNAPSHOT_PATH)) if Path(SNAPSHOT_PATH).exists() else {
        "arm_init_qpos": {}, "box_start_xyz": DESIRED_GOAL.tolist(),
        "box_start_rpy": [0.0, 0.0, 0.0], "gripper_start": [0.0, 0.0]
    }
    
    dof = (ARC_DOF_SEQUENCE if phase == "arc" else DESCENT_DOF_SEQUENCE)[stage]
    env = gym.make(
        "BaseRobot-v0", render_mode=render_mode,
        init_qpos_config=pose["arm_init_qpos"], desired_goal=DESIRED_GOAL,
        box_start_xyz=pose["box_start_xyz"], box_start_rpy=pose["box_start_rpy"],
        gripper_start=pose["gripper_start"], frozen_joints=frozen_joints_for_dof(dof),
        reward_config={
            "arc_height": ARC_HEIGHT, "progress_radius": PROGRESS_RADIUS,
            "orientation_tolerance_deg": ORIENTATION_TOL_DEG,
            "arc_checkpoint_radius": ARC_CHECKPOINT_RADIUS,
            "arc_checkpoint_speed": ARC_CHECKPOINT_SPEED,
        },
        dof_sequence_arc=ARC_DOF_SEQUENCE, dof_sequence_descent=DESCENT_DOF_SEQUENCE,
        training_phase=phase, starting_stage=stage, arc_final_poses_dir=ARC_FINAL_POSES_DIR
    )
    env.unwrapped.set_training_phase(phase, starting_stage=stage)
    env.unwrapped.set_curriculum_stage(stage)
    return UR5eTaskWrapper(env, ARC_DOF_SEQUENCE, DESCENT_DOF_SEQUENCE)

def load_model(phase, stage, env):
    dof = (ARC_DOF_SEQUENCE if phase == "arc" else DESCENT_DOF_SEQUENCE)[stage]
    path = LOG_DIR / f"best_{phase}_dof_{dof}.zip"
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return PPO.load(path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")

def sample_random_goal():
    if not USE_GOAL_RANDOMIZATION:
        return DESIRED_GOAL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler = GoalGMMSampler(GOAL_CENTER, GOAL_RADIUS, GOAL_PEAK_HEIGHTS, GOAL_COVS, device=device)
    return sampler.sample_xyz()

def set_goal_before_reset(env, goal):
    base = env.unwrapped
    base.desired_goal = np.asarray(goal, float).reshape(3)
    base.box_start_xyz = np.asarray(goal, float).reshape(3)

def apply_random_descent_pose(env):
    pose_dir = ARC_FINAL_POSES_DIR
    if not pose_dir.exists():
        raise FileNotFoundError(f"No poses found in {ARC_FINAL_POSES_DIR}")
    poses = sorted(pose_dir.glob("pose_*.json"))
    if not poses:
        raise FileNotFoundError(f"No pose_*.json files in {ARC_FINAL_POSES_DIR}")
    
    pose_data = json.load(open(np.random.choice(poses)))
    base = env.unwrapped
    base._apply_pose_data(pose_data)

    box_bid = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_BODY, "box")
    quat_adr = base.model.jnt_qposadr[base.model.body_jntadr[box_bid]] + 3
    base.arc_reward.set_box_orientation(base.data.qpos[quat_adr:quat_adr + 4])
    
    if "box_start_xyz" in pose_data:
        base.desired_goal = np.asarray(pose_data["box_start_xyz"], float).reshape(3)
    mujoco.mj_forward(base.model, base.data)

# Tolerances
def get_ee_pos(env):
    base = env.unwrapped
    site = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:eef_site")
    return base.data.site_xpos[site].copy()

def avg_speed(positions):
    if len(positions) < 2:
        return np.inf
    recent = np.array(positions[-10:])
    if len(recent) < 2:
        return np.inf
    return float(np.mean(np.linalg.norm(np.diff(recent, axis=0), axis=1) / 0.08))

def orientation_error_deg(env):
    base = env.unwrapped
    site = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:eef_site")
    ee_z = base.data.site_xmat[site].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(np.dot(ee_z, [0, 0, -1]), -1.0, 1.0))))

def gripper_alignment_error_deg(env):
    base = env.unwrapped
    site = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:eef_site")
    gripper_x = base.data.site_xmat[site].reshape(3, 3)[:, 0]
    gripper_yaw = np.arctan2(gripper_x[1], gripper_x[0])
    
    box_yaw = float(base.arc_reward.box_yaw) if base.arc_reward.box_yaw is not None else 0.0
    grasp_angles = (np.array([box_yaw, box_yaw + np.pi/2, box_yaw + np.pi, box_yaw + 3*np.pi/2]) + np.pi) % (2*np.pi) - np.pi
    angular_diffs = np.abs((gripper_yaw - grasp_angles + np.pi) % (2*np.pi) - np.pi)
    return float(np.degrees(np.min(angular_diffs)))

def tolerances_met(env, positions, phase, stage, cascade_mode=False):
    base = env.unwrapped
    obs = base._get_obs()
    ee, goal = np.asarray(obs["achieved_goal"], float), np.asarray(obs["desired_goal"], float)
    
    dof = (ARC_DOF_SEQUENCE if phase == "arc" else DESCENT_DOF_SEQUENCE)[stage]
    cfg = (ARC_STAGE_CONFIGS if phase == "arc" else DESCENT_STAGE_CONFIGS)[dof]
    
    target = goal.copy()
    if phase == "arc":
        target[2] += ARC_HEIGHT
    
    # Override distance tolerance in cascade mode
    dist_tol = CASCADE_ARC_DISTANCE_TOL if (cascade_mode and phase == "arc" and CASCADE_ARC_DISTANCE_TOL) else cfg.get("distance_tol", np.inf)
    
    checks = [
        np.linalg.norm(ee - target) <= dist_tol,
        abs(ee[2] - target[2]) < cfg.get("z_distance_tol", np.inf),
        avg_speed(positions) <= cfg.get("speed_tol", np.inf),
        orientation_error_deg(env) <= cfg.get("orientation_tol", np.inf),
        gripper_alignment_error_deg(env) <= cfg.get("gripper_align_tol", np.inf) if "gripper_align_tol" in cfg else True,
    ]
    return all(checks)

# Phase tests
def test_arc(stage):
    print(f"[ARC TEST] Stage {stage} (DoF {ARC_DOF_SEQUENCE[stage]})")
    env = make_env("arc", stage)
    set_goal_before_reset(env, sample_random_goal())
    obs, _ = env.reset()
    model = load_model("arc", stage, env)
    
    positions, step = [], 0
    while True:
        step += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        positions.append(get_ee_pos(env))
        
        if tolerances_met(env, positions, "arc", stage) or done or truncated:
            print(f"[ARC] Finished at step {step}")
            break
    env.close()

def test_descent(stage):
    print(f"[DESCENT TEST] Stage {stage} (DoF {DESCENT_DOF_SEQUENCE[stage]})")
    env = make_env("descent", stage)
    obs, _ = env.reset()
    apply_random_descent_pose(env)
    model = load_model("descent", stage, env)
    
    positions, step = [], 0
    while True:
        step += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        positions.append(get_ee_pos(env))
        
        if tolerances_met(env, positions, "descent", stage) or done or truncated:
            print(f"[DESCENT] Finished at step {step}")
            break
    env.close()

def test_cascade():
    print("[CASCADE TEST] ARC(DoF6) → DESCENT(DoF6)")
    arc_stage = len(ARC_DOF_SEQUENCE) - 1
    descent_stage = len(DESCENT_DOF_SEQUENCE) - 1
    
    env = make_env("arc", arc_stage)
    set_goal_before_reset(env, sample_random_goal())
    obs, _ = env.reset()
    model = load_model("arc", arc_stage, env)
    
    positions, step, transitioned = [], 0, False
    while True:
        step += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        positions.append(get_ee_pos(env))
        
        if not transitioned and tolerances_met(env, positions, "arc", arc_stage, cascade_mode=True):
            print(f"[CASCADE] Arc→Descent transition at step {step}")
            base = env.unwrapped
            base.training_phase, base.dof_sequence = "descent", DESCENT_DOF_SEQUENCE
            base.curriculum_stage, base._frozen_idx = descent_stage, np.array(frozen_joints_for_dof(6), dtype=int)
            model = load_model("descent", descent_stage, env)
            positions.clear()
            transitioned = True
        elif transitioned and (tolerances_met(env, positions, "descent", descent_stage) or done or truncated):
            print(f"[CASCADE] Complete at step {step}")
            break
        elif done or truncated:
            break
    env.close()

# Main
if __name__ == "__main__":
    if TEST_PHASE == "arc":
        test_arc(TEST_STAGE)
    elif TEST_PHASE == "descent":
        test_descent(TEST_STAGE)
    elif TEST_PHASE == "cascade":
        test_cascade()
    else:
        raise ValueError(f"Invalid TEST_PHASE: {TEST_PHASE}")