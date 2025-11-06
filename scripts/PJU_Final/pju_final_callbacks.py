
import os
import json
import numpy as np
from pathlib import Path
import torch
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from syllabus.core import TaskWrapper
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList



# Handles DoF freezing
class UR5eTaskWrapper(TaskWrapper):
    def __init__(self, env, arc_dof_sequence, descent_dof_sequence):
        super().__init__(env)
        self.arc_dof_sequence = arc_dof_sequence
        self.descent_dof_sequence = descent_dof_sequence
        self.task = int(getattr(env.unwrapped, "curriculum_stage", 0))

        # Map of frozen joints
        self.frozen_map = {
            1: [1, 2, 3, 4, 5],  
            2: [2, 3, 4, 5],     
            3: [3, 4, 5],      
            4: [4, 5],       
            5: [5],           
            6: [],           
        }

    # Update of frozen joints for stage or phase change
    def reset(self, **kwargs):
        if "new_task" in kwargs:
            self.task = int(kwargs.pop("new_task"))
        if "training_phase" in kwargs:
            self.env.unwrapped.training_phase = kwargs.pop("training_phase")

        base = self.env.unwrapped
        base.curriculum_stage = self.task

        # Pick correct sequence based on current phase
        if base.training_phase == "arc":
            dof_sequence = self.arc_dof_sequence
        else:
            dof_sequence = self.descent_dof_sequence

        # Freeze based on correct sequence
        current_dof = dof_sequence[self.task]
        base._frozen_idx = np.array(self.frozen_map[current_dof], dtype=int)
        return self.env.reset(**kwargs)



# Curriculum for managing arc and descent phases
class DualCurriculum:
    def __init__(self, arc_dof_sequence, descent_dof_sequence,
                 arc_configs, descent_configs, initial_phase):
        self.arc_dof_sequence = arc_dof_sequence
        self.descent_dof_sequence = descent_dof_sequence
        self.arc_configs = arc_configs
        self.descent_configs = descent_configs
        self.arc_idx = 0
        self.descent_idx = 0
        self.training_phase = initial_phase

        # Dwell initialisation
        self.arc_dwell = {d: 0 for d in arc_dof_sequence}
        self.descent_dwell = {d: 0 for d in descent_dof_sequence}
        self.arc_stable = {d: False for d in arc_dof_sequence}
        self.descent_stable = {d: False for d in descent_dof_sequence}

        # Flags
        self.arc6_ready_flag = False
        self.arc_training_complete = False
        self.descent_training_complete = False

    # Training for arc and descent complete
    def is_training_complete(self):
        return self.descent_training_complete


# Handles stage transitions and dwell logic for arc and descent phases
class DualCurriculumCallback(BaseCallback):
    def __init__(self, curriculum, arc_configs, descent_configs,
                 tw_eval, log_dir, verbose=0,
                 use_goal_randomization=False, num_arc_final_poses=1):
        super().__init__(verbose)
        self.cur = curriculum
        self.arc_configs = arc_configs
        self.descent_configs = descent_configs
        self.tw_eval = tw_eval
        self.log_dir = log_dir
        self.use_goal_randomization = use_goal_randomization
        self.num_arc_final_poses = num_arc_final_poses

        self.positions = []
        self.episode_success = False
        self.pending_transition = None
        self.wait_for_eval = False
        self.pose_saved_this_episode = False
        self.eval_cb_ref = None
    
    # Evaluates each simulation step for progression
    def _on_step(self):
        if self.cur.is_training_complete():
            return False

        # Phase and environment initialisation
        env = self.training_env.envs[0].unwrapped
        phase = self.cur.training_phase
        dof = (self.cur.arc_dof_sequence[self.cur.arc_idx]
            if phase == "arc" else self.cur.descent_dof_sequence[self.cur.descent_idx])
        cfg = self.arc_configs[dof] if phase == "arc" else self.descent_configs[dof]

        # Reset pose saving flag when episode ends
        if self.locals.get("dones") is not None and any(self.locals["dones"]):
            self.pose_saved_this_episode = False

        # Observations
        obs = env._get_obs()
        ee, goal = obs["achieved_goal"], obs["desired_goal"]
        self.positions.append(ee)

        # Average speed over the last 10 steps
        if len(self.positions) > 1:
            recent = np.array(self.positions[-10:])
            avg_speed = np.mean(np.linalg.norm(np.diff(recent, axis=0), axis=1) / 0.08)
        else:
            avg_speed = np.inf

        # Distance & height tolerances
        if phase == "arc":
            arc_height = getattr(env.arc_reward, "arc_height", 0.328)
            target = goal.copy()
            target[2] += arc_height
        else:
            target = goal

        # Distance tolerance
        dist = np.linalg.norm(ee - target)
        dist_ok = dist <= cfg.get("distance_tol", np.inf)

        # Z-height tolerance
        z_ok = abs(ee[2] - target[2]) < cfg.get("z_distance_tol", np.inf)

        # Speed tolerance
        speed_ok = avg_speed <= cfg.get("speed_tol", np.inf)

        # Orientation tolerance
        eef_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:eef_site")
        ee_rot = env.data.site_xmat[eef_site].reshape(3, 3)
        ee_z = ee_rot[:, 2]
        mis_deg = np.degrees(np.arccos(np.clip(np.dot(ee_z, [0, 0, -1]), -1.0, 1.0)))
        orientation_ok = mis_deg <= cfg.get("orientation_tol", np.inf)

        # Gripper alignment tolerance
        qpos = env.data.qpos
        eef_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:eef_site")
        ee_rot = env.data.site_xmat[eef_site].reshape(3, 3)
        gripper_x = ee_rot[:, 0]
        gripper_yaw = np.arctan2(gripper_x[1], gripper_x[0])

        if phase == "arc":
            box_yaw = float(env.arc_reward.box_yaw) if env.arc_reward.box_yaw is not None else 0.0
        else:
            box_yaw = float(env.arc_reward.box_yaw) if env.arc_reward.box_yaw is not None else 0.0

        grasp_angles = np.array([
            box_yaw,
            box_yaw + np.pi/2,
            box_yaw + np.pi,
            box_yaw + 3*np.pi/2
        ])

        grasp_angles = (grasp_angles + np.pi) % (2*np.pi) - np.pi
        angular_diffs = np.abs((gripper_yaw - grasp_angles + np.pi) % (2*np.pi) - np.pi)
        min_error = np.min(angular_diffs)
        err_deg = np.degrees(min_error)

        gripper_ok = err_deg <= cfg.get("gripper_align_tol", np.inf)


        # All tolerances met
        if all([dist_ok, z_ok, speed_ok, orientation_ok, gripper_ok]):
            self.episode_success = True
            # Pose saving
            if phase == "arc" and dof == 6 and self.use_goal_randomization and not self.pose_saved_this_episode:
                self._save_arc_endpoint_pose()
                self.pose_saved_this_episode = True

        return True


    # Manages dwell and pending transitions
    def _on_rollout_end(self):
        phase = self.cur.training_phase
        dof = (self.cur.arc_dof_sequence[self.cur.arc_idx]
               if phase == "arc" else self.cur.descent_dof_sequence[self.cur.descent_idx])
        dwell = self.cur.arc_dwell if phase == "arc" else self.cur.descent_dwell
        stable = self.cur.arc_stable if phase == "arc" else self.cur.descent_stable
        cfg = self.arc_configs[dof] if phase == "arc" else self.descent_configs[dof]

        if self.pending_transition:
            self.positions.clear()
            self.episode_success = False
            return

        if self.episode_success:
            if stable[dof]:
                dwell[dof] += 1
                
                if dwell[dof] >= cfg["dwell_episodes"]:
                    seq = self.cur.arc_dof_sequence if phase == "arc" else self.cur.descent_dof_sequence
                    idx = self.cur.arc_idx if phase == "arc" else self.cur.descent_idx
                    is_last = (idx + 1 >= len(seq))

                    # Transition from arc to descent
                    if phase == "arc" and dof == 6:
                        self.cur.arc6_ready_flag = True
                        self.pending_transition = "to_descent"
                        self.wait_for_eval = True

                    # Continue to next stage
                    elif not is_last:
                        next_dof = seq[idx + 1]
                        self.pending_transition = next_dof
                        self.wait_for_eval = True

                    # Descent complete - all training terminate
                    elif phase == "descent":
                        self.cur.descent_training_complete = True

                    # Reset stable and dwell
                    dwell[dof] = 0
                    stable[dof] = False

            else:
                stable[dof] = True
                dwell[dof] = 1
        else:
            stable[dof] = False
            dwell[dof] = 0

        self.positions.clear()
        self.episode_success = False

    # Transition to next stage in current phase
    def _execute_transition(self, next_dof, phase):
        if phase == "arc":
            self.cur.arc_idx += 1
        else:
            self.cur.descent_idx += 1

        for env_ref in [self.training_env, self.tw_eval]:
            env = getattr(env_ref, "envs", [env_ref])[0]
            env.reset(new_task=self.cur.arc_idx if phase == "arc" else self.cur.descent_idx,
                      training_phase=phase)
        print(f"Switched to {next_dof} DoF")
        
    
    # Manages transition from arc to descent
    def _transition_to_descent(self):
        self.cur.training_phase = "descent"
        self.cur.descent_idx = 0

        # Build the arc_final_poses directory path
        arc_pose_dir = Path(self.log_dir) / "arc_final_poses"

        for env_ref in [self.training_env, self.tw_eval]:
            if hasattr(env_ref, "envs"):
                inner_env = env_ref.envs[0]
            else:
                inner_env = env_ref
                while hasattr(inner_env, "env"):
                    inner_env = inner_env.env

            base = inner_env.unwrapped
            base.arc_final_poses_dir = str(arc_pose_dir)

            try:
                inner_env.reset(new_task=0, training_phase="descent")
            except TypeError:
                inner_env.reset()

        self.descent_triggered = True
        
        # Reset best models for descent phase
        if self.eval_cb_ref is not None:
            self.eval_cb_ref.reset_for_descent(self.cur.descent_dof_sequence)
        

    # Saves pose at arc endpoint
    def _save_arc_endpoint_pose(self):
        env = self.training_env.envs[0].unwrapped
        qpos = env.data.qpos.copy()
        arm_qpos = {
            j: float(qpos[env.model.jnt_qposadr[env.model_names.joint_name2id[j]]])
            for j in env.controller.joint_names
        }
        eef_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:eef_site")
        ee_pos = env.data.site_xpos[eef_site].copy()

        pose_data = {
            "arm_init_qpos": arm_qpos,
            "ee_position": ee_pos,
            "box_start_xyz": getattr(env, "box_start_xyz", None),
            "box_start_rpy": getattr(env, "box_start_rpy", None),
            "gripper_start": getattr(env, "gripper_start", None),
        }

        # Convert data for making json pose file
        def _make_json_safe(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_make_json_safe(v) for v in obj]
            else:
                return obj

        pose_data = _make_json_safe(pose_data)

        base_dir = Path(self.log_dir)
        save_dir = base_dir / "arc_final_poses"
        save_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(save_dir.glob("pose_*.json"))
        next_idx = len(existing) if len(existing) < self.num_arc_final_poses else np.random.randint(0, self.num_arc_final_poses)
        save_path = save_dir / f"pose_{next_idx:03d}.json"

        with open(save_path, "w") as f:
            json.dump(pose_data, f, indent=2)


# Evaluates model, and manages model saving and stage transitions
class StageEvalCallback(EvalCallback):
    def __init__(self, curriculum, arc_seq, descent_seq,
                 linked_curriculum_cb, eval_env, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.cur = curriculum
        self.arc_seq = arc_seq
        self.descent_seq = descent_seq
        self.linked_curriculum_cb = linked_curriculum_cb
        self.best_rewards = {d: -np.inf for d in arc_seq + descent_seq}
        
        # Extra measurement metrics based on tolerances
        self.extra_metrics = {"dist": [], "align": [], "orient": []}

    def _on_step(self):
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            phase = self.cur.training_phase
            dof = (self.arc_seq[self.cur.arc_idx] if phase == "arc" else self.descent_seq[self.cur.descent_idx])
            mean_reward = np.mean(self.evaluations_results[-1])

            # Compute additional evaluation metrics from last eval episode
            env = self.eval_env.envs[0].unwrapped
            obs = env._get_obs()
            ach, goal = obs["achieved_goal"], obs["desired_goal"]

            dist = np.linalg.norm(ach - goal)
            align_err = np.linalg.norm((ach - goal)[:2])

            eef_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:eef_site")
            ee_rot = env.data.site_xmat[eef_site].reshape(3, 3)
            orient_err = np.degrees(np.arccos(np.clip(np.dot(ee_rot[:, 2], [0, 0, -1]), -1, 1)))

            # Append to SB3’s evaluations_results list
            self.extra_metrics["dist"].append(dist)
            self.extra_metrics["align"].append(align_err)
            self.extra_metrics["orient"].append(orient_err)

            os.makedirs(os.path.dirname(os.path.join(self.log_path, "evaluations.npz")), exist_ok=True)

            # Log to the .npz when EvalCallback saves
            np.savez(
                os.path.join(self.log_path, "evaluations.npz"),
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                distances=np.array(self.extra_metrics["dist"]),
                alignments=np.array(self.extra_metrics["align"]),
                orientations=np.array(self.extra_metrics["orient"]),
            )

            # Save best reward per stage
            if mean_reward > self.best_rewards[dof]:
                self.best_rewards[dof] = mean_reward
                model_path = os.path.join(self.best_model_save_path, f"best_{phase}_dof_{dof}.zip")
                self.model.save(model_path)
                print(f"New best mean reward {mean_reward:.2f} for DoF {dof}")

            # Handle stage transitions
            cb = self.linked_curriculum_cb
            if cb and cb.wait_for_eval and cb.pending_transition:
                if cb.pending_transition == "to_descent":
                    cb._save_arc_endpoint_pose()
                    pose_dir = Path(cb.log_dir) / "arc_final_poses"
                    num_collected = len(list(pose_dir.glob("pose_*.json")))
                    if self.cur.arc6_ready_flag and num_collected >= cb.num_arc_final_poses:
                        cb._transition_to_descent()
                        cb.pending_transition = None
                        cb.wait_for_eval = False
                        self.cur.arc6_ready_flag = False
                else:
                    next_dof = cb.pending_transition
                    cb._execute_transition(next_dof, phase)
                    cb.pending_transition = None
                    cb.wait_for_eval = False
        return result
    

    def reset_for_descent(self, descent_seq):
        self.best_rewards = {d: -np.inf for d in descent_seq}
        self.evaluations_results.clear()
        self.evaluations_timesteps.clear()
        self.extra_metrics = {"dist": [], "align": [], "orient": []}



# Samples random goal location from Guassian Mixture
class GoalGMMSampler:
    def __init__(self, center_xyz, radius, heights, covs, device="cpu"):
        self.center_xy = np.asarray(center_xyz[:2])
        self.z = float(center_xyz[2])
        self.radius = float(radius)
        means_np = self._four_peak_means(self.center_xy, self.radius)
        weights = torch.tensor(heights, dtype=torch.float32, device=device)
        weights = weights / weights.sum()
        covs_t = torch.stack([torch.tensor(C, dtype=torch.float32, device=device) for C in covs])
        comps = MultivariateNormal(torch.tensor(means_np, dtype=torch.float32, device=device),
                                   scale_tril=torch.linalg.cholesky(covs_t))
        self._dist = MixtureSameFamily(Categorical(probs=weights), comps)

    def sample_xyz(self):
        for _ in range(1000):
            xy = self._dist.sample((1,)).squeeze(0).cpu().numpy()
            if np.linalg.norm(xy - self.center_xy) <= self.radius:
                return np.array([xy[0], xy[1], self.z])
        return np.array([*self.center_xy, self.z])

    def _four_peak_means(self, center_xy, r):
        d = r / np.sqrt(2)
        angles = np.deg2rad([45, 135, 225, 315])
        offsets = np.stack([d * np.cos(angles), d * np.sin(angles)], axis=1)
        return center_xy[None, :] + offsets


# Randomised goal location wrapper
class RandomizedGoalEnv(gym.Wrapper):
    def __init__(self, env, sampler):
        super().__init__(env)
        self._sampler = sampler

    def reset(self, **kwargs):
        new_goal = self._sampler.sample_xyz()
        base = self.env
        while hasattr(base, "env"):
            base = base.env
            
        base.desired_goal = new_goal
        base.box_start_xyz = new_goal
        return self.env.reset(**kwargs)
