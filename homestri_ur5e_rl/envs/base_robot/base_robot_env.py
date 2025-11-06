
from os import path
import os, json
import mujoco
import numpy as np
from pathlib import Path
from gymnasium import spaces
from homestri_ur5e_rl.controllers.joint_position_controller import JointPositionController
from homestri_ur5e_rl.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
from homestri_ur5e_rl.envs.base_robot.pju_final_rewards import ArcReward, DescentReward

SUCCESS_THRESHOLD = 0.01

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.2,
    "azimuth": np.pi / 4,
    "elevation": -20.0,
    "lookat": np.array([0, 0, 1]),
}

# Base robot env
class BaseRobot(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 12}

    def __init__(
        self,
        model_path="../assets/base_robot/scene.xml",
        frame_skip=40,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        init_qpos_config=None,
        desired_goal=None,
        goal_box=None,
        use_grasp_start=None,
        box_start_xyz=None,
        box_start_rpy=None,
        gripper_start=None,
        frozen_joints=None,
        reward_config=None,
        dof_sequence_arc=None,
        dof_sequence_descent=None,
        training_phase="arc",
        starting_stage=None,
        arc_final_poses_dir=None,
        **kwargs,
    ):
        xml_path = path.join(path.dirname(path.realpath(__file__)), model_path)
        super().__init__(
            xml_path,
            frame_skip,
            spaces.Dict({
                "observation": spaces.Box(-np.inf, np.inf, (12,), np.float64),
                "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), np.float64),
                "desired_goal": spaces.Box(-np.inf, np.inf, (3,), np.float64),
            }),
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # Configuration details
        self.init_qpos_config = init_qpos_config or {}
        self.desired_goal = desired_goal
        self.goal_box = goal_box
        self.use_grasp_start = use_grasp_start
        self.box_start_xyz = box_start_xyz
        self.box_start_rpy = box_start_rpy
        self.gripper_start = gripper_start
        self.gripper_closed = bool(use_grasp_start)
        self._frozen_idx = np.array(frozen_joints or [], dtype=int)
        self.curriculum_stage = int(starting_stage)
        self.training_phase = training_phase
        self.arc_final_poses_dir = None

        # Passing to reward function for arc build
        self.arc_reward = ArcReward(
            arc_height=reward_config["arc_height"],
            progress_radius=reward_config["progress_radius"],
            orientation_tolerance_deg=reward_config["orientation_tolerance_deg"],
            arc_checkpoint_radius=reward_config["arc_checkpoint_radius"],
            arc_checkpoint_speed=reward_config["arc_checkpoint_speed"],
        )
        self.descent_reward = DescentReward()

        # DOF sequences
        self.dof_sequence_arc = dof_sequence_arc
        self.dof_sequence_descent = dof_sequence_descent
        self.dof_sequence = (
            self.dof_sequence_descent if training_phase == "descent" else self.dof_sequence_arc
        )

        # Mujoco initialisation
        self.init_qvel = self.data.qvel.copy()
        self.init_ctrl = self.data.ctrl.copy()
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float64)
        self.model_names = MujocoModelNames(self.model)
        self.max_episode_steps = 100

        # JointPositionController base for our new algorithm
        self.controller = JointPositionController(
            model=self.model,
            data=self.data,
            model_names=self.model_names,
            eef_name="robot0:eef_site",
            joint_names=[
                "robot0:ur5e:shoulder_pan_joint",
                "robot0:ur5e:shoulder_lift_joint",
                "robot0:ur5e:elbow_joint",
                "robot0:ur5e:wrist_1_joint",
                "robot0:ur5e:wrist_2_joint",
                "robot0:ur5e:wrist_3_joint",
            ],
            actuator_names=[
                "robot0:ur5e:shoulder_pan",
                "robot0:ur5e:shoulder_lift",
                "robot0:ur5e:elbow",
                "robot0:ur5e:wrist_1",
                "robot0:ur5e:wrist_2",
                "robot0:ur5e:wrist_3",
            ],
            min_effort=[-150] * 6,
            max_effort=[150] * 6,
            min_position=[-np.pi] * 6,
            max_position=[np.pi] * 6,
            kp=[1.2, 0.8, 0.8, 0.8, 0.8, 0.8],
            kd=[0.5] * 6,
        )

        for jname, jpos in self.init_qpos_config.items():
            qid = self.model.jnt_qposadr[self.model_names.joint_name2id[jname]]
            self.init_qpos[qid] = jpos

    # Set training stage
    def set_curriculum_stage(self, stage: int):
        self.curriculum_stage = int(stage)
        self.curr_dof = self.dof_sequence[self.curriculum_stage]

    # Set training phase
    def set_training_phase(self, phase, starting_stage=None):
        self.training_phase = phase
        self.dof_sequence = (
            self.dof_sequence_descent if phase == "descent" else self.dof_sequence_arc
        )
        self.curriculum_stage = starting_stage
        self.curr_dof = self.dof_sequence[self.curriculum_stage]

    # Step method for simulation
    def step(self, action):
        self.episode_step_count += 1
        qpos_ids = [
            self.model.jnt_qposadr[self.model_names.joint_name2id[j]]
            for j in self.controller.joint_names
        ]
        
        q = self.data.qpos[qpos_ids].astype(float)
        a = np.asarray(action, float)[:6]

        # Freeze inactive joints (set their actions to zero)
        if len(self._frozen_idx) > 0:
            a[self._frozen_idx] = 0.0

        # Delta step method
        dt = float(self.model.opt.timestep) * int(self.frame_skip)
        step_scale = 15 * np.pi * np.array([0.4, 0.4, 0.5, 0.6, 0.6, 0.6]) * dt

        # Compute target joint angles
        q_target = q.copy()
        movable = np.setdiff1d(np.arange(6), self._frozen_idx)
        q_target[movable] = np.clip(
            q[movable] + a[movable] * step_scale[movable],
            np.array(self.controller.min_position)[movable],
            np.array(self.controller.max_position)[movable],
        )

        # End-effector location
        eef_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:eef_site")
        # Current end-effector position
        prev_pos = self.data.site_xpos[eef_site].copy()

        # Moves robotic joints
        for _ in range(self.frame_skip):
            ctrl = self.data.ctrl.copy()
            self.controller.run(q_target, ctrl)
            ctrl[6] = 255.0 if self.gripper_closed else 0.0
            self.do_simulation(ctrl, 1)
            if self.render_mode == "human":
                self.render()

        ee_velocity = (self.data.site_xpos[eef_site] - prev_pos) / dt
        ee_rotation = self.data.site_xmat[eef_site].reshape(3, 3).copy()
        obs = self._get_obs()
        current_dof = self.dof_sequence[self.curriculum_stage]

        # Rewards
        if self.training_phase == "arc":
            reward = self.arc_reward.compute(
                obs["achieved_goal"], obs["desired_goal"], current_dof, ee_rotation, self.data.qpos
            )
        else:
            reward = self.descent_reward.compute(
                obs["achieved_goal"], obs["desired_goal"], current_dof, ee_rotation,
                qpos=self.data.qpos, box_yaw=self.arc_reward.box_yaw
            )

        # Terminations
        success = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]) < SUCCESS_THRESHOLD
        truncated = self.episode_step_count >= self.max_episode_steps
        return obs, reward, success, truncated, {"is_success": success}

    # Obain observations compatible with HER
    def _get_obs(self):
        jids = [self.model_names.joint_name2id[j] for j in self.controller.joint_names]
        qpos_ids = [int(self.model.jnt_qposadr[j]) for j in jids]
        qvel_ids = [int(self.model.jnt_dofadr[j]) for j in jids]
        obs = np.concatenate(
            (self.data.qpos[qpos_ids], self.data.qvel[qvel_ids])
        ).astype(np.float64)
        eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:eef_site")
        return {
            "observation": obs,
            "achieved_goal": self.data.site_xpos[eef_id].astype(np.float64, copy=True),
            "desired_goal": np.asarray(self.desired_goal, np.float64).reshape(3),
        }

    # Reset model at end of episode
    def reset_model(self, training_phase=None):
        if training_phase:
            self.set_training_phase(training_phase)

        self.arc_reward.wrist_target_joint = 0.0
        self.descent_reward.wrist_target_joint = 0.0

        if self.training_phase == "descent":
            return self._reset_for_descent()
        else:
            return self._reset_for_arc()


    # Arc reset for starting at original end-effector location and building new arc for random goal location
    def _reset_for_arc(self):
        self.set_state(self.init_qpos.copy(), self.init_qvel.copy())
        self.episode_step_count = 0
        obs = self._get_obs()
        
        box_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        j_id = int(self.model.body_jntadr[box_bid])
        qpos_adr = int(self.model.jnt_qposadr[j_id])
        qvel_adr = int(self.model.jnt_dofadr[j_id])

        # Move box to desired goal position
        goal = np.asarray(self.desired_goal, float)
        self.data.qpos[qpos_adr:qpos_adr+3] = goal.copy()
        self.data.qvel[qvel_adr:qvel_adr+6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        
        box_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        quat_adr = self.model.jnt_qposadr[self.model.body_jntadr[box_bid]] + 3
        box_quat = self.data.qpos[quat_adr:quat_adr + 4].copy()
        self.arc_reward.set_box_orientation(box_quat)

        self.arc_reward.build_arc(obs["achieved_goal"], obs["desired_goal"])
        return obs

    # Descent reset for starting at randomly selected pose from those saved at the end of the arc
    def _reset_for_descent(self):
        pose_path = None
        pose_dir = Path(self.arc_final_poses_dir or ".")

        if pose_dir.is_dir():
            poses = sorted(pose_dir.glob("pose_*.json"))
            if poses:
                pose_path = str(np.random.choice(poses))

        if pose_path and os.path.exists(pose_path):
            with open(pose_path, "r") as f:
                pose_data = json.load(f)
            self._apply_pose_data(pose_data)
        else:
            self.set_state(self.init_qpos.copy(), self.init_qvel.copy())
            
        box_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        quat_adr = self.model.jnt_qposadr[self.model.body_jntadr[box_bid]] + 3
        box_quat = self.data.qpos[quat_adr:quat_adr + 4].copy()
        self.arc_reward.set_box_orientation(box_quat)

        self.episode_step_count = 0
        obs = self._get_obs()
        return obs


    # Apply pose to start of descent
    def _apply_pose_data(self, pose_data):
        for joint, pos in pose_data.get("arm_init_qpos", {}).items():
            qid = self.model.jnt_qposadr[self.model_names.joint_name2id[joint]]
            self.data.qpos[qid] = pos
            
        # Apply saved box position and orientation
        if "box_start_xyz" in pose_data:
            box_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
            j_id = int(self.model.body_jntadr[box_bid])
            qpos_adr = int(self.model.jnt_qposadr[j_id])
            self.data.qpos[qpos_adr:qpos_adr+3] = np.array(pose_data["box_start_xyz"])

        if "box_start_rpy" in pose_data:
            qpos_adr = int(self.model.jnt_qposadr[j_id])
            # Convert RPY (Euler) to quaternion for Mujoco
            r, p, y = pose_data["box_start_rpy"]
            cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
            cp, sp = np.cos(p * 0.5), np.sin(p * 0.5)
            cr, sr = np.cos(r * 0.5), np.sin(r * 0.5)
            quat = np.array([
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy
            ])
            self.data.qpos[qpos_adr+3:qpos_adr+7] = quat
            
        mujoco.mj_forward(self.model, self.data)
