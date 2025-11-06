
import numpy as np


# Arc rewards
class ArcReward:
    def __init__(self, arc_height, progress_radius, orientation_tolerance_deg,
                 arc_checkpoint_radius, arc_checkpoint_speed):
        self.arc_height = float(arc_height)
        self.progress_radius = float(progress_radius)
        self.orientation_tolerance_deg = float(orientation_tolerance_deg)
        self.orientation_tolerance_rad = np.radians(orientation_tolerance_deg)
        self.arc_checkpoint_radius = float(arc_checkpoint_radius)
        self.arc_checkpoint_speed = float(arc_checkpoint_speed)

        # Arc states
        self.C = None; self.r = None
        self.phi_s = None; self.phi_e = None
        self.z0 = None
        self.waypoints = None
        self.waypoints_captured = None
        self.last_waypoint_idx = -1
        self.box_yaw = None


    # Building arc
    def build_arc(self, start, goal, base_xy=(0.0, 0.0)):
        start, goal = np.asarray(start,float), np.asarray(goal,float)
        sx, sy, sz = start; gx, gy, gz = goal; bx, by = base_xy

        # Vector from orgin base to end-effector start point
        v_base = np.array([sx - bx, sy - by], float)
        if np.linalg.norm(v_base) < 1e-8:
            v_base = np.array([1.0, 0.0])
        else:
            v_base /= np.linalg.norm(v_base)

        # Choose tangent direction requiring least rotation from shoulder pan joint
        t1 = np.array([-v_base[1], v_base[0]], float); t2 = -t1
        to_goal = np.array([gx-sx, gy-sy], float); to_goal /= np.linalg.norm(to_goal)
        T = t1 if abs(np.arctan2(t1[1],t1[0]) - np.arctan2(to_goal[1],to_goal[0])) <= \
                 abs(np.arctan2(t2[1],t2[0]) - np.arctan2(to_goal[1],to_goal[0])) else t2
        T /= np.linalg.norm(T)

        # Compute arc center, radius and start angle
        n = np.array([-T[1], T[0]], float)
        S, G = np.array([sx,sy]), np.array([gx,gy])
        alpha = 0.5 * np.dot(G-S, G-S) / np.dot(G-S, n)
        Cxy = S + alpha*n
        r = np.linalg.norm(Cxy - S)

        # Angular sweep direction and angular difference
        phi_s = np.arctan2(sy - Cxy[1], sx - Cxy[0])
        phi_g = np.arctan2(gy - Cxy[1], gx - Cxy[0])
        R_s = np.array([sx - Cxy[0], sy - Cxy[1]])
        t_ccw = np.array([-R_s[1], R_s[0]]) / np.linalg.norm(R_s)
        ccw = np.dot(T, t_ccw) >= 0.0
        dphi_raw = (phi_g - phi_s + np.pi) % (2*np.pi) - np.pi
        dphi = dphi_raw if ccw == (dphi_raw >= 0.0) else dphi_raw + (2*np.pi if ccw else -2*np.pi)

        self.C, self.r, self.phi_s, self.phi_e = Cxy.copy(), float(r), float(phi_s), float(phi_s + dphi)
        self.z0 = float(sz)

        # Waypoints for reward and closest point at distance of 0.01 apart
        arc_len = abs(self.r * dphi)
        n_wp = max(int(arc_len / 0.01), 10)
        ts = np.linspace(0,1,n_wp)
        self.waypoints = np.stack([
            self.C[0] + self.r*np.cos(self.phi_s + ts*(self.phi_e-self.phi_s)),
            self.C[1] + self.r*np.sin(self.phi_s + ts*(self.phi_e-self.phi_s)),
            self.z0 + ts*(self.arc_height - self.z0)
        ], axis=1)
        self.waypoints_captured = np.zeros(len(self.waypoints), dtype=bool)
        self.last_waypoint_idx = -1
        
        self.ee_pos = np.asarray(start, float).copy()



    # Distance to arc endpoint
    def _distance_term(self, ee, goal):
        arc_endpoint = np.array([goal[0], goal[1], self.arc_height])
        return -np.linalg.norm(ee - arc_endpoint)

    # Bonus for passing through arc waypoint
    def _waypoint_bonus(self, ee):
        idx = np.argmin(np.linalg.norm(self.waypoints - ee, axis=1))
        closest = self.waypoints[idx]
        bonus = 0.0
        if np.linalg.norm(ee - closest) < self.progress_radius:
            for k in range(self.last_waypoint_idx + 1, idx + 1):
                if not self.waypoints_captured[k]:
                    self.waypoints_captured[k] = True
                    bonus += 0.5
            self.last_waypoint_idx = max(self.last_waypoint_idx, idx)
        return bonus

    # Penalty for end-effector not being perpendicular to ground plane
    def _orientation_penalty(self, ee_rotation):
        ee_z = ee_rotation[:,2]
        mis_deg = np.degrees(np.arccos(np.clip(np.dot(ee_z,[0,0,-1]), -1.0, 1.0)))
        return -0.005 * mis_deg

    # Penalty for gripper not being aligned to pick up box
    def _gripper_penalty(self, qpos, ee_rotation):
        gripper_x = ee_rotation[:, 0] 
        gripper_yaw = np.arctan2(gripper_x[1], gripper_x[0])
        box_yaw = float(self.box_yaw) if self.box_yaw is not None else 0.0

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
        
        return -0.05 * err_deg


    # Main reward computation for all DoFs
    def compute(self, ee, goal, dof, ee_rotation, qpos=None):
        ee = np.asarray(ee, float); goal = np.asarray(goal, float)

        # DoF 1 is purely distance for shoulder pan angle
        if dof == 1:
            return -np.linalg.norm(ee - goal)
        # DoF 2 is maintaining z-height with panning motion
        if dof == 2:
            return -np.linalg.norm([ee[0]-goal[0], ee[1]-goal[1]])

        # DoF 3 combines distance to arc end with waypoint bonuses
        total = 0.0
        total += self._distance_term(ee, goal)
        total += self._waypoint_bonus(ee)

        # DoF 4 and 5 combines rewards associated with DoF 3 with orientation penalty
        if dof >= 4:
            total += self._orientation_penalty(ee_rotation)
        # DoF 6 penalises gripper alignment with box
        if dof > 5:
            total += self._gripper_penalty(qpos, ee_rotation)
        return total


    # Extracts boxes orientation to align gripper
    def set_box_orientation(self, box_quat):
        w, x, y, z = box_quat
        self.box_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))



# Descent rewards
class DescentReward:
    def __init__(self):
        pass

    # Distance to goal reward
    def _distance_term(self, ee, goal):
        return -np.linalg.norm(ee - goal)

    # Penalty for end-effector not being perpendicular to ground plane
    def _orientation_penalty(self, ee_rotation):
        ee_z = ee_rotation[:,2]
        mis_deg = np.degrees(np.arccos(np.clip(np.dot(ee_z,[0,0,-1]), -1.0, 1.0)))
        return -0.01 * mis_deg

    # Penalty for gripper not being aligned to pick up box
    def _gripper_penalty(self, qpos, box_yaw, ee_rotation):
        gripper_x = ee_rotation[:, 0]
        gripper_yaw = np.arctan2(gripper_x[1], gripper_x[0])
        box_yaw = float(box_yaw) if box_yaw is not None else 0.0

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
        
        return -0.05 * err_deg


    # Main reward computing all DoFs
    def compute(self, ee, goal, dof, ee_rotation, qpos=None, box_yaw=None):
        ee = np.asarray(ee,float); goal = np.asarray(goal,float)

        # DoF 1, 2 and 3 purely focus on reducing distance to object
        total = self._distance_term(ee, goal)
        # DoF 4 and 5 combines distance reward with orientation penalty
        if dof >= 4:
            total += self._orientation_penalty(ee_rotation)
        # DoF 6 penalises gripper alignment with box
        if dof > 5:
            total += self._gripper_penalty(qpos, box_yaw, ee_rotation)
        return total
