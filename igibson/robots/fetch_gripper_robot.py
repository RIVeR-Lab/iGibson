import gym
import numpy as np
import pybullet as p

import igibson.utils.transform_utils as T
from igibson.controllers.ik_controller import IKController
from igibson.controllers.js_controller import JointSpaceController
from igibson.external.pybullet_tools.utils import (
    get_child_frame_pose,
    get_constraint_violation,
    get_joint_info,
    get_relative_pose,
    joints_from_names,
    set_coll_filter,
    set_joint_positions,
)
from igibson.robots.robot_locomotor import LocomotorRobot

# Assisted grasping parameters
ASSIST_FRACTION = 1.0
ARTICULATED_ASSIST_FRACTION = 0.7
MIN_ASSIST_FORCE = 0
MAX_ASSIST_FORCE = 500
ASSIST_FORCE = MIN_ASSIST_FORCE + (MAX_ASSIST_FORCE - MIN_ASSIST_FORCE) * ASSIST_FRACTION
CONSTRAINT_VIOLATION_THRESHOLD = 0.1
RELEASE_WINDOW = 1 / 30.0  # release window in seconds

# GRIPPER index constants
GRIPPER_BASE_CENTER_OFFSET = [0.1, 0, 0]


class FetchGripper(LocomotorRobot):
    """
    Fetch Robot
    Reference: https://fetchrobotics.com/robotics-platforms/fetch-mobile-manipulator/
    Uses joint velocity control
    """

    def __init__(self, config, simulator):
        self.simulator = simulator
        self.config = config

        self.base_max_linear_vel = config.get("base_max_linear_vel", 1.0)  # m/s
        self.base_max_angular_vel = config.get("base_max_angular_vel", np.pi)  # rad/second

        self.default_arm_pose = config.get("default_arm_pose", "vertical")
        self.trunk_offset = config.get("trunk_offset", 0.0)
        self.use_ag = config.get("use_ag", True)  # Use assisted grasping
        self.ag_strict_mode = config.get("ag_strict_mode", True)  # Require object to be contained by forks for AG

        self.wheel_axle_half = 0.186  # half of the distance between the wheels
        self.wheel_radius = 0.0613  # radius of the wheels
        self.head_limit_epsilon = 1e-2
        self.gripper_limit_epsilon = 1e-2

        # Joint ids
        self.base_joint_ids = np.array([1, 2])
        self.head_joint_ids = np.array([4, 5])
        self.torso_and_arm_joint_ids = np.array([3, 12, 13, 14, 15, 16, 17, 18])
        self.gripper_joint_ids = np.array([19, 20, 21])  # These are fixed joints
        self.gripper_fingers_joint_ids = np.array([20, 21])

        # Indices of the joints in the self.joint_ids vector
        self.base_joint_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.base_joint_ids]
        self.head_joint_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.head_joint_ids]
        self.torso_and_arm_joint_idx = [
            i for i, idn in enumerate(self.joint_ids) if idn in self.torso_and_arm_joint_ids
        ]
        self.gripper_fingers_joint_idx = [
            i for i, idn in enumerate(self.joint_ids) if idn in self.gripper_fingers_joint_ids
        ]

        # Robot DoFs
        self.base_dofs = len(
            self.base_joint_idx
        )  # No matter if we control each wheel or linear (1dof) + angular (1dof) velocity
        self.head_dofs = len(self.head_joint_ids)
        self.torso_and_arm_dofs = len(self.torso_and_arm_joint_ids)
        self.gripper_fingers_dofs = len(self.gripper_fingers_joint_ids)

        self.base_action_dim = 2
        self.head_action_dim = 2
        self.gripper_action_dim = 1

        self.arm_controller_type = self.config["controller"].get("type", "cartesian_ik")
        joint_control = "velocity"
        if self.arm_controller_type == "cartesian_ik":
            self.arm_controller = IKController(robot=self, config=self.config)
            self.arm_max_delta_pos = config.get("arm_max_delta_pos", 0.25)  # delta_pos = 0.25m
            self.arm_max_delta_orn = config.get("arm_max_delta_orn", np.deg2rad(30))  # delta_orn = 30deg
            self.arm_delta_pos_dim = 3
            self.arm_delta_orn_dim = 3
            self.torso_and_arm_action_dim = self.arm_delta_pos_dim + self.arm_delta_orn_dim
        elif "joint_space" in self.arm_controller_type:
            self.torso_and_arm_action_dim = self.torso_and_arm_dofs
            self.arm_controller = JointSpaceController(robot=self, config=self.config)
            joint_control = self.arm_controller_type.replace("joint_space_", "")
            if joint_control not in ["position", "velocity", "torque"]:
                print("The joint space control type should define either position, velocity or torque")
                exit(-1)
        else:
            print("Unsupported arm_controller type")
            exit(-1)

        LocomotorRobot.__init__(
            self,
            "fetch/fetch_gripper.urdf",
            action_dim=self.base_action_dim
            + self.head_action_dim
            + self.torso_and_arm_action_dim
            + self.gripper_action_dim,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control=["differential_drive"] * 2
            + [joint_control]
            + ["velocity"] * 2
            + [joint_control] * 7
            + ["velocity"] * 2,
            self_collision=False,
        )

        # Assistive grasp params
        self.object_in_hand = None
        self.obj_cid = None
        self.obj_cid_params = {}
        self.should_freeze_joints = False
        self.release_counter = None
        self.freeze_vals = {}

    @property
    def joint_ids(self):
        return np.sort(
            np.concatenate(
                (self.base_joint_ids, self.head_joint_ids, self.torso_and_arm_joint_ids, self.gripper_fingers_joint_ids)
            )
        )

    @property
    def joint_names(self):
        return [
            "r_wheel_joint",
            "l_wheel_joint",
            "torso_lift_joint",
            "head_pan_joint",
            "head_tilt_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "r_gripper_finger_joint",
            "l_gripper_finger_joint",
        ]

    @property
    def joint_damping(self):
        return np.array([get_joint_info(self.robot_ids[0], joint_id)[6] for joint_id in self.joint_ids])

    @property
    def num_joints(self):
        return len(self.joint_ids)

    @property
    def lower_joint_limits(self):
        return np.array(
            [
                -100.0,
                -100.0,
                0.0,
                -1.57,
                -0.76,
                -1.6056,
                -1.221,
                -100.0,
                -2.251,
                -100.0,
                -2.16,
                -100.0,
                0.0,
                0.0,
            ]
        )

    @property
    def upper_joint_limits(self):
        return np.array(
            [
                100.0,
                100.0,
                0.38615,
                1.57,
                1.45,
                1.6056,
                1.518,
                100.0,
                2.251,
                100.0,
                2.16,
                100.0,
                0.05,
                0.05,
            ]
        )

    @property
    def joint_range(self):
        return self.upper_joint_limits - self.lower_joint_limits

    @property
    def max_joint_velocities(self):
        return np.array(
            [
                17.4,  # rad/s
                17.4,  # rad/s
                0.1,  # m/s
                1.57,  # rad/s
                1.57,  # rad/s
                1.256,  # rad/s
                1.454,  # rad/s
                1.571,  # rad/s
                1.521,  # rad/s
                1.571,  # rad/s
                2.268,  # rad/s
                2.268,  # rad/s
                0.05,  # m/s
                0.05,  # m/s
            ]
        )

    @property
    def eef_link_id(self):
        """
        Link corresponding to eef
        """
        return 19

    @property
    def tucked_default_joints(self):
        return np.array(
            [
                0.0,
                0.0,  # wheels
                0.02,  # trunk
                0.0,
                0.0,  # head
                1.1707963267948966,
                1.4707963267948965,
                -0.4,
                1.6707963267948966,
                0.0,
                1.5707963267948966,
                0.0,  # arm
                0.05,
                0.05,  # gripper
            ]
        )

    @property
    def untucked_default_joints(self):
        if self.default_arm_pose == "vertical":
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -0.94121,
                    -0.64134,
                    1.55186,
                    1.65672,
                    -0.93218,
                    1.53416,
                    2.14474,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )
        elif self.default_arm_pose == "diagonal15":
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -0.95587,
                    -0.34778,
                    1.46388,
                    1.47821,
                    -0.93813,
                    1.4587,
                    1.9939,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )
        elif self.default_arm_pose == "diagonal30":
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -1.06595,
                    -0.22184,
                    1.53448,
                    1.46076,
                    -0.84995,
                    1.36904,
                    1.90996,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )
        elif self.default_arm_pose == "diagonal45":
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -1.11479,
                    -0.0685,
                    1.5696,
                    1.37304,
                    -0.74273,
                    1.3983,
                    1.79618,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )
        else:  # horizontal
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -1.43016,
                    0.20965,
                    1.86816,
                    1.77576,
                    -0.27289,
                    1.31715,
                    2.01226,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )

        return pose

    def force_wakeup(self):
        """
        compatibility hack - mjlbach
        """
        pass

    def get_proprioception_dim(self):
        return 49

    def get_proprioception(self):
        relative_eef_pos = self.get_relative_eef_position()
        relative_eef_orn = p.getEulerFromQuaternion(self.get_relative_eef_orientation())
        joint_states = np.array([j.get_state() for j in self.ordered_joints]).astype(np.float32).flatten()
        self.ag_data = self.calculate_ag_object()
        is_grasping = np.array([self.object_in_hand is not None and self.release_counter is None]).astype(np.float32)
        return np.concatenate([relative_eef_pos, relative_eef_orn, is_grasping, joint_states])

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        if self.arm_controller_type == "cartesian_ik":
            self.action_high = np.array(
                [self.base_max_linear_vel]
                + [self.base_max_angular_vel]
                + [self.max_joint_velocities[3], self.max_joint_velocities[4]]
                + [self.arm_max_delta_pos] * self.arm_delta_pos_dim
                + [self.arm_max_delta_orn] * self.arm_delta_orn_dim
                + [1]  # Will be converted into binary signal for fingers to close or open
            )
            self.action_low = -self.action_high
        elif self.arm_controller_type == "joint_space_velocity":
            self.action_high = np.array(
                [self.base_max_linear_vel]  # Base is always controlled in diff drive
                + [self.base_max_angular_vel]  # Base is always controlled in diff drive
                + list(self.max_joint_velocities[2:12])  # torso, head, arm
                + [1]  # Will be converted into binary signal for fingers to close or open
            )
            self.action_low = -self.action_high
        elif self.arm_controller_type == "joint_space_position":
            self.action_high = np.array(
                [self.base_max_linear_vel]  # Base is always controlled in diff drive
                + [self.base_max_angular_vel]  # Base is always controlled in diff drive
                + list(self.upper_joint_limits[2:12])  # torso, head, arm
                + [1]  # Will be converted into binary signal for fingers to close or open
            )
            self.action_low = np.array(
                [-self.base_max_linear_vel]  # Base is always controlled in diff drive
                + [-self.base_max_angular_vel]  # Base is always controlled in diff drive
                + list(self.lower_joint_limits[2:12])  # torso, head, arm
                + [-1]  # Will be converted into binary signal for fingers to close or open
            )
        else:
            print("wrong type")
            exit(-1)

        self.action_space = gym.spaces.Box(shape=(self.action_dim,), low=-1.0, high=1.0, dtype=np.float32)

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        assert False, "Fetch does not support discrete actions"

    def robot_specific_reset(self):
        """
        Fetch robot specific reset.
        Reset the torso lift joint and tuck the arm towards the body
        """
        super(FetchGripper, self).robot_specific_reset()

        untucked_q = self.untucked_default_joints
        set_joint_positions(self.robot_ids[0], self.joint_ids, untucked_q)

        self.arm_controller.reset()

    def get_end_effector_position(self):
        """
        Get end-effector position
        """
        return self.parts["gripper_link"].get_position()

    def end_effector_part_index(self):
        """
        Get end-effector link id
        """
        return self.parts["gripper_link"].body_part_index

    def get_relative_eef_pose(self):
        """
        Get relative end-effector pose wrt robot base (returns 4x4 homogenous array)
        """
        return T.pose2mat(get_relative_pose(body=self.robot_ids[0], link1=self.eef_link_id))

    def get_relative_eef_position(self):
        """
        Get relative end-effector position wrt robot base
        """
        return self.get_relative_eef_pose()[:3, -1]

    def get_relative_eef_orientation(self):
        """
        Get relative end-effector orientation wrt robot base, in quaternion form
        """
        return T.mat2quat(self.get_relative_eef_pose()[:3, :3])

    def load(self):
        """
        Load the robot into pybullet. Filter out unnecessary self collision
        due to modeling imperfection in the URDF
        """
        ids = super(FetchGripper, self).load()
        robot_id = self.robot_ids[0]

        disable_collision_names = [
            ["torso_lift_joint", "shoulder_lift_joint"],
            ["torso_lift_joint", "torso_fixed_joint"],
            ["caster_wheel_joint", "estop_joint"],
            ["caster_wheel_joint", "laser_joint"],
            ["caster_wheel_joint", "torso_fixed_joint"],
            ["caster_wheel_joint", "l_wheel_joint"],
            ["caster_wheel_joint", "r_wheel_joint"],
        ]
        for names in disable_collision_names:
            link_a, link_b = joints_from_names(robot_id, names)
            p.setCollisionFilterPair(robot_id, robot_id, link_a, link_b, 0)

        # Increase lateral friction for end effector
        for link in self.gripper_joint_ids:
            p.changeDynamics(self.get_body_id(), link, lateralFriction=500)

        return ids

    def apply_action(self, action):
        """
        Apply policy action
        """
        if self.use_ag:
            self.handle_assisted_grasping(action)

        real_action = self.policy_action_to_robot_action(action)
        self.apply_robot_action(real_action)
        if self.should_freeze_joints:
            self.freeze_joints()

    def policy_action_to_robot_action(self, action):
        self.calc_state()
        if self.arm_controller_type == "cartesian_ik":
            # action has 2 + 2 + 6 + 1 = 11 dimensional
            # linear and angular base vels + pan and tilt head joint vels + ee position and vel + grasping open/close

            # 1) Unnormalize (possibly) the given action
            robot_action = super(FetchGripper, self).policy_action_to_robot_action(action)

            # 2) Some actions need to be transformed into commands for the DoFs of the robot
            robot_joint_commands = np.zeros(self.num_joints)

            # 3) Copy directly the first two dimensions (we do not transform here into wheel motion)
            # dim 0 and 1 of the action space: linear and angular velocities of robot base
            robot_joint_commands[self.base_joint_idx] = robot_action[:2]

            # 3) Process the head joint velocities
            # dim 2 and 3 of the action space: head joint velocities
            # we check that we do not command velocities that would bring the head out of the joint limits
            # if that happens, we command 0 vels
            current_joint_position = np.array(
                [item[0] for item in p.getJointStates(self.robot_ids[0], self.head_joint_ids)]
            )
            violate_lower_limit = current_joint_position < (
                self.lower_joint_limits[self.head_joint_idx] + self.head_limit_epsilon
            )
            negative_action = robot_action[2:4] < 0.0

            violate_upper_limit = current_joint_position > (
                self.upper_joint_limits[self.head_joint_idx] - self.head_limit_epsilon
            )
            positive_action = robot_action[2:4] > 0.0
            # zero out head movement velocity if it gets close to the joint limit
            violation = np.logical_or(violate_lower_limit * negative_action, violate_upper_limit * positive_action)
            robot_joint_commands[self.head_joint_idx] = robot_action[2:4] * (~violation)

            # 4) Process the desired delta in end-effector pose
            # dim 4-9: eef delta pos and orn
            q_arm_controller = self.arm_controller.control(robot_action[4:10])  # velocities of all joints
            # TODO: why are we dividing by max_joint_velocities here? are the velocities not coming unnormalized already??
            robot_joint_commands[self.torso_and_arm_joint_idx] = (
                q_arm_controller[self.torso_and_arm_joint_idx] / self.max_joint_velocities[self.torso_and_arm_joint_idx]
            )

            # 5) Process the desired gripper action (binary: open/close)
            # dim 10: gripper open/close (with maximum joint velocity)
            # we check that we do not command velocities that would bring the fingers out of the joint limits
            # if that happens, we command 0 vels
            if robot_action[10] > 0.0:
                robot_joint_commands[self.gripper_fingers_joint_idx] = 1.0
            else:
                robot_joint_commands[self.gripper_fingers_joint_idx] = -1.0
            current_joint_position = np.array(
                [item[0] for item in p.getJointStates(self.robot_ids[0], self.gripper_fingers_joint_ids)]
            )
            violate_lower_limit = current_joint_position < (
                self.lower_joint_limits[self.gripper_fingers_joint_idx] + self.gripper_limit_epsilon
            )
            negative_action = robot_joint_commands[self.gripper_fingers_joint_idx] < 0.0

            violate_upper_limit = current_joint_position > (
                self.upper_joint_limits[self.gripper_fingers_joint_idx] - self.gripper_limit_epsilon
            )
            positive_action = robot_joint_commands[self.gripper_fingers_joint_idx] > 0.0
            # zero out gripper velocity if it gets close to the joint limit
            violation = np.logical_or(violate_lower_limit * negative_action, violate_upper_limit * positive_action)
            robot_joint_commands[self.gripper_fingers_joint_idx] *= ~violation

            return robot_joint_commands
        else:
            # action has 2 + 1 + 2 + 7 + 1 = 13 dimensional
            # linear and angular base vels + torso + head pan/tilt + arm + gripper

            # 1) Unnormalize (possibly) the given action
            # robot_action = super(FetchGripper, self).policy_action_to_robot_action(action)
            robot_action = action  # TODO: Set the action space to be physical units

            # 2) Actions are directly commands for the DoFs of the robot except for the gripper
            robot_joint_commands = np.zeros(self.num_joints)
            robot_joint_commands[:12] = robot_action[:12]

            # 5) Process the desired gripper action (binary: open/close)
            # dim 10: gripper open/close (with maximum joint velocity)
            # we check that we do not command velocities that would bring the fingers out of the joint limits
            # if that happens, we command 0 vels
            if robot_action[12] > 0.0:
                robot_joint_commands[self.gripper_fingers_joint_idx] = 1.0
            else:
                robot_joint_commands[self.gripper_fingers_joint_idx] = -1.0
            current_joint_position = np.array(
                [item[0] for item in p.getJointStates(self.robot_ids[0], self.gripper_fingers_joint_ids)]
            )
            violate_lower_limit = current_joint_position < (
                self.lower_joint_limits[self.gripper_fingers_joint_idx] + self.gripper_limit_epsilon
            )
            negative_action = robot_joint_commands[self.gripper_fingers_joint_idx] < 0.0

            violate_upper_limit = current_joint_position > (
                self.upper_joint_limits[self.gripper_fingers_joint_idx] - self.gripper_limit_epsilon
            )
            positive_action = robot_joint_commands[self.gripper_fingers_joint_idx] > 0.0
            # zero out gripper velocity if it gets close to the joint limit
            violation = np.logical_or(violate_lower_limit * negative_action, violate_upper_limit * positive_action)
            robot_joint_commands[self.gripper_fingers_joint_idx] *= ~violation

            return robot_joint_commands

    def calculate_ag_object(self):
        """
        Calculates which object to assisted-grasp. Returns an (object_id, link_id) tuple or None
        if no valid AG-enabled object can be found.
        """
        # Step 1 - find all objects in contact with both gripper forks
        gripper_fork_1_contacts = p.getContactPoints(
            bodyA=self.get_body_id(), linkIndexA=self.gripper_fingers_joint_ids[0]
        )
        gripper_fork_2_contacts = p.getContactPoints(
            bodyA=self.get_body_id(), linkIndexA=self.gripper_fingers_joint_ids[1]
        )

        contact_dict = {}
        set_1_contacts = set()
        for contact in gripper_fork_1_contacts:
            set_1_contacts.add(contact[2])
            if contact[2] not in contact_dict:
                contact_dict[contact[2]] = []
            contact_dict[contact[2]].append({"contact_position": contact[5], "target_link": contact[4]})

        set_2_contacts = set()
        for contact in gripper_fork_2_contacts:
            set_2_contacts.add(contact[2])
            if contact[2] not in contact_dict:
                contact_dict[contact[2]] = []
            contact_dict[contact[2]].append({"contact_position": contact[5], "target_link": contact[4]})

        candidates = list(set_1_contacts.intersection(set_2_contacts))

        if len(candidates) == 0:
            return None

        # Step 2, check if contact with target is inside bounding box
        # Might be easier to check if contact normal points towards or away from center of gripper from
        # getContact Points

        if self.ag_strict_mode:
            # Compute gripper bounding box
            corners = []

            eef_pos, eef_orn, _, _, _, _ = p.getLinkState(self.get_body_id(), self.eef_link_id)
            i_eef_pos, i_eef_orn = p.invertTransform(eef_pos, eef_orn)

            gripper_fork_1_state = p.getLinkState(self.get_body_id(), self.gripper_fingers_joint_ids[0])
            local_corners = [
                [0.04, -0.012, 0.014],
                [0.04, -0.012, -0.014],
                [-0.04, -0.012, 0.014],
                [-0.04, -0.012, -0.014],
            ]
            for coord in local_corners:
                corner, _ = p.multiplyTransforms(gripper_fork_1_state[0], gripper_fork_1_state[1], coord, [0, 0, 0, 1])
                corners.append(corner)

            gripper_fork_2_state = p.getLinkState(self.get_body_id(), self.gripper_fingers_joint_ids[1])
            local_corners = [
                [0.04, 0.012, 0.014],
                [0.04, 0.012, -0.014],
                [-0.04, 0.012, 0.014],
                [-0.04, 0.012, -0.014],
            ]
            for coord in local_corners:
                corner, _ = p.multiplyTransforms(gripper_fork_2_state[0], gripper_fork_2_state[1], coord, [0, 0, 0, 1])
                corners.append(corner)

            eef_local_corners = []
            for coord in corners:
                corner, _ = p.multiplyTransforms(i_eef_pos, i_eef_orn, coord, [0, 0, 0, 1])
                eef_local_corners.append(corner)

            eef_local_corners = np.stack(eef_local_corners)
            for candidate in candidates:
                new_contact_point_data = []
                for contact_point_data in contact_dict[candidate]:
                    pos = contact_point_data["contact_position"]
                    local_pos, _ = p.multiplyTransforms(i_eef_pos, i_eef_orn, pos, [0, 0, 0, 1])
                    x_inside = local_pos[0] < np.max(eef_local_corners[:, 0]) and local_pos[0] > np.min(
                        eef_local_corners[:, 0]
                    )
                    y_inside = local_pos[1] < np.max(eef_local_corners[:, 1]) and local_pos[1] > np.min(
                        eef_local_corners[:, 1]
                    )
                    z_inside = local_pos[2] < np.max(eef_local_corners[:, 2]) and local_pos[2] > np.min(
                        eef_local_corners[:, 2]
                    )
                    if x_inside and y_inside and z_inside:
                        new_contact_point_data.append(contact_point_data)
                contact_dict[candidate] = new_contact_point_data

        # Step 3 - find the closest object to the gripper center among these "inside" objects
        gripper_state = p.getLinkState(self.get_body_id(), self.eef_link_id)
        # Compute gripper bounding box
        gripper_center_pos = np.copy(GRIPPER_BASE_CENTER_OFFSET)
        gripper_center_pos, _ = p.multiplyTransforms(
            gripper_state[0], gripper_state[1], gripper_center_pos, [0, 0, 0, 1]
        )

        self.candidate_data = []
        for candidate in candidates:
            for contact_point_data in contact_dict[candidate]:
                dist = np.linalg.norm(np.array(contact_point_data["contact_position"]) - np.array(gripper_center_pos))
                self.candidate_data.append((candidate, contact_point_data["target_link"], dist))

        self.candidate_data = sorted(self.candidate_data, key=lambda x: x[2])
        if len(self.candidate_data) > 0:
            ag_bid, ag_link, _ = self.candidate_data[0]
        else:
            return None

        # Return None if any of the following edge cases are activated
        if not self.simulator.can_assisted_grasp(ag_bid, ag_link) or (self.get_body_id() == ag_bid):
            return None

        return ag_bid, ag_link

    def release_grasp(self):
        p.removeConstraint(self.obj_cid)
        self.obj_cid = None
        self.obj_cid_params = {}
        self.should_freeze_joints = False
        self.release_counter = 0

    def handle_release_window(self):
        self.release_counter += 1
        time_since_release = self.release_counter * self.simulator.render_timestep
        if time_since_release >= RELEASE_WINDOW:
            set_coll_filter(
                target_id=self.object_in_hand,
                body_id=self.get_body_id(),
                body_links=self.gripper_joint_ids,
                enable=True,
            )
            self.object_in_hand = None
            self.release_counter = None

    def freeze_joints(self):
        """
        Freezes gripper finger joints - used in assisted grasping.
        """
        for joint_index, j_val in self.freeze_vals.items():
            p.resetJointState(self.get_body_id(), joint_index, targetValue=j_val, targetVelocity=0.0)

    def establish_grasp(self, ag_data):
        ag_bid, ag_link = ag_data

        child_frame_pos, child_frame_orn = get_child_frame_pose(
            parent_bid=self.get_body_id(), parent_link=self.eef_link_id, child_bid=ag_bid, child_link=ag_link
        )

        # If we grab a child link of a URDF, create a p2p joint
        if ag_link == -1:
            joint_type = p.JOINT_FIXED
        else:
            joint_type = p.JOINT_POINT2POINT

        self.obj_cid = p.createConstraint(
            parentBodyUniqueId=self.get_body_id(),
            parentLinkIndex=self.eef_link_id,
            childBodyUniqueId=ag_bid,
            childLinkIndex=ag_link,
            jointType=joint_type,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=child_frame_pos,
            childFrameOrientation=child_frame_orn,
        )
        # Modify max force based on user-determined assist parameters
        if ag_link == -1:
            max_force = ASSIST_FORCE
        else:
            max_force = ASSIST_FORCE * ARTICULATED_ASSIST_FRACTION
        p.changeConstraint(self.obj_cid, maxForce=max_force)

        self.obj_cid_params = {
            "childBodyUniqueId": ag_bid,
            "childLinkIndex": ag_link,
            "jointType": joint_type,
            "maxForce": max_force,
        }
        self.object_in_hand = ag_bid
        self.should_freeze_joints = True
        # Disable collisions while picking things up
        set_coll_filter(target_id=ag_bid, body_id=self.get_body_id(), body_links=self.gripper_joint_ids, enable=False)
        for joint_index in self.gripper_fingers_joint_ids:
            j_val = p.getJointState(self.get_body_id(), joint_index)[0]
            self.freeze_vals[joint_index] = j_val

    def handle_assisted_grasping(self, action):
        """
        Handles assisted grasping.
        :param action: numpy array of actions.
        """

        applying_grasp = action[10] < 0.0
        releasing_grasp = action[10] > 0.0

        # Execute gradual release of object
        if self.object_in_hand is not None and self.release_counter is not None:
            self.handle_release_window()

        elif self.object_in_hand and self.release_counter is None:
            constraint_violated = get_constraint_violation(self.obj_cid) > CONSTRAINT_VIOLATION_THRESHOLD
            if constraint_violated or releasing_grasp:
                self.release_grasp()

        elif not self.object_in_hand and applying_grasp:
            ag_data = self.calculate_ag_object()
            if ag_data:
                self.establish_grasp(ag_data)

    def is_grasping(self, candidate_obj):
        return np.array([self.object_in_hand == candidate_obj])

    # significant overlap with BehaviorRobot
    def dump_state(self):
        if not self.use_ag:
            return None

        # Recompute child frame pose because it could have changed since the
        # constraint has been created
        if self.obj_cid is not None:
            ag_bid = self.obj_cid_params["childBodyUniqueId"]
            ag_link = self.obj_cid_params["childLinkIndex"]
            child_frame_pos, child_frame_orn = child_frame_pos, child_frame_orn = get_child_frame_pose(
                parent_bid=self.get_body_id(), parent_link=self.eef_link_id, child_bid=ag_bid, child_link=ag_link
            )
            self.obj_cid_params.update(
                {
                    "childFramePosition": child_frame_pos,
                    "childFrameOrientation": child_frame_orn,
                }
            )

        return {
            "object_in_hand": self.object_in_hand,
            "release_counter": self.release_counter,
            "should_freeze_joints": self.should_freeze_joints,
            "freeze_vals": self.freeze_vals,
            "obj_cid": self.obj_cid,
            "obj_cid_params": self.obj_cid_params,
        }

    def load_state(self, dump):
        if not self.use_ag:
            return

        # Cancel the previous AG if exists
        if self.obj_cid is not None:
            p.removeConstraint(self.obj_cid)

        if self.object_in_hand is not None:
            set_coll_filter(
                target_id=self.object_in_hand,
                body_id=self.get_body_id(),
                body_links=self.gripper_joint_ids,
                enable=True,
            )

        self.object_in_hand = dump["object_in_hand"]
        self.release_counter = dump["release_counter"]
        self.should_freeze_joints = dump["should_freeze_joints"]
        self.freeze_vals = {int(key): val for key, val in dump["freeze_vals"].items()}
        self.obj_cid = dump["obj_cid"]
        self.obj_cid_params = dump["obj_cid_params"]
        if self.obj_cid is not None:
            self.obj_cid = p.createConstraint(
                parentBodyUniqueId=self.get_body_id(),
                parentLinkIndex=self.eef_link_id,
                childBodyUniqueId=dump["obj_cid_params"]["childBodyUniqueId"],
                childLinkIndex=dump["obj_cid_params"]["childLinkIndex"],
                jointType=dump["obj_cid_params"]["jointType"],
                jointAxis=(0, 0, 0),
                parentFramePosition=(0, 0, 0),
                childFramePosition=dump["obj_cid_params"]["childFramePosition"],
                childFrameOrientation=dump["obj_cid_params"]["childFrameOrientation"],
            )
            p.changeConstraint(self.obj_cid, maxForce=dump["obj_cid_params"]["maxForce"])

        if self.object_in_hand is not None:
            set_coll_filter(
                target_id=self.object_in_hand,
                body_id=self.get_body_id(),
                body_links=self.gripper_joint_ids,
                enable=False,
            )

    def can_toggle(self, toggle_position, toggle_distance_threshold):
        for joint_id in self.gripper_joint_ids:
            finger_link_state = p.getLinkState(self.get_body_id(), joint_id)
            link_pos = finger_link_state[0]
            if np.linalg.norm(np.array(link_pos) - np.array(toggle_position)) < toggle_distance_threshold:
                return True
        return False
