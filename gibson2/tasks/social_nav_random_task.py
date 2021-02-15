from gibson2.tasks.point_nav_random_task import PointNavRandomTask
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.pedestrian import Pedestrian
from gibson2.termination_conditions.pedestrian_collision import PedestrianCollision

import pybullet as p
import numpy as np
import rvo2
from IPython import embed


class SocialNavRandomTask(PointNavRandomTask):
    """
    Social Navigation Random Task
    The goal is to navigate to a random goal position, in the presence of pedestrians
    """

    def __init__(self, env):
        """
        numStepsStop        A list of number of consecutive timesteps
                            each pedestrian had to stop for.
        numStepsStopThresh  The maximum number of consecutive timesteps
                            the pedestrian should stop for before sampling
                            a new waypoint.
        maxNeighborRadius   Maximum distance to be considered a nearby
                            a new waypoint.
        backoffRadianThresh If the angle (in radian) between the pedestrian's
                            orientation and the next direction of the next
                            goal is greater than the backoffRadianThresh,
                            then the pedestrian is considered backing off.
        """
        super(SocialNavRandomTask, self).__init__(env)
        # For debugging purposes, so that we can simulate colliding pedestrians
        # np.random.seed(1)
        self.termination_conditions.append(PedestrianCollision(self.config))
        # Each pixel is 0.01 square meter
        num_sqrt_meter = env.scene.floor_map[0].nonzero()[0].shape[0] / 100.0
        self.num_sqrt_meter_per_ped = self.config.get(
            'num_sqrt_meter_per_ped', 10)
        self.num_pedestrians = int(
            num_sqrt_meter / self.num_sqrt_meter_per_ped)

        self.num_steps_stop = [0] * self.num_pedestrians
        self.neighbor_stop_radius = self.config.get(
            'neighbor_stop_radius', 1.0)
        self.num_steps_stop_thresh = self.config.get(
            'num_steps_stop_thresh', 5)
        # backoff when angle is greater than 2.7 radians
        self.backoff_radian_thresh = self.config.get(
            'backoff_radian_thresh', 2.7)

        self.neighbor_dist = self.config.get('orca_neighbor_dist', 5)
        self.max_neighbors = self.num_pedestrians + 1
        self.time_horizon = self.config.get('orca_time_horizon', 2.0)
        self.time_horizon_obst = self.config.get('orca_time_horizon_obst', 2.0)
        self.radius = self.config.get('orca_radius', 0.3)
        self.max_speed = self.config.get('orca_max_speed', 0.5)
        self.pedestrian_velocity = self.config.get('pedestrian_velocity', 1.0)
        self.pedestrian_goal_thresh = \
            self.config.get('pedestrian_goal_thresh', 0.3)
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).
        """
        self.orca_sim = rvo2.PyRVOSimulator(
            env.action_timestep,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            self.radius,
            self.max_speed)
        self.pedestrians, self.orca_pedestrians = self.load_pedestrians(env)
        self.pedestrian_goals = self.load_pedestrian_goals(env)
        self.load_obstacles(env)

    def load_pedestrians(self, env):
        """
        Load pedestrians

        :param env: environment instance
        :return: a list of pedestrians
        """
        self.robot_orca_ped = self.orca_sim.addAgent((0, 0))
        pedestrians = []
        orca_pedestrians = []
        colors = [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1]
        ]
        for i in range(self.num_pedestrians):
            # ped = VisualMarker(
            #     visual_shape=p.GEOM_CYLINDER,
            #     rgba_color=colors[i % 3],
            #     radius=0.3,
            #     length=1.8,
            #     initial_offset=[0, 0, 1.8 / 2])
            ped = Pedestrian()
            env.simulator.import_object(ped)
            pedestrians.append(ped)
            orca_ped = self.orca_sim.addAgent((0, 0))
            orca_pedestrians.append(orca_ped)
        return pedestrians, orca_pedestrians

    def load_pedestrian_goals(self, env):
        pedestrian_goals = []
        colors = [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1]
        ]
        for i, ped in enumerate(self.pedestrians):
            ped_goal = VisualMarker(
                visual_shape=p.GEOM_CYLINDER,
                rgba_color=colors[i % 3][:3] + [0.5],
                radius=0.3,
                length=0.2,
                initial_offset=[0, 0, 0.2 / 2])
            env.simulator.import_object(ped_goal)
            pedestrian_goals.append(ped_goal)
        return pedestrian_goals

    def load_obstacles(self, env):
        for obj_name in env.scene.objects_by_name:
            obj = env.scene.objects_by_name[obj_name]
            if obj.category in ['walls', 'floors', 'ceilings']:
                continue
            # body_id = obj.body_ids[0]
            # if p.getBodyInfo(body_id)[0].decode('utf-8') == 'world':
            #     aabb = p.getAABB(body_id, 0)
            # else:
            #     aabb = p.getAABB(body_id, -1)

            # (x_min, y_min, _), (x_max, y_max, _) = aabb
            # # self.orca_sim.addObstacle([
            # #     (x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)
            # # ])
            # self.orca_sim.addObstacle([
            #     (x_max, y_max), (x_min, y_max), (x_min, y_min), (x_max, y_min)
            # ])
            x_extent, y_extent = obj.bounding_box[:2]
            initial_bbox = np.array([
                [x_extent / 2.0, y_extent / 2.0],
                [-x_extent / 2.0, y_extent / 2.0],
                [-x_extent / 2.0, -y_extent / 2.0],
                [x_extent / 2.0, -y_extent / 2.0]
            ])
            yaw = obj.bbox_orientation_rpy[2]
            rot_mat = np.array([
                [np.cos(-yaw), -np.sin(-yaw)],
                [np.sin(-yaw), np.cos(-yaw)],
            ])
            initial_bbox = initial_bbox.dot(rot_mat)
            initial_bbox = initial_bbox + obj.bbox_pos[:2]
            self.orca_sim.addObstacle([
                tuple(initial_bbox[0]),
                tuple(initial_bbox[1]),
                tuple(initial_bbox[2]),
                tuple(initial_bbox[3]),
            ])

        self.orca_sim.processObstacles()

    def reset_pedestrians(self, env):
        """
        Reset the poses of pedestrians to have no collisions with the scene or the robot and set waypoints to follow

        :param env: environment instance
        """
        self.pedestrian_waypoints = []
        for id, (ped, orca_ped) in enumerate(zip(self.pedestrians, self.orca_pedestrians)):
            initial_pos = None
            must_resample_pos = True

            # resample pedestrian's initial position
            while must_resample_pos:
                _, initial_pos = env.scene.get_random_point(
                    floor=self.floor_num)
                must_resample_pos = False

                # If too close to the robot, resample
                dist = np.linalg.norm(initial_pos[:2] - self.initial_pos[:2])
                if dist < self.radius:
                    must_resample_pos = True
                    continue

                # If too close to the previous pedestrians, resample
                for neighbor_id in range(id):
                    neighbor_ped = self.pedestrians[neighbor_id]
                    neighbor_pos_xyz = neighbor_ped.get_position()
                    dist = np.linalg.norm(
                        np.array(neighbor_pos_xyz)[:2] -
                        initial_pos[:2])
                    if dist < self.radius:
                        must_resample_pos = True
                        break

            ped.set_position_orientation(
                initial_pos, p.getQuaternionFromEuler(ped.default_orn_euler))
            self.orca_sim.setAgentPosition(orca_ped, tuple(initial_pos[0:2]))
            waypoints = self.sample_new_target_pos(env, initial_pos)
            self.pedestrian_waypoints.append(waypoints)

    def reset_agent(self, env):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.

        :param env: environment instance
        """
        super(SocialNavRandomTask, self).reset_agent(env)
        self.orca_sim.setAgentPosition(self.robot_orca_ped,
                                       tuple(self.initial_pos[0:2]))
        self.reset_pedestrians(env)

    def sample_new_target_pos(self, env, initial_pos):
        while True:
            _, target_pos = env.scene.get_random_point(floor=self.floor_num)
            # print('initial_pos', initial_pos)
            shortest_path, _ = env.scene.get_shortest_path(
                self.floor_num,
                initial_pos[:2],
                target_pos[:2],
                entire_path=True)
            if len(shortest_path) > 1:
                break
        waypoints = self.shortest_path_to_waypoints(shortest_path)
        return waypoints

    def shortest_path_to_waypoints(self, shortest_path):
        assert len(shortest_path) > 0
        waypoints = []
        valid_waypoint = None
        prev_waypoint = None
        cached_slope = None
        for waypoint in shortest_path:
            if valid_waypoint is None:
                valid_waypoint = waypoint
            elif cached_slope is None:
                cached_slope = waypoint - valid_waypoint
            else:
                cur_slope = waypoint - prev_waypoint
                cosine_angle = np.dot(cached_slope, cur_slope) / \
                    (np.linalg.norm(cached_slope) * np.linalg.norm(cur_slope))
                if np.abs(cosine_angle - 1.0) > 1e-3:
                    waypoints.append(valid_waypoint)
                    valid_waypoint = prev_waypoint
                    cached_slope = waypoint - valid_waypoint

            prev_waypoint = waypoint

        # Add the last two valid waypoints
        waypoints.append(valid_waypoint)
        waypoints.append(shortest_path[-1])

        # Remove the first waypoint because it's the same as the initial pos
        waypoints.pop(0)

        return waypoints

    def step(self, env):
        """
        Perform task-specific step: move the dynamic objects with action repeat

        :param env: environment instance
        """
        super(SocialNavRandomTask, self).step(env)
        self.orca_sim.setAgentPosition(
            self.robot_orca_ped,
            tuple(env.robots[0].get_position()[0:2]))

        for i, (ped, orca_ped, waypoints) in \
                enumerate(zip(self.pedestrians,
                              self.orca_pedestrians,
                              self.pedestrian_waypoints)):
            current_pos = np.array(ped.get_position())

            # Sample new waypoints if empty OR if the pedestrian froze for x amount of time.
            if len(waypoints) == 0 or self.num_steps_stop[i] >= self.num_steps_stop_thresh:
                # if self.num_steps_stop[i] >= self.num_steps_stop_thresh:
                #     print("sampling new point because pedestrian #${} \
                #           stoped for too long".format(i))
                waypoints = self.sample_new_target_pos(env, current_pos)
                self.pedestrian_waypoints[i] = waypoints
                self.num_steps_stop[i] = 0

            next_goal = waypoints[0]
            self.pedestrian_goals[i].set_position(
                np.array([next_goal[0], next_goal[1], current_pos[2]]))
            yaw = np.arctan2(next_goal[1] - current_pos[1],
                             next_goal[0] - current_pos[0])
            ped.set_yaw(yaw)
            desired_vel = next_goal - current_pos[0:2]
            desired_vel = desired_vel / \
                np.linalg.norm(desired_vel) * self.pedestrian_velocity
            self.orca_sim.setAgentPrefVelocity(orca_ped, tuple(desired_vel))

        self.orca_sim.doStep()

        next_peds_pos_xyz, next_peds_stop_flag = self.update_pos_and_stop_flags()

        # Update the pedestrian position in PyBullet if it does not stop
        # or update the position in RVO2 simulator if it needs to stop
        for i, (ped, orca_pred, waypoints) in \
                enumerate(zip(self.pedestrians,
                              self.orca_pedestrians,
                              self.pedestrian_waypoints)):
            pos_xyz = next_peds_pos_xyz[i]
            if next_peds_stop_flag[i] is True:
                # revert back ORCA sim pedestrian to the previous time step
                self.num_steps_stop[i] += 1
                self.orca_sim.setAgentPosition(orca_pred, pos_xyz[:2])
            else:
                # advance pybullet pedstrian to the current time step
                self.num_steps_stop[i] = 0
                ped.set_position(pos_xyz)
                next_goal = waypoints[0]
                if np.linalg.norm(next_goal - np.array(pos_xyz[:2])) \
                        <= self.pedestrian_goal_thresh:
                    waypoints.pop(0)

    def update_pos_and_stop_flags(self):
        """
        Wrapper function that updates pedestrians' next position and whether
        they should stop for the next time step

        :return: the list of next position for all pedestrians,
                 the list of flags whether the pedestrian should stop for the
                 next time step
        """
        next_peds_pos_xyz = \
            {i: ped.get_position() for i, ped in enumerate(self.pedestrians)}
        next_peds_stop_flag = [False for i in range(len(self.pedestrians))]

        for i, (ped, orca_ped, waypoints) in \
                enumerate(zip(self.pedestrians,
                              self.orca_pedestrians,
                              self.pedestrian_waypoints)):
            pos_xy = self.orca_sim.getAgentPosition(orca_ped)
            prev_pos_xyz = ped.get_position()
            next_pos_xyz = np.array([pos_xy[0], pos_xy[1], prev_pos_xyz[2]])

            if self.detect_backoff(ped, orca_ped):
                self.stop_neighbor_pedestrians(i,
                                               next_peds_stop_flag,
                                               next_peds_pos_xyz)
            elif next_peds_stop_flag[i] is False:
                # If there are no other neighboring pedestrians that forces
                # this pedestrian to stop, then simply update next position.
                next_peds_pos_xyz[i] = next_pos_xyz

        return next_peds_pos_xyz, next_peds_stop_flag

    def stop_neighbor_pedestrians(self, id, peds_stop_flags, peds_next_pos_xyz):
        """
        If the pedestrian whose instance stored in self.pedestrians with
        index |id| is attempting to backoff, all the other neighboring
        pedestrians within |self.neighbor_stop_radius| will stop

        :param id: the index of the pedestrian object
        :param peds_stop_flags: list of boolean corresponding to if the pestrian
                                at index i should stop for the next
        :param peds_next_pos_xyz: list of xyz position that the pedestrian would
                            move in the next timestep or the position in the
                            PyRVOSimulator that the pedestrian would revert to
        """
        ped = self.pedestrians[id]
        ped_pos_xyz = ped.get_position()

        for i, neighbor in enumerate(self.pedestrians):
            if id == i:
                continue
            neighbor_pos_xyz = neighbor.get_position()
            dist = np.linalg.norm([neighbor_pos_xyz[0] - ped_pos_xyz[0],
                                   neighbor_pos_xyz[1] - ped_pos_xyz[1]])
            if dist <= self.neighbor_stop_radius:
                peds_stop_flags[i] = True
                peds_next_pos_xyz[i] = neighbor_pos_xyz
        peds_stop_flags[id] = True
        peds_next_pos_xyz[id] = ped_pos_xyz

    def detect_backoff(self, ped, orca_ped):
        """
        Detects if the pedestrian is attempting to perform a backoff
        due to some form of imminent collision

        :param ped: the pedestrain object
        :param orca_ped: the pedestrian id in the orca simulator
        :return: whether the pedestrian is backing off
        """
        pos_xy = self.orca_sim.getAgentPosition(orca_ped)
        prev_pos_xyz = ped.get_position()

        yaw = ped.get_yaw()

        # Computing the directional vectors from yaw
        normalized_dir = np.array([np.cos(yaw), np.sin(yaw)])

        next_dir = np.array([pos_xy[0] - prev_pos_xyz[0],
                             pos_xy[1] - prev_pos_xyz[1]])

        if np.linalg.norm(next_dir) == 0.0:
            return False

        next_normalized_dir = next_dir / np.linalg.norm(next_dir)

        angle = np.arccos(np.dot(normalized_dir, next_normalized_dir))
        return angle >= self.backoff_radian_thresh