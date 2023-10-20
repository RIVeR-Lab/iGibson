import logging
import os

import gym

import igibson
from igibson.object_states import AABB
from igibson.object_states.utils import detect_closeness
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots import REGISTERED_ROBOTS
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.scenes.stadium_scene import StadiumScene
from igibson.simulator import Simulator
from igibson.simulator_vr import SimulatorVR
from igibson.utils.utils import parse_config

log = logging.getLogger(__name__)


class BaseEnv(gym.Env):
    """
    Base Env class that handles loading scene and robot, following OpenAI Gym interface.
    Functions like reset and step are not implemented.
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        rendering_settings=None,
        vr_settings=None,
        device_idx=0,
        use_pb_gui=False,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, headless_tensor, gui_interactive, gui_non_interactive, vr
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param rendering_settings: rendering_settings to override the default one
        :param vr_settings: vr_settings to override the default one
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """

        #print("[env_base::__init__] START")

        self.config = parse_config(config_file)
        if scene_id is not None:
            self.config["scene_id"] = scene_id

        image_width = self.config["image_width"]
        image_height = self.config["image_height"]

        #print("[env_base::__init__] image_width: " + str(image_width))
        #print("[env_base::__init__] image_height: " + str(image_height))
        
        #print("[env_base::__init__] DEBUG INF")
        #while(1):
        #    continue

        self.mode = mode
        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep
        self.rendering_settings = rendering_settings
        self.vr_settings = vr_settings
        self.texture_randomization_freq = self.config.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config.get("object_randomization_freq", None)
        self.object_randomization_idx = 0
        self.num_object_randomization_idx = 10

        default_enable_shadows = False  # What to do if it is not specified in the config file
        enable_shadow = self.config.get("enable_shadow", default_enable_shadows)
        default_enable_pbr = False  # What to do if it is not specified in the config file
        enable_pbr = self.config.get("enable_pbr", default_enable_pbr)
        texture_scale = self.config.get("texture_scale", 1.0)

        if self.rendering_settings is None:
            # TODO: We currently only support the optimized renderer due to some issues with obj highlighting.
            self.rendering_settings = MeshRendererSettings(
                enable_shadow=enable_shadow,
                enable_pbr=enable_pbr,
                msaa=False,
                texture_scale=texture_scale,
                optimized=self.config.get("optimized_renderer", True),
                load_textures=self.config.get("load_texture", True),
                hide_robot=self.config.get("hide_robot", True),
            )

        if mode == "vr":
            if self.vr_settings is None:
                self.vr_settings = VrSettings(use_vr=True)
            self.simulator = SimulatorVR(
                physics_timestep=physics_timestep,
                render_timestep=action_timestep,
                image_width=self.config.get("image_width", 128),
                image_height=self.config.get("image_height", 128),
                vertical_fov=self.config.get("vertical_fov", 90),
                device_idx=device_idx,
                rendering_settings=self.rendering_settings,
                vr_settings=self.vr_settings,
                use_pb_gui=use_pb_gui,
            )
        else:
            #print("[env_base::__init__] START Simulator")
            self.simulator = Simulator(
                mode=mode,
                physics_timestep=physics_timestep,
                render_timestep=action_timestep,
                image_width=self.config.get("image_width", 128),
                image_height=self.config.get("image_height", 128),
                vertical_fov=self.config.get("vertical_fov", 90),
                device_idx=device_idx,
                rendering_settings=self.rendering_settings,
                use_pb_gui=use_pb_gui,
            )
            #print("[env_base__init__] END Simulator")
        
        #print("[env_base::__init__] START load")
        self.load()
        #print("[env_base::__init__] END load")
        #print("[env_base::__init__] END")


    def reload(self, config_file):
        """
        Reload another config file.
        This allows one to change the configuration on the fly.

        :param config_file: new config file path
        """
        self.config = parse_config(config_file)
        self.simulator.reload()
        self.load()

    def reload_model(self, scene_id):
        """
        Reload another scene model.
        This allows one to change the scene on the fly.

        :param scene_id: new scene_id
        """
        self.config["scene_id"] = scene_id
        self.simulator.reload()
        self.load()

    def reload_model_object_randomization(self):
        """
        Reload the same model, with the next object randomization random seed.
        """
        if self.object_randomization_freq is None:
            return
        self.object_randomization_idx = (self.object_randomization_idx + 1) % (self.num_object_randomization_idx)
        self.simulator.reload()
        self.load()

    def load(self):
        """
        Load the scene and robot specified in the config file.
        """

        #print("[env_base::load] START")

        if self.config["scene"] == "empty":
            floor_plane_rgba=self.config.get("floor_plane_rgba", [1, 1, 1, 1])
            scene = EmptyScene(floor_plane_rgba=floor_plane_rgba)
        elif self.config["scene"] == "stadium":
            scene = StadiumScene()
        elif self.config["scene"] == "gibson":
            scene = StaticIndoorScene(
                self.config["scene_id"],
                waypoint_resolution=self.config.get("waypoint_resolution", 0.2),
                num_waypoints=self.config.get("num_waypoints", 10),
                build_graph=self.config.get("build_graph", False),
                trav_map_resolution=self.config.get("trav_map_resolution", 0.1),
                trav_map_erosion=self.config.get("trav_map_erosion", 2),
                pybullet_load_texture=self.config.get("pybullet_load_texture", False),
            )
        elif self.config["scene"] == "igibson":
            urdf_file = self.config.get("urdf_file", None)
            if urdf_file is None and not self.config.get("online_sampling", True):
                urdf_file = "{}_task_{}_{}_{}_fixed_furniture".format(
                    self.config["scene_id"],
                    self.config["task"],
                    self.config["task_id"],
                    self.config["instance_id"],
                )
            include_robots = self.config.get("include_robots", True)
            scene = InteractiveIndoorScene(
                self.config["scene_id"],
                urdf_file=urdf_file,
                waypoint_resolution=self.config.get("waypoint_resolution", 0.2),
                num_waypoints=self.config.get("num_waypoints", 10),
                build_graph=self.config.get("build_graph", False),
                trav_map_resolution=self.config.get("trav_map_resolution", 0.1),
                trav_map_erosion=self.config.get("trav_map_erosion", 2),
                trav_map_type=self.config.get("trav_map_type", "with_obj"),
                texture_randomization=self.texture_randomization_freq is not None,
                object_randomization=self.object_randomization_freq is not None,
                object_randomization_idx=self.object_randomization_idx,
                should_open_all_doors=self.config.get("should_open_all_doors", False),
                load_object_categories=self.config.get("load_object_categories", None),
                not_load_object_categories=self.config.get("not_load_object_categories", None),
                load_room_types=self.config.get("load_room_types", None),
                load_room_instances=self.config.get("load_room_instances", None),
                merge_fixed_links=self.config.get("merge_fixed_links", True)
                and not self.config.get("online_sampling", False),
                include_robots=include_robots,
            )
            # TODO: Unify the function import_scene and take out of the if-else clauses.
            first_n = self.config.get("_set_first_n_objects", -1)
            if first_n != -1:
                scene._set_first_n_objects(first_n)

        self.simulator.import_scene(scene)

        #print("[env_base::load] START ROBOT")

        # Get robot config
        robot_name = self.config.get("robot_name", "")

        if robot_name != "":
            for robot_config_file in os.listdir(os.path.join(igibson.configs_path, "robots")):
                config = parse_config(os.path.join(igibson.configs_path, "robots", robot_config_file))
                robot_config = config["robot"]
                robot_name_tmp = robot_config.pop("name")
                if robot_name != robot_name_tmp:
                    continue
        else:
            robot_config = self.config["robot"]
            robot_name = robot_config.pop("name")
        
        #print("[env_base::load] robot_config: " + str(robot_config))

        #print("[env_base::load] DEBUG_INF")
        #while 1:
        #    continue

        # If no robot has been imported from the scene
        if len(scene.robots) == 0:
            # Get corresponding robot class
            #robot_name = robot_config.pop("name")

            #print("[env_base::load] robot_name: " + str(robot_name))

            assert robot_name in REGISTERED_ROBOTS, "Got invalid robot to instantiate: {}".format(robot_name)
            robot = REGISTERED_ROBOTS[robot_name](**robot_config)

            #print("[env_base::load] START import_object")
            self.simulator.import_object(robot)
            #print("[env_base::load] END import_object")

            position = [0, 0, 0]
            robot.set_position(position)
            robot.reset()
            robot.keep_still()
            #robots[robot_name] = (robot, position[1])
            #print("[env_base::load] Loaded " + robot_name)

            #print("[env_base::load] DEBUG_INF")
            #while 1:
            #    continue

            # The scene might contain cached agent pose
            # By default, we load the agent pose that matches the robot name (e.g. Fetch, BehaviorRobot)
            # The user can also specify "agent_pose" in the config file to use the cached agent pose for any robot
            # For example, the user can load a BehaviorRobot and place it at Fetch's agent pose
            agent_pose_name = self.config.get("agent_pose", robot_name)
            if isinstance(scene, InteractiveIndoorScene) and agent_pose_name in scene.agent_poses:
                pos, orn = scene.agent_poses[agent_pose_name]

                if agent_pose_name != robot_name:
                    # Need to change the z-pos - assume we always want to place the robot bottom at z = 0
                    lower, _ = robot.states[AABB].get_value()
                    pos[2] = -lower[2]

                #print("[env_base::load] START set_position_orientation")
                robot.set_position_orientation(pos, orn)
                #print("[env_base::load] END set_position_orientation")

                if any(
                    detect_closeness(
                        bid, exclude_bodyB=scene.objects_by_category["floors"][0].get_body_ids(), distance=0.01
                    )
                    for bid in robot.get_body_ids()
                ):
                    log.warning("Robot's cached initial pose has collisions.")

        self.scene = scene
        self.robots = scene.robots

        #print("[env_base::load] END")

    def clean(self):
        """
        Clean up the environment.
        """
        if self.simulator is not None:
            self.simulator.disconnect()

    def close(self):
        """
        Synonymous function with clean.
        """
        self.clean()

    def simulator_step(self):
        """
        Step the simulation.
        This is different from environment step that returns the next
        observation, reward, done, info.
        """
        self.simulator.step()

    def step(self, action):
        """
        Overwritten by subclasses.
        """
        return NotImplementedError()

    def reset(self):
        """
        Overwritten by subclasses.
        """
        return NotImplementedError()
