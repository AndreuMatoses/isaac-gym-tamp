# creating a test environment for my tamp furniture problem

import numpy as np
from math import sqrt
from isaacgym import gymapi, gymutil
from isaacgym import gymapi


##### CLASSES AND FUNCTIONS:
# Simplified asset class with templates for the assets i use
class SimplifiedAsset:
    def __init__(self, gym_handle, sim_handle, asset_type:str, name:str, location:tuple, size_xyz = None, z_rotation = 0.0, color_rgb=None ):
        # type "robot", "box", "cabinet"
        self.asset_type = asset_type
        self.size_xyz = size_xyz
        self.location = location
        self.z_rotation = z_rotation
        self.ahandles = []
        self.name = name
        self.gym_handle = gym_handle
        self.sim_handle = sim_handle

        if color_rgb is not None:
            if color_rgb == "random":
                # generate random bright color
                c = 0.5 + 0.5 * np.random.random(3)
                self.color_rgb = gymapi.Vec3(c[0], c[1], c[2])
            else:
                self.color_rgb = gymapi.Vec3(color_rgb[0], color_rgb[1], color_rgb[2])
        else:
            self.color_rgb = None

        self.create_asset_obj(gym_handle, sim_handle)

    def create_asset_obj(self, gym_handle, sim_handle):

        if self.asset_type == "cabinet":
            asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf"
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.use_mesh_materials = True
            self.asset_gym = gym_handle.load_asset(sim_handle, "../thirdparty/isaacgym/assets", asset_file, asset_options)

        elif self.asset_type == "box":
            asset_options = gymapi.AssetOptions()
            asset_options.density = 10.0
            self.asset_gym = gym_handle.create_box(sim_handle, self.size_xyz[0], self.size_xyz[1], self.size_xyz[2], asset_options)
        
        elif self.asset_type == "table":
            asset_file = "urdf/square_table.urdf"
            asset_options = gymapi.AssetOptions()
            self.asset_gym = gym_handle.load_asset(sim_handle, "../assets", asset_file, asset_options)

        elif self.asset_type == "jackal":
            asset_file = "urdf/jackal/jackal.urdf"
            asset_options = gymapi.AssetOptions()
            self.asset_gym = gym_handle.load_asset(sim_handle, "../assets", asset_file, asset_options)

        elif self.asset_type == "boxer":
            asset_file = "urdf/boxer/boxer.urdf"
            asset_options = gymapi.AssetOptions()
            self.asset_gym = gym_handle.load_asset(sim_handle, "../assets", asset_file, asset_options)
        else:
            print("*** Type of asset not supported. Add it in the method: create_asset_obj()")
            quit()

    def add2env(self ,env_handle):

        collisions_in_env = True
        collision_filter = 0
        if collisions_in_env:
            collision_group = len(self.ahandles)
        else:
            collision_group = 0

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(self.location[0], self.location[1], self.location[2])
        pose.r = gymapi.Quat.from_euler_zyx(0, 0, self.z_rotation)

        actor_instance = self.gym_handle.create_actor(env_handle, self.asset_gym, pose, self.name, collision_group, collision_filter)
        self.ahandles.append(actor_instance)

        if self.color_rgb is not None:
            self.gym_handle.set_rigid_body_color(env_handle, actor_instance, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.color_rgb)

#### GYM 

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 4, "help": "Number of environments to create"},
        {"name": "--all_collisions", "action": "store_true", "help": "Simulate all collisions"},
        {"name": "--no_collisions", "action": "store_true", "help": "Ignore all collisions"}])

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU torch pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.restitution = 0
plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
plane_params.distance = 0
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
gym.add_ground(sim, plane_params)

# create viewer
# position the camera
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
cam_pos = gymapi.Vec3(-6, -6, 5)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
if viewer is None:
    print("*** Failed to create viewer")
    quit()    

assets_list = [SimplifiedAsset(gym, sim, "box", name="box1", location=(2.0,2.0,1.0), size_xyz=(1, 2, 0.5), color_rgb="random"),
               SimplifiedAsset(gym, sim, "box", name="box2",location=(2.0,2.0,2.0), size_xyz=(0.5, 0.5, 0.5), color_rgb="random"),
               SimplifiedAsset(gym, sim, "cabinet", name="cabinet",location=(-0.5, -0.5, 0.0), z_rotation=3.1415, color_rgb="random"),
               SimplifiedAsset(gym, sim, "jackal", name="robot",location=(0.0, 2.0, -0.5))]


# set up the env grid
num_envs = args.num_envs
num_per_row = int(sqrt(num_envs))
env_spacing = 4
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache useful handles
envs = []

# Set up envs
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    for simple_asset in assets_list:
        simple_asset.add2env(env)

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
