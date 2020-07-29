import pybullet as p
import numpy as np
import time

from gibson2.core.physics.robot_locomotors import Fetch
from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj, GripperObj
from gibson2.core.simulator import Simulator
from gibson2 import assets_path, dataset_path
from gibson2.utils.utils import parse_config
from math import sqrt

model_path = assets_path + '\\models\\'
gripper_folder = model_path + '\\gripper\\'
configs_folder = '..\\configs\\'
fetch_config = parse_config(configs_folder + 'fetch_p2p_nav.yaml')

optimize = True
# Toggle this to only use renderer without VR, for testing purposes
vrMode = True
# Possible types: hmd_relative, torso_relative
movement_type = 'torso_relative'

# Timestep should always be set to 1/90 to match VR system's 90fps
s = Simulator(mode='vr', timestep = 1/90.0, msaa=True, vrFullscreen=False, vrEyeTracking=False, optimize_render=optimize, vrMode=vrMode)
scene = BuildingScene('Placida', is_interactive=False)
scene.sleep = optimize
s.import_scene(scene)

# Grippers represent hands
lGripper = GripperObj()
s.import_articulated_object(lGripper)
lGripper.set_position([0.0, 0.0, 1.5])

rGripper = GripperObj()
s.import_articulated_object(rGripper)
rGripper.set_position([0.0, 0.0, 1.0])

# Load objects in the environment
for i in range(5):
    bottle = YCBObject('006_mustard_bottle')
    s.import_object(bottle)
    _, org_orn = p.getBasePositionAndOrientation(bottle.body_id)
    bottle_pos = [1 ,0 - 0.2 * i, 1]
    p.resetBasePositionAndOrientation(bottle.body_id, bottle_pos, org_orn)

# Controls how closed each gripper is (maximally open to start)
leftGripperFraction = 0.0
rightGripperFraction = 0.0

if optimize:
    s.optimize_data()

# Account for Gibson floors not being at z=0 - shift user height down by 0.2m
s.setVROffset([0, 0, -0.2])

def normalizeListVec(v):
    length = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
    if length <= 0:
        length = 1
    v = [val/sqrt(length) for val in v]
    return v

vr_movement_speed = 0.01

# Runs simulation
while True:
    s.step(shouldTime=False)

    if vrMode:
        hmdIsValid, hmdTrans, hmdRot = s.getDataForVRDevice('hmd')
        lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')
        rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')
        # TODO: Make nice interface functions for this function
        lTrig, lTouchX, lTouchY = s.getButtonDataForController('left_controller')
        rTrig, rTouchX, rTouchY = s.getButtonDataForController('right_controller')

        if lIsValid:
            lGripper.move_gripper(lTrans, lRot)
            lGripper.set_close_fraction(lTrig)

        if rIsValid:
            rGripper.move_gripper(rTrans, rRot)
            rGripper.set_close_fraction(rTrig)

        current_offset = s.getVROffset()

        relative_device = 'hmd'
        if movement_type == 'torso_relative':
            relative_device = 'right_controller'
        right, up, forward = s.getDeviceCoordinateSystem(relative_device)

        # Move the VR player in the direction of the analog stick
        # In this implementation, +ve x corresponds to right and +ve y corresponds to forward
        # relative to the HMD
        # Only uses data from right controller
        if rIsValid:
            vr_offset_vec = [right[i] * rTouchX + forward[i] * rTouchY for i in range(3)]
            vr_offset_vec[2] = 0
            vr_offset_vec = normalizeListVec(vr_offset_vec)
            final_offset = [current_offset[i] + vr_offset_vec[i] * vr_movement_speed for i in range(3)]
            s.setVROffset(final_offset)

s.disconnect()