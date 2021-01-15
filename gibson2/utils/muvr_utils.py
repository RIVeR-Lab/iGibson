""" Utility classes and functions needed for the multi-user VR experience. """


from collections import defaultdict
import copy
import numpy as np
import time
from time import sleep

from gibson2.render.mesh_renderer.mesh_renderer_cpu import Instance, InstanceGroup
from gibson2.utils.vr_utils import calc_offset, VrData

from PodSixNet.Channel import Channel
# TODO: Can use connection as an endpoint - perhaps endpoint is already created so I can't use it?
from PodSixNet.Connection import connection, ConnectionListener
# TODO: Experiment with endpoints once everything else is working
from PodSixNet.EndPoint import EndPoint
from PodSixNet.Server import Server


# An FPS cap is needed to ensure that the client and server don't fall too far out of sync
# 90 is the FPS cap of VR, so is the fastest speed we realistically need for any MUVR-related work
MUVR_FPS_CAP = 90.0

# TODO: Add some kind of accrediation to PodSixNet here, since we were inspired by them, but ended up modifying their code

class QueueChannel(Channel):
    """ A Channel subclass that stores all incoming data in a queue, rather than immediately
    triggering asynchronous callbacks. This stops the MUVR networking from interrupting the main
    simulation loop, which causes severe slowdown.
    """


class IGVRClient(ConnectionListener):
    """ TODO: Add comments everywhere! """
    def __init__(self, host, port):
        self.Connect((host, port))
        self.frame_data = {}
        self.frame_start = 0

    def register_data(self, sim, client_agent):
        self.s = sim
        self.renderer = sim.renderer
        self.client_agent = client_agent
        self.vr_device = '{}_controller'.format(self.s.vr_settings.movement_controller)
        self.devices = ['left_controller', 'right_controller', 'hmd']

    def ingest_frame_data(self):
        self.frame_start = time.time()
        if not self.frame_data:
            return

        # Deep copy frame data so it doesn't get overwritten by a random async callback
        self.latest_frame_data = copy.deepcopy(self.frame_data)
        for instance in self.renderer.get_instances():
            data = frame_data[instance.pybullet_uuid]
            if isinstance(instance, Instance):
                trans = np.array(data[0])
                rot = np.array(data[1])
                instance.pose_trans = trans
                instance.pose_rot = rot
            elif isinstance(instance, InstanceGroup):
                poses_trans = []
                poses_rot = []
                data_trans = data[0]
                data_rot = data[1]
                num_links = len(data_trans)
                for i in range(num_links):
                    next_trans = np.array(data_trans[i])
                    next_rot = np.array(data_rot[i])
                    poses_trans.append(np.ascontiguousarray(next_trans))
                    poses_rot.append(np.ascontiguousarray(next_rot))

                instance.poses_trans = poses_trans
                instance.poses_rot = poses_rot

    def client_step(self):
        self.s.viewer.update()
        """
        TODO: Work on these VR features!
        if self.s.can_access_vr_context:
            self.s.poll_vr_events()
            # Sets the VR starting position if one has been specified by the user
            self.s.perform_vr_start_pos_move()

            # Update VR offset so updated value can be used in server
            self.client_agent.update_frame_offset()
        """

    def gen_vr_data(self):
        if not self.s.can_access_vr_context:
            self.vr_data = []
        else:
            print("Gen VR data coming soon!")

    def send_vr_data(self):
        if self.vr_data:
            self.Send({"action": "vr_data", "vr_data": self.vr_data})

    def Network_frame_data(self, data):
        # Store frame data until it is needed during rendering
        # This avoids the overhead of updating the renderer every single time this function is called
        self.frame_data = data["frame_data"]

    def Refresh(self):
        # Receive data from connection's queue
        self.Pump()
        # Push data out to the network
        connection.Pump()
        # Keep client at FPS cap if it is running too fast
        frame_dur = time.time() - self.frame_start
        time_until_min_dur = (1 / MUVR_FPS_CAP) - frame_dur
        if time_until_min_dur > 0:
            sleep(time_until_min_dur)


class IGVRChannel(Channel):
    """ TODO: Add comments everywhere! """
    def __init__(self, *args, **kwargs):
        Channel.__init__(self, *args, **kwargs)
        self.vr_data = {}
    
    def Close(self):
        print(self, "Client disconnected")

    def Network_vr_data(self, data):
        # Store vr data until it is needed for physics simulation
        # This avoids the overhead of updating the physics simulation every time this function is called
        self.vr_data = data["vr_data"]
        print("Received vr data: {}".format(data["vr_data"]))

    def send_frame_data(self, frame_data):
        self.Send({"action": "frame_data", "frame_data": frame_data})



class IGVRServer(Server):
    """ TODO: Add comments everywhere! """
    channelClass = IGVRChannel
    
    def __init__(self, *args, **kwargs):
        Server.__init__(self, *args, **kwargs)
        self.client = None
        self.latest_vr_data = None
        self.frame_start = 0

    def Connected(self, channel, addr):
        #print("Someone connected to the server!")
        self.client = channel

    def register_data(self, sim, client_agent):
        self.s = sim
        self.renderer = sim.renderer
        self.client_agent = client_agent

    def client_connected(self):
        return self.client is not None

    def ingest_vr_data(self):
        self.frame_start = time.time()
        if not self.client:
            return

        # Make a copy of channel's most recent VR data, so it doesn't get mutated if new requests arrive
        self.latest_vr_data = copy.deepcopy(self.client.vr_data)

    def gen_frame_data(self):
        # TODO: Implement this properly!
        self.frame_data = [6,7,8,9,10]

    def send_frame_data(self):
        if self.client:
            self.client.send_frame_data(self.frame_data)
    
    def Refresh(self):
        self.Pump()

        # Keep server at FPS cap if it is running too fast
        frame_dur = time.time() - self.frame_start
        time_until_min_dur = (1 / MUVR_FPS_CAP) - frame_dur
        if time_until_min_dur > 0:
            sleep(time_until_min_dur)
