#!/usr/bin/env python3

import os
import numpy as np
from numpy import exp
import rospy
from mujoco_py import load_model_from_path, MjSim, MjSimPool, MjViewer, MjSimState
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
# from franka_mujoco_sim.msg import JointCommand
from lula_ros_util_internal.msg import RosVector
from scipy.stats import multivariate_normal
from scipy.signal import savgol_filter
from task import Task

np.random.seed(10)

class MultModelMPPI(object):

    def __init__(self): ### since this is a ros class, I can just get away with writting this here

        ### some parameters
        self.horizon = 20
        self.frame_skip = 5
        self.num_samples_per_model = 1
        self.num_samples = 20
        self.noise = 1.0
        self.lam = 0.6

        self.num_tot_trajectories = self.num_samples * self.num_samples_per_model
        self.model_probs = np.ones(self.num_tot_trajectories)
        self.model_probs /= np.sum(self.model_probs)

        script_path = os.path.dirname(os.path.realpath(__file__))
        model_path = script_path + '/../assets/franka-door.xml'
        self.model = load_model_from_path(model_path)
        self.sim = MjSim(self.model)
        self.models = []
        sim_pool = []

        for i in range(self.num_tot_trajectories): ### generate the sims
            self.models.append(load_model_from_path(model_path))
            sim_pool.append(MjSim(self.models[-1]))

        self.pool = MjSimPool(sim_pool, nsubsteps=self.frame_skip)
        self.data = [sim.data for sim in self.pool.sims] ### reference to the read-only data wrapper

        model = self.models[0]
        sim = sim_pool[0]
        self.num_states = model.nq + model.nv
        self.num_actions = model.nu
        self.nq = model.nq
        self.nv = model.nv

        self.task = Task(model)

        ### create some storage
        self.sk = np.zeros((self.num_tot_trajectories, self.horizon)) ### entropy of the trajectory
        self.delta = np.zeros((self.num_tot_trajectories, self.horizon, self.num_actions))
        ### mean actions to be shifted as each action is computed and returned
        self.mean_actions = np.random.normal(0., self.noise, size=(self.horizon, self.num_actions))
        self.act_mid = np.mean(model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (model.actuator_ctrlrange[:,1]-model.actuator_ctrlrange[:,0])

        self.handle_bid = model.body_name2id('handle')


        self.sim.data.qpos[0] = -0.3
        self.sim.data.qpos[1] = -1.
        self.sim.data.qpos[3] = -1.7
        self.sim.data.qpos[5] = 1.4
        self.sim.forward()

        self.state = MjSimState(0.0, sim.data.qpos.copy(),
                                    sim.data.qvel.copy(), None, {})
        self.handle_pos = np.zeros(3)
        self.handle_quat = np.zeros(4)



        self.jnt_state_sub = rospy.Subscriber('/robot/joint_states',
                                                JointState, self.get_jnt_state)
        self.handle_pose_sub = rospy.Subscriber('/env/handle',
                                                Pose, self.get_handle_pose)
        self.jnt_cmd_pub= rospy.Publisher('/emppi/joint/command',
                                                RosVector, queue_size=1)
        self.viewer = MjViewer(self.sim)

    def __call__(self):

        self.update_model() ### the subs have the details
        state = self.sim.get_state()
        for sim in self.pool.sims: ### reset each of the simulators initial state
            sim.set_state(state)

        self.task.update_stage(self.sim.data) ### check to see if the statge is good
        print('stage is : ',self.task.stage)

        ### pre-shift the mean actions -- this does nothing in the beginning
        self.mean_actions[:-1] = self.mean_actions[1:]
        self.mean_actions[-1] = np.random.normal(0., self.noise, size=(self.num_actions,))

        self.delta = np.random.normal(0., self.noise, size=self.delta.shape) ### create the random action perturbation

        for t in range(self.horizon): ### for each time step
            for k, data in enumerate(self.data): ### do each parameter fixing
                _ctrl = self.mean_actions[t] + self.delta[k, t, :]
                data.ctrl[:] = self.scale_ctrl(_ctrl.copy())
                self.sk[k,t] = self.task(data) \
                                + self.lam * np.dot(self.mean_actions[t], self.delta[k,t,:])/self.noise
            self.pool.step() ### step the simulator
        for k, data in enumerate(self.data):
            self.sk[k,-1] += self.task(data, terminal_cost=True)

        self.sk = np.cumsum(self.sk[:, ::-1], axis=1)[:,::-1] ### in theory this should do what I think

        for t in range(self.horizon):### loop through each action and do the following
            self.sk[:,t] -= np.min(self.sk[:,t]) ### sutract that out so you shift the weight
            _w = self.model_probs * exp(-self.sk[:,t]/self.lam) + 1e-5 ### this should be of size no_trajectories
            _w /= np.sum(_w) ### normalize the weights
            self.mean_actions[t] += np.dot(_w, self.delta[:,t,:])

        self.mean_actions = savgol_filter(self.mean_actions, len(self.mean_actions)-1, 3, axis=0)

        if self.task.stage == 'two': ### this will force the gripper to close
            self.mean_actions[0,-1] = 1.

        self.sim.data.ctrl[:] = self.scale_ctrl(self.mean_actions[0])
        for _ in range(self.frame_skip):
            self.sim.step()
        self.viewer.render()
        jnt_cmd = RosVector() #JointCommand()
        # jnt_cmd.header = Header()
        # jnt_cmd.ctrls = self.sim.data.qpos[:7]
        jnt_cmd.values = self.sim.data.qpos[:7]
        self.jnt_cmd_pub.publish(jnt_cmd)

    def scale_ctrl(self, ctrl):
        ## first clip
        ctrl = np.clip(ctrl, -1.0, 1.0)
        ## now scale
        ctrl = self.act_mid + ctrl * self.act_rng
        return ctrl

    def update_model(self):
        alpha = 0.9995 #### alpha blend the measured state and the current sim state
        self.sim.data.qpos[:9] = (1-alpha) * self.m_qpos[:9] + alpha * self.sim.data.qpos[:9]
        self.sim.data.qvel[:9] = (1-alpha) * self.m_qvel[:9] + alpha * self.sim.data.qvel[:9]

        for sim in self.pool.sims: ## the update to the state of the handle pose has to happen at the model level
            ## this is because of the generalized coordinate system
            sim.model.body_pos[self.handle_bid,:] = self.handle_pos[:].copy()
            sim.model.body_quat[self.handle_bid,:] = self.handle_quat[:].copy()
            sim.forward()
        self.sim.model.body_pos[self.handle_bid,:] = self.handle_pos[:].copy()
        self.sim.model.body_quat[self.handle_bid,:] = self.handle_quat[:].copy()
        self.sim.forward() ## this forward computes the tree

    def get_jnt_state(self, data): ### subscribes to the robot joint states and updates them
        qpos = np.zeros(self.nq)
        qvel = np.zeros(self.nv)
        qpos[:7] = data.position[:7]
        qvel[:7] = data.position[:7]
        qpos[6] -= 0.78539816339
        qpos[7] = data.position[7]
        qpos[8] = -data.position[8] ## gripper weird
        # alpha = 1.0
        # qpos[:9] = (1-alpha) * qpos[:9] + alpha * self.sim.data.qpos[:9]
        # qvel[:9] = (1-alpha) * qvel[:9] + alpha * self.sim.data.qpos[:9]
        # self.state = MjSimState(0.0, qpos, qvel, None, {})
        self.m_qpos = qpos
        self.m_qvel = qvel

    def get_handle_pose(self, data): ### TODO: make sure that this is subscribed to the dart state
        ### TODO: check to see if the quaternions mess with the hinge location + orientation in the simulator
        self.handle_pos[0] = data.position.x
        self.handle_pos[1] = data.position.y
        self.handle_pos[2] = data.position.z

        self.handle_quat[0] = data.orientation.x
        self.handle_quat[1] = data.orientation.y
        self.handle_quat[2] = data.orientation.z
        self.handle_quat[3] = data.orientation.w

    def view_sample_sim(self): ### look at the simulation state
        self.viewer.render()

def main():
    import time
    rospy.init_node('emppi_ctrlr')
    rate = rospy.Rate(10)

    emppi = MultModelMPPI()
    while not rospy.is_shutdown():
        start = time.time()
        emppi()
        print('time : ', time.time() - start)
        # print(emppi.state)
        rate.sleep()

if __name__ == '__main__':
    main()
