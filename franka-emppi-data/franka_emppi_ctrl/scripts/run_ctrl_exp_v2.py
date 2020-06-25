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
from std_msgs.msg import Bool

# np.random.seed(234)

class MultModelMPPI(object):

    def __init__(self): ### since this is a ros class, I can just get away with writting this here

        ### some parameters
        self.horizon = 20
        self.frame_skip = 5
        self.num_samples_per_model = 10
        self.num_samples = 2
        self.noise = 1.0
        self.lam = 0.8

        self.num_tot_trajectories = self.num_samples * self.num_samples_per_model
        self.model_probs = np.ones(self.num_tot_trajectories)
        self.model_probs /= np.sum(self.model_probs)

        self.__batch_size = 100
        self.__dump_model_probs = np.zeros((self.__batch_size, self.num_tot_trajectories))
        self.__dump_sampled_jnts = np.zeros((self.__batch_size, self.num_tot_trajectories, 3))
        self.__dump_sampled_poses = np.zeros((self.__batch_size, self.num_tot_trajectories, 3))
        self.__dump_mean_jnts = np.zeros((self.__batch_size, 3))
        self.__dump_mean_poses = np.zeros((self.__batch_size, 3))
        self.__t = 0

        script_path = os.path.dirname(os.path.realpath(__file__))
        model_path = script_path + '/../assets/franka-door.xml'
        ff_model_path = script_path + '/../assets/franka-feedforward.xml'
        self.script_path = script_path

        self.model = load_model_from_path(ff_model_path)
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



        self.state = MjSimState(0.0, sim.data.qpos.copy(),
                                    sim.data.qvel.copy(), None, {})
        self.handle_pos = np.zeros(3)
        self.handle_quat = np.zeros(4)



        self.jnt_state_sub = rospy.Subscriber('/robot/joint_states_relay',
                                                JointState, self.get_jnt_state)
        self.handle_pose_sub = rospy.Subscriber('/env/handle',
                                                Pose, self.get_handle_pose)
        self.jnt_cmd_pub= rospy.Publisher('/emppi/joint/command',
                                                RosVector, queue_size=1)
        self.gripper_cmd_pub = rospy.Publisher('/emppi/gripper', Bool, queue_size=1)
        self.phase_two_sub = rospy.Subscriber('/phase_two_ago', Bool, self.phase_two_callback)
        self.sent_close_gripper = False
        rospy.sleep(2) ### see if this calls the subscriber
        self.sim.data.qpos[:7] = self.m_qpos[:7]
        #self.sim.data.qpos[0] = -0.3
        #self.sim.data.qpos[1] = -1.
        #self.sim.data.qpos[3] = -1.5
        #self.sim.data.qpos[4] = 0.6
        #self.sim.data.qpos[5] = 1.4
        self.sim.forward()
        self.viewer = MjViewer(self.sim)

    def phase_two_callback(self, msg):
        if msg.data is True:
            self.task.stage = 'two'

    def __call__(self):

        self.update_model() ### the subs have the details
        ## TODO: make sure that I don't actually need this
        #state = self.sim.get_state()
        #for sim in self.pool.sims: ### reset each of the simulators initial state
        #    sim.set_state(state)

        #self.task.update_stage(self.sim.data) ### check to see if the statge is good
        #print('stage is : ',self.task.stage)

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
        print(self.task.stage)
        if self.task.stage == 'two':
            pass
            #self.mean_actions[0,-1] = -1.
            #if self.sent_close_gripper == False:
            #    close_grip_msg = Bool()
            #    close_grip_msg.data = True 
            #    self.gripper_cmd_pub.publish(close_grip_msg)
            #    self.sent_close_gripper = True
        
        self.eval_ctrl = self.scale_ctrl(self.mean_actions[0])
        
        # TODO: self.update_distribution()
        self.sim.data.ctrl[:] = self.eval_ctrl[:]
        
        for _ in range(self.frame_skip):
            self.sim.step()
        jnt_cmd = RosVector() #JointCommand()
        # jnt_cmd.header = Header()
        # jnt_cmd.ctrls = self.sim.data.qpos[:7]
        jnt_cmd.values = self.sim.data.qpos[:7]
        self.jnt_cmd_pub.publish(jnt_cmd)

        self.viewer.render()
        
   
    def update_distribution(self):
        for sim in self.pool.sims:
            sim.data.qpos[:9] = self.sim.data.qpos[:9].copy()
            sim.data.qvel[:9] = self.sim.data.qvel[:9].copy()
            sim.forward()
            sim.data.ctrl[:] = self.eval_ctrl[:]

        handle_state = np.hstack((self.handle_pos, self.handle_quat)) 
        self.pool.step()
        mean_handle_jnt = 0.0 
        for i, (data, model_prob) in enumerate(zip(self.data, self.model_probs)):
            simulated_measurements = data.sensordata[:] # TODO: get handle pose + quat 
            diff = handle_state - simulated_measurements
            likelihood = np.prod(multivariate_normal.pdf(diff, 0., 0.02))
            self.model_probs[i] = model_prob * likelihood  

        self.model_probs += 1e-10
        self.model_probs /= np.sum(self.model_probs)
        print('updating distribution...')
        print(1/np.sum(np.square(self.model_probs)) , self.num_tot_trajectories/2)
        
        sampled_jnts, sampled_poses , mean_jnt, mean_pose =\
                self.task.get_mean_vals(self.pool.sims,  self.model_probs)
        '''        
        self.__dump_model_probs[self.__t, :] = self.model_probs.copy()
        self.__dump_sampled_jnts[self.__t, :, :] = sampled_jnts.copy()
        self.__dump_sampled_poses[self.__t, :, :] = sampled_poses.copy()

        self.__dump_mean_jnts[self.__t, :] = mean_jnt.copy()
        self.__dump_mean_poses[self.__t, :] = mean_pose.copy()
        self.__t = 0
        
        self.__t += 1
        if self.__t == self.__batch_size:
            np.save(self.script_path + '/model_probs.npy',self.__dump_model_probs)
            np.save(self.script_path + '/sampled_jnts.npy', self.__dump_sampled_jnts)
            np.save(self.script_path + '/sampled_poses.npy', self.__dump_sampled_poses)
            np.save(self.script_path + '/mean_jnts.npy'
        '''
        if 1/np.sum(np.square(self.model_probs)) < self.num_tot_trajectories/2:
            print('resampling!!!!!!!!!!!!!!!!!')
            self.task.resample(self.pool.sims, self.model_probs, self.sim)
            self.model_probs = np.ones(self.num_tot_trajectories)
            self.model_probs /= np.sum(self.model_probs)

    def scale_ctrl(self, ctrl):
        ## first clip
        ctrl = np.clip(ctrl, -1.0, 1.0)
        ## now scale
        ctrl = self.act_mid + ctrl * self.act_rng
        return ctrl

    def update_model(self):
        alpha = 0.999 ##0.995 works well #### alpha blend the measured state and the current sim state
        v_alpha = 0.999
        self.sim.data.qpos[:9] = (1-alpha) * self.m_qpos[:9] + alpha * self.sim.data.qpos[:9]
        self.sim.data.qvel[:9] = (1-v_alpha) * self.m_qvel[:9] + v_alpha * self.sim.data.qvel[:9]
        self.sim.data.qpos[7] = 0.03
        self.sim.data.qpos[8] = -0.03
        self.sim.data.qvel[7:9] = 0.
        #self.sim.data.qvel[9:] = 0. 
        self.sim.forward()
        for sim in self.pool.sims: ## the update to the state of the handle pose has to happen at the model level
            ## this is because of the generalized coordinate system
            sim.data.qpos[:9] = self.sim.data.qpos[:9].copy()
            sim.data.qvel[:9] = self.sim.data.qvel[:9].copy()
            self.sim.data.qpos[7] = 0.03
            self.sim.data.qpos[8] = -0.03
            self.sim.data.qvel[7:9] = 0.
            sim.model.body_pos[self.handle_bid,:] = self.handle_pos[:].copy()
            sim.model.body_quat[self.handle_bid,:] = self.handle_quat[:].copy()
            sim.forward()
        # TODO: uncomment to return back to the original version without feedforward
        #self.sim.model.body_pos[self.handle_bid,:] = self.handle_pos[:].copy()
        #self.sim.model.body_quat[self.handle_bid,:] = self.handle_quat[:].copy()
        #self.sim.forward() ## this forward computes the tree

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

        alpha = 0 
        handle_pos = np.zeros(3)
        handle_pos[0] = data.position.x
        handle_pos[1] = data.position.y
        handle_pos[2] = data.position.z

        self.handle_pos = (1-alpha) * handle_pos + alpha * (self.handle_pos)

        handle_quat = np.zeros(4)
        handle_quat[0] = data.orientation.x
        handle_quat[1] = data.orientation.y
        handle_quat[2] = data.orientation.z
        handle_quat[3] = data.orientation.w
        # self.handle_state = np.hstack((self.handle_pos, self.handle_quat))
        
        self.handle_quat = (1-alpha) * handle_quat + alpha * (self.handle_quat)

    def view_sample_sim(self): ### look at the simulation state
        self.viewer.render()

def main():
    import time
    rospy.init_node('emppi_ctrlr')
    rate = rospy.Rate(10)
    
    print('init controller')
    emppi = MultModelMPPI()
    print('waiting a few seconds for measurements to be collected')
    rospy.sleep(2)
    # TODO: emppi.task.randomize_param(emppi.models, emppi.handle_pos)
    
    
    while not rospy.is_shutdown():
        start = time.time()
        emppi()
        print('time : ', time.time() - start)
        # print(emppi.state)
        rate.sleep()

if __name__ == '__main__':
    main()
