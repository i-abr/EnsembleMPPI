#!/usr/bin/env python3

import numpy as np
from scipy.stats import multivariate_normal 
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimPool
import os
from mult_model_mppi import MultModelMPPIMujoco
from numpy.random import choice
import pickle 
from mujoco_py.generated import const 

frame_skip = 2

sim_model_path = 'assets/DAPG_door.xml'

model_path = 'assets/DAPG_door.xml'

model = load_model_from_path(model_path)
sim = MjSim(model)

viewer = MjViewer(sim)

### data idx that are necessary in this example
door_hinge_did = model.jnt_dofadr[model.joint_name2id('door_hinge')]
grasp_sid = model.site_name2id('S_grasp')
handle_sid = model.site_name2id('S_handle')
door_bid = model.body_name2id('frame')

#### TODO: TESTING CODE REMOVE LATER
sim.model.jnt_axis[door_hinge_did] = np.array([-1.,0., 0])
sim.model.jnt_pos[door_hinge_did][2] = 0.2

np.random.seed(451)

def update_actuator_gains(_sim):
    _sim.model.actuator_gainprm[_sim.model.actuator_name2id('A_WRJ1'):_sim.model.actuator_name2id('A_WRJ0')+1,:] = np.array([10,0,0])
    _sim.model.actuator_gainprm[_sim.model.actuator_name2id('A_FFJ3'):_sim.model.actuator_name2id('A_THJ0')+1,:] = np.array([1,0,0])
    _sim.model.actuator_biasprm[_sim.model.actuator_name2id('A_WRJ1'):_sim.model.actuator_name2id('A_WRJ0')+1,:] = np.array([0,-10,0])
    _sim.model.actuator_biasprm[_sim.model.actuator_name2id('A_FFJ3'):_sim.model.actuator_name2id('A_THJ0')+1,:] = np.array([0,-1,0])


def task(data, action):### this function assumes that the input data is a numpy array
    touch_sensor = data.sensordata[:]  
    palm_pos = data.site_xpos[grasp_sid].ravel()
    handle_pos = data.site_xpos[handle_sid].ravel()
    door_pos = data.qpos[door_hinge_did]
    loss = 0.0
    #if door_pos < 1.57:
    loss += 400.0*np.linalg.norm(palm_pos - handle_pos) + 0.001*np.linalg.norm(data.qvel[:])
    return loss - 100.0 * door_pos

def terminal_cost(data):
    touch_sensor = data.sensordata[:]
    palm_pos = data.site_xpos[grasp_sid].ravel()
    handle_pos = data.site_xpos[handle_sid].ravel()
    door_pos = data.qpos[door_hinge_did]
    loss = 0.0
    # if door_pos < 1.4:
    loss += 10.0 * np.linalg.norm(palm_pos - handle_pos) #- 1000* np.sum(touch_sensor)
    return 0*np.linalg.norm(data.qvel[:])

def update_param(_model):
    axis_choices = [np.array([-1., 0., 0.]), np.array([0., 0., 1.])]
    idx = np.random.choice(len(axis_choices))
    chosen_axis = axis_choices[idx]
    _model.jnt_axis[door_hinge_did] = chosen_axis
    if idx == 0:
        _model.jnt_pos[door_hinge_did][:] = 0. 
        _model.jnt_pos[door_hinge_did][2] = np.random.uniform(0.1, 0.5) #* chosen_axis
    else:
        _model.jnt_pos[door_hinge_did][:] = 0.
        _model.jnt_pos[door_hinge_did][0] = np.random.uniform(0.1, 0.5)

def update_distribution(sims, probs):

    var_joint_pos = 0.0
    for _sim, m_prob in zip(mppi.pool.sims, mppi.model_probs):
        _diff = _sim.model.jnt_pos[door_hinge_did].ravel() - mean_joint_pos 
        var_joint_pos += m_prob * np.outer(_diff, _diff)

    #print('Mean estimated Door hinge pos: {}, axis : {}, var : {}'.format(mean_joint_pos, mean_joint_axis, var_joint_pos))

    for sim in sims:
        sampled_sim = choice(sims, p=probs)
        sim.model.jnt_pos[door_hinge_did][:] = sampled_sim.model.jnt_pos[door_hinge_did][:].copy()
        jnt_ax = sampled_sim.model.jnt_axis[door_hinge_did][:].copy()
        jnt_pos = sampled_sim.model.jnt_pos[door_hinge_did][:].copy()
        if np.argmax(np.abs(jnt_ax)) == 0:
            sim.model.jnt_pos[door_hinge_did][2] = np.random.normal(jnt_pos[2], 0.01) #* chosen_axis
        else:
            sim.model.jnt_pos[door_hinge_did][0] = np.random.normal(jnt_pos[0], 0.01) #* chosen_axis
        sim.model.jnt_axis[door_hinge_did][:] = jnt_ax

        
    



no_trajectories = 40
sim_model_pool = []
for i in range(no_trajectories):
    sim_model = load_model_from_path(sim_model_path)
    update_param(sim_model) ### this updates the distribution of door hinges? that sounds dumb
    sim_model_pool.append(sim_model)


mppi = MultModelMPPIMujoco(sim_model_pool, task, terminal_cost, 
        frame_skip=frame_skip, 
        horizon=30, no_trajectories=no_trajectories , noise=0.2, lam=.8)
print(mppi.num_states, mppi.num_actions)
input()
### update actuator
for m_sim in mppi.pool.sims:
    update_actuator_gains(m_sim)
update_actuator_gains(sim)

### I need a filter
door_hinge_distr = []
hinge_poses = []
hinge_axis = []
hinge_probs = []
counter = 0
while True:
    counter += 1
    state = sim.get_state()
    ctrl, pred_meas = mppi(state, predict_measurements=True)
    sim.data.ctrl[:] = ctrl
    for _ in range(frame_skip):
        sim.step()
    real_meas = sim.data.sensordata[:].copy()
    real_meas += np.random.normal(0., 0.01, size=real_meas.shape)
    ### Use the measurements to update the probability of the models
    logl = np.array([multivariate_normal.logpdf(real_meas-s_meas, 0., 0.01).sum()
                                    for s_meas in pred_meas])
    logl -= np.max(logl)
    mppi.model_probs *= np.exp(logl)
    mppi.model_probs += 1e-5
    mppi.model_probs /= np.sum(mppi.model_probs) 
    norm_prob = np.linalg.norm(mppi.model_probs)    
    mean_joint_pos = 0.0
    mean_joint_axis = 0.0
    _hinge_poses = []
    _hinge_axis = []
    _hinge_probs = [] 
    for _sim, m_prob in zip(mppi.pool.sims, mppi.model_probs):
        mean_joint_pos += _sim.model.jnt_pos[door_hinge_did] * m_prob
        mean_joint_axis += _sim.model.jnt_axis[door_hinge_did] * m_prob 
        _hinge_poses.append(_sim.model.jnt_pos[door_hinge_did].ravel().copy())
        _hinge_axis.append(_sim.model.jnt_axis[door_hinge_did].ravel().copy())
        _hinge_probs.append(m_prob.copy())
        if abs(_hinge_axis[-1])[0] > 0:
            rot = np.array([
                    [0., 0., -1],
                    [0., 1., 0.],
                    [1., 0., 0.]
                ]).flatten()
        else:
            rot = np.eye(3).flatten()
        viewer.add_marker(pos=_sim.data.xanchor[door_hinge_did].flatten() + np.array([0., -0.1, 0.]), 
                size=np.array([0.01,0.01,0.4]), type=const.GEOM_ARROW, label='',
                rgba=np.array([1.,1.,1.,m_prob/norm_prob]),
                mat=rot)

    #hinge_poses.append(_hinge_poses)
    #hinge_axis.append(_hinge_axis)
    #hinge_probs.append(_hinge_probs)

    

    viewer.render()


    
    if 1/np.sum(np.square(mppi.model_probs)) < no_trajectories/2:
        print('RESAMPLING POOL')
        update_distribution(mppi.pool.sims, mppi.model_probs)
        mppi.model_probs = np.ones(mppi.model_probs.shape)
        mppi.model_probs /= np.sum(mppi.model_probs)
    #if counter % 200 == 0:
    #    file_pi = open('door-hinge-data.pickle', 'wb')
    #    pickle.dump({ 'hinge_poses': hinge_poses,
    #                'hinge_axis' : hinge_axis,
    #               'hinge_probs' : hinge_probs 
    #         }, file_pi)

    if sim.data.qpos[door_hinge_did] >= 1.2:#1.2:
        sim.reset()
        mppi.reset()
    if os.getenv('TESTING') is not None:
        break

