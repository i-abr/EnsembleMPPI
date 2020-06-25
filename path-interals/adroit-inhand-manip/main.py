#!/usr/bin/env python3

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
# from mult_model_mppi import MultModelMPPI

from mult_model_pi2 import MultModelPI2

frame_skip = 4
model_path = '../adroit-hand-assets/DAPG_pen.xml'
model = load_model_from_path(model_path)
sim = MjSim(model)

viewer = MjViewer(sim)

### data idx that are necessary in this example
target_obj_bid = model.body_name2id('target')
grasp_sid = model.site_name2id('S_grasp')
obj_bid = model.body_name2id('Object')
eps_ball_sid = model.site_name2id('eps_ball')
obj_t_sid = model.site_name2id('object_top')
obj_b_sid = model.site_name2id('object_bottom')
tar_t_sid = model.site_name2id('target_top')
tar_b_sid = model.site_name2id('target_bottom')



def update_actuator_gains(_sim):
    _sim.model.actuator_gainprm[_sim.model.actuator_name2id('A_WRJ1'):_sim.model.actuator_name2id('A_WRJ0')+1,:] = np.array([10,0,0])
    _sim.model.actuator_gainprm[_sim.model.actuator_name2id('A_FFJ3'):_sim.model.actuator_name2id('A_THJ0')+1,:] = np.array([4,0,0])
    _sim.model.actuator_biasprm[_sim.model.actuator_name2id('A_WRJ1'):_sim.model.actuator_name2id('A_WRJ0')+1,:] = np.array([0,-10,0])
    _sim.model.actuator_biasprm[_sim.model.actuator_name2id('A_FFJ3'):_sim.model.actuator_name2id('A_THJ0')+1,:] = np.array([0,-4,0])

def randomize_param(models):
    mean_mass = models[0].body_mass[obj_bid]
    mean_intertia = models[0].body_inertia[obj_bid,:]
    scale = 0.01
    for model in models:
        model.body_mass[obj_bid] = np.random.uniform(mean_mass-mean_mass*scale, mean_mass+mean_mass*scale)
        model.body_inertia[obj_bid, :] = np.random.uniform(mean_intertia-mean_intertia*scale, mean_intertia+mean_intertia*scale)

def update_distribution(sims, probs):
    mean_mass = 0.0
    mean_intertia = 0.0

    for sim, prob in zip(sims, probs):
        mean_mass += sim.model.body_mass[obj_bid] * prob
        mean_intertia += sim.model.body_inertia[obj_bid, :] * prob

    for sim in sims:
        sim.model.body_mass[obj_bid] = np.random.normal(mean_mass, 0.01)
        sim.model.body_inertia[obj_bid, :] = np.random.normal(mean_intertia, 0.01)

def get_stats_from_prob(sims, probs):
    mean_inertia = 0.0
    mean_mass = 0.0

    for sim, prob in zip(sims, probs):
        mean_mass += sim.model.body_mass[obj_bid] * prob
        mean_inertia += sim.model.body_inertia[obj_bid] * prob

    stats = {
        'mean mass' : mean_mass,
        'mean inertia' : mean_inertia
    }

    # # var = 0.0
    # #
    # # for sim, prob in zip(sims, probs):
    # #     param = np.hstack((
    # #         sim.model.dof_damping[3:],
    # #         sim.model.jnt_stiffness[3:],
    # #         sim.model.body_mass[1:]
    # #     ))
    # #     delta_param = param - mean_param
    # #     var += np.outer(delta_param, delta_param) * prob
    #
    # stats = {
    #     'mean dof damping' : mean_dof_damping,
    #     'mean jnt stiffness' : mean_jnt_stiff,
    #     'mean mass' : mean_mass,
    #     'var' : var
    #     # 'dof damping var' : dof_damping_var,
    #     # 'jnt stiffness var' : jnt_stiff_var,
    #     # 'mass var' : mass_var
    # }
    return stats


def task(data, action):### this function assumes that the input data is a numpy array

    palm_pos = data.site_xpos[grasp_sid].ravel()
    pen_length = np.linalg.norm(data.site_xpos[obj_t_sid].ravel() - data.site_xpos[obj_b_sid].ravel())
    tar_length = np.linalg.norm(data.site_xpos[tar_t_sid].ravel() - data.site_xpos[tar_b_sid].ravel())

    if pen_length < 1e-8 or tar_length < 1e-8:
        pen_length = 1.0
        tar_length = 1.0

    obj_pos = data.body_xpos[obj_bid].ravel()
    desired_loc = data.site_xpos[eps_ball_sid].ravel()
    obj_orien = (data.site_xpos[obj_t_sid].ravel() - data.site_xpos[obj_b_sid].ravel())/pen_length
    desired_orien = (data.site_xpos[tar_t_sid].ravel() - data.site_xpos[tar_b_sid].ravel())/tar_length
    ## pose cost
    dist = np.linalg.norm(obj_pos - desired_loc)
    ## orientation cost
    orien = np.dot(obj_orien, desired_orien)

    loss = 100.0*dist - 800.0*orien + 10000.0 * np.linalg.norm(palm_pos - obj_pos)
    if obj_pos[2] < 0.15:
        loss += 10.0
    return loss + np.linalg.norm(data.qvel[:])

def terminal_cost(data):
    return 0.0



def main():

    #### --- initial parameters
    num_models      = 1#10
    num_trajectories = 20#4
    horizon         = 40
    noise           = 0.01
    lam             = 0.06

    print('Generating the candidate models ...')
    model_pool = []
    for i in range(num_models):
        _model = load_model_from_path(model_path)
        model_pool.append(_model)

    print('Randomizing parameters')
    # randomize_param(model_pool)

    # emppi = MultModelMPPI(model_pool, task, terminal_cost,
    #             frame_skip=frame_skip,
    #             horizon=horizon, num_trajectories=num_trajectories, noise=noise, lam=lam)

    emppi = MultModelPI2(model_pool, task, terminal_cost,
                frame_skip=frame_skip,
                horizon=horizon, num_trajectories=num_trajectories, noise=noise, lam=lam,
                default_hidden_layers=[32])

    ### update actuator
    for m_sim in emppi.pool.sims:
        update_actuator_gains(m_sim)
    update_actuator_gains(sim)


    while True:

        state = sim.get_state()
        ctrl = emppi(state, sim.data.sensordata[:])
        sim.data.ctrl[:] = ctrl
        for _ in range(frame_skip):
            sim.step()

        viewer.render()
        sensor_measurements = sim.data.sensordata[:]
        emppi.update_distribution(sensor_measurements, state, ctrl)

        if 1/np.sum(np.square(emppi.model_probs)) < emppi.num_tot_trajectories/2:
            # print('resampling!!!!!', 1/np.sum(np.square(emppi.model_probs)), emppi.num_tot_trajectories)
            update_distribution(emppi.pool.sims, emppi.model_probs)
            emppi.model_probs = np.ones(emppi.num_tot_trajectories)
            emppi.model_probs /= np.sum(emppi.model_probs)

        # print(emppi.model_probs)
        # stats = get_stats_from_prob(emppi.pool.sims, emppi.model_probs)
        # print('----- predicted -------')
        # print(stats['mean mass'])
        # print(stats['mean inertia'])
        #
        # print('----- measured ------')
        # print(sim.model.body_mass[obj_bid])
        # print(sim.model.body_inertia[obj_bid,:])

        if os.getenv('TESTING') is not None:
            break

if __name__ == '__main__':
    main()
