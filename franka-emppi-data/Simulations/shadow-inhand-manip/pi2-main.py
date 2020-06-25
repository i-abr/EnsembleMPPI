#!/usr/bin/env python3

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from quatmath import quat2euler, euler2quat
# from mult_model_mppi import MultModelMPPI
from mult_model_pi2 import MultModelPI2


frame_skip = 10
model_path = '../shadow-hand-assets/hand/manipulate_block.xml'
model = load_model_from_path(model_path)
sim = MjSim(model)

viewer = MjViewer(sim)

## data idx that are necessary in this example
target_obj_bid = model.body_name2id('target')
target_obj_sid = model.site_name2id('target:center')
obj_sid = model.site_name2id('object:center')
obj_bid = model.body_name2id('object')
palm_sid = model.site_name2id('robot0:Palm')
# eps_ball_sid = model.site_name2id('eps_ball')


def randomize_param(models):
    mean_mass = models[0].body_mass[obj_bid]
    mean_intertia = models[0].body_inertia[obj_bid,:]
    scale = 0.4
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
        sim.model.body_inertia[obj_bid, :] = np.random.normal(mean_intertia, 1e-6)


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

    return stats

def task(data, action):### this function assumes that the input data is a numpy array

    obj_config = data.site_xmat[obj_sid].ravel()
    obj_pos = data.site_xpos[obj_sid].ravel()
    desired_config = data.site_xmat[target_obj_sid].ravel()
    palm_pos = data.site_xpos[palm_sid].ravel()

    orien = np.linalg.norm(obj_config - desired_config)
    dist = np.linalg.norm(palm_pos - obj_pos)

    vel = data.qvel[:]
    return 200.0*dist + 10.0*orien + 0.1 * np.linalg.norm(vel)


def terminal_cost(data):
    obj_config = data.site_xmat[obj_sid].ravel()
    obj_pos = data.site_xpos[obj_sid].ravel()
    desired_config = data.site_xmat[target_obj_sid].ravel()
    palm_pos = data.site_xpos[palm_sid].ravel()

    orien = np.linalg.norm(obj_config - desired_config)
    dist = np.linalg.norm(palm_pos - obj_pos)

    vel = data.qvel[:]
    return 100.0*dist + 0.0 * orien #+ 100.0 * np.dot(vel , vel)

def termination_condition(data):

    obj_config = data.site_xmat[obj_sid].ravel()
    obj_pos = data.site_xpos[obj_sid].ravel()
    desired_config = data.site_xmat[target_obj_sid].ravel()
    palm_pos = data.site_xpos[palm_sid].ravel()

    orien = np.linalg.norm(obj_config - desired_config)
    dist = np.linalg.norm(palm_pos - obj_pos)

    if orien < 0.1:
        return True
    else:
        return False

def get_obs(data):
    qp = data.qpos.ravel()
    qvel = data.qvel.ravel()
    obj_pos = data.site_xpos[obj_sid].ravel()
    obj_orien = data.site_xmat[obj_sid].ravel()

    obj_vel = data.qvel[-6:].ravel()

    desired_orien = data.site_xmat[target_obj_sid].ravel()
    palm_pos = data.site_xpos[palm_sid].ravel()


    return np.concatenate([qp[:-6], qvel[:-6], obj_pos, obj_vel, obj_orien,desired_orien,
                            obj_pos - palm_pos, obj_orien - desired_orien ])


def main():

    #### --- initial parameters
    num_models      = 10
    num_trajectories = 2
    horizon         = 20
    noise           = 0.01
    lam             = 0.1
    final_time      = 200
    print('Generating the candidate models ...')
    model_pool = []
    for i in range(num_models):
        _model = load_model_from_path(model_path)
        model_pool.append(_model)

    print('Randomizing parameters')
    randomize_param(model_pool)

    emppi = MultModelPI2(model_pool, task, terminal_cost, get_obs,
                frame_skip=frame_skip,
                horizon=horizon, final_time=final_time, num_trajectories=num_trajectories, noise=noise, lam=lam,
                default_hidden_layers=[128])
                # default_hidden_layers=[128,64])

    while True:
        print('resettting')
        # sim.reset()
        desired_orien = np.zeros(3)
        desired_orien[0] = np.random.uniform(low=-2.0, high=2.0)
        desired_orien[1] = np.random.uniform(low=-2.0, high=2.0)
        desired_orien[2] = np.random.uniform(low=-2.0, high=2.0)
        sim.model.body_quat[target_obj_bid] = euler2quat(desired_orien)
        sim.forward()
        for _sim in emppi.pool.sims:
            _sim.reset()
            _sim.model.body_quat[target_obj_bid] = euler2quat(desired_orien)
            _sim.forward()
        for t in range(final_time):
            state = sim.get_state()


            ctrl = emppi(state, get_obs(sim.data), t)
            sim.data.ctrl[:] = ctrl
            for _ in range(frame_skip):
                sim.step()

            if termination_condition(sim.data):
                break
            viewer.render()

            sensor_measurements = sim.data.sensordata[:]
            emppi.update_distribution(sensor_measurements, state, ctrl)

            print(1/np.sum(np.square(emppi.model_probs)) , emppi.num_tot_trajectories/2)
            if 1/np.sum(np.square(emppi.model_probs)) < emppi.num_tot_trajectories/2:
                print('resampling!!!!!', 1/np.sum(np.square(emppi.model_probs)), emppi.num_tot_trajectories)
                update_distribution(emppi.pool.sims, emppi.model_probs)
                emppi.model_probs = np.ones(emppi.num_tot_trajectories)
                emppi.model_probs /= np.sum(emppi.model_probs)

        # emppi.reset()

if __name__ == '__main__':
    main()
