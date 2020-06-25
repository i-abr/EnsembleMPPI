#!/usr/bin/env python3

import numpy as np
import numpy.random as npr
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import time
# from mult_model_mppi import MultModelMPPI

from mult_model_pi2 import MultModelPI2


frame_skip = 2
model_path = '../kuka-assets/kuka-allegro.xml'
model = load_model_from_path(model_path)
sim = MjSim(model)

viewer = MjViewer(sim)

target_obj_sid = model.site_name2id('target:center')
obj_sid = model.site_name2id('object:center')
palm_bid = model.site_name2id('allegro:palm')

def task(data, action):### this function assumes that the input data is a numpy array

    # obj_config = data.site_xmat[obj_sid].ravel()
    obj_pos = data.site_xpos[obj_sid].ravel()
    desired_pos = data.site_xpos[target_obj_sid].ravel()
    palm_pos = data.site_xpos[palm_bid].ravel()

    dist = np.linalg.norm(palm_pos - obj_pos)
    target_dist = np.linalg.norm(desired_pos - obj_pos)
    vel = data.qvel[:]
    return 100.0 * dist + 0.8 * np.dot(vel, vel)

def terminal_cost(data):
    return 0

def main():
    #### --- initial parameters
    num_models      = 1
    num_trajectories = 40
    horizon         = 80
    noise           = 0.005
    lam             = 0.05

    print('Generating the candidate models ...')
    model_pool = []
    for i in range(num_models):
        _model = load_model_from_path(model_path)
        model_pool.append(_model)

    print('Randomizing parameters')
    # randomize_param(model_pool)

    emppi = MultModelPI2(model_pool, task, terminal_cost,
                frame_skip=frame_skip,
                horizon=horizon, num_trajectories=num_trajectories, noise=noise, lam=lam,
                default_hidden_layers=[128, 64])

    # emppi = MultModelMPPI(model_pool, task, terminal_cost,
    #             frame_skip=frame_skip,
    #             horizon=horizon, num_trajectories=num_trajectories, noise=noise, lam=lam)


    while True:
        state = sim.get_state()
        ctrl = emppi(state, sim.data.sensordata[:])
        print(ctrl)
        sim.data.ctrl[:] = ctrl
        for _ in range(frame_skip):
            sim.step()
        viewer.render()




if __name__ == '__main__':
    main()
