#!/usr/bin/env python3

import numpy as np
import numpy.random as npr
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from quatmath import quat2euler, euler2quat
from mult_model_mppi import MultModelMPPI
from mujoco_py.generated import const
from task import Task
import time

def main():
    frame_skip = 5#2
    script_path = os.path.dirname(os.path.realpath(__file__))
    env_model_path = script_path + '/../assets/franka-door.xml'
    env_model = load_model_from_path(env_model_path)
    env = MjSim(env_model)
    viewer = MjViewer(env)

    num_models      = 10
    num_trajectories = 2#4
    horizon         = 40
    simulation_time   = 200
    noise           = 1.0#0.1
    lam             = 0.6


    model_path = script_path + '/../assets/franka-door.xml'

    model_pool = []
    for i in range(num_models):
        _model = load_model_from_path(model_path)
        model_pool.append(_model)


    task = Task(model_pool[0])

    print('Randomizing parameters')
    task.randomize_param(model_pool)

    emppi = MultModelMPPI(model_pool, task,
                frame_skip=frame_skip,
                horizon=horizon, num_trajectories=num_trajectories,
                noise=noise, lam=lam)

    while  True:
        task.stage = 'one'
        env.reset()
        desired_pos = np.zeros(3)
        desired_pos[0] = np.random.uniform(0.5, 0.7)
        desired_pos[1] = np.random.uniform(-0.3, 0.3)
        desired_pos[2] = np.random.uniform(0.3, 0.7)
        env.model.body_pos[task.handle_bid,:] = desired_pos
        env.data.qpos[0] = -0.3
        env.data.qpos[1] = -1.
        env.data.qpos[3] = -1.7
        env.data.qpos[5] = 1.4
        env.forward()
        for _sim in emppi.pool.sims:
            _sim.reset()
            _sim.model.body_pos[task.handle_bid,:] = desired_pos
            _sim.forward()

        state = emppi.pool.sims[0].get_state()
        for t in range(simulation_time):

            state = env.get_state() ## TODO: make this a function that is more similar to real world where I can only reset a few things
            start_time = time.time()
            ctrl = emppi(state)
            print('target - vel : ', ctrl)
            print(time.time() - start_time)
            env.data.ctrl[:] = ctrl

            for _ in range(frame_skip):
                env.step()
            task.update_stage(env.data)
            viewer.render()

            sensor_measurements = env.data.sensordata[:]
            emppi.update_distribution(sensor_measurements, state, ctrl)


if __name__ == '__main__':
    main()