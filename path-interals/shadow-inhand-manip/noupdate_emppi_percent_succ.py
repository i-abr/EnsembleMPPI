#!/usr/bin/env python3

import matplotlib.pyplot as plt

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from quatmath import quat2euler, euler2quat, mat2quat
from mult_model_mppi import MultModelMPPI
# from mult_model_pi2 import MultModelPI2
import pickle

from gym.envs.robotics import rotations


frame_skip = 5#10
model_path = '../shadow-hand-assets/hand/manipulate_block_gain.xml'
env_model = load_model_from_path(model_path)
env = MjSim(env_model)

# viewer = MjViewer(env)
# viewer._render_every_frame = True
## data idx that are necessary in this example
target_obj_bid = env_model.body_name2id('target')
target_obj_sid = env_model.site_name2id('target:center')
obj_sid = env_model.site_name2id('object:center')
obj_bid = env_model.body_name2id('object')
palm_sid = env_model.site_name2id('robot0:Palm')
# eps_ball_sid = model.site_name2id('eps_ball')

# env.model.body_mass[obj_bid] = 0.5

def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat

def randomize_param(models):
    mean_mass = models[0].body_mass[obj_bid]
    for model in models:
        model.actuator_gainprm[2:, 0] = np.random.uniform(1.0, 5.0, size=(model.nu-2))
        model.body_mass[obj_bid] = np.clip(np.random.normal(mean_mass, 0.1), 0.001, np.inf)

def update_distribution(sims, probs):
    mean_gains = 0.0
    mean_mass = 0.0

    for sim, prob in zip(sims, probs):
        mean_mass += sim.model.body_mass[obj_bid] * prob
        mean_gains += sim.model.actuator_gainprm[2:,0] * prob

    for sim in sims:
        sim.model.body_mass[obj_bid] = np.clip(np.random.normal(mean_mass, 0.01), 0.01, np.inf)
        sim.model.actuator_gainprm[2:,0] = np.random.normal(mean_gains, 0.15)


def get_stats_from_prob(sims, probs):
    mean_gains = 0.0
    mean_mass = 0.0

    for sim, prob in zip(sims, probs):
        mean_mass += sim.model.body_mass[obj_bid] * prob
        mean_gains += sim.model.actuator_gainprm[2:,0] * prob

    var_gains = 0.0
    var_mass = 0.0

    ### get the variance
    for sim, prob in zip(sims, probs):
        delta_gain = mean_gains - sim.model.actuator_gainprm[2:,0]
        delta_mass = mean_mass - sim.model.body_mass[obj_bid]
        var_mass += np.outer(delta_mass, delta_mass) * prob
        var_gains += np.outer(delta_gain, delta_gain) * prob

    stats = {
        'mean gain' : mean_gains,
        'var gain' : var_gains,
        'mean mass' : mean_mass,
        'var mass' : var_mass,
    }

    return stats


def task(data, action):### this function assumes that the input data is a numpy array

    obj_config = data.site_xmat[obj_sid].reshape((3,3))
    obj_pos = data.site_xpos[obj_sid].ravel()
    desired_config = data.site_xmat[target_obj_sid].reshape((3,3))
    palm_pos = data.site_xpos[palm_sid].ravel()

    tr_RtR = np.trace(obj_config.T.dot(desired_config))
    _arc_c_arg = (tr_RtR - 1)/2.0
    _th = np.arccos(_arc_c_arg)


    orien = _th**2#np.linalg.norm(obj_config - desired_config)
    dist = np.linalg.norm(palm_pos - obj_pos)

    vel = data.qvel[:]
    return 400.0*dist + 10.0*orien + 1e-1 * np.linalg.norm(vel)


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

    obj_config     =  data.body_xquat[obj_bid]
    desired_config =  data.body_xquat[target_obj_bid]
    _th = np.linalg.norm(obj_config - desired_config)
    if _th < 0.15:
        return True
    else:
        return False

    # tr_RtR = np.trace(obj_config.T.dot(desired_config))
    # _arc_c_arg = (tr_RtR - 1)/2.0
    # _th = np.arccos(_arc_c_arg)
    # _th = np.linalg.norm(mat2quat(obj_config) - mat2quat(desired_config))
    # if _th < 0.15:
    #     return True
    # else:
    #     return False

def get_distance(data):

    obj_config = data.site_xmat[obj_sid].reshape((3,3))
    obj_pos = data.site_xpos[obj_sid].ravel()
    desired_config = data.site_xmat[target_obj_sid].reshape((3,3))
    palm_pos = data.site_xpos[palm_sid].ravel()

    tr_RtR = np.trace(obj_config.T.dot(desired_config))
    _arc_c_arg = (tr_RtR - 1)/2.0
    _th = np.arccos(_arc_c_arg)
    return _th

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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("trial_num", type=int)
args = parser.parse_args()


def main():

    trial_num        = args.trial_num
    max_attempts     = 25

    #### --- initial parameters
    num_models      = 20
    num_trajectories = 2
    horizon         = 20
    noise           = 0.8
    lam             = 0.001
    final_time      = 200
    print('Generating the candidate models ...')
    model_pool = []
    for i in range(num_models):
        _model = load_model_from_path(model_path)
        model_pool.append(_model)

    print('Randomizing parameters')
    randomize_param(model_pool)

    emppi = MultModelMPPI(model_pool, task, terminal_cost,
                frame_skip=frame_skip,
                horizon=horizon, num_trajectories=num_trajectories,
                noise=noise, lam=lam)

    obs_action_pairs = []

    mean_mass = env_model.body_mass[obj_bid]
    mean_intertia = env_model.body_inertia[obj_bid,:]

    cnt = 0
    performance_data = {'successes' : [], 'time2success' : []}

    np.random.seed(10)
    win_cnt = 0
    while cnt < max_attempts:

        succeeded = False
        for t in range(final_time):

            stats = get_stats_from_prob(emppi.pool.sims, emppi.model_probs)

            state = env.get_state()
            ctrl = emppi(state, env.data.sensordata[:])
            scaled_ctrl = emppi.scale_ctrl(ctrl.copy())
            env.data.ctrl[:] = scaled_ctrl

            for _ in range(frame_skip):
                env.step()

            sensor_measurements = env.data.sensordata[:] + np.random.normal(0., 0.001, size=(emppi.num_sensors,))
            # emppi.update_distribution(sensor_measurements, state, scaled_ctrl)

            # if 1/np.sum(np.square(emppi.model_probs)) < emppi.num_tot_trajectories/2:
            #     # print('resampling!!!!!', 1/np.sum(np.square(emppi.model_probs)), emppi.num_tot_trajectories)
            #     update_distribution(emppi.pool.sims, emppi.model_probs)
            #     emppi.model_probs = np.ones(emppi.num_tot_trajectories)
            #     emppi.model_probs /= np.sum(emppi.model_probs)

            # viewer.render()

            if termination_condition(env.data):
                win_cnt += 1
                print('success')
                succeeded = True
                break
        print('success rate :', win_cnt/max_attempts)
        performance_data['successes'].append(succeeded)
        performance_data['time2success'].append(t)

        if not succeeded:
            print('failed')
            env.reset()

        print('resettting target')
        np.random.seed(cnt + 100)
        initial_quat = env.data.body_xquat[target_obj_bid]

        desired_orien = np.zeros(3)
        desired_orien[0] = np.random.uniform(low=-1.0, high=1.0)
        desired_orien[1] = np.random.uniform(low=-1.0, high=1.0)
        # desired_orien[0] = np.random.uniform(low=-2.0, high=2.0)
        # desired_orien[1] = np.random.uniform(low=-2.0, high=2.0)
        # desired_orien[2] = np.random.uniform(low=-2.0, high=2.0)

        angle = np.random.uniform(-np.pi, np.pi)
        axis = np.random.uniform(-1., 1., size=3)
        offset_quat = quat_from_angle_and_axis(angle, axis)
        initial_quat = rotations.quat_mul(initial_quat, offset_quat)
        # initial_quat /= np.linalg.norm(initial_quat)

        # env.model.body_quat[target_obj_bid] = euler2quat(desired_orien)
        env.model.body_quat[target_obj_bid] = initial_quat

        env.forward()

        for _sim in emppi.pool.sims:
            # _sim.model.body_quat[target_obj_bid] = euler2quat(desired_orien)
            _sim.model.body_quat[target_obj_bid] = initial_quat
            _sim.forward()

        if cnt % 2 == 0:
            print('saving at :', cnt)
            path = './data/percent_success/noupdate_param/'
            pickle.dump(performance_data, open(path + 'performance_data{}.pkl'.format(trial_num), 'wb'))

        # sample_error.append(error)
        # cnt += 1
        # data_path = 'data/actuator_param_known/'
        # print('saving data ....')
        # file_pi = open(data_path + 'sample_error.pickle', 'wb')
        # pickle.dump(sample_error, file_pi)
        # file_pi.close()
        # print('saved data! number : ', cnt)
        #
        # file_pi = open(data_path + 'estimation_data.pickle', 'wb')
        # pickle.dump(estimation_data, file_pi)
        # file_pi.close()
        cnt += 1
        if cnt > 100:
            break

if __name__ == '__main__':
    main()
