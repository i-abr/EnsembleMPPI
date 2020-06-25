#!/usr/bin/env python3

import matplotlib.pyplot as plt

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from quatmath import quat2euler, euler2quat
from mult_model_mppi import MultModelMPPI
# from mult_model_pi2 import MultModelPI2
import pickle

frame_skip = 5#10
model_path = '../shadow-hand-assets/hand/manipulate_block_gain.xml'
env_model = load_model_from_path(model_path)
env = MjSim(env_model)

viewer = MjViewer(env)
viewer._render_every_frame = True
## data idx that are necessary in this example
target_obj_bid = env_model.body_name2id('target')
target_obj_sid = env_model.site_name2id('target:center')
obj_sid = env_model.site_name2id('object:center')
obj_bid = env_model.body_name2id('object')
palm_sid = env_model.site_name2id('robot0:Palm')
# eps_ball_sid = model.site_name2id('eps_ball')

# env.model.body_mass[obj_bid] = 0.5

def randomize_param(models):
    # wrong_param = models[0].actuator_gainprm[2:,0].copy() - 0.2
    wrong_param = np.random.uniform(1.0, 5.0, size=(models[0].nu-2))
    for model in models:
        model.actuator_gainprm[2:, 0] = wrong_param

def update_distribution(sims, probs):
    mean_gains = 0.0

    for sim, prob in zip(sims, probs):
        mean_gains += sim.model.actuator_gainprm[2:,0] * prob

    for sim in sims:
        sim.model.actuator_gainprm[2:,0] = np.random.normal(mean_gains, 0.15)


def get_stats_from_prob(sims, probs):
    mean_gains = 0.0

    for sim, prob in zip(sims, probs):
        mean_gains += sim.model.actuator_gainprm[2:,0] * prob

    var_gains = 0.0

    ### get the variance
    for sim, prob in zip(sims, probs):
        delta_gain = mean_gains - sim.model.actuator_gainprm[2:,0]
        var_gains += np.outer(delta_gain, delta_gain) * prob

    stats = {
        'mean gain' : mean_gains,
        'var gain' : var_gains
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
    return 400.0*dist + 10.0*orien + 0.1 * np.linalg.norm(vel)


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

    obj_config = data.site_xmat[obj_sid].reshape((3,3))
    obj_pos = data.site_xpos[obj_sid].ravel()
    desired_config = data.site_xmat[target_obj_sid].reshape((3,3))
    palm_pos = data.site_xpos[palm_sid].ravel()

    tr_RtR = np.trace(obj_config.T.dot(desired_config))
    _arc_c_arg = (tr_RtR - 1)/2.0
    _th = np.arccos(_arc_c_arg)

    if _th < 0.02:
        return True
    else:
        return False
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


def main():

    #### --- initial parameters
    num_models      = 10
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
    state_data = []
    estimation_data = []
    target_data = []
    completion_time_per_dist_data = []

    sample_error = []
    np.random.seed(10)

    while True:
        env.reset()
        print('resettting')
        np.random.seed(cnt + 100)
        desired_orien = np.zeros(3)
        desired_orien[0] = np.random.uniform(low=-2.0, high=2.0)
        desired_orien[1] = np.random.uniform(low=-2.0, high=2.0)
        desired_orien[2] = np.random.uniform(low=-2.0, high=2.0)
        env.model.body_quat[target_obj_bid] = euler2quat(desired_orien)
        target_data.append(
            env.model.body_quat[target_obj_bid].ravel().copy()
        )
        env.forward()

        for _sim in emppi.pool.sims:
            _sim.reset()
            _sim.model.body_quat[target_obj_bid] = euler2quat(desired_orien)
            _sim.forward()
        emppi.mean_actions = emppi.mean_actions*0.0
        #emppi.model_probs = np.ones(emppi.num_tot_trajectories)
        #emppi.model_probs /= np.sum(emppi.model_probs)
        print('updated sim')
        error = []

        for t in range(final_time):
            error.append(
                task(env.data, None)
            )
            ### get the stats first
            stats = get_stats_from_prob(emppi.pool.sims, emppi.model_probs)

            state = env.get_state()
            ctrl = emppi(state, env.data.sensordata[:])
            scaled_ctrl = emppi.scale_ctrl(ctrl.copy())
            env.data.ctrl[:] = scaled_ctrl

            for _ in range(frame_skip):
                env.step()

            # sensor_measurements = env.data.sensordata[:] + np.random.normal(0., 0.001, size=(emppi.num_sensors,))
            # emppi.update_distribution(sensor_measurements, state, scaled_ctrl)
            #img = env.render(720, 540, camera_name="fixed")
            #plt.clf()
            #plt.imshow(img, origin='lower')
            #plt.pause(0.01)

            viewer.render()

            ### Store the data
            state_data.append(
                (state, env.model.body_quat[target_obj_bid].ravel().copy())
            )
            estimation_data.append(stats)

            print(stats['mean gain'])
            print(1/np.sum(np.square(emppi.model_probs)))

            # if 1/np.sum(np.square(emppi.model_probs)) < emppi.num_tot_trajectories/2:
            #     print('resampling!!!!!', 1/np.sum(np.square(emppi.model_probs)), emppi.num_tot_trajectories)
            #     update_distribution(emppi.pool.sims, emppi.model_probs)
            #     emppi.model_probs = np.ones(emppi.num_tot_trajectories)
            #     emppi.model_probs /= np.sum(emppi.model_probs)

            #if termination_condition(env.data):
            #    break

        sample_error.append(error)
        cnt += 1
        data_path = 'data/actuator_param_wrong/'
        print('saving data ....')
        file_pi = open(data_path + 'sample_error.pickle', 'wb')
        pickle.dump(sample_error, file_pi)
        file_pi.close()
        print('saved data! number : ', cnt)

        file_pi = open(data_path + 'estimation_data.pickle', 'wb')
        pickle.dump(estimation_data, file_pi)
        file_pi.close()

        if cnt >20:
            break

if __name__ == '__main__':
    main()
