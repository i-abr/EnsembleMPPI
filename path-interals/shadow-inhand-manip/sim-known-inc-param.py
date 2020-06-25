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
model_path = '../shadow-hand-assets/hand/manipulate_block.xml'
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

env.model.body_mass[obj_bid] = 0.5

def randomize_param(models):
    mean_mass = models[0].body_mass[obj_bid]
    mean_intertia = models[0].body_inertia[obj_bid,:]
    scale = 0.8
    _masses = np.linspace(0.01, 0.1, len(models))
    mass = 0.1 #np.random.normal(mean_mass, 0.2)
    for model, _mass in zip(models, _masses):
        model.body_mass[obj_bid] = mass
        #model.body_mass[obj_bid] = np.clip(np.random.uniform(0.01, 0.1), 0.01, np.inf)
        #model.body_inertia[obj_bid] = np.clip(np.random.uniform(1e-5,4e-5, size=(3,)), 2e-7, np.inf)
        #model.body_inertia[obj_bid, :] = np.random.uniform(mean_intertia-mean_intertia*scale, mean_intertia+mean_intertia*scale)
        #model.body_inertia[obj_bid, :] = np.random.uniform(mean_intertia-mean_intertia*scale, mean_intertia+mean_intertia*scale)

def update_distribution(sims, probs):
    mean_mass = 0.0
    mean_intertia = 0.0

    for sim, prob in zip(sims, probs):
        mean_mass += sim.model.body_mass[obj_bid] * prob
        mean_intertia += sim.model.body_inertia[obj_bid, :] * prob

    for sim in sims:
        sim.model.body_mass[obj_bid] = np.clip(np.random.normal(mean_mass, 0.01), 0.01, np.inf)
        #sim.model.body_inertia[obj_bid, :] = np.clip(np.random.normal(mean_intertia, 5e-6), 2e-7, np.inf)


def get_stats_from_prob(sims, probs):
    mean_inertia = 0.0
    mean_mass = 0.0

    for sim, prob in zip(sims, probs):
        mean_mass += sim.model.body_mass[obj_bid] * prob
        mean_inertia += sim.model.body_inertia[obj_bid] * prob

    var_mass = 0.0
    var_inertia = 0.0

    ### get the variance
    for sim, prob in zip(sims, probs):
        delta_mass = mean_mass - sim.model.body_mass[obj_bid]
        delta_inertia = mean_inertia - sim.model.body_inertia[obj_bid]
        var_mass += np.outer(delta_mass, delta_mass) * prob
        var_inertia += np.outer(delta_inertia, delta_inertia) * prob

    stats = {
        'mean mass' : mean_mass,
        'mean inertia' : mean_inertia,
        'var mass' : var_mass,
        'var inertia' : var_inertia
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

    _filter = 0.0
    alpha = 0.1
    cnt = 0
    state_data = []
    estimation_data = []
    target_data = []
    completion_time_per_dist_data = []
    
    sample_error = []

    ### We initialize the seed here to ensure no other call to rand will change the outcome of param
    while True:
        env.reset()
        print('resettting')
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
            # obs_action_pairs.append(
            #     [get_obs(env.data).copy(), ctrl.copy()]
            # )
            env.data.ctrl[:] = scaled_ctrl

            for _ in range(frame_skip):
                env.step()

            sensor_measurements = env.data.sensordata[:]
            #emppi.update_distribution(sensor_measurements, state, scaled_ctrl)
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

            #if termination_condition(env.data):
            #    break
        sample_error.append(error)
        cnt += 1 
        data_path = 'data/baseline_data-inc-param/'
        print('saving data ....')
        file_pi = open(data_path + 'sample_error.pickle', 'wb')
        pickle.dump(sample_error, file_pi)
        print('saved data! number : ', cnt)
        if cnt >20:
            break
if __name__ == '__main__':
    main()
