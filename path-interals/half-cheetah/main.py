#!/usr/bin/env python3

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimPool
import os
from mult_model_mppi import MultModelMPPI
from mult_model_pi2_v2 import MultModelPI2
from task import task, terminal_cost
from mult_model_pi2 import MultModelPI2 as MultModelPI2_def
import pickle
# import matplotlib.pyplot as plt
# np.set_printoptions(precision=3, suppress=True)

frame_skip = 2
model_path = 'half_cheetah.xml'

model = load_model_from_path(model_path)
sim = MjSim(model)
viewer = MjViewer(sim)

np.random.seed(50)

def randomize_param(models):
    mean_mass = models[0].body_mass[1:]
    mean_jnt_stiff = models[0].jnt_stiffness[3:]
    mean_dof_damping = models[0].dof_damping[3:]
    print('blah blah : ')
    print(mean_mass, mean_dof_damping)
    min_jnt_stuff = np.min(mean_jnt_stiff)
    for model in models:
        model.dof_damping[3:] = np.random.uniform(mean_dof_damping * 0 + 1., mean_dof_damping*0 + 6.0)
        #model.dof_damping[3:] = np.random.normal(mean_dof_damping, 0.4)
        #model.jnt_stiffness[3:] = np.random.normal(mean_jnt_stiff, 10.0)
        model.body_mass[1:] = np.random.uniform(mean_mass * 0 + 1., mean_mass*0 + 6.0)

        #model.body_mass[1:] = np.random.normal(mean_mass, 0.4)

def update_distribution(sims, probs):
    mean_dof_damping = 0.0
    mean_jnt_stiff = 0.0
    mean_mass = 0.0

    for sim, prob in zip(sims, probs):
        mean_dof_damping += sim.model.dof_damping[3:] * prob
        # mean_jnt_stiff += sim.model.jnt_stiffness[3:] * prob
        mean_mass += sim.model.body_mass[1:] * prob

    for sim in sims:
        sim.model.dof_damping[3:] = np.random.normal(mean_dof_damping, 0.1)
        #sim.model.jnt_stiffness[3:] = np.random.normal(mean_jnt_stiff, 10.0)
        sim.model.body_mass[1:] = np.random.normal(mean_mass, 0.1)

def get_stats_from_prob(sims, probs):
    mean_dof_damping = 0.0
    mean_jnt_stiff = 0.0
    mean_mass = 0.0

    for sim, prob in zip(sims, probs):
        mean_dof_damping += sim.model.dof_damping[3:] * prob
        #mean_jnt_stiff += sim.model.jnt_stiffness[3:] * prob
        mean_mass += sim.model.body_mass[1:] * prob

    mean_param = np.hstack((
        mean_dof_damping,
        #mean_jnt_stiff,
        mean_mass
    ))

    dof_damping_var = 0.0
    jnt_stiff_var = 0.0
    mass_var = 0.0

    var = 0.0

    for sim, prob in zip(sims, probs):
        param = np.hstack((
            sim.model.dof_damping[3:],
            #sim.model.jnt_stiffness[3:],
            sim.model.body_mass[1:]
        ))
        delta_param = param - mean_param
        var += np.outer(delta_param, delta_param) * prob
        # delta_dof_damping = sim.model.dof_damping[3:]-mean_dof_damping
        # dof_damping_var += np.outer(delta_dof_damping, delta_dof_damping)
        #
        # delta_jnt_stiff = sim.model.jnt_stiffness[3:]-mean_jnt_stiff
        # jnt_stiff_var += np.outer(delta_jnt_stiff, delta_jnt_stiff)
        #
        # delta_mass = sim.model.body_mass[1:]-mean_mass
        # mass_var += np.outer(delta_mass, delta_mass)



    # dof_damping_var /= len(sims)
    # jnt_stiff_var /= len(sims)
    # mass_var /= len(sims)

    # var /= len(sims)

    stats = {
        'mean dof damping' : mean_dof_damping,
        'mean jnt stiffness' : mean_jnt_stiff,
        'mean mass' : mean_mass,
        'var' : var
        # 'dof damping var' : dof_damping_var,
        # 'jnt stiffness var' : jnt_stiff_var,
        # 'mass var' : mass_var
    }
    return stats




def main():

    #### --- initial parameters
    num_models      = 20
    num_trajectories = 4
    horizon         = 40
    final_time      = 400
    noise           = 0.5
    lam             = 0.1

    print('Generating the candidate models ...')
    model_pool = []
    for i in range(num_models):
        _model = load_model_from_path(model_path)
        model_pool.append(_model)

    print('Randomizing parameters')
    randomize_param(model_pool)

    print('creating the controller')
    emppi = MultModelMPPI(model_pool, task, frame_skip=frame_skip,
           horizon=horizon, num_trajectories=num_trajectories, noise=noise, lam=lam)
    print(emppi.num_states)
    print(emppi.num_actions)
    input()
    #emppi = MultModelPI2(model_pool, task, terminal_cost, frame_skip=frame_skip,
    #        horizon=horizon, final_time=final_time, num_trajectories=num_trajectories, noise=noise, lam=lam,
    #        default_hidden_layers=[])
    #emppi = MultModelPI2_def(model_pool, task, terminal_cost, frame_skip=frame_skip,
    #       horizon=horizon, num_trajectories=num_trajectories, noise=noise, lam=lam,
    #        default_hidden_layers=[])


    ##--- sanity check
    # print(emppi.pool.sims[2*num_trajectories+1].model.jnt_stiffness[4])
    # print(emppi.pool.sims[6*num_trajectories+1].model.jnt_stiffness[4])
    
    est_data = []
    state_data = []
    
    cnt =0
    while True:
        tot_cost = 0.0
        use_policy=False
        emppi.mean_actions *= 0.0
        for t in range(200):
            state = sim.get_state() ### get the state

            # ctrl = emppi(state, sim.data.sensordata[:], t, use_policy=use_policy)
            ctrl = emppi(state)
            sim.data.ctrl[:] = ctrl
            tot_cost += task(sim.data, ctrl)
            for _ in range(frame_skip):
                sim.step()

            viewer.render()

            ### -- updating the model distribution
            sensor_measurements = sim.data.sensordata[:] + np.random.normal(0., 0.001, size=(emppi.num_sensors,))
            emppi.update_distribution(sensor_measurements, state, ctrl)
            print( 1/np.sum(np.square(emppi.model_probs)) )
            
            stats = get_stats_from_prob(emppi.pool.sims, emppi.model_probs)
            est_data.append(stats)
            state_data.append(state)
            
            if 1/np.sum(np.square(emppi.model_probs)) < emppi.num_tot_trajectories/2:
                # print('resampling!!!!!', 1/np.sum(np.square(emppi.model_probs)), emppi.num_tot_trajectories)
                update_distribution(emppi.pool.sims, emppi.model_probs)
                emppi.model_probs = np.ones(emppi.num_tot_trajectories)
                emppi.model_probs /= np.sum(emppi.model_probs)


            print('----- predicted -------')
            print(stats['mean dof damping'])
            print(stats['mean mass'])

            print('----- measured ------')
            print(sim.model.dof_damping[3:])
            print(sim.model.body_mass[1:])
        sim.reset()
        '''
        print('Total cost : ', tot_cost)
        sim.reset()
        cnt += 1
        data_path = 'data/trial2/'
        print('saving data.....')
        file_pi = open(data_path + 'state-data.pickle', 'wb')
        pickle.dump(state_data, file_pi)
        file_pi.close()
        file_pi = open(data_path + 'estimation_data.pickle', 'wb')
        pickle.dump(est_data, file_pi)
        file_pi.close()
        np.save(data_path + 'true-mass.npy', sim.model.body_mass[1:].ravel())
        np.save(data_path + 'true-damping.npy', sim.model.dof_damping[3:].ravel())
        '''
if __name__ == '__main__':
    main()
