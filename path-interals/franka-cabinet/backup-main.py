#!/usr/bin/env python3

import numpy as np
import numpy.random as npr
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from quatmath import quat2euler, euler2quat
from mult_model_mppi import MultModelMPPI
from mult_model_pi2 import MultModelPI2
from mujoco_py.generated import const

# np.random.seed(800)

frame_skip = 2
real_model_path = 'assets/franka-cabinet.xml'
model_path = 'assets/franka-door.xml'

real_model = load_model_from_path(real_model_path)
real_sim = MjSim(real_model)
real_sim.step()
model = load_model_from_path(model_path)
sim = MjSim(model)

viewer = MjViewer(real_sim)

door_bid = model.body_name2id('Door')
handle_sid = model.site_name2id('Handle')
handle_measurement_sid = real_model.site_name2id('measurement-spot')
# drawer_hinge_did = model.jnt_dofadr[model.joint_name2id('drawer_top')]
# drawer_hinge_did = model.jnt_dofadr[model.joint_name2id('door_left')]
drawer_hinge_did = model.jnt_dofadr[model.joint_name2id('hinge')]
grasp_sid = model.site_name2id('grasp')

target_gripper_config = np.array([
    [0., 0., 1.],
    [1., 0., 0.],
    [0., 1., 0.]
]).ravel()

def rotationMatrixToEulerAngles(R) :

    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


stage_two = False

def task(data):
    global stage_two
    jnt = data.get_joint_qpos('finger1') - data.get_joint_qpos('finger2')
    handle_pos = data.site_xpos[handle_sid].ravel()
    grasp_pos = data.site_xpos[grasp_sid].ravel()
    grasp_config = data.site_xmat[grasp_sid].ravel()
    velocity_loss = np.dot(data.qvel, data.qvel)

    grasp_err = np.linalg.norm(grasp_pos - handle_pos)
    robot_ang = data.qpos[1] * data.qpos[1]
    num_contacts = len(data.contact)
    grasp_config_err = np.linalg.norm(grasp_config - target_gripper_config)
    loss = 0. * robot_ang + 10.0 * grasp_config_err \
            + 80.0 * grasp_err\
            + 0.01 * np.sqrt(velocity_loss)

    loss += 100.0 * num_contacts
    loss -= 100.0 * np.clip(jnt, 0., 0.06)
    if grasp_err < 0.04:
        stage_two = True
    # loss += 50.0 * handle_pos[0]
    if grasp_err > 0.04 and stage_two is False:
        return loss
    else:
        return 100.0 * handle_pos[0] + 80.0 * grasp_err + 80.0 * grasp_config_err + 80.0 * np.clip(jnt, 0., 0.06)
def terminal_cost():
    return 0

def randomize_param(models, sim):

    mean_jnt_axis = models[0].jnt_axis[drawer_hinge_did].ravel()
    mean_jnt_pos = models[0].jnt_pos[drawer_hinge_did].ravel()
    handle_pos = sim.data.site_xpos[handle_sid].ravel()
    tot_models = len(models)
    for i,_model in enumerate(models):
        # _model.jnt_axis[drawer_hinge_did][:] =
        # ax = npr.normal(mean_jnt_axis, 0.1)#, size=(3,))
        if i < tot_models/2:
            ax = npr.normal(np.array([0., 0., 1.]), 0.1)#, size=(3,))
        if i >= tot_models/2:
            ax =  npr.normal(np.array([1., 0., 0.]), 0.1)
        # ax = npr.uniform(-1, 1, size=(3,))
        ax /= np.linalg.norm(ax)
        _model.jnt_axis[drawer_hinge_did][:] = ax.copy()
        if np.argmax(ax) == 1 or np.argmax(ax) == 2:
            _model.jnt_type[drawer_hinge_did] = 3
        elif np.argmax(ax) == 0:
            _model.jnt_type[drawer_hinge_did] = 2
        # _model.jnt_pos[drawer_hinge_did][0] = 0.
        # _model.jnt_pos[drawer_hinge_did][1] = npr.uniform(-0.2, 0.2)
        # _model.jnt_pos[drawer_hinge_did][2] = npr.uniform(-0.05, 0.05)

def update_distribution(sims, probs):
    mean_jnt_axis = 0.0
    mean_jnt_pos = 0.0

    for sim, prob in zip(sims, probs):
        mean_jnt_axis += sim.model.jnt_axis[drawer_hinge_did].ravel() * prob
        mean_jnt_pos += sim.model.jnt_pos[drawer_hinge_did].ravel() * prob

    mean_jnt_axis /= np.linalg.norm(mean_jnt_axis)

    for sim in sims:
        ax = np.random.normal(mean_jnt_axis, 0.01)
        ax /= np.linalg.norm(ax)
        pos = np.random.normal(mean_jnt_pos, 0.01)

        sim.model.jnt_axis[drawer_hinge_did][:] = ax
        sim.model.jnt_pos[drawer_hinge_did][:] = pos

        if np.argmax(ax) == 1 or np.argmax(ax) == 2:
            sim.model.jnt_type[drawer_hinge_did] = 3
        elif np.argmax(ax) == 0:
            sim.model.jnt_type[drawer_hinge_did] = 2


def main():
    #### --- initial parameters
    num_models      = 10
    num_trajectories = 4
    horizon         = 40
    final_time      = 200
    noise           = 0.1
    lam             = 0.6

    print('Generating the candidate models ...')
    model_pool = []
    for i in range(num_models):
        _model = load_model_from_path(model_path)
        model_pool.append(_model)

    print('Randomizing parameters')
    randomize_param(model_pool, sim)

    # emppi = MultModelPI2(model_pool, task, terminal_cost,
    #             frame_skip=frame_skip,
    #             horizon=horizon, num_trajectories=num_trajectories, noise=noise, lam=lam,
    #             default_hidden_layers=[128])
    #             #default_hidden_layers=[128,64])
    emppi = MultModelMPPI(model_pool, task, terminal_cost,
                frame_skip=frame_skip,
                horizon=horizon, num_trajectories=num_trajectories,
                noise=noise, lam=lam)

    while True:


        sim.reset()
        desired_pos = np.zeros(3)
        desired_pos[0] = np.random.uniform(0.6, 0.8)
        desired_pos[1] = np.random.uniform(-0.5, 0.5)
        desired_pos[2] = np.random.uniform(0.3, 0.7)

        real_sim.reset()
        desired_pos = real_sim.data.site_xpos[handle_measurement_sid,:]
        print(desired_pos)
        sim.model.body_pos[door_bid,:] = desired_pos

        sim.data.qpos[0] = -0.3
        sim.data.qpos[1] = -1.
        sim.data.qpos[3] = -1.7
        sim.data.qpos[5] = 1.4
        sim.forward()
        for _sim in emppi.pool.sims:
            _sim.reset()
            _sim.model.body_pos[door_bid,:] = desired_pos
            _sim.forward()
        traj_pos = []
        traj_vel = []
        est_axis = []
        est_pos = []
        model_prob_traj = []
        cnt = 0

        for t in range(final_time):
            state = sim.get_state()
            # ctrl = emppi(state, sim.data.sensordata[:], t)
            ctrl= emppi(state)
            sim.data.ctrl[:] = ctrl
            for _ in range(frame_skip):
                sim.step()
            # traj_pos.append(sim.data.qpos[:].copy())
            # traj_vel.append(sim.data.qvel[:].copy())
            # cnt += 1
            # if cnt % 200 == 0:
            #     sim.reset()
            #     sim.data.qpos[0] = -0.3
            #     sim.data.qpos[1] = -1.
            #     sim.data.qpos[3] = -1.7
            #     sim.data.qpos[5] = 1.4
            #     sim.forward()
            sensor_measurements = sim.data.sensordata[:]
            emppi.update_distribution(sensor_measurements, state, ctrl)
            if 1/np.sum(np.square(emppi.model_probs)) < emppi.num_tot_trajectories/2:
                # print('resampling!!!!!', 1/np.sum(np.square(emppi.model_probs)), emppi.num_tot_trajectories)
                update_distribution(emppi.pool.sims, emppi.model_probs)
                emppi.model_probs = np.ones(emppi.num_tot_trajectories)
                emppi.model_probs /= np.sum(emppi.model_probs)
            # mean_joint_pos = 0.0
            # mean_joint_axis = 0.0
            # _hinge_poses = []
            # _hinge_axis = []
            # _hinge_probs = []
            # norm_prob = np.linalg.norm(emppi.model_probs)
            #
            # for _sim, m_prob in zip(emppi.pool.sims, emppi.model_probs):
            #     mean_joint_pos += _sim.model.jnt_pos[drawer_hinge_did] * m_prob
            #     mean_joint_axis += _sim.model.jnt_axis[drawer_hinge_did] * m_prob
            #     _hinge_poses.append(_sim.data.xanchor[drawer_hinge_did].ravel().copy())
            #     _hinge_axis.append(_sim.data.xaxis[drawer_hinge_did].ravel().copy())
            #     _hinge_probs.append(m_prob.copy())
            #     if abs(_hinge_axis[-1])[0] > 0:
            #         rot = np.array([
            #                 [0., 0., -1],
            #                 [0., 1., 0.],
            #                 [1., 0., 0.]
            #             ]).flatten()
            #     else:
            #         rot = np.eye(3).flatten()
            #     a = _sim.model.jnt_axis[drawer_hinge_did].ravel()
            #     a /= np.linalg.norm(a)
            #     b = np.array([0.,0., 1.])
            #     rot = eulerAnglesToRotationMatrix(a,b)
            #     viewer.add_marker(pos=_sim.data.xanchor[drawer_hinge_did].flatten(),
            #             size=np.array([0.01,0.01,0.4]), type=const.GEOM_ARROW, label='',
            #             rgba=np.array([1.,1.,1.,m_prob/norm_prob]),
            #             mat=rot)
            #
            # est_axis.append(_hinge_axis)
            # est_pos.append(_hinge_poses)
            # model_prob_traj.append(_hinge_probs)


            viewer.render()

        # emppi.lam /= (cnt%100)
    # np.save('door-many-ax-pos.npy', traj_pos)
    # np.save('door-many-ax-vel.npy', traj_vel)
    # np.save('door-est-axis.npy', est_axis)
    # np.save('door-est-pos.npy', est_pos)
    # np.save('door-model-prob.npy', model_prob_traj)

if __name__ == '__main__':
    main()
