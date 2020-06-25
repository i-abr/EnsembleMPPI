#!/usr/bin/env python3

import numpy as np
import rospy
import rosbag

from mujoco_py import load_model_from_path, MjSim, MjSimPool, MjViewer, MjSimState
from mujoco_py.generated import const
from quatmath import euler2mat

def rot2vect(R, vector_orig, vector_fin):

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)

    # The rotation axis (normalised).
    axis = np.cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = np.arccos(np.dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = np.cos(angle)
    sa = np.sin(angle)

    # Calculate the rotation matrix elements.
    R[0,0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0,1] = -z*sa + (1.0 - ca)*x*y
    R[0,2] = y*sa + (1.0 - ca)*x*z
    R[1,0] = z*sa+(1.0 - ca)*x*y
    R[1,1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1,2] = -x*sa+(1.0 - ca)*y*z
    R[2,0] = -y*sa+(1.0 - ca)*x*z
    R[2,1] = x*sa+(1.0 - ca)*y*z
    R[2,2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)
    return R

def get_handle_data(npy_file_name, num_files):

    data = np.load(npy_file_name + '_{}.npy'.format(0))
    for i in range(num_files):
        _data = np.load(npy_file_name + '_{}.npy'.format(i))
        data = np.vstack((data, _data))
    return data

def main():

    model_path = '../assets/franka-cabinet.xml'
    model = load_model_from_path(model_path)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    drawer_jnt_did = model.jnt_dofadr[model.joint_name2id('drawer_top')]

    handle_pose_data = get_handle_data('./data/trial4/sampled_poses',20)
    handle_axis_data = get_handle_data('./data/trial4/sampled_jnts', 20)
    model_prob_data = get_handle_data('./data/trial4/model_probs', 20)

    bag = rosbag.Bag('./data/franka_emppi_2019-02-15-08-51-45.bag')
    franka_jnts = '/robot/joint_states_relay'
    sektion_jnts = '/tracker/sektion_cabinet/joint_states'
    marker_rot = np.eye(3)
    shift_forward = euler2mat(np.array([0., -np.pi/2, 0.]))
    cnt = 0
    tic = 0.0
    toc = 0.12 ### effective time

    t0 = None
    t_prev = None
    start_exp_time = 0.0
    t_prev = 0.
    for topic, msg, t in bag.read_messages(topics=[franka_jnts, sektion_jnts]):
        if t0 is None:
            t0 = t.to_sec()
        print('relative time : ', t.to_sec() - t0)
        if topic == franka_jnts:
            jnts = msg.position
            sim.data.qpos[:7] = jnts[:7]
            sim.data.qpos[6] -= 0.78539816339
            sim.data.qpos[7] = jnts[7]
            sim.data.qpos[8] = -jnts[8]
            # if start_exp_time == None and np.linalg.norm(msg.velocity[:7]) > 0.1:
            #     start_exp_time = t.to_sec() - t0
        elif topic == sektion_jnts:

            jnt_name = list(msg.name)
            jnt_idx = jnt_name.index('drawer_top_joint')
            sim.data.qpos[drawer_jnt_did] = msg.position[jnt_idx]

        sim.forward()

        if start_exp_time is not None:
            tic += t.to_sec() - t0 - t_prev
            if tic >= toc:
                cnt += 1
                tic = 0.0

        if start_exp_time is not None:
            if (t.to_sec()-t0) >= start_exp_time:
                #### add the sampled joints
                sampled_jnts = handle_axis_data[cnt]
                sampled_poses = handle_pose_data[cnt]
                model_probs = model_prob_data[cnt]
                model_prob_norm = np.linalg.norm(model_probs)
                for m_prob, _jnt, _pose in zip(model_probs, sampled_jnts, sampled_poses):
                    marker_rot = rot2vect(marker_rot, _jnt, np.array([1., 0., 0.]))
                    viewer.add_marker(pos=_pose,
                                    size=np.array([0.01, 0.01, 0.4]), type=const.GEOM_ARROW, label='',
                                    rgba=np.array([1.0, 1.0, 1.0, m_prob/model_prob_norm]),
                                    mat = shift_forward.dot(marker_rot))
        else:
                #### add the sampled joints
                sampled_jnts = handle_axis_data[0]
                sampled_poses = handle_pose_data[0]
                model_probs = model_prob_data[0]
                model_prob_norm = np.linalg.norm(model_probs)
                for m_prob, _jnt, _pose in zip(model_probs, sampled_jnts, sampled_poses):
                    marker_rot = rot2vect(marker_rot, _jnt, np.array([1., 0., 0.]))
                    viewer.add_marker(pos=_pose,
                                    size=np.array([0.01, 0.01, 0.4]), type=const.GEOM_ARROW, label='',
                                    rgba=np.array([1.0, 1.0, 1.0, m_prob/model_prob_norm]),
                                    mat = shift_forward.dot(marker_rot))
        # rospy.sleep(t.to_sec() - t0 - t_prev)
        viewer.render()

        t_prev = t.to_sec() - t0
        print(cnt)
    bag.close()


'''
def main():

    model_path = '../assets/franka-cabinet.xml'
    model = load_model_from_path(model_path)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    jnts = get_jnt_data('./data/franka_emppi_2019-02-14-16-20-11.bag')

    handle_pose_data = get_handle_data('./data/trial3/sampled_poses',22)

    sektion_jnts = get_drawer_jnt_data('./data/franka_emppi_2019-02-14-16-20-11.bag')

    print(len(sektion_jnts), len(jnts))

    print(handle_pose_data.shape)
    for jnt in jnts:
        sim.data.qpos[:7] = jnt[:7]
        sim.forward()
        viewer.render()
'''


if __name__=='__main__':
    main()
