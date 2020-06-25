#!/usr/bin/env python3

import numpy as np
import numpy.random as npr
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import time
from mujoco_py.generated import const


frame_skip = 2
model_path = 'assets/franka-cabinet.xml'
model = load_model_from_path(model_path)
sim = MjSim(model)

viewer = MjViewer(sim)

def eulerAnglesToRotationMatrix(a, b):
    v = np.cross(a,b)
    c = np.dot(a,b)
    s = np.linalg.norm(v)
    I = np.identity(3)
    k = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
    ])
    return I + k + np.dot(k, k) * 1/(1+c)

def main():
    poses = np.load('many-ax-pos.npy')
    vels = np.load('many-ax-vel.npy')
    est_axis= np.load('est-axis.npy')
    est_pos = np.load('est-pos.npy')
    model_prob_traj = np.load('model-prob.npy')


    while True:

        for pose, vel, jnt_axiss, jnt_poss, model_probs in zip(poses, vels, est_axis, est_pos, model_prob_traj):

            sim.data.qpos[:] = pose
            sim.data.qvel[:] = vel

            sim.forward()
            # sim.step()


            norm_prob = np.linalg.norm(model_probs)

            for jnt_axis, jnt_pos, m_prob in zip(jnt_axiss, jnt_poss, model_probs):
                a = jnt_axis
                a /= np.linalg.norm(a)
                b = np.array([0.,0., -1.])
                rot = eulerAnglesToRotationMatrix(a,b)
                viewer.add_marker(pos=jnt_pos,
                        size=np.array([0.01,0.01,0.4]), type=const.GEOM_ARROW, label='',
                        rgba=np.array([1.,1.,1.,m_prob/norm_prob]),
                        mat=rot)


            viewer.render()
            time.sleep(0.01)




if __name__ == '__main__':
    main()
