#!/usr/bin/env python3

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from quatmath import quat2euler, euler2quat
import pickle
from behavior_cloning import policy

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

def main():

    final_time      = 400

    param = pickle.load(open('bc-policy.pickle', 'rb'))
    cnt = 0
    while True:
        # sim.reset()
        print('resettting')
        desired_orien = np.zeros(3)
        desired_orien[0] = np.random.uniform(low=-2.0, high=2.0)
        desired_orien[1] = np.random.uniform(low=-2.0, high=2.0)
        desired_orien[2] = np.random.uniform(low=-2.0, high=2.0)
        sim.model.body_quat[target_obj_bid] = euler2quat(desired_orien)
        sim.forward()

        print('updated sim')
        for t in range(final_time):
            ctrl = policy(param, sim.data.sensordata[:])
            print(ctrl)
            sim.data.ctrl[:] = ctrl
            for _ in range(frame_skip):
                sim.step()
            if termination_condition(sim.data):
                break
            viewer.render()

if __name__ == '__main__':
    main()
