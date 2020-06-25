#!/usr/bin/env python3

import numpy as np
import pickle
from mujoco_py import load_model_from_path, MjSim, MjViewer
import time
import matplotlib.pyplot as plt
frame_skip = 5

model_path = '../../shadow-hand-assets/hand/manipulate_block.xml'
env_model = load_model_from_path(model_path)
env = MjSim(env_model)

viewer = MjViewer(env)

target_obj_bid = env_model.body_name2id('target')
target_obj_sid = env_model.site_name2id('target:center')
obj_sid = env_model.site_name2id('object:center')
obj_bid = env_model.body_name2id('object')
palm_sid = env_model.site_name2id('robot0:Palm')

file = open('./actuator_update/state-data.pickle', 'rb')
data = pickle.load(file)
file.close()


def termination_condition(data):

    obj_config = data.site_xmat[obj_sid].reshape((3,3))
    obj_pos = data.site_xpos[obj_sid].ravel()
    desired_config = data.site_xmat[target_obj_sid].reshape((3,3))
    palm_pos = data.site_xpos[palm_sid].ravel()

    tr_RtR = np.trace(obj_config.T.dot(desired_config))
    _arc_c_arg = (tr_RtR - 1)/2.0
    _th = np.arccos(_arc_c_arg)

    if _th < 0.2:
        return True
    else:
        return False

while True:
    # viewer.render()
    # time.sleep(10.0)
    data_len = len(data)
    num_frames = 10
    for i, (state, target) in enumerate(data):
        env.model.body_quat[target_obj_bid] = target
        env.set_state(state)
        env.forward()
        # img = env.render(1024, 786, camera_name="fixed")
        # plt.clf()
        # plt.imshow(img, origin='lower')
        # plt.pause(0.01)
        # if i % num_frames == 0:
        viewer.render()
        time.sleep(0.05)
