#!/usr/bin/env python3

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from quatmath import quat2euler, euler2quat
import pickle
from behavior_cloning import policy

import torch

# device = torch.device("cpu")
device = torch.device("cuda:0")

frame_skip = 10
model_path = '../shadow-hand-assets/hand/manipulate_block.xml'
model = load_model_from_path(model_path)
sim = MjSim(model)

viewer = MjViewer(sim)
act_mid = np.mean(model.actuator_ctrlrange, axis=1)
act_rng = 0.5 * (model.actuator_ctrlrange[:,1]-model.actuator_ctrlrange[:,0])
def scale_ctrl(ctrl, act_mid, act_rng):
    ## first clip
    ctrl = np.clip(ctrl, -1.0, 1.0)
    ## now scale
    ctrl = act_mid + ctrl * act_rng
    return ctrl



## data idx that are necessary in this example
target_obj_bid = model.body_name2id('target')
target_obj_sid = model.site_name2id('target:center')
obj_sid = model.site_name2id('object:center')
obj_bid = model.body_name2id('object')
palm_sid = model.site_name2id('robot0:Palm')
# eps_ball_sid = model.site_name2id('eps_ball')
def get_obs(data):
    qp = data.qpos.ravel()
    qvel = data.qvel.ravel()
    obj_pos = data.site_xpos[obj_sid].ravel()
    obj_orien = data.site_xmat[obj_sid].ravel()

    obj_vel = data.qvel[-6:].ravel()

    desired_orien = data.site_xmat[target_obj_sid].ravel()
    palm_pos = data.site_xpos[palm_sid].ravel()


    return np.concatenate([qp[:-6], qvel[:-6], obj_pos, obj_vel, obj_orien, desired_orien,
                            obj_pos - palm_pos, obj_orien - desired_orien ])

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

    final_time = 400
    policy = torch.load('torch_policy.pt')
    policy.to(device=device)
    dtype = torch.float
    obs = torch.randn(1, len(get_obs(sim.data)), device=device, dtype=dtype)
    # param = pickle.load(open('bc-policy.pickle', 'rb'))
    cnt = 0
    while True:
        sim.reset()
        print('resettting')
        desired_orien = np.zeros(3)
        desired_orien[0] = np.random.uniform(low=-2.0, high=2.0)
        desired_orien[1] = np.random.uniform(low=-2.0, high=2.0)
        desired_orien[2] = np.random.uniform(low=-2.0, high=2.0)
        sim.model.body_quat[target_obj_bid] = euler2quat(desired_orien)
        sim.forward()

        print('updated sim')
        for t in range(final_time):
            # ctrl = policy(param, sim.data.sensordata[:])
            obs = torch.as_tensor(get_obs(sim.data), device=device, dtype=dtype)
            ctrl = policy(obs)
            sim.data.ctrl[:] = scale_ctrl(ctrl.cpu().detach().numpy()[:], act_mid, act_rng)
            for _ in range(frame_skip):
                sim.step()
            if termination_condition(sim.data):
                break
            viewer.render()

if __name__ == '__main__':
    main()
