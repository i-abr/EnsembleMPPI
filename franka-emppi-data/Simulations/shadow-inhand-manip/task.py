import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer


class Task(object):

    def __init__(self, model):
        self.target_obj_bid = model.body_name2id('target')
        self.target_obj_sid = model.site_name2id('target:center')
        self.obj_sid = model.site_name2id('object:center')
        self.obj_bid = model.body_name2id('object')
        self.palm_sid = model.site_name2id('robot0:Palm')

    def __call__(self, data, terminal=False):### this function assumes that the input data is a numpy array

        obj_config = data.site_xmat[self.obj_sid].ravel()
        obj_pos = data.site_xpos[self.obj_sid].ravel()
        desired_config = data.site_xmat[self.target_obj_sid].ravel()
        palm_pos = data.site_xpos[self.palm_sid].ravel()

        orien = np.linalg.norm(obj_config - desired_config)
        dist = np.linalg.norm(palm_pos - obj_pos)

        vel = data.qvel[:]
        if terminal:
            return 100.0 * dist + .01 * np.linalg.norm(vel)
        else:
            return 200.0 * dist + 10.0 * orien + 0.01 * np.linalg.norm(vel)

    def get_obs(self, data):
        return data.sensordata[:]
        # qp = data.qpos.ravel()
        # qvel = data.qvel.ravel()
        # obj_pos = data.site_xpos[self.obj_sid].ravel()
        # obj_orien = data.site_xmat[self.obj_sid].ravel()
        #
        # obj_vel = data.qvel[-6:].ravel()
        #
        # desired_orien = data.site_xmat[self.target_obj_sid].ravel()
        # palm_pos = data.site_xpos[self.palm_sid].ravel()
        #
        #
        # return np.concatenate([qp[:-6], qvel[:-6], obj_pos, obj_vel, obj_orien, desired_orien,
        #                         obj_pos - palm_pos, obj_orien - desired_orien ])
