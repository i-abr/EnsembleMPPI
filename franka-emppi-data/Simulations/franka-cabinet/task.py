from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import numpy.random as npr


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

class Task(object):



    def __init__(self, model):

        self.stage = 'one'

        self.handle_bid = model.body_name2id('handle')
        # self.handle_sid = model.site_name2id('Handle')
        # self.handle_measurement_sid = model.site_name2id('measurement-spot')
        self.articulation_did = model.jnt_dofadr[model.joint_name2id('articulation')]
        self.grasp_sid = model.site_name2id('grasp')
        self.target_gripper_config = np.array([
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.]
        ]).ravel()


    def __call__(self, data, terminal_cost=False):
        jnt = data.get_joint_qpos('finger1') - data.get_joint_qpos('finger2')
        handle_pos = data.body_xpos[self.handle_bid].ravel()
        grasp_pos = data.site_xpos[self.grasp_sid].ravel()
        grasp_config = data.site_xmat[self.grasp_sid].ravel()
        velocity_loss = np.dot(data.qvel, data.qvel)

        grasp_err = np.linalg.norm(grasp_pos - handle_pos)
        robot_ang = data.qpos[1] * data.qpos[1]
        num_contacts = len(data.contact)
        grasp_config_err = np.linalg.norm(grasp_config - self.target_gripper_config)

        if terminal_cost:
            return 10.0 * np.sqrt(velocity_loss)

        if self.stage == 'one':
            loss = 0. * robot_ang + 10.0 * grasp_config_err \
                    + 80.0 * grasp_err\
                    + 0.01 * np.sqrt(velocity_loss)

            loss += 100.0 * num_contacts
            loss -= 100.0 * np.clip(jnt, 0., 0.06)
            return loss
        else:
            loss = 500.0 * handle_pos[0] \
                    + 100.0 * np.clip(jnt, 0., 0.06) + 100.0* grasp_err + 100. * grasp_config_err
            return loss

    def update_stage(self, data):
        jnt = data.get_joint_qpos('finger1') - data.get_joint_qpos('finger2')
        handle_pos = data.body_xpos[self.handle_bid].ravel()
        grasp_pos = data.site_xpos[self.grasp_sid].ravel()
        grasp_config = data.site_xmat[self.grasp_sid].ravel()
        velocity_loss = np.dot(data.qvel, data.qvel)

        grasp_err = np.linalg.norm(grasp_pos - handle_pos)
        if grasp_err < 0.05:
            self.stage = 'two'

    def randomize_param(self, models):

        mean_jnt_axis = models[0].jnt_axis[self.articulation_did].ravel()
        mean_jnt_pos = models[0].jnt_pos[self.articulation_did].ravel()
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
            _model.jnt_axis[self.articulation_did][:] = ax.copy()
            if np.argmax(ax) == 1 or np.argmax(ax) == 2:
                _model.jnt_type[self.articulation_did] = 3
            elif np.argmax(ax) == 0:
                _model.jnt_type[self.articulation_did] = 2
            # _model.jnt_pos[drawer_hinge_did][0] = 0.
            # _model.jnt_pos[drawer_hinge_did][1] = npr.uniform(-0.2, 0.2)
            # _model.jnt_pos[drawer_hinge_did][2] = npr.uniform(-0.05, 0.05)

    def resample(self, sims, probs):
        mean_jnt_axis = 0.0
        mean_jnt_pos = 0.0

        for sim, prob in zip(sims, probs):
            mean_jnt_axis += sim.model.jnt_axis[self.articulation_did].ravel() * prob
            mean_jnt_pos += sim.model.jnt_pos[self.articulation_did].ravel() * prob

        mean_jnt_axis /= np.linalg.norm(mean_jnt_axis)
        print(mean_jnt_pos, mean_jnt_axis)


        for sim in sims:
            ax = np.random.normal(mean_jnt_axis, 0.01)
            ax /= np.linalg.norm(ax)
            pos = np.random.normal(mean_jnt_pos, 0.01)

            sim.model.jnt_axis[self.articulation_did][:] = ax
            sim.model.jnt_pos[self.articulation_did][:] = pos

            if np.argmax(ax) == 1 or np.argmax(ax) == 2:
                sim.model.jnt_type[self.articulation_did] = 3
            elif np.argmax(ax) == 0:
                sim.model.jnt_type[self.articulation_did] = 2
