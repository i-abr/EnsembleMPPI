from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import numpy.random as npr
from quatmath import euler2mat

class Task(object):

    def __init__(self, model):

        self.stage = 'one'
        self.handle_bid = model.body_name2id('handle')
        self.articulation_did = model.jnt_dofadr[model.joint_name2id('articulation')]
        self.grasp_sid = model.site_name2id('grasp')
        self.target_gripper_config = np.array([
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.]
        ]).ravel()


    def get_rot_err(self, data):
        handle_mat = data.body_xmat[self.handle_bid].reshape((3,3))
        grasp_mat = data.site_xmat[self.grasp_sid].reshape((3,3))
        roty = euler2mat(np.array([0., np.pi/2, 0.])) 
        rotx = euler2mat(np.array([-np.pi, 0., 0.]))
        rotz = euler2mat(np.array([0., 0., -np.pi/2]))
        handle_mat = np.dot(handle_mat, roty.dot(rotx.dot(rotz)))
        tr_RtR = np.trace(grasp_mat.T.dot(handle_mat))
        _arc_c_arg = (tr_RtR - 1.0) / 2.0
        _th = np.arccos(_arc_c_arg)
        return _th 

    def __call__(self, data, terminal_cost=False):
        jnt = data.get_joint_qpos('finger1') - data.get_joint_qpos('finger2')
        handle_pos = data.body_xpos[self.handle_bid].ravel()
        grasp_pos = data.site_xpos[self.grasp_sid].ravel()
        grasp_config = data.site_xmat[self.grasp_sid].ravel()
        velocity_loss = np.dot(data.qvel, data.qvel)

        #grasp_err = np.linalg.norm(grasp_pos - handle_pos)
        grasp_err = grasp_pos - handle_pos
        grasp_err = np.dot(grasp_err, grasp_err)
        robot_ang = data.qpos[1] * data.qpos[1]
        num_contacts = len(data.contact)
        #grasp_config_err = np.linalg.norm(grasp_config - self.target_gripper_config)
        grasp_config_err = self.get_rot_err(data)**2

        if terminal_cost:
            return 0.0 * np.sqrt(velocity_loss) + 0.0 * grasp_config_err 

        if self.stage == 'one':
            loss = 0. * robot_ang + 200.0 * grasp_config_err \
                    + 200.0 * grasp_err\
                    + 0.01 * np.sqrt(velocity_loss)

            loss += .0 * num_contacts
            loss -= 10.0 * np.clip(jnt, 0., 0.06)
            if grasp_err < 0.03:
                loss -= 100.0
            return loss
        else:
            loss = 100.0 * handle_pos[0] \
                    + 40.0* grasp_err + 40. * grasp_config_err \
                    + 0.0 * np.clip(jnt, 0., 0.06)
            # if grasp_err < 0.02:
            #     loss -= 100.0
            return loss

    def update_stage(self, data):
        jnt = data.get_joint_qpos('finger1') - data.get_joint_qpos('finger2')
        handle_pos = data.body_xpos[self.handle_bid].ravel()
        grasp_pos = data.site_xpos[self.grasp_sid].ravel()
        grasp_config = data.site_xmat[self.grasp_sid].ravel()
        velocity_loss = np.dot(data.qvel, data.qvel)

        grasp_err = np.linalg.norm(grasp_pos - handle_pos)
        if grasp_err < 0.01:
            self.stage = 'two'

    def randomize_param(self, models, handle_pos):

        tot_models = len(models)
        for i,_model in enumerate(models):
            ax = np.random.normal(np.array([1.,0., 0.]), 0.6)
            # ax = np.random.uniform(0., 1.0, size=(3,))
            max_idx = np.argmax(np.abs(ax))
            _jnt = ax[max_idx]
            ax /= np.linalg.norm(ax)
            #ax = np.zeros(3)
            #ax[max_idx] = 1.0
            _model.jnt_axis[self.articulation_did][:] = ax.copy()
            if max_idx == 1 or max_idx == 2:
                _model.jnt_type[self.articulation_did] = 3
            elif max_idx == 0:
                print('THERE ARE THIS MANY PRISMATIC JOINTS')
                _model.jnt_type[self.articulation_did] = 2

            #_model.jnt_pos[self.articulation_did][:] = np.random.normal(handle_pos, 0.2)
            dim = np.array([0.1, 0.2, 0.05])
            #_model.jnt_pos[self.articulation_did][:] = np.random.uniform(handle_pos-0.1, handle_pos+0.1)
            _model.jnt_pos[self.articulation_did][:] = np.random.uniform(handle_pos-dim, handle_pos+dim)
            #_model.jnt_pos[self.articulation_did][0] = 0.0

    def get_mean_vals(self, sims, probs):
        mean_jnt_axis = 0.0
        mean_jnt_pos = 0.0
        N = len(sims)
        sampled_jnts = np.zeros((N, 3))
        sampled_poses = np.zeros((N, 3))
        for i, (sim, prob) in enumerate(zip(sims, probs)):
            sampled_jnts[i,:] = sim.model.jnt_axis[self.articulation_did].ravel()
            sampled_poses[i,:] = sim.model.jnt_pos[self.articulation_did].ravel()
            mean_jnt_axis += sim.model.jnt_axis[self.articulation_did].ravel() * prob
            mean_jnt_pos += sim.model.jnt_pos[self.articulation_did].ravel() * prob

        mean_jnt_axis /= np.linalg.norm(mean_jnt_axis)
        print(mean_jnt_pos, mean_jnt_axis)
        return sampled_jnts, sampled_poses, mean_jnt_axis, mean_jnt_pos


    def resample(self, sims, probs, mean_sim):
        mean_jnt_axis = 0.0
        mean_jnt_pos = 0.0

        for sim, prob in zip(sims, probs):
            mean_jnt_axis += sim.model.jnt_axis[self.articulation_did].ravel() * prob
            mean_jnt_pos += sim.model.jnt_pos[self.articulation_did].ravel() * prob

        mean_jnt_axis /= np.linalg.norm(mean_jnt_axis)

        # TODO: technically this is independent of the algorithm 
        # I am introducing a LOT of bias to the algorithm...
        #mean_sim.model.jnt_axis[self.articulation_did][:] = mean_jnt_axis[:]
        #mean_sim.model.jnt_pos[self.articulation_did][:] = mean_jnt_pos[:]


        if np.argmax(mean_jnt_axis) == 1 or np.argmax(mean_jnt_axis) == 2:
            #mean_sim.model.jnt_type[self.articulation_did] = 3
            print(mean_jnt_pos, mean_jnt_axis, 'rotational')
        elif np.argmax(mean_jnt_axis) == 0:
            #mean_sim.model.jnt_type[self.articulation_did] = 2
            print(mean_jnt_pos, mean_jnt_axis, 'prismatic')
        #print(mean_knt_pose, mean_jnt_axis)
        
        for sim in sims:
            ax = np.random.normal(mean_jnt_axis, 0.1)
            ax /= np.linalg.norm(ax)
            pos = np.random.normal(mean_jnt_pos, 0.01)

            sim.model.jnt_axis[self.articulation_did][:] = ax
            sim.model.jnt_pos[self.articulation_did][:] = pos

            if np.argmax(np.abs(ax)) == 1 or np.argmax(np.abs(ax)) == 2:
                sim.model.jnt_type[self.articulation_did] = 3
            elif np.argmax(np.abs(ax)) == 0:
                print('created sim with prismatic joint!!!!!')
                sim.model.jnt_type[self.articulation_did] = 2
