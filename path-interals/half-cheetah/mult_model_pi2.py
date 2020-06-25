import numpy as np
from numpy import exp
import numpy.random as npr
import time
from mujoco_py import MjSim, MjSimPool, load_model_from_path
from scipy.stats import multivariate_normal


def policy(param, obs):
    x = obs
    for (W, b) in param[:-1]:
        x = np.dot(x, W) + b
        # x = np.maximum(0, x)
        x = np.sin(x)
    W, b = param[-1]
    return np.dot(x, W) + b

def gen_rand_perturb(layer_sizes, noise=0.01):
    return [[noise*npr.randn(insize, outsize), noise*npr.randn(outsize)]
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

class MultModelPI2(object):

    ### Now I am going to pass in multiple models to see what happenes (need to speed this up)
    def __init__(self, models, task, terminal_cost,
                        frame_skip=1, horizon=10, num_trajectories=5, noise=0.2, lam=10.0,
                        default_hidden_layers=[]):

        ### setup the simulator env
        self.models = models
        sim_pool = [] ## need to make a seperate list

        for i, model in enumerate(self.models): ### need to loop through each model now
            # self.model_probs[i*no_trajectories:i*no_trajectories+no_trajectories] = model_probs[i]
            for _ in range(num_trajectories):
                sim_pool.append(MjSim(model))

        ## need to pull out actuator mean for the ranges (I think this matter)
        self.act_mid = np.mean(self.models[0].actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.models[0].actuator_ctrlrange[:,1]-self.models[0].actuator_ctrlrange[:,0])


        self.model_probs = np.ones(len(sim_pool))
        self.model_probs /= np.sum(self.model_probs)

        self.pool = MjSimPool(sim_pool) ### simulators that are to be used

        self.frame_skip = frame_skip

        ## also need to add the task into the system
        self.task = task
        self.terminal_cost = terminal_cost

        self.data = [sim.data for sim in self.pool.sims] ### reference to the read-only data wrapper

        ### hopefully the number of states and actions are not changeing across each model
        self.num_states = self.models[0].nq + self.models[0].nv
        self.num_actions = self.models[0].nu
        self.num_sensors = len(self.data[0].sensordata[:]) # TODO: this does not correspond
        ### setup the parameters
        self.horizon                        = horizon
        self.num_trajectories_per_model     = num_trajectories
        self.num_tot_trajectories           = num_trajectories * len(self.models)
        self.noise                          = noise
        self.inv_sigma                      = np.eye(self.num_actions)/self.noise
        self.lam                            = lam

        # self.sk = np.zeros((self.num_tot_trajectories, self.horizon)) ### entropy of the trajectory
        # self.delta = np.zeros((self.num_tot_trajectories, self.horizon, self.num_actions))
        ### mean actions to be shifted as each action is computed and returned
        # self.mean_actions = np.random.normal(0., self.noise, size=(self.horizon, self.num_actions))
        self.sk = np.zeros(self.num_tot_trajectories)
        self.layer_sizes = [self.num_sensors] + default_hidden_layers + [self.num_actions]
        


        self.mean_param = gen_rand_perturb(self.layer_sizes, noise=self.noise)
        # npr.normal(0., self.noise, size=(self.num_actions, self.num_sensors))
        # self.mean_bias = npr.normal(0., self.noise, size=(self.num_actions,))
        #
        # self.delta = [gen_rand_perturb(self.mean_params) for _ in range(self.num_tot_trajectories)]

        self.iter = 0

        self.prev_sk = np.zeros(self.num_tot_trajectories)

    def scale_ctrl(self, ctrl):
        ## first clip
        ctrl = np.clip(ctrl, -1.0, 1.0)
        ## now scale
        ctrl = self.act_mid + ctrl * self.act_rng
        return ctrl


    def __call__(self, state, obs, predict_measurements=False, just_run_policy=False): ### call to compute the control and return it



        if just_run_policy:
            return policy(self.mean_param, obs)

        ### pre-shift the mean actions -- this does nothing in the beginning
        # self.mean_actions[:-1] = self.mean_actions[1:]
        # self.mean_actions[-1] = np.random.normal(0., self.noise, size=(self.num_actions,))

        self.sk = np.zeros(self.num_tot_trajectories)

        for sim in self.pool.sims: ### reset each of the simulators initial state
            sim.set_state(state)

        # self.delta = np.random.normal(0., self.noise, size=self.delta.shape) ### create the random action perturbation
        self.delta = [gen_rand_perturb(self.layer_sizes, noise=self.noise)
                        for _ in range(self.num_tot_trajectories)]
        for t in range(self.horizon): ### for each time step

            for k, data in enumerate(self.data): ### do each parameter fixing

                delta_param = [[W1+W2, b1+b2]
                    for (W1, b1), (W2, b2) in zip(self.mean_param, self.delta[k])]

                _ctrl = policy(delta_param, data.sensordata[:])

                data.ctrl[:] = self.scale_ctrl(_ctrl.copy())
                self.sk[k] += self.task(data, _ctrl)*t #\
                                # + self.lam * np.dot(_ctrl, self.inv_sigma).dot(_ctrl)
                # if t == self.horizon-1:
                    # self.sk[k] += self.terminal_cost(data)

            for _ in range(self.frame_skip):
                self.pool.step() ### step the simulator


        self.sk -= np.min(self.sk)
        w = self.model_probs * exp(- self.sk / self.lam) + 1e-4
        w /= np.sum(w)
        for k in range(self.num_tot_trajectories):
            self.mean_param = [[W1+w[k]*W2, b1+w[k]*b2]
                    for (W1, b1), (W2, b2) in zip(self.mean_param, self.delta[k])]

        # min_noise = 0.0
        # for k in range(self.num_tot_trajectories):

        self.iter += 1


        ctrl = policy(self.mean_param, obs)
        ### I don't need to copy anything because the ctrl action would have been applied
        return self.scale_ctrl(ctrl)

    def update_distribution(self, true_measurements, state, ctrl):

        for sim in self.pool.sims:
            sim.set_state(state)
            sim.data.ctrl[:] = ctrl

        for _ in range(self.frame_skip):
            self.pool.step()

        for i, (data, model_prob) in enumerate(zip(self.data, self.model_probs)):
            simulated_measurements = data.sensordata[:]
            diff = true_measurements - simulated_measurements
            likelihood = np.prod(multivariate_normal.pdf(diff, 0., 0.1))
            self.model_probs[i] = model_prob * likelihood

        self.model_probs += 1e-10
        self.model_probs /= np.sum(self.model_probs)
        '''
        if predict_measurements:
            sensordata = []
            for sim in self.pool.sims:
                sim.set_state(state)
                sim.data.ctrl[:] = ctrl
            for _ in range(self.frame_skip):
                self.pool.step()

            for i, sim in enumerate(self.pool.sims):
                sensordata.append(sim.data.sensordata[:].copy())
            return ctrl, sensordata
        else:
            return ctrl
        '''

    # def single_step_measurement(self, state, ctrl):
    #
    #     for sim in self.pool.sims:
    #         sim.set_state(state)
    #         sim.data.ctrl[:] = ctrl
    #
    #     for _ in range(self.frame_skip):
    #         self.pool.step()
    #
    #     for k, data in enumerate(self.data):
    #         self.sensordata_hyp[k, :] = data.sensordata[:]
