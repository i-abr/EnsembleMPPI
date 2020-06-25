import numpy as np
from numpy import exp
import time
from mujoco_py import MjSim, MjSimPool, load_model_from_path 
from scipy.stats import multivariate_normal
from scipy.signal import savgol_filter


class MultModelMPPIMujoco(object):

    ### Now I am going to pass in multiple models to see what happenes (need to speed this up)
    def __init__(self, models, task, terminal_cost, frame_skip=1, horizon=10, no_trajectories=5, noise=0.2, lam=10.0):

        ### setup the simulator env
        self.models = models 
        sim_pool = [] ## need to make a seperate list

        model_probs = np.ones(len(models))
        model_probs /= np.sum(model_probs)
        self.model_probs = model_probs 

        for i, model in enumerate(self.models): ### need to loop through each model now 
            sim_pool.append(MjSim(model))
            #sim_pool = sim_pool + [MjSim(model) for _ in range(no_trajectories)] ### going to add the previous list to the new set of models
        self.model_probs /= np.sum(self.model_probs) 
        self.pool = MjSimPool(sim_pool, nsubsteps=frame_skip) ### simulators that are to be used
        
        ### need to grab the actuator scales
        self.act_mid = np.mean(self.models[0].actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.models[0].actuator_ctrlrange[:,1]-self.models[0].actuator_ctrlrange[:,0])

        self.frame_skip = frame_skip 

        ## also need to add the task into the system 
        self.task = task
        self.terminal_cost = terminal_cost 

        self.data = [sim.data for sim in self.pool.sims] ### reference to the read-only data wrapper
        
        ### hopefully the number of states and actions are not changeing across each model
        self.no_states = self.models[0].nq + self.models[0].nv
        self.no_actions = self.models[0].nu
        self.num_states = self.models[0].nq + self.models[0].nv
        self.num_actions = self.models[0].nu
        self.no_sensors = self.models[0].nsensor ### I dont think i really need this now 
    
        ### setup the parameters
        self.horizon = horizon 
        self.no_trajectories = no_trajectories 
        self.noise = noise
        self.inv_sigma = np.eye(self.no_actions)/self.noise
        self.lam = lam
        
        self.sk = np.zeros((self.no_trajectories, self.horizon)) ### entropy of the trajectory
        self.delta = np.zeros((self.no_trajectories, self.horizon, self.no_actions))        
        ### mean actions to be shifted as each action is computed and returned 
        self.mean_actions = np.random.normal(0., self.noise, size=(self.horizon, self.no_actions))
    
    def reset(self):
        self.mean_actions = np.random.normal(0., self.noise, size=self.mean_actions.shape)

    def scale_ctrl(self, ctrl):
        ctrl = np.clip(ctrl, -1.0, 1.0)
        ctrl = self.act_mid + ctrl * self.act_rng 
        return ctrl 

    def __call__(self, state, predict_measurements=False): ### call to compute the control and return it 

        self.mean_actions[:-1] = self.mean_actions[1:]
        self.mean_actions[-1] = 0*np.random.normal(0., self.noise, size=(self.no_actions,))
        
        for sim in self.pool.sims: ### reset each of the simulators initial state
            sim.set_state(state)
        
        self.delta = np.random.normal(0., self.noise, size=self.delta.shape)

        for t in range(self.horizon): ### for each time step
            for k, data in enumerate(self.data): ### do each parameter fixing    
                #self.delta[k, t, :] = (t+1) * np.random.normal(0., self.noise, size=(self.no_actions,))
                _ctrl = self.mean_actions[t] + self.delta[k, t, :]
                data.ctrl[:] = self.scale_ctrl(_ctrl.copy())
                self.sk[k,t] = self.task(data, _ctrl) + self.delta[k, t, :].dot(self.inv_sigma).dot(self.delta[k, t, :])
                if t == self.horizon-1:
                    self.sk[k, t] = self.terminal_cost(data)
            #for _ in range(self.frame_skip): ### simulate each parameter
            self.pool.step()

        ### reverse cumulative sum in time
        self.sk = np.cumsum(self.sk[:, ::-1],axis=1)[:,::-1] ### in theory this should do what I think
        for t in range(self.horizon):### loop through each action and do the following
            _beta = np.min(self.sk[:,t]) ### fine the min sk 
            self.sk[:,t] -= _beta ### sutract that out so you shift the weight
            _w = self.model_probs * exp(-self.sk[:,t]/self.lam) + 1e-5 ### this should be of size no_trajectories
            _w /= np.sum(_w) ### normalize the weights
            self.mean_actions[t] += np.dot(_w, self.delta[:,t,:])
        
        self.mean_actions = savgol_filter(self.mean_actions, len(self.mean_actions)-1, 3, axis=0)
        ctrl = self.mean_actions[0].copy()
        ctrl = self.scale_ctrl(ctrl) 
        if predict_measurements:
            return ctrl, self.predict_measurements(state, ctrl)
        else:
            return ctrl

    def predict_measurements(self, state, ctrl):
        sensordata = []
        for sim in self.pool.sims:
            sim.set_state(state)
            sim.data.ctrl[:] = ctrl

        #for _ in range(self.frame_skip):
        self.pool.step()


        for sim in self.pool.sims:
            sensordata.append(sim.data.sensordata[:].copy())

        return sensordata 












