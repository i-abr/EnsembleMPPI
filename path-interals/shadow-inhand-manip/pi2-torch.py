import pickle
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjSimPool, MjViewer
import os
from quatmath import quat2euler, euler2quat
import torch
import time
from task import Task

torch.set_grad_enabled(False)
## --- first I am going to initialize the device code
dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0")

#### CONST PARAMETERS ----
frame_skip = 5
final_time = 200
max_iter = 400
num_models = 1
lam = 0.01
num_samples_per_model = 40
num_tot_trajectories = num_models * num_samples_per_model

### --- create the mujoco models
model_path = '../shadow-hand-assets/hand/manipulate_block.xml'
model_pool = []
sim_pool = []
for _ in range(num_tot_trajectories):
    model_pool.append(load_model_from_path(model_path))
    sim_pool.append(MjSim(model_pool[-1]))
sim_pool = MjSimPool(sim_pool) ### -- simulator pool
data_pool = [sim.data for sim in sim_pool.sims]

model = load_model_from_path(model_path) ###--- model we will use for reference
sim = MjSim(model) ### ---- sim we will use for esting
viewer = MjViewer(sim)
task = Task(model)

act_mid = np.mean(model.actuator_ctrlrange, axis=1)
act_rng = 0.5 * (model.actuator_ctrlrange[:,1]-model.actuator_ctrlrange[:,0])

def scale_ctrl(_ctrl):
    ## first clip
    _ctrl = np.clip(_ctrl, -1.0, 1.0)
    ## now scale
    _ctrl = act_mid + _ctrl * act_rng
    return _ctrl
num_obs = len(task.get_obs(sim.data)) ### need to do this because of stupid reasons
num_ctrl = sim.model.nu

noise = 0.1

sk = torch.zeros(num_tot_trajectories, dtype=dtype, device=device)
obs = torch.zeros(num_obs, dtype=dtype, device=device) ### need to preallocate this

class Sin(torch.nn.Module):
    def __init__(self):
        super(Sin, self).__init__()
    def forward(self, x):
        return torch.sin(x)

policies = []
for _ in range(num_tot_trajectories):
    policy = torch.nn.Sequential(
                torch.nn.Linear(num_obs, 200),
                Sin(),
                torch.nn.Linear(200, num_ctrl)
            )
    policy.to(device=device)
    policies.append(policy) ### --- it is more efficient to just set each policy at the begining and update

mean_param = [param.data.clone() for param in policies[0].parameters()]
delta_weights = [
            [torch.randn(param.shape, device=device, dtype=dtype).mul(noise) for param in policies[0].parameters()]
            for _ in range(num_tot_trajectories)
        ]

omega = torch.ones(num_tot_trajectories, dtype=dtype, device=device)
omega /= torch.sum(omega)

for i in range(max_iter):
    sk.mul(0.)
    sim.reset()
    sim_pool.reset()
    # desired_orien = np.zeros(3)
    # desired_orien[0] = np.random.uniform(low=-2.0, high=2.0)
    # desired_orien[1] = np.random.uniform(low=-2.0, high=2.0)
    # desired_orien[2] = np.random.uniform(low=-2.0, high=2.0)
    # sim.model.body_quat[task.target_obj_bid] = euler2quat(desired_orien)
    # sim.forward()
    # for _sim in sim_pool.sims:
    #     _sim.reset()
    #     _sim.model.body_quat[task.target_obj_bid] = euler2quat(desired_orien)
    #     _sim.forward()

    print('iter {}.... updating candidate policies'.format(i))
    start_time = time.time()
    for k in range(num_tot_trajectories):
        for i, (param, m_param) in enumerate(zip(policies[k].parameters(), mean_param)):
            d_param = torch.randn(param.shape, device=device, dtype=dtype).mul(noise)
            delta_weights[k][i] = d_param
            param.data = m_param + d_param

    end_time = time.time()
    print('updating policies took : {} s'.format(end_time - start_time))

    start_time = time.time()
    print('testing policies in sim pool')
    for t in range(final_time):

        for k, data in enumerate(data_pool):
            obs[:] = torch.from_numpy(task.get_obs(data))
            ctrl = policies[k](obs)
            data.ctrl[:] = scale_ctrl(ctrl.cpu().detach().numpy()) ### holy fuck
            sk[k] += task(data)

            if t == final_time-1:
                sk[k] += task(data, terminal=True)

        for _ in range(frame_skip):
            sim_pool.step()
    end_time = time.time()
    print('testing took {}'.format(end_time - start_time))

    print('updating mean parameters of policy ')
    sk -= sk.min()
    omega = torch.exp(-sk / lam).add(1e-4)
    omega /= torch.sum(omega)
    ### update mean parameters
    for k in range(num_tot_trajectories):
        for i, d_param in enumerate(delta_weights[k]):
            mean_param[i] += d_param.mul(omega[k])

    print('testing the policy out ')
    # for param in policies[0].parameters():
    #     print(param.data)
    for param, m_param in zip(policies[0].parameters(), mean_param):
        param.data = m_param.clone()
    # for param in policies[0].parameters():
    #     print(param.data)

    cost = 0.0
    for t in range(final_time):
        obs[:] = torch.from_numpy(task.get_obs(sim.data))
        ctrl = policies[0](obs)
        sim.data.ctrl[:] = scale_ctrl(ctrl.cpu().detach().numpy()) ### holy fuck
        cost += task(sim.data)
        if t == final_time -1:
            cost += task(sim.data, terminal=True)
        for _ in range(frame_skip):
            sim.step()
        viewer.render()
    print('Cost : ', cost)
