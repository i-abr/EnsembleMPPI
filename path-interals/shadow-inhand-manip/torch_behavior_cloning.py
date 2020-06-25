
import pickle
import numpy as np
import torch
from torch.distributions import MultivariateNormal
dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0")

demos = pickle.load(open('./bc_data/obs-act-data.pickle', 'rb'))
obs, ctrl = demos[0]
batch_size, num_obs, num_ctrl = len(demos), obs.shape[0], ctrl.shape[0]
h1,h2 = 256, 128
h3 = 64
input_data = torch.randn(batch_size, num_obs, device=device, dtype=dtype)
output_data = torch.randn(batch_size, num_ctrl, device=device, dtype=dtype)

for i, (obs, ctrl) in enumerate(demos):
    input_data[i,:] = torch.from_numpy(obs)
    output_data[i,:] = torch.from_numpy(ctrl)

class Sin(torch.nn.Module):

    def __init__(self):
        super(Sin, self).__init__()
    def forward(self, x):
        return torch.sin(x)

policy = torch.nn.Sequential(
    torch.nn.Linear(num_obs, h1, bias=True),
    Sin(),
    torch.nn.Linear(h1, h2, bias=True),
    Sin(),
    # torch.nn.Linear(h2, h3, bias=True),
    # torch.nn.Tanh(),
    torch.nn.Linear(h2, num_ctrl, bias=True),
)
policy.to(device=device)
loss_fn = torch.nn.MSELoss(reduction='mean')
# loss_fn = torch.nn.L1Loss(reduction='mean')
# loss_fn = MultivariateNormal(torch.zeros(num_ctrl,device=device, dtype=dtype),
#                             torch.eye(num_ctrl, device=device, dtype=dtype).mul(1.0) ).log_prob
learning_rate = 1e-3
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=1e-8)
# optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate, momentum=0.9)
import matplotlib.pyplot as plt
error = []
for i in range(4000):

    y_pred = policy(input_data)
    # loss = -loss_fn(y_pred-output_data).sum()
    loss = loss_fn(y_pred, output_data)
    print(i, loss.item())
    error.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # plt.clf()
    # plt.plot(error)
    # plt.pause(0.0001)

torch.save(policy, 'torch_policy.pt')
