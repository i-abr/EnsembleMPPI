#!/usr/bin/env python3
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc import flatten
from random import choice
import pickle

def gen_rand_perturb(layer_sizes, noise=0.1):
    return [[noise*npr.randn(insize, outsize), noise*npr.randn(outsize)]
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def policy(param, obs):
    x = obs
    for (W, b) in param[:-1]:
        x = np.dot(x, W) + b
        # x = np.maximum(0, x)
        x = np.sin(x)
    W, b = param[-1]
    return np.dot(x, W) + b

def max_likelihood(param, unflatten, policy, obs_action_pair, noise=0.8):
    obs, action = obs_action_pair
    predicted_action = policy(unflatten(param), obs)
    return np.sum(norm.logpdf(predicted_action, action, noise))
    # return -np.linalg.norm(predicted_action-action)

grad_max_likelihood = grad(max_likelihood)


if __name__ == '__main__':
    demos = pickle.load(open('obs-act-data.pickle', 'rb'))
    obs, ctrl = demos[0]
    num_obs = len(obs)
    num_ctrl = len(ctrl)
    print(len(demos), num_obs, num_ctrl)

    layer_sizes = [num_obs] + [46,256,128] + [num_ctrl]
    # param = gen_rand_perturb(layer_sizes)
    param = [[0.1*npr.randn(insize, outsize), 0*npr.randn(outsize)]
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]
    param, unflatten = flatten(param)

    batch_size = 100
    num_iters = 20
    max_iter = 1000
    b1=0.9
    b2=0.999
    eps=1e-8
    for k in range(max_iter):
        randomly_selected_data = [choice(demos) for _ in range(batch_size)]
        gradient = np.zeros(param.shape)
        _m = np.zeros(param.shape)
        _v = np.zeros(param.shape)
        _error = 0.0
        for i in range(num_iters):
            for data in randomly_selected_data:
                gradient += grad_max_likelihood(param, unflatten, policy, data)
                _error += max_likelihood(param, unflatten, policy, data)
            gradient /= batch_size
            _error /= batch_size
            _m = (1-b1) * gradient + b1 * _m
            _v = (1-b2) * (gradient**2) + b2*_v
            mhat = _m/(1 - b1**(i+1))
            vhat = _v/(1 - b2**(i+1))
            # param += 0.0001 * gradient * (0.999**(k+1))
            param += 0.0001 * mhat / (np.sqrt(vhat) + eps)
            # param += npr.normal(0., 0.001, size=param.shape)
            print('Iteration : {}, Error : {}'.format(k, _error))
        if k % 5 == 0:
            file_pi = open('bc-policy.pickle', 'wb')
            pickle.dump(unflatten(param), file_pi)
            print('Saved Policy')
