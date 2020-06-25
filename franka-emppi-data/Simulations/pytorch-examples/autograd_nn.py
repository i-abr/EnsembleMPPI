import torch
import matplotlib.pyplot as plt

torch.manual_seed(10)
torch.cuda.manual_seed(10)

dtype = torch.float
device = torch.device("cpu") ### this runs on the cpu
#device = torch.device("cuda:0") ### this runs on the gpu

### N is the batch size; D_in is the input dimension
### H is the hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

### Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

### initialize the weights randomly
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

loss_dump = []

import time

learning_rate = 1e-6
for t in range(500):
    start = time.time()

    ## forward pass: compute prediction on y
    ## you can do this is one line apparently
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    ## find out the loss
    loss = (y_pred - y).pow(2).sum() ### item gets the single number in the scalar array

    loss_dump.append(loss.item())

    print(t, loss.item())


    ### now we use autograd to compute the backward pass
    ### now w1.grad and w2.grad will be tensors that hold the gradient values
    ### of the loss wrt w1 and w2 respectively
    loss.backward()

    ## update the weights
    with torch.no_grad(): ### this needs to be done so that we dont track the tensor update
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # manually zero gradients after update
        w1.grad.zero_()
        w2.grad.zero_()

    end = time.time()
    print('Elapsed time : {}'.format(end - start))

plt.plot(loss_dump)
plt.show()
