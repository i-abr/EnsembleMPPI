import torch


dtype = torch.float
# device = torch.device("cpu") ### this runs on the cpu
device = torch.device("cuda:0") ### this runs on the gpu

### N is the batch size; D_in is the input dimension
### H is the hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

### Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

### initialize the weights randomly
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    ## forward pass: compute prediction on y
    h = x.mm(w1) ### I think this means multipy
    h_relu = h.clamp(min=0) ### clamp the bottom part to 0
    y_pred = h_relu.mm(w2) ### multiply by the output weights

    ## find out the loss
    loss = (y_pred - y).pow(2).sum().item() ### item gets the single number in the scalar array

    print(loss)
    
    ## ok now back prob
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred) ### the .t() is transpose
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone() ### just clones the data
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)


    ## update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
