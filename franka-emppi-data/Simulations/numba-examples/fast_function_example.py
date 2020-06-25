from numba import jit
import numpy as np

x = np.arange(100).reshape(10,10)


@jit(nopython=True) ## this makes it go fasterr
def go_fast(a): ## numba compiles function to code at the first instance
    trace = 0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i,i])

    return a + trace ## return the trace of matrix

print(go_fast(x))
