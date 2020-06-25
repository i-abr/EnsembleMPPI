import numpy as np

def task(data, action):### this function assumes that the input data is a numpy array

    rot = data.sensordata[0]
    x_vel = data.sensordata[13]
    z_vel = data.sensordata[-1]
    # return - 10.0*x_vel + (rot * rot) #+ 0.01 * z_vel**2 #+ np.dot(action, action) * 0.1

    return  10.0 * x_vel**2 + 60.0 * (rot+2*np.pi)**2 #+ 0.01 * z_vel**2
    #return -0*sensor_data[0] -20.*sensor_data[2] +  100.0*sensor_data[1]**2 + 0.001 * np.dot(action, action)

def terminal_cost():
    pass
