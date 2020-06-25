import numpy as np
import matplotlib.pyplot as plt
import pickle


data = pickle.load(open('door-hinge-data.pickle', 'rb'))

hinge_poses = data['hinge_poses']

hinge_poses = np.array(hinge_poses)
hinge_probs = np.array(data['hinge_probs'])

print(hinge_poses.shape, hinge_probs.shape)

mean_hinge_poses = []
var_hinge_poses = []
for i in range(len(hinge_poses)):
    mean_hinge_poses.append(
                   np.dot(hinge_probs[i] , hinge_poses[i,:])
                )

mean_hinge_poses = np.array(mean_hinge_poses)
std_hinge_poses = np.std(hinge_poses, axis=1)


time = [i for i in range(len(mean_hinge_poses))]
for m_hinge_pos, std_hinge_pos in zip(mean_hinge_poses.T, std_hinge_poses.T):
    plt.plot(time, m_hinge_pos)
    plt.fill_between(time, m_hinge_pos-std_hinge_pos, m_hinge_pos+std_hinge_pos, alpha=0.6)

target_pos = [0.2 for i in range(len(mean_hinge_poses))]
plt.plot(time, target_pos, ls='--')


plt.show()
