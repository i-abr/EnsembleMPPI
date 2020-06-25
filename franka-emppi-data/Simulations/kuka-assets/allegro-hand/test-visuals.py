#!/usr/bin/env python3

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer


model = load_model_from_path('allegro_hand_description_right.xml')
sim = MjSim(model)

viewer = MjViewer(sim)

t = 0
while True:
    t += 1
    eps = np.random.normal(0., 2.0, size=(len(sim.data.ctrl[:]),))
    sim.data.ctrl[:] = np.sin(t*0.01 + eps)
    sim.step()
    viewer.render()
