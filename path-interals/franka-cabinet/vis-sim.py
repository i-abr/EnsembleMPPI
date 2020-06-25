#!/usr/bin/env python3

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer

model_path = 'assets/franka-door.xml'
model = load_model_from_path(model_path)
sim = MjSim(model)
viewer = MjViewer(sim)

door_bid = model.body_name2id('Door')

t_model_path = 'assets/franka-cabinet.xml'
t_model = load_model_from_path(t_model_path)
t_sim = MjSim(t_model)
t_viewer = MjViewer(t_sim)

handle_sid = t_model.site_name2id('Handle')

while True:

    sim.data.ctrl[:] = np.random.normal(0., 0.1, size=(sim.model.nu,))
    sim.step()
    viewer.render()

    t_sim.data.ctrl[:] = np.random.normal(0., 0.1, size=(sim.model.nu,))

    t_sim.step()
    t_viewer.render()
