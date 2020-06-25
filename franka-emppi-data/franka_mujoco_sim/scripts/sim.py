#!/usr/bin/env python3

import os
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from quatmath import mat2quat
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from franka_mujoco_sim.msg import JointCommand

class Franka(object):

    def __init__(self):

        script_path = os.path.dirname(os.path.realpath(__file__))
        model_path = script_path + '/../assets/franka-cabinet.xml'
        self.model = load_model_from_path(model_path)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)

        self.handle_sid = self.model.site_name2id('handle')

        #############------------------- robot stuff
        self.jnt_names = ['panda_joint1', 'panda_joint2', 'panda_joint3',
                        'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7',
                        'panda_finger_joint1', 'panda_finger_joint2']
        rospy.Subscriber('/robot/joint/command', JointCommand, self.command_callback)
        self.jnt_state_pub = rospy.Publisher('/robot/joint/state', JointState, queue_size=1)
        self.jnt_state = JointState()
        self.jnt_state.name = self.jnt_names

        self.sim.data.qpos[0] = -0.3
        self.sim.data.qpos[1] = -1.
        self.sim.data.qpos[3] = -1.7
        self.sim.data.qpos[5] = 1.4

        # self.sim.data.ctrl[:] = np.zeros(self.model.nu)
        # self.sim.data.ctrl[0] = -0.3
        # self.sim.data.ctrl[1] = -1
        # self.sim.data.ctrl[3] = -1.7
        # self.sim.data.ctrl[5] = 1.4

        ##########------------------ handle stuff
        self.handle_pose = Pose()
        self.handle_pose_pub = rospy.Publisher('/env/handle', Pose, queue_size=1)


    def command_callback(self, data):
        ### this callback just sets the sim.data.ctrl[:] to
        ### the assigned ctrl value. the loop will run on 10 hz and update
        ### the step accordingly
        self.sim.data.ctrl[:] = data.ctrls[:]
    def update_sim(self):

        # for _ in range(5): ### frame skip
        self.sim.step()
        self.viewer.render()

        handle_pos = self.sim.data.site_xpos[self.handle_sid].ravel()
        handle_quat = mat2quat(self.sim.data.site_xmat[self.handle_sid].reshape((3,3)))
        self.handle_pose.position.x = handle_pos[0]
        self.handle_pose.position.y = handle_pos[1]
        self.handle_pose.position.z = handle_pos[2]
        self.handle_pose.orientation.x = handle_quat[0]
        self.handle_pose.orientation.y = handle_quat[1]
        self.handle_pose.orientation.z = handle_quat[2]
        self.handle_pose.orientation.w = handle_quat[3]
        self.handle_pose_pub.publish(self.handle_pose)

        self.jnt_state.header = Header()
        self.jnt_state.position = self.sim.data.qpos[:9]
        self.jnt_state.velocity = self.sim.data.qvel[:9]
        self.jnt_state_pub.publish(self.jnt_state)


def main():
    rospy.init_node('franka_mujoco_sim')
    rate = rospy.Rate(20)

    robot = Franka()

    #model = load_model_from_path(model_path)
    #sim = MjSim(model)
    #viewer = MjViewer(sim)

    #rate = rospy.Rate(100)
    #robot_pub = rospy.Publisher('/robot/state', JointState, queue_size=1)
    #jnt_cmd = np.array([0.1, 1.2, 1.2, -1.3, 1.4, 0.5, 0.2, 0.2])
    while not rospy.is_shutdown():
        #state = sim.get_state()
        #sim.data.ctrl[:] = jnt_cmd
        #sim.step()
        #viewer.render()
        robot.update_sim()
        rate.sleep()
if __name__ == '__main__':
    main()
