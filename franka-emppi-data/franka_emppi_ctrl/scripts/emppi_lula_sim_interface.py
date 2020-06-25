#!/usr/bin/env python

import rospy
import numpy as np
from lula_franka.franka import Franka
from franka_mujoco_sim.msg import JointCommand
from lula_ros_util_internal.msg import RosVector
from geometry_msgs.msg import Pose

def jnt_command_callback(data):
    global cmd
    cmd = data.values[:]
    cmd = list(cmd)
    cmd[-1] = cmd[-1] + 0.78539816339 ### this reshifts the joint sent from sim

if __name__=='__main__':
    rospy.init_node('interface_code')
    fake_handle_pose_pub = rospy.Publisher('/env/handle', Pose, queue_size=1)
    rospy.Subscriber('/emppi/joint/command', RosVector, jnt_command_callback)

    _handle_pos = [0.7,0.0,0.6]
    _handle_quat = [0, 0, 0, 1]
    handle_pose = Pose()
    handle_pose.position.x = _handle_pos[0]
    handle_pose.position.y = _handle_pos[1]
    handle_pose.position.z = _handle_pos[2]
    handle_pose.orientation.x = _handle_quat[0]
    handle_pose.orientation.y = _handle_quat[1]
    handle_pose.orientation.z = _handle_quat[2]
    handle_pose.orientation.w = _handle_quat[3]
    fake_handle_pose_pub.publish(handle_pose)


    rate = rospy.Rate(40)

    robot = Franka(False)
    target_joints = [-0.010528470875401246, -0.4039533993954269, 0.00967197220002067,
                    -2.810112469210358, 2.002982503168678603, 3.034502274857627,  0.7620898691520851]
    target_joints[-1] -= 1.51
    robot.end_effector.go_config(target_joints, wait_for_target=False)
    rospy.sleep(4.)
    cmd = target_joints

    robot.set_speed('med') #f.set_speed('slow')
    while not rospy.is_shutdown():
        fake_handle_pose_pub.publish(handle_pose)

        # target_joints = [alpha * t_jnt + _cmd * (1-alpha)
        #                 for t_jnt,_cmd in zip(target_joints, cmd)]

        # target_joints = [t_jnt + _cmd * dt
        #                 for t_jnt,_cmd in zip(target_joints, cmd)]
        robot.end_effector.go_config(cmd, wait_for_target=False)
        print('+++---')
        rate.sleep()
    print('done')
