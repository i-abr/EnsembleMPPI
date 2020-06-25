#!/usr/bin/env python

'''
    Author : Ian Abraham 
    Script is written to relay information from dart + franka robot 
    to a path integral controller running on a seperate computer.
'''


# Command line: franka switch_base --base=00_base_link  (for right side)
# frame_id = <handle_frame_name>
# world.get_transform_matrix(frame_id, wait_until_available = True)
# import os
# import signal
# os.kill(os.getpid(), signal.SIGKILL)

import rospy
import numpy as np
from lula_franka.franka import Franka
from lula_control.world import make_basic_world
from lula_control.object import ControllableObject
from lula_ros_util_internal.msg import RosVector
from quatmath import mat2quat
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
import os
import signal
import traceback



def jnt_command_callback(data):
    global cmd 
    cmd = data.values[:]
    cmd = list(cmd)
    cmd[-1] = cmd[-1] + 0.78539816339


def main():
    global cmd
    rospy.init_node('interface_code')

    print('linking with franka')
    robot = Franka(is_physical_robot=True)

    print('world client')
    world = make_basic_world()
    world.wait_for_objs()

    #TODO: make sure that I set the name
    handle_name = 'chewie_door_right_handle'#'indigo_drawer_handle_top'
    drawer_name = 'chewie_door_right_link'#'indigo_drawer_top'
    # drawer = ControllableObject(world.get_object(drawer_name), robot=robot)

    ### TODO: uncomment when I am sure I am reading in the right things
    handle_pose_pub = rospy.Publisher('/env/handle', Pose, queue_size=1)
    rospy.Subscriber('/emppi/joint/command', RosVector, jnt_command_callback)
    rospy.Subscriber('/emppi/gripper', Bool, gripper_command_callback)
    _handle_pos = [0.6,0.0,0.8]
    _handle_quat = [0, 0, 0, 1]
    handle_pose = Pose()
    handle_pose.position.x = _handle_pos[0]
    handle_pose.position.y = _handle_pos[1]
    handle_pose.position.z = _handle_pos[2]
    handle_pose.orientation.x = _handle_quat[0]
    handle_pose.orientation.y = _handle_quat[1]
    handle_pose.orientation.z = _handle_quat[2]
    handle_pose.orientation.w = _handle_quat[3]
    handle_pose_pub.publish(handle_pose) ## true handle pose is handle.pose


    #cmd = [-0.010528470875401246, -0.4039533993954269, 0.00967197220002067,
    #        -2.810112469210358, -0.002982503168678603, 3.034502274857627,  0.7620898691520851]
    
    cmd = [0.13899624057616405, -1.2389129968136638, 0.206734661129483, 
            -2.1975005022350103, -0.6279011405926266, 2.126850206662556, 0.18986900396002324]
    
    rate_hz = 40
    robot.set_speed('slow') ## TODO: change when running the real experiment
    rate = rospy.Rate(rate_hz)
    robot.end_effector.gripper.open()
    robot.end_effector.go_config(cmd, wait_for_target=False)
    rospy.sleep(4.)
    robot.set_speed('med')
    
    # drawer.suppress()
    while not rospy.is_shutdown():
        handle_tf = world.get_transform_matrix(handle_name, wait_until_available = True)
        # print(handle_tf)
        _handle_quat = mat2quat(handle_tf[:3,:3])
        _handle_pos = handle_tf[:3,-1]
        '''
        handle_pose = Pose()
        handle_pose.position.x = _handle_pos[0]
        handle_pose.position.y = _handle_pos[1]
        handle_pose.position.z = _handle_pos[2]
        handle_pose.orientation.x = _handle_quat[0]
        handle_pose.orientation.y = _handle_quat[1]
        handle_pose.orientation.z = _handle_quat[2]
        handle_pose.orientation.w = _handle_quat[3]
        '''
        handle_pose_pub.publish(handle_pose) ## true handle pose is handle.pose
        robot.end_effector.go_config(cmd, wait_for_target=False)
        
        print('-=-=-=-=-=')
        rate.sleep()



if __name__=='__main__':
    try:
        main()
    except Exception as e: # something is hanging up and this is needed 
        print(e)
        traceback.print_exc()
    finally:
        os.kill(os.getpid(), signal.SIGKILL)
