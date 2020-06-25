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
from lula_controller_msgs.msg import ForwardKinematics
from lula_ros_util_internal.msg import RosVector
from quatmath import mat2quat
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState 

import os
import signal
import traceback


### topci : /robot/kinematics/fk

### create a class object to handle the relays of messages from the franka to my comp

class CustomSignalRelay(object):

    def __init__(self, robot):
        self.robot = robot
        #self.cmd = [0.013899624057616405, -1.2389129968136638, 0.206734661129483,
        #        -2.1975005022350103, 0.6279011405926266, 2.126850206662556, 1.08986900396002324]
        self.default_cmd = [-0.3532721232154913, -1.3421308260131302, -0.025498443449520482, 
                        -2.633722574600121, 1.1662470502815794, 2.509950220098635, 1.1872866369788737]
        
        
        self.cmd = [-0.3532721232154913, -1.3421308260131302, -0.025498443449520482, 
                        -2.633722574600121, 1.1662470502815794, 2.509950220098635, 1.1872866369788737]
        
        self.ee_pos = np.array([0., 0., 0.])

        self.handle_pose_pub = rospy.Publisher('/env/handle', Pose, queue_size=1)
        self.phase_two_pub = rospy.Publisher('/phase_two_ago', Bool, queue_size=1)

        rospy.Subscriber('/emppi/joint/command', RosVector, self.jnt_command_callback)
        #rospy.Subscriber('/emppi/gripper', Bool, self.gripper_command_callback)
        #TODO: remember to remove this as we are now using the 
        # measured values to close the gripper 
        
        ### we are now using the fk from the measured robot as the criteria to close gripper 
        rospy.Subscriber('/robot/kinematics/fk', ForwardKinematics, self.get_fk)
        rospy.Subscriber('/tracker/sektion_cabinet/joint_states', JointState, self.articulation_callback)

    def articulation_callback(self, data):
        self.articulation_name = list(data.name)
        self.articulation_pose = list(data.position)

    def jnt_command_callback(self, data):
        self.cmd[:] = data.values[:]
        # cmd = list(cmd)
        self.cmd[-1] = self.cmd[-1] + 0.78539816339

    def get_fk(self, data):
        self.ee_pos[0] = data.orig[0]
        self.ee_pos[1] = data.orig[1]
        self.ee_pos[2] = data.orig[2]

    def gripper_command_callback(self, msg):
        if msg.data is True:
            self.robot.end_effector.gripper.close()
            rospy.sleep(2.0) ### sleep for two seconds
        if msg.data is False:
            self.robot.end_effector.gripper.open()
            rospy.sleep(2.0)

def main():
    rospy.init_node('interface_code')

    print('linking with franka')
    robot = Franka(is_physical_robot=True)

    print('world client')
    world = make_basic_world()
    world.wait_for_objs()

    #TODO: make sure that I set the name
    handle_name = 'drawer_handle_top_frame'#'chewie_door_right_handle'#'indigo_drawer_handle_top'
    #handle_name = 'drawer_handle_bottom_frame'#'chewie_door_right_handle'#'indigo_drawer_handle_top'
    drawer_name = 'chewie_door_right_link'#'indigo_drawer_top'
    # drawer = ControllableObject(world.get_object(drawer_name), robot=robot)
    print('creating relay')
    signal_relay = CustomSignalRelay(robot) ### this deals with sending messages from comp to robot

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
    signal_relay.handle_pose_pub.publish(handle_pose) ## true handle pose is handle.pose

    rate_hz = 20
    robot.set_speed('slow') ## TODO: change when running the real experiment
    rate = rospy.Rate(rate_hz)
    robot.end_effector.gripper.open()
    robot.end_effector.go_config(signal_relay.cmd, wait_for_target=False)
    rospy.sleep(4.)
    robot.set_speed('med')

    hold_for_n_cnts = 0
    cnt = 0
    call_once = True
    # drawer.suppress()
    while not rospy.is_shutdown():
        handle_tf = world.get_transform_matrix(handle_name, wait_until_available = True)
        # print(handle_tf)
        _handle_quat = mat2quat(handle_tf[:3,:3])
        _handle_pos = handle_tf[:3,-1]
        off_set = np.array([0.04, 0., 0.])
        gripper_err = np.linalg.norm(_handle_pos - signal_relay.ee_pos)
        print('gripper erro : ', gripper_err)
        if gripper_err < 0.03 and call_once:
            hold_for_n_cnts += 1
            if hold_for_n_cnts > 40:
                robot.end_effector.gripper.close()
                rospy.sleep(0.5)
                phase_two_ago = Bool()
                phase_two_ago.data = True
                signal_relay.phase_two_pub.publish(phase_two_ago)
                call_once = False


        handle_pose = Pose()
        handle_pose.position.x = _handle_pos[0]
        handle_pose.position.y = _handle_pos[1]
        handle_pose.position.z = _handle_pos[2]
        handle_pose.orientation.x = _handle_quat[0]
        handle_pose.orientation.y = _handle_quat[1]
        handle_pose.orientation.z = _handle_quat[2]
        handle_pose.orientation.w = _handle_quat[3]
        
        signal_relay.handle_pose_pub.publish(handle_pose) ## true handle pose is handle.pose
        
        
        if signal_relay.articulation_pose[-1] > 0.2:
            print('successful opening of drawer/door/cabinet')
            print("who\'s a good robot! Yes you are !")
            print('opening gripper and killing relay')
            robot.end_effector.gripper.open()
            rospy.sleep(0.5)
            break
        
        robot.end_effector.go_config(signal_relay.cmd, wait_for_target=False)
        cnt += 1
        print(cnt)
        
        print(signal_relay.articulation_name, signal_relay.articulation_pose)
        
        
        
        
        rate.sleep()



if __name__=='__main__':
    try:
        main()
    except Exception as e: # something is hanging up and this is needed
        print(e)
        traceback.print_exc()
    finally:
        os.kill(os.getpid(), signal.SIGKILL)

