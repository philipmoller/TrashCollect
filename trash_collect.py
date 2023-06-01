from __future__ import print_function

import datetime
import argparse
import sys
import cv2
from cv2 import aruco
import numpy as np
import datetime
import time
from scipy import ndimage
import math
from matplotlib import pyplot as plt
import torch
import pandas as pd

from bosdyn.api import image_pb2
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2, manipulation_api_pb2
from bosdyn.client.frame_helpers import get_a_tform_b, get_vision_tform_body, BODY_FRAME_NAME, HAND_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, VISION_FRAME_NAME
from bosdyn.client import math_helpers

import bosdyn.client.lease
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api import arm_command_pb2, robot_command_pb2, synchronized_command_pb2, trajectory_pb2, geometry_pb2, gripper_command_pb2
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.client.lease import LeaseClient
from bosdyn.util import seconds_to_duration

from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf import wrappers_pb2


class TrashCollect():
    def __init__(self, argv):
        parser = argparse.ArgumentParser()
        bosdyn.client.util.add_base_arguments(parser)
        options = parser.parse_args(argv)

        bosdyn.client.util.setup_logging(options.verbose)

        self.sdk = bosdyn.client.create_standard_sdk('ArmTrajectory')
        self.robot = self.sdk.create_robot(options.hostname)
        bosdyn.client.util.authenticate(self.robot)
        self.robot.time_sync.wait_for_sync()

        assert not self.robot.is_estopped(), "Robot is estopped!"

        self.motion_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.manipulation_api_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)

        self.aruco_size = 0.02
        self.aruco_ids = [1, 2] # shelf and trolley IDs 
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_parameters =  aruco.DetectorParameters()
        self.aruco_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_parameters)

        # Maximum speeds.
        self._max_x_vel = 0.5
        self._max_y_vel = 0.5
        self._max_ang_vel = 1.0

        # Image callbacks for clicking
        self.g_image_click = None
        self.g_image_display = None

        # Buffers for previous images
        self.front_left_buffer = None
        self.front_right_buffer = None
        self.right_buffer = None
        self.left_buffer = None
        self.rear_buffer = None
        self.image_log_path = '/home/iemn/Desktop/folkemode/trash_collect/log/'

    #####################################################################
    ####################### UTILITY FUNCTIONS ###########################
    #####################################################################
    def init_spot(self):
        """
        Turn Spot motors on and stand up - blocks until finished
        """
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), "Robot power on failed."
        command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)

    def poweroff_spot(self):
        """
        Sit Spot down and turn motors off - blocks until finished
        """
        self.robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not self.robot.is_powered_on(), "Robot power off failed."

    def get_camera_intrinsics(self, image_response):
        ints = image_response.source.pinhole.intrinsics
        intr_mat = np.array([[ints.focal_length.x, ints.skew.x,         ints.principal_point.x],
                             [ints.skew.y,         ints.focal_length.y, ints.principal_point.y],
                             [0,                   0,                   1]])
        return intr_mat

    def get_body_tform_in_vision(self):
        body_tform_in_vision = get_vision_tform_body(self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
        return body_tform_in_vision

    def cv_mouse_callback(self, event, x, y, flags, param):
        clone = self.g_image_display.copy()
        if event == cv2.EVENT_LBUTTONUP:
            self.g_image_click = (x, y)
        else:
            # Draw some lines on the image.
            #print('mouse', x, y)
            color = (30, 30, 30)
            thickness = 2
            image_title = 'Click to grasp'
            height = clone.shape[0]
            width = clone.shape[1]
            cv2.line(clone, (0, y), (width, y), color, thickness)
            cv2.line(clone, (x, 0), (x, height), color, thickness)
            cv2.imshow(image_title, clone)
    
    #####################################################################
    ########################## PERCEPTION ###############################
    #####################################################################
    def locate_aruco_from_body(self, id, size_of_marker):
        # List of body camera references
        image_sources = ['frontleft_fisheye_image',
                         'frontright_fisheye_image',
                         'left_fisheye_image',
                         'right_fisheye_image',
                         'back_fisheye_image']

        tf_sources = ['frontleft_fisheye',
                      'frontright_fisheye',
                      'left_fisheye',
                      'right_fisheye',
                      'back_fisheye']

        # Grab images and their transforms
        images = []
        intrinsics = []
        transforms = []
        for source in image_sources:
            image_request = [(build_image_request(source, quality_percent=100, pixel_format=None))]
            image_response = self.image_client.get_image(image_request)
            image = np.frombuffer(image_response[0].shot.image.data, dtype=np.uint8)
            images.append(cv2.imdecode(image, -1))
            intrinsics.append(self.get_camera_intrinsics(image_response[0]))
            transforms.append(image_response[0].shot.transforms_snapshot)
        
        self.front_left_buffer = images[0]
        self.front_right_buffer = images[1]
        self.left_buffer = images[2]
        self.right_buffer = images[3]
        self.rear_buffer = images[4]

        
        # Check if requested ArUco marker ID is present in each image, return the first image its found in or return -1 if not found at all       
        for i in range(len(images)):
            #corners, ids, _ = aruco.detectMarkers(images[i], aruco_dict, parameters=parameters)
            corners, ids, _ = self.aruco_detector.detectMarkers(images[i])
            print(i, ":", ids)
            #try:
            if ids is None: # If no IDS are found
                pass
            else:
                if id in ids: # If ID is present, compute location
                    dist_coeffs = np.zeros((5,1))
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners=corners, markerLength=size_of_marker, cameraMatrix=intrinsics[i], distCoeffs=dist_coeffs)
                    rmats = []
                    if rvecs is None:
                        cv2.imshow('Failed ArUco reading', image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        raise RuntimeError('Error: ArUco marker not found!')
                    # Convert rotation vectors to matrices and draw the coordinate frame axis
                    for j in range(len(tvecs)):
                        rmat, _ = cv2.Rodrigues(rvecs[j])
                        # We only use 1 ArUco marker, so save the right id
                        if ids[j] == id:
                            rmats = rmat
                            tvecs = tvecs[j]
                    # Convert ArUco pose to SE3Pose
                    camera_tform_aruco = math_helpers.SE3Pose(x=tvecs[0][0]*10, y=tvecs[0][1]*10, z=tvecs[0][2]*10, rot=math_helpers.Quat.from_matrix(rmats))
                    # Get transformation from vision frame to camera frame
                    vision_tform_camera = get_a_tform_b(transforms[i], VISION_FRAME_NAME, tf_sources[i])
                    # Get transformation from vision frame to aruco marker
                    vision_tform_aruco = vision_tform_camera.mult(camera_tform_aruco)

                    return 1, vision_tform_aruco, image_sources[i]
                else:
                    pass
            #except Exception as e:
            #    print("Error:", e)
            #    return -1, None, None
        return -1, None, None

    #####################################################################
    ########################## NAVIGATION ###############################
    #####################################################################
    def move_to_NoWait(self, pose, dist_offset=0.5, side_offset=0.0, angle=0.0):
        """
        Function to move to vertically placed ArUco markers with a specific offset and angle

        Input
            pose: SE3Pose, defining the pose of a an object in the world frame
            offset: double, offset along z axis for approach point
            angle: double, body alignment to goal pose given in degrees
            thresholds: [double, double], precision thresholds for when to accept the movement as done, first entry is position, second is rotation
        """

        # Create an approach point by offsetting the goal point along the Z axis
        R = pose.rot.to_matrix()
        z_axis = R[:,2] # THIS IS 1M UNIT VECTOR
        x_axis = R[:,0] 
        pose_arr = np.array([pose.x, pose.y, pose.z])
        target = pose_arr + (dist_offset*z_axis) + (side_offset*x_axis)
        target_x = target[0]
        target_y = target[1]

        # Compute heading from approach pose to marker pose and add desired angle
        approach_to_marker = pose_arr[0:2] - target[0:2]
        angle_in_frame = math.atan2(approach_to_marker[1], approach_to_marker[0])
        heading = angle_in_frame + (angle*(math.pi/180))

        # Command the robot to go to the tag 
        speed_limit = geometry_pb2.SE2VelocityLimit(max_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=self._max_x_vel, y=self._max_y_vel), angular=self._max_ang_vel))
        mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit, locomotion_hint=spot_command_pb2.HINT_AUTO)
        tag_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(goal_x=target_x, 
                                                                           goal_y=target_y,
                                                                           goal_heading=heading, 
                                                                           frame_name=VISION_FRAME_NAME, 
                                                                           params=mobility_params,
                                                                           body_height=0.0, 
                                                                           locomotion_hint=spot_command_pb2.HINT_AUTO)
        end_time = 10.0 # this doesnt matter since we pause in the while loop
        # Issue the command to the robot
        self.command_client.robot_command(lease=None, 
                                          command=tag_cmd,
                                          end_time_secs=time.time() + end_time)
        return target, heading
    
    def move_to_aruco(self, ID, marker_size, dist_offset=1.0, side_offset=0.0, global_orientation=False, update_global=False, cutoff_time=5):
        """
        Makes Spot move to a shelf head-on with a given offset.
        Returns 1 if successful, returns 0 if failed.
        """
        # Find initial estimate of ArUco marker
        aruco = self.locate_aruco_from_body(ID, marker_size)
        
        if aruco[0] == -1:
            id_found = False
            iterations = 0
            while not id_found:
                self.rotate_body(45, 4)
                time.sleep(1.0)
                aruco = self.locate_aruco_from_body(ID, marker_size)
                if aruco[0] == -1:
                    iterations += 1
                    cv2.imwrite("FailToFind{}_FrontLeft_iter{}.png".format(ID, iterations), self.front_left_buffer)
                    cv2.imwrite("FailToFind{}_FrontRight_iter{}.png".format(ID, iterations), self.front_right_buffer)
                    cv2.imwrite("FailToFind{}_Left_iter{}.png".format(ID, iterations), self.left_buffer)
                    cv2.imwrite("FailToFind{}_Right_iter{}.png".format(ID, iterations), self.right_buffer)
                    cv2.imwrite("FailToFind{}_Rear_iter{}.png".format(ID, iterations), self.rear_buffer)
                    #print("ID search iterations: {}".format(iterations))
                else:
                    id_found = True
            
        ret_val, target, source = aruco
        
        if update_global:
            self.global_rot = target.rot

        if global_orientation:
            if self.global_rot == None:
                print("No prior global orientation stored in buffer!")
            else:    
                target.rot = self.global_rot

        target_pos, target_angle = self.move_to_NoWait(target, dist_offset, side_offset)

        moving = True
        start_t = time.time()
        while moving:
            current_pos = self.get_body_tform_in_vision()
            current_angle = current_pos.rot.to_yaw()
            x_diff = abs(target_pos[0] - current_pos.x)
            y_diff = abs(target_pos[1] - current_pos.y)
            rot_diff = abs(target_angle - current_angle)
            if x_diff < 0.075 and y_diff < 0.075 and (rot_diff < 0.075 or rot_diff > (math.pi*2)-0.075):
                moving = False
                break
            elif time.time()-start_t > cutoff_time:
                break

        self.gripper_search_height = target.z - current_pos.z
        return 1

    def move_simple(self, pose, end_time):
        """
        Moves to a SE3Pose with a given maximum time
        """
        target_x = pose.x
        target_y = pose.y
        heading = pose.rot.to_yaw()
        # Command the robot to go to the tag in kinematic odometry frame
        speed_limit = geometry_pb2.SE2VelocityLimit(max_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=self._max_x_vel, y=self._max_y_vel), angular=self._max_ang_vel))
        mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit, locomotion_hint=spot_command_pb2.HINT_AUTO)
        tag_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(goal_x=target_x, 
                                                                           goal_y=target_y,
                                                                           goal_heading=heading, 
                                                                           frame_name=VISION_FRAME_NAME, 
                                                                           params=mobility_params,
                                                                           body_height=0.0, 
                                                                           locomotion_hint=spot_command_pb2.HINT_AUTO)
        # Issue the command to the robot
        self.command_client.robot_command(lease=None, 
                                          command=tag_cmd,
                                          end_time_secs=time.time() + end_time)
        time.sleep(end_time)
        return

    def rotate_body(self, angle, end_time=2):
        """
        Function for turning the body in place
        """
        # Extract current pose
        original_pose = self.get_body_tform_in_vision()
        original_yaw = original_pose.rot.to_yaw()

        new_yaw = original_yaw + (angle*(math.pi/180))
        new_rot = math_helpers.Quat.from_yaw(new_yaw)
        new_pose = math_helpers.SE3Pose(x=original_pose.position.x, y=original_pose.position.y, z=original_pose.position.z, rot=new_rot)
        self.move_simple(new_pose, end_time=end_time)


    #####################################################################
    ######################### MANIPULATION ##############################
    #####################################################################
    def movel(self, pos, q, t, blocking=False):
        """
        Move end-effector linearly to a given Cartesian and quaternion position. Optional bool to force the trajectory to block the robot until complete.
        """
        self.robot.time_sync.wait_for_sync()

        target_ori = geometry_pb2.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
        target_pose = math_helpers.SE3Pose(x=pos[0], y=pos[1], z=pos[2], rot=target_ori)

        target_traj_point = trajectory_pb2.SE3TrajectoryPoint(pose=target_pose.to_proto(), time_since_reference=seconds_to_duration(t))
        target_traj = trajectory_pb2.SE3Trajectory(points=[target_traj_point])

        arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(pose_trajectory_in_task=target_traj, root_frame_name=BODY_FRAME_NAME)
        arm_command = arm_command_pb2.ArmCommand.Request(arm_cartesian_command=arm_cartesian_command)
        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
        robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)
        cmd_id = self.command_client.robot_command(robot_command) # Execute command

        # If blocking is enabled this will force the robot to wait until trajectory is complete
        if blocking == True: # TO DO: Change to look at the command id instead
            start_t = time.time()
            while time.time()-start_t < t:
                pass

    def movej(self, q, t=5, deg=True):
        """
        Rotates the joints with the specified angles (in radians). The trajectory is executed in the specified time in seconds, defaults to 5 seconds.
        """
        assert len(q)==6, "Invalid joint input!"
        if deg == True:
            radians = []
            for angle in q:
                radians.append(self.deg_to_rad(angle))
            q1, q2, q3, q4, q5, q6 = radians
        else:
            q1, q2, q3, q4, q5, q6 = q

        if type(q1) == wrappers_pb2.DoubleValue:
            arm_position = arm_command_pb2.ArmJointPosition(sh0=q1, 
                                                            sh1=q2,
                                                            el0=q3, 
                                                            el1=q4,
                                                            wr0=q5, 
                                                            wr1=q6)
        else:            
            arm_position = arm_command_pb2.ArmJointPosition(sh0=wrappers_pb2.DoubleValue(value=q1), 
                                                            sh1=wrappers_pb2.DoubleValue(value=q2),
                                                            el0=wrappers_pb2.DoubleValue(value=q3), 
                                                            el1=wrappers_pb2.DoubleValue(value=q4),
                                                            wr0=wrappers_pb2.DoubleValue(value=q5), 
                                                            wr1=wrappers_pb2.DoubleValue(value=q6))
        # Wrap position in trajectory structure
        arm_joint_trajectory_point = arm_command_pb2.ArmJointTrajectoryPoint(position=arm_position,
                                                                             time_since_reference=seconds_to_duration(t))
        arm_joint_trajectory = arm_command_pb2.ArmJointTrajectory(points=[arm_joint_trajectory_point], 
                                                                  maximum_velocity=wrappers_pb2.DoubleValue(value=4.0))
        arm_joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_trajectory)
        arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=arm_joint_move_command)
        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
        robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)
        # Issue command to robot
        cmd_id = self.command_client.robot_command(robot_command)

    def arm_drop_position(self, side_offset=0.0):
        traj_time = 3
        position = [0.9, 0.0+side_offset, 0.1]
        orientation = [0.707, 0.0, 0.707, 0.0]
        self.movel(position, orientation, traj_time, True)

    def arm_retract_position(self):
        traj_time = 3
        position = [0.75, 0.0, 0.3]
        orientation = [0.707, 0.0, 0.707, 0.0]
        self.movel(position, orientation, traj_time, True)

    def stow_arm(self):
        stow = RobotCommandBuilder.arm_stow_command()
        self.command_client.robot_command(stow)
        time.sleep(2)

    def open_gripper(self,open_fraction=1.0):
        theta = open_fraction * (-1.5708) # Max opening angle in radians
        claw_gripper_command = gripper_command_pb2.ClawGripperCommand.Request(trajectory=trajectory_pb2.ScalarTrajectory(points=[trajectory_pb2.ScalarTrajectoryPoint(point=theta, time_since_reference=seconds_to_duration(1.0))]),
                                                                              maximum_open_close_velocity=wrappers_pb2.DoubleValue(value=6.28),
                                                                              maximum_torque=wrappers_pb2.DoubleValue(value=0.5))
        gripper_command = gripper_command_pb2.GripperCommand.Request(claw_gripper_command=claw_gripper_command)
        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(gripper_command=gripper_command)
        command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)
        cmd_id = self.command_client.robot_command(command)

    def add_grasp_constraint(self, grasp):
        grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME
        axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
        axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(axis_to_align_with_ewrt_vo)
        # Anything within about 10 degrees for top-down is fine
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    def arm_object_grasp(self):
        self.g_image_click = None
        self.g_image_display = None
        image_sources = ['frontleft_fisheye_image',
                         'frontright_fisheye_image',
                         'left_fisheye_image',
                         'right_fisheye_image',
                         'back_fisheye_image']

        image_responses = self.image_client.get_image_from_sources([image_sources[0]])

        if len(image_responses) != 1:
            print('Got invalid number of images: ' + str(len(image_responses)))
            print(image_responses)
            assert False

        image = image_responses[0]
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
        img = np.fromstring(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)

        # Show the image to the user and wait for them to click on a pixel
        image_title = 'Click to grasp'
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(image_title, self.cv_mouse_callback)

        self.g_image_display = img
        cv2.imshow(image_title, self.g_image_display)
        while self.g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                exit(0)
        cv2.destroyAllWindows()

        pick_vec = geometry_pb2.Vec2(x=self.g_image_click[0], y=self.g_image_click[1])

        grasp = manipulation_api_pb2.PickObjectInImage(pixel_xy=pick_vec,
                                                       transforms_snapshot_for_camera=image.shot.transforms_snapshot,
                                                       frame_name_image_sensor=image.shot.frame_name_image_sensor,
                                                       camera_model=image.source.pinhole)
        # Force top-down grasp
        self.add_grasp_constraint(grasp)
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
        cmd_response = self.manipulation_api_client.manipulation_api_command(manipulation_api_request=grasp_request)
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(manipulation_cmd_id=cmd_response.manipulation_cmd_id)
            response = self.manipulation_api_client.manipulation_api_feedback_command(manipulation_api_feedback_request=feedback_request)
            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                return 1
            elif response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                return 0
            time.sleep(0.25)

    def lock_arm_in_body(self):
        """
        Reads current joint configuration for the arm and executes a joint command to the same configuration to lock the arm in the body frame
        """
        joint_states = self.robot_state_client.get_robot_state().kinematic_state.joint_states
        arm_idx = [12, 13, 15, 16, 17, 18]
        joint_positions = []
        for idx in arm_idx:
            joint_positions.append(joint_states[idx].position)
        self.movej(joint_positions, t=1, deg=False)
        time.sleep(1.5)








    def main(self):
        with bosdyn.client.lease.LeaseKeepAlive(self.motion_client, must_acquire=True, return_at_exit=True):
            # Turn Spot on and stand up

            self.init_spot()

            i = 0
            while 1:
                print("Enter 1 for banana, enter 2 for bottle:")
                input_val = int(input())

                if input_val == 1: # If banana
                    ID = 1
                    s_offset1 = 0.7
                    s_offset2 = 0.5
                    arm_offset = -0.1
                    rot_val = 135
                elif input_val == 2: # If bottle
                    ID = 2
                    s_offset1 = -0.7
                    s_offset2 = -0.5
                    arm_offset = 0.1
                    rot_val = -135

                # Show image, get input pixel and grasp item
                grasp_success = self.arm_object_grasp()
                # TODO: Add error handling when grasp fails
                if not grasp_success:
                    print("FAILED GRASP")
                # If grasp is successful, carry it to trash bin
                else:
                    # Retract arm from ground to carry position
                    self.arm_retract_position()
                    time.sleep(0.25)
                    self.lock_arm_in_body()
                    # Move to corresponding trash bin and drop the item into it
                    self.move_to_aruco(ID, marker_size=self.aruco_size, dist_offset=0.7, side_offset=s_offset1)
                    time.sleep(0.25)
                    self.move_to_aruco(ID, marker_size=self.aruco_size, dist_offset=0.5, side_offset=s_offset2, update_global=True)
                    time.sleep(0.25)
                    #self.move_to_aruco(ID, marker_size=self.aruco_size, dist_offset=0.7, side_offset=0.0, global_orientation=True)
                    #time.sleep(0.25)
                    self.arm_drop_position(arm_offset)
                    self.open_gripper(1.0)
                    time.sleep(2)
                    self.open_gripper(0.0)
                    # Stow arm and return to "ready" pose
                    self.stow_arm()
                    
                    self.move_to_aruco(ID, marker_size=self.aruco_size, dist_offset=0.7, side_offset=s_offset2, global_orientation=True)
                    
                    self.rotate_body(rot_val, 7)

                    time.sleep(1)



                    
            self.poweroff_spot()

TC = TrashCollect(sys.argv[1:])
TC.main()
