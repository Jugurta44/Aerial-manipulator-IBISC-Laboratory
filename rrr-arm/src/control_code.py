#!/usr/bin/env python

from pid import PID, generate_trajectory
import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray, Float32
from tf.transformations import euler_from_quaternion

def control_kwad(msg, args):
    # Keep track of state and errors across calls
    global roll, pitch, yaw, err_roll, err_pitch, err_yaw

    # Current simulation time
    start_time = rospy.get_time()
    t = rospy.get_time() - start_time

    # Container for motor velocity commands
    f = Float64MultiArray()

    # Desired trajectory setpoints
    xd, yd, zd, psi_d = generate_trajectory(t, mode=1)

    # Get the index of the robot model in /gazebo/model_states
    ind = msg.name.index("rrr_arm")
    
    # Extract orientation and convert quaternion → roll, pitch, yaw
    orientationObj = msg.pose[ind].orientation
    orientationList = [
        orientationObj.x,
        orientationObj.y,
        orientationObj.z,
        orientationObj.w,
    ]
    roll, pitch, yaw = euler_from_quaternion(orientationList)

    # Run PID control to update motor velocities
    fUpdated, err_roll, err_pitch, err_yaw = PID(roll, pitch, yaw, f, xd, yd, zd, psi_d)

    # Publish control outputs and errors
    args[0].publish(fUpdated)
    args[1].publish(err_roll)
    args[2].publish(err_pitch)
    args[3].publish(err_yaw)

    # Debug output
    print("Roll:", roll * (180 / 3.141592653),
          "Pitch:", pitch * (180 / 3.141592653),
          "Yaw:", yaw * (180 / 3.141592653))
    print(orientationObj)

# ROS node initialization
rospy.init_node("Control")

# Publishers for roll, pitch, yaw errors
err_rollPub = rospy.Publisher("roll", Float32, queue_size=1)
err_pitchPub = rospy.Publisher("pitch", Float32, queue_size=1)
err_yawPub = rospy.Publisher("yaw", Float32, queue_size=1)

# Publisher for motor velocity commands
velPub = rospy.Publisher(
    "/rrr_arm/joint_motor_controller/command", Float64MultiArray, queue_size=4
)

# Subscriber to Gazebo model states, triggers control callback
PoseSub = rospy.Subscriber(
    "/gazebo/model_states",
    ModelStates,
    control_kwad,
    (velPub, err_rollPub, err_pitchPub, err_yawPub),
)

# Keep the node running
rospy.spin()
