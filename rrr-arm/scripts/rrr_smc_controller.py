#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np

# --- Constants (adjust based on your actual robot model)
m1, m2, m3 ,mb= 0.07, 0.082, 0.06 , 1.15 # Segment masses
d2, d3 = 0.15, 0.10           # Link lengths
lambda_gain = 1.5
k_gain = 0.2
epsilon = 0.1             # For smooth SMC

class SMCController:
  
    def __init__(self):
        rospy.init_node("smc_com_controller")
        self.joint_1 = 0.0
        self.joint_2 = 0.0
        self.joint_3 = 0.0
        self.joint_indices = {}
        self.new_r0=0
        self.y_des = 0
        self.prev_time = rospy.Time.now()
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.joint_pos = {'joint1': 0.0, 'joint2': 0.0, 'joint3': 0.0}
        self.joint_vel = {'joint1': 0.0, 'joint2': 0.0, 'joint3': 0.0}
        self.prev_error = 0.0
        self.prev_time = rospy.Time.now()
        self.start_time = rospy.Time.now()

        # Subscribers
        #rospy.Subscriber("/rrr_arm/joint_states", JointState, self.joint_state_cb)
        rospy.Subscriber('/rrr_arm/joint_states', JointState, self.joint_state_callback)

        # Publishers
        self.pub_joint1 = rospy.Publisher(
            "/rrr_arm/joint1_position_controller/command", Float64, queue_size=10)
        self.pub_joint2 = rospy.Publisher(
            "/rrr_arm/joint2_position_controller/command", Float64, queue_size=10)
        self.pub_joint3 = rospy.Publisher(
            "/rrr_arm/joint3_position_controller/command", Float64, queue_size=10)

        self.x_cm_pub = rospy.Publisher('/x_cm', Float64, queue_size=10)
        self.error_pub = rospy.Publisher('/x_cm_error', Float64, queue_size=10)

        self.r0_desired = rospy.Publisher('/r0_desired', Float64, queue_size=10)
        self.x_desired = rospy.Publisher('/x_desired', Float64, queue_size=10)

        rospy.Timer(rospy.Duration(0.005), self.control_loop)  # 100 Hz
        rospy.spin()

    #def joint_state_cb(self, msg):
        #for i, name in enumerate(msg.name):
            #self.joint_pos[name] = msg.position[i]
            #self.joint_vel[name] = msg.velocity[i]
            
    def joint_state_callback(self, msg):
        self.joint_indices = {}
        for i, name in enumerate(msg.name):
            if name in ['joint_1', 'joint_2', 'joint_3']:
                self.joint_indices[name] = i

        self.joint_1 = msg.position[self.joint_indices.get('joint_1', 0)]
        self.joint_2 = msg.position[self.joint_indices.get('joint_2', 0)]
        self.joint_3 = msg.position[self.joint_indices.get('joint_3', 0)]


    def compute_xcm(self, r0, theta2, theta3):
        mG = m1 + m2 + m3 +mb
        xcm = (1/mG) * (
            (m1 + m2 + m3) * r0
            + (0.5 * m2 + m3) * d2 * np.sin(theta2)
            + 0.5 * m3 * d3 * np.sin(theta2 + theta3)
        )
        return xcm
    
  

    def publish_trajectory(self, t):

        if t < 5:
            self.theta2_cmd = 0.2 *np.sin(0.5*t)
            self.theta3_cmd = 0 
        elif t < 10:
            self.theta2_cmd = -0.4 
            self.theta3_cmd = -0.4 *np.sin(0.5*t)
        elif t < 15:
            self.theta2_cmd = -0.4 
            self.theta3_cmd = -0.2 
        else:
            # Final steady values
            self.theta2_cmd = 0.8
            self.theta3_cmd = -1.2
        # Publish to revolute joints
        self.pub_joint2.publish(Float64(self.theta2_cmd))
        self.pub_joint3.publish(Float64(self.theta3_cmd))

        rospy.loginfo(f"[Trajectory] Joint2: {self.theta2_cmd:.3f}, Joint3: {self.theta3_cmd:.3f}")
        
      

    def control_loop(self, event):
        
        now = rospy.Time.now()
        t = (now - self.start_time).to_sec()

        # === PID Controller ===
        kp = 3.0     # Proportional gain
        ki = 0.1     # Integral gain
        kd = 0.01     # Derivative gain

        # Publish desired joint2 and joint3 trajectory
        self.publish_trajectory(t)

       
        r0 = self.joint_1
        theta2 = self.joint_2
        theta3 = self.joint_3

        x_cm = self.compute_xcm(r0, theta2, theta3)
        error = x_cm  
        desired_xcm = 0

        #y_cm = self.compute_y_cm(r0, theta2, theta3)
        #error = self.y_des - y_cm  # controlling y_cm#
        
        # Time delta
        
        dt = (now - self.prev_time).to_sec()
        self.prev_time = now

        if dt == 0:
            return

        d_error = (error - self.prev_error) / dt
        self.prev_error = error

        s = d_error + lambda_gain * error
        if abs(s) <= epsilon:
            sat = s / epsilon
        else:
            sat = np.sign(s)

        # SMC Law with boundary layer (tanh version)
        u = -k_gain * sat #np.sat(s / epsilon)
        
        #PID Update error terms
        #error = self.y_des - x_cm
        #d_error = (error - self.prev_error) / dt
        #self.integral_error += error * dt
        #self.prev_error = error
        # PID output
        u_pid = kp*error + ki*self.integral_error + kd*d_error

        # Compute new joint1 command (desired prismatic position)
        self.new_r0 = r0 + u * dt

        # publishing values
        self.pub_joint1.publish(Float64(self.new_r0))
        self.r0_desired.publish(Float64(r0))
        self.x_desired.publish(Float64(desired_xcm))
        self.x_cm_pub.publish(Float64(x_cm))
        self.error_pub.publish(Float64(s))


        
         # Logging for visualization
        rospy.loginfo("=== SMC CONTROL LOOP ===")
        rospy.loginfo(f"Joint1 (Prismatic):     {r0:.4f} -> {self.new_r0:.4f}")
        rospy.loginfo(f"Joint2 (Revolute):      {theta2} rad")
        rospy.loginfo(f"Joint3 (Revolute):      {theta3} rad")
        rospy.loginfo(f"y_cm:                   {x_cm:.4f} m")
        rospy.loginfo(f"Error (y_des - y_cm):   {error:.4f}")
        rospy.loginfo(f"SMC output u:           {s:.4f}")
        #rospy.loginfo(f"Sliding surface:        {s:.4f}")
        rospy.loginfo("===========================")

if __name__ == "__main__":
    try:
        SMCController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass