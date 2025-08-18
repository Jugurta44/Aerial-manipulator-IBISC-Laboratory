import time
import numpy as np

def generate_trajectory(t, mode=1):
    """Generate desired position and yaw setpoints."""
    if mode == 1:
        if t < 5:
            xd, yd, zd = 0, 0, 1
            psi_d = np.pi / 6
        else:
            xd, yd, psi_d = 0, 0, 0
            zd = 1
    return xd, yd, zd, psi_d

def calculate_euler_angles(xd, yd, zd, psi_d):
    """Convert desired position into desired Euler angles."""
    d = np.sqrt(xd**2 + yd**2 + zd**2)
    if d == 0:
        return 0.0, 0.0
    theta_d = np.arcsin((xd * np.sin(psi_d) + yd * np.cos(psi_d)) / d)
    phi_d = np.arctan2(xd * np.cos(psi_d) + yd * np.sin(psi_d), zd)
    return theta_d, phi_d

def PID(roll, pitch, yaw, f, xd, yd, zd, psi_d, set_roll=0, set_pitch=0, set_yaw=0, base_thrust=1845):
    """PID controller for roll, pitch, yaw to generate motor commands."""
    global kp_roll, ki_roll, kd_roll
    global kp_pitch, ki_pitch, kd_pitch
    global kp_yaw, ki_yaw, kd_yaw
    global prevErr_roll, prevErr_pitch, prevErr_yaw
    global pMem_roll, pMem_pitch, pMem_yaw
    global iMem_roll, iMem_pitch, iMem_yaw
    global dMem_roll, dMem_pitch, dMem_yaw
    global prevTime, flag, sampleTime

    # PID gains
    kp_roll, ki_roll, kd_roll = 70, 0.0, 0.0
    kp_pitch, ki_pitch, kd_pitch = kp_roll, ki_roll, kd_roll
    kp_yaw, ki_yaw, kd_yaw = 0.1, 0.0, 0.0

    sampleTime = 0.001

    # Desired angles
    theta_d, phi_d = calculate_euler_angles(xd, yd, zd, psi_d)

    # Current errors (deg)
    err_pitch = float(pitch) * (180 / np.pi) - set_pitch
    err_roll = float(roll) * (180 / np.pi) - set_roll
    err_yaw = float(yaw) * (180 / np.pi) - set_yaw

    currTime = time.time()

    # Initialize on first run
    if 'flag' not in globals() or flag == 0:
        prevTime = 0
        prevErr_roll = prevErr_pitch = prevErr_yaw = 0
        pMem_roll = pMem_pitch = pMem_yaw = 0
        iMem_roll = iMem_pitch = iMem_yaw = 0
        dMem_roll = dMem_pitch = dMem_yaw = 0
        flag = 1

    # Time and error deltas
    dTime = currTime - prevTime
    dErr_pitch = err_pitch - prevErr_pitch
    dErr_roll = err_roll - prevErr_roll
    dErr_yaw = err_yaw - prevErr_yaw

    # PID update if sample time reached
    if dTime >= sampleTime:
        # Proportional
        pMem_roll = kp_roll * err_roll
        pMem_pitch = kp_pitch * err_pitch
        pMem_yaw = kp_yaw * err_yaw

        # Integral with clamping
        iMem_roll = max(min(iMem_roll + err_pitch * dTime, 400), -400)
        iMem_pitch = max(min(iMem_pitch + err_roll * dTime, 400), -400)
        iMem_yaw = max(min(iMem_yaw + err_yaw * dTime, 400), -400)

        # Derivative
        dMem_roll = dErr_roll / dTime
        dMem_pitch = dErr_pitch / dTime
        dMem_yaw = dErr_yaw / dTime

    # Save state for next iteration
    prevTime = currTime
    prevErr_roll = err_roll
    prevErr_pitch = err_pitch
    prevErr_yaw = err_yaw

    # PID outputs
    output_roll = pMem_roll + ki_roll * iMem_roll + kd_roll * dMem_roll
    output_pitch = pMem_pitch + ki_pitch * iMem_pitch + kd_pitch * dMem_pitch
    output_yaw = pMem_yaw + ki_yaw * iMem_yaw + kd_yaw * dMem_yaw

    # ESC pulse values
    esc_br = base_thrust + output_roll + output_pitch - output_yaw
    esc_bl = base_thrust + output_roll - output_pitch + output_yaw
    esc_fl = base_thrust - output_roll - output_pitch - output_yaw
    esc_fr = base_thrust - output_roll + output_pitch + output_yaw

    # Constrain ESC values
    for esc in ['esc_br', 'esc_bl', 'esc_fl', 'esc_fr']:
        val = locals()[esc]
        val = min(max(val, 1100), 2500)
        locals()[esc] = val

    # Convert ESC pulses to motor velocities
    br_motor_vel = ((esc_br - base_thrust) / 25) + 55.5
    bl_motor_vel = ((esc_bl - base_thrust) / 25) + 55.5
    fr_motor_vel = ((esc_fr - base_thrust) / 25) + 55.5
    fl_motor_vel = ((esc_fl - base_thrust) / 25) + 55.5

    # Assign to message (motor order: CW, CCW, CW, CCW)
    f.data = [fr_motor_vel, -fl_motor_vel, bl_motor_vel, -br_motor_vel]

    return f, err_roll, err_pitch, err_yaw
