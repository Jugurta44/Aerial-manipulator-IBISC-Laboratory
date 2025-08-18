
# ROS Drone Simulation

Credits to the original [owner](https://github.com/NishanthARao/ROS-Quadcopter-Simulation). This repository is a few tweaks to allow easier integration for personal use

# Prerequesties

 - ROS1
 - Gazebo
 - ROS Control
 - ROS Controllers

# Setup

```bash
rosdep install --from-paths --src --ignore-src -r -y --rosdistro $ROS_DISTRO

# Build the workspace
catkin_make
```

### Usage

To start the simulation, run

```bash
roslaunch drone_simulation gazebo.launch

# A small script allows to showcase the drone control by hovering
rosrun drone_simulation control.py
```

### Other Useful Informaiton

This provides a speed of:

 - 50 units to front_right motor
 - -50 units to front_left motor
 - 50 units to back_left motor
 - -50 units to back_right motor

Here, the negative sign denotes rotation in the opposite direction.