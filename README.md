# Capstone Proposal: Autonomous Drone Navigation

Is it possible to train a drone — through supervised or reinforcement learning — to navigate by visual information alone?

Consider how a teenager learns to drive a car... before they drive solo, they have hours of reinforcement learning with the help of a trainer — sometimes in the form of a parent screaming “STOP!” at critical times.  And before that, they have years of learning by example as they watch from the backseat as mom and dad drive.

This project will demonstrate how similar training techniques can be used to train a drone to navigate autonomously.  Specifically, I will explore how one can train a convolutional neural network (CNN) via supervised learning to guide operate a quadcopter to accomplish a simple navigational task, such as to identify and move toward a target in a previously unseen indoor environment.  A successful capstone demonstration would involve the flight of an actual drone in a real-world, indoor setting, guided by a pre-trained CNN in real time.

## Resources

* [Parrot AR.Drone 2.0](https://www.parrot.com/us/drones/parrot-ardrone-20-power-édition#ar-drone-20-power-edition): A consumer-grade drone excellent stability control and an [SDK]( http://developer.parrot.com/docs/SDK2/) for programmatic navigation control and sensor data collection.  The AR.Drone has a forward-facing camera capable of sending individual snapshots and streaming video to a supervising program.

* [Robot Operating System (ROS)](http://www.ros.org): A well-established open source framework for controlling robotic arms, mobile robots and drones.  With Python library bindings and support for Ubuntu.

* ["AR.Drone" Autonomy” ROS Node](https://ardrone-autonomy.readthedocs.io): A ROS module based on the Parrot SDK that permits an easy-to-use API for controlling drone flight and reading drone sensors.

The AR.Drone Autonomy ROS Node defers complex stability control functions to the Parrot firmware and exposes simplified navigation operations via the ROS interface.  Navigation operations include:

Takeoff / land / emergency
Translate forward / backward
Translate left / right
Ascend / descend
Rotate clockwise / counter clockwise

<!-- The AR.Drone Autonomy ROS Node is also capable of publishing high-resolution camera snapshots and 640x480 resolution H.264 via the ROS interface.-->


## Data
For training data, I will collect frames of video from the Bebop drone in flight, synchronized with a recording of the navigational commands sent through the AR.Drone Autonomy ROS node.  For the sake of efficient transmission and storage (as well as later model training), I will downsample the video data both spatially and temporally — perhaps to 320x240 and 1 frame per second.  To simplify the training problem it will also be ideal to initially capture only 8-bit grayscale video frames.
