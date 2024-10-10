# Autonomous Parking Environment Based on Gymnasium
COPYRIGHT VIOLATION NOTICE: In the spirit of open source, we have not set our project as private. However, please do not directly copy from our code, we have a full submission record, showing that we have build our project completly by ourselves. Any plagiarism will be considered a breach of the NUS Academic Integrity Regulations.

This project is created for the NUS ME5418 Machine Learning in Robotics. We are the Group 39.
> Authors: [Yimin](https://github.com/ztony0712), [Guorong](https://github.com/z492x), [Zeren](https://github.com/HardyPavel)

![Ubuntu 20.04](https://img.shields.io/badge/OS-Ubuntu_20.04-informational?style=flat&logo=ubuntu&logoColor=white&color=2bbc8a)
![Gymnasium](https://img.shields.io/badge/Tools-Gymnasium_1.0.0-informational?style=flat&logo=OpenAI&logoColor=white&color=2bbc8a)
![Python](https://img.shields.io/badge/Code-Python_3.12-informational?style=flat&logo=Python&logoColor=white&color=2bbc8a)

<!-- > Video Presentation -->
<!-- [![ROS SLAM, Perception, and Navigation based on Gazebo simulation](https://img.youtube.com/vi/WiEzSJmcEQE/0.jpg)](https://www.youtube.com/watch?v=WiEzSJmcEQE) -->

<!-- As you see in the video, we have implemented the following techniques: -->
<!-- - For SLAM: The Fast-lio algorithm is chosen for 3D LiDAR SLAM  -->

For more detail information, please check our report pdf file.

## Dependencies
The following dependencies are required to visualize the parking environment:
```bash
# No need if you don't have visualization problems
sudo apt-get update -y
sudo apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev
    libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev
    ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc
```

## Installation
First of all, clone this repo to your root directory:
```bash
# Clone and go to the directory
cd
git clone https://github.com/ztony0712/5418_autopark_env.git
cd 5418_autopark_env
```

Then, install the required packages using conda:
```bash
# Create the conda environment
conda create -f environment.yml

# Activate the environment
conda activate 5418_autopark_env
```

## Usage
### 0. Install Enviroment
```bash
# Development Env, no need reinstall after change
pip install -e .

# Production Env, need upgrade
pip install git+https://github.com/ztony0712/5418_autopark_env
```

### 1. Test Parking Environment
Run test_env.py script to check the environment

```bash
# Display the parking environment visualization
python test_env.py
```

