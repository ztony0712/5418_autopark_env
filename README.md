# Autonomous Parking based on SAC Motion Control Method
COPYRIGHT VIOLATION NOTICE: In the spirit of open source, we have not set our project as private. However, please do not directly copy from our code, we have a full submission record, showing that we have build our project completly by ourselves. Any plagiarism will be considered a breach of the NUS Academic Integrity Regulations.

This project is created for the NUS ME5418 Machine Learning in Robotics. We are the Group 39.
> Authors: [Yimin](https://github.com/ztony0712), [Guorong](https://github.com/z492x), [Zeren](https://github.com/HardyPavel)

![Ubuntu 20.04](https://img.shields.io/badge/OS-Ubuntu_20.04-informational?style=flat&logo=ubuntu&logoColor=white&color=2bbc8a)
![Gymnasium](https://img.shields.io/badge/Tools-Gymnasium_1.0-informational?style=flat&logo=OpenAI&logoColor=white&color=2bbc8a)
![Python](https://img.shields.io/badge/Code-Python_3.8-informational?style=flat&logo=Python&logoColor=white&color=2bbc8a)
![Gymnasium](https://img.shields.io/badge/Tools-Gymnasium_1.0-informational?style=flat&logo=OpenAI&logoColor=white&color=2bbc8a)
![HighwayEnv](https://img.shields.io/github/v/release/Farama-Foundation/HighwayEnv?style=flat&color=2bbc8a&label=HighwayEnv)


<!-- > Video Presentation -->
<!-- [![ROS SLAM, Perception, and Navigation based on Gazebo simulation](https://img.youtube.com/vi/WiEzSJmcEQE/0.jpg)](https://www.youtube.com/watch?v=WiEzSJmcEQE) -->

As you see in the video, we have implemented the following techniques:
<!-- - For SLAM: The Fast-lio algorithm is chosen for 3D LiDAR SLAM  -->

For more detail information, please check our report pdf file.

## Installation
First of all, clone this repo to your root directory:
```bash
# Clone and go to the directory
cd
git clone https://github.com/ztony0712/5418_sac_parking.git
cd 5418_sac_parking
```

Then, install the required packages using conda:
```bash
# No need now, just create a new one
conda create -n 5418_sac_parking python=3.8

# Activate the environment
conda activate 5418_sac_parking
```

This repo is developed based on gym highway-env. To install it, follow the instructions below:
```bash
# Install the dependencies of gym highway-env
sudo apt-get update -y
sudo apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev
    libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev
    ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc

# Install the gym highway-env development version
pip install --user git+https://github.com/eleurent/highway-env
```




## Usage

### 0. Install Environment
```bash
# Install customized gym parking environment
pip install -e .
```

### 1. Test Parking Environment
Run test_env.py script to check the environment

```bash
# Display the parking environment visualization
python test_env.py
```

### 2. Second Step
Next Step

## License

<!-- The [ME5413_Final_Project](https://github.com/NUS-Advanced-Robotics-Centre/ME5413_Final_Project) is released under the [MIT License](https://github.com/NUS-Advanced-Robotics-Centre/ME5413_Final_Project/blob/main/LICENSE) -->
