# Tutorial for Hardware and AI Tools Setup

## Installing Jetson Nano
1. [Get Started With Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)

## Installation Python3.8 and Ultralytics
1. Check update and upgrade for Jetson Nano system
   ```bash
   sudo apt-get update && sudo apt-get upgrade
   ```
2. Install essential library for python3.8
   ```bash
   sudo apt install build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev libffi-dev libc6-dev
   ```
3. Install Python 3.8
   ```bash
   sudo apt install python3.8
   ```
4. Create python3.8 environment
   ```bash
   python3.8 -m venv myenv
   source myenv/bin/activate
   ```
5. Install Ultralytics for YOLOv8
   ```bash
   pip install ultralytics
   ```
6. 