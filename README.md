<h1 align="center">SMART CART</h1>

## Hardware Component
- NVIDIA Jetson Nano Development Kit B01
- IMX219-77 Camera for Jetson Nano
- LCD Waveshare 7" HDMI IPS Capacitive Touchscreen
- Samsung Evo Plus 64GB UHS-I Class 10
- Intel 8265NGW Wireless Network Card
- Power Supply 5V/4A for Jetson Nano
- UPS Power Module (B)
- Panda 2D Barcode Scanner PRJ-888

## Hardware & Environment Setup

### Ubuntu 20.04 for Jetson
1. Download the Linux ISO from this [link](https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image).
2. Installing the operating system through the [guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit).
3. Increasing swap memory allocation:
   ```bash
   sudo fallocate -l 8G /var/swapfile 
   sudo chmod 600 /var/swapfile
   sudo mkswap /var/swapfile
   sudo swapon /var/swapfile
   sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0"  >> /etc/fstab'
   sudo reboot
   ```

### Installing Ultralytics and OpenCV with Gstreamer 
1. Check for updates and upgrades for the Jetson Nano system.
   ```bash
   sudo apt-get update && sudo apt-get upgrade
   ```  
2. Install pip3 for Python 3.
   ```bash
   sudo apt install python3-pip
   ```
3. Install virtual environment for Python 3.
   ```bash
   sudo apt install python3-venv -y
   ```
4. Create a virtual environment.
   ```bash
   # for python3
   python3 -m venv <your_env>
   
   # for python3.8
   python3.8 -m venv <your_env>

   source <your_env>/bin/activate
   ```
5. Install Python 3.8 or higher for Ultralytics YOLOv8. If your default Python 3 version is 3.8 or higher, skip this step.
   ```bash
   sudo apt install python3.8
   alias python=/usr/bin/python3.8
   ```
6. Install Ultralytics YOLOv8 using pip.
   ```bash
   pip install ultralytics
   ```
7. Remove OpenCV default version.
   ```bash
   pip uninstall opencv-python
   ```
8. Install gstreamer1.0
   ```bash
   sudo apt-get install gstreamer1.0*
   sudo apt install ubuntu-restricted-extras
   ```
9. Install gstreamer library and numpy.
   ```bash
   sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
   pip install numpy
   ```
10. Clone OpenCV official repository.
    ```bash
    git clone https://github.com/opencv/opencv.git
    cd opencv/
    git checkout 4.x
    ```
11. Building Cmake.
    ```bash
    mkdir build
    cd build
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D PYTHON_EXECUTABLE=$(which python3) \
    -D BUILD_opencv_python2=OFF \
    -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D WITH_GSTREAMER=ON \
    -D BUILD_EXAMPLES=ON ..
    ```
    Check out the Gstreamer section. It should show “YES (version number)". Otherwise, please check your GStreamer lib package.
    _If something went wrong, remove the build folder and try again._
    ```bash
    sudo make -j$(nproc)
    sudo make install
    sudo ldconfig
    ```
12. Check OpenCV build information.
    ```bash
    import cv2
    print(cv2.getBuildInformation())
    ```

## Reference
- [Get Started With Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)
- [IMX219-77 Camera](https://www.waveshare.com/wiki/IMX219-77_Camera)
- [CSI-Camera Python Program](https://github.com/JetsonHacksNano/CSI-Camera)
- [UPS Power Module Battery](https://www.waveshare.com/wiki/UPS_Power_Module)

