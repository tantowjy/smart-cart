<h1 align="center">Smart Cart</h1>

## Hardware

#### Component
- NVIDIA Jetson Nano Development Kit B01
- IMX219-77 Camera for Jetson Nano - [*documentation*](https://www.waveshare.com/wiki/IMX219-77_Camera)
- LCD Waveshare 7" HDMI IPS Capacitive Touchscreen
- Samsung Evo Plus 64GB UHS-I Class 10
- Intel 8265NGW Wireless Network Card
- Power Supply 5V/4A for Jetson Nano
- Panda 2D Barcode Scanner PRJ-888
- Acrylic Clear Case for Jetson Nano

#### Reference
- [Get Started With Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)
- [CSI-Camera Python Program](https://github.com/JetsonHacksNano/CSI-Camera)
- [UPS Power Module Battery](https://www.waveshare.com/wiki/UPS_Power_Module)

## Software

### Setup Ubuntu 20.04 for Jetson
1. Download the ISO from this [link](https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image).
2. Switch the display manager to LightDM:
   ```bash
   sudo dpkg-reconfigure lightdm
   ```
3. Increase swap memory allocation:
   ```bash
   sudo fallocate -l 8G /var/swapfile 
   sudo chmod 600 /var/swapfile
   sudo mkswap /var/swapfile
   sudo swapon /var/swapfile
   sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0"  >> /etc/fstab'
   ```
4. 


## Requirement
- Python 3.8.10
- OpenCV 4.8.0
- Ultralytics 8.0.255