[![CI](https://github.com/Keys-481/fa25-prime-directive/actions/workflows/ci.yml/badge.svg)](https://github.com/Keys-481/fa25-prime-directive/actions/workflows/ci.yml)

## Setup
Some setup beyond what is provided in this repository is necessary for the rover to work properly

Create a crontab file that will launch the program when the Pi is booted up using the following command:

  crontab -e
  
Then add the following to the bottom of the file:

  @reboot "command or path to script"

Notes for device functionality:

The camera device module index will almost certainly be either 1 or 0.
In our use case, the index is 1

The libcamera stack does not work properly on the Raspberry Pi 5, but the cam command provides an alternative.
The following command allows for the storage of images in the .ppm format which is compatible with the OpenCV
library:

cam --camera 1 --capture --file="$HOME/ws/captures/frame#.ppm" --stream width=820,height=640,pixelformat=BGR888
