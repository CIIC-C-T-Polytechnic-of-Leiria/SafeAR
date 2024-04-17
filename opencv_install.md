### OpenCV with CUDA and cuDNN on Ubuntu 22.04 LTS, System-wide Installation

https://gist.github.com/minhhieutruong0705/8f0ec70c400420e0007c15c98510f133

tutorial 2:
https://www.samontab.com/web/2023/02/installing-opencv-4-7-0-in-ubuntu-22-04-lts/ 

tutorial 3:
https://www.youtube.com/watch?v=8zXHSfFyXZs

tutorial 4:
https://www.youtube.com/watch?v=Nn5OfJjBGmg

tutorial 5:
https://www.youtube.com/watch?v=whAFl-izD-4

Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux. It is used to install, update, and manage packages and their dependencies. 
To create a new Conda environment, run the following command:

```bash
conda create -n safeAR python=3.10 # Create a new Conda environment with Python 3.10
conda activate safeAR              # Activate the created Conda environment
```

Advanced Package Tool (APT) is a package management system used by Debian-based distributions. It is used to install, update, and remove packages on the system. To update the package list and upgrade the installed packages, run the following commands:

```bash
sudo apt-get update
sudo apt-get upgrade
```

Install the required packages using APT:
```bash
sudo apt-get install build-essential cmake python3-numpy python3-dev python3-tk libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libdc1394-dev libeigen3-dev libgtk-3-dev libvtk7-qt-dev
```

- `build-essential`: This package contains a list of packages which are considered essential for building Ubuntu packages including gcc compiler, make and other required tools.
- `cmake`: A cross-platform open-source tool for managing the build process of software using a compiler-independent method.
- `python3-dev`: This package contains the header files and static libraries for Python (default Python 3 version). It's usually needed if you want to compile Python extensions.
- `python3-numpy`: NumPy is a package for scientific computing with Python. It contains among other things a powerful N-dimensional array object and capabilities for linear algebra, Fourier transform, and random number capabilities.
- `python3-tk`: Tkinter is Python's standard GUI (Graphical User Interface) package.
- `libavcodec-dev`, `libavformat-dev`, `libavutil-dev`, `libswscale-dev`: These are development files for the libav* series of libraries, which are a very comprehensive and highly efficient suite of libraries for audio/video processing.
- `libdc1394-dev`: This is a library that provides a high level programming interface for application developers who wish to control and capture streams from IEEE 1394 based cameras.
- `libeigen3-dev`: Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.
- `libgtk-3-dev`: This is the development files for the GTK+ library. GTK+ is a multi-platform toolkit for creating graphical user interfaces.
- `libvtk7-qt-dev`: This is the development files for the Visualization Toolkit (VTK), which is an open-source, freely available software system for 3D computer graphics, image processing and visualization. This package is specifically for the QT interface of VTK.


Download the OpenCV and OpenCV contrib source code, version 4.7.0:
```bash
wget https://github.com/opencv/opencv/archive/refs/tags/4.7.0.tar.gz
tar -xvzf 4.7.0.tar.gz
rm 4.7.0.tar.gz

wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.tar.gz
tar -xvzf 4.7.0.tar.gz
rm 4.7.0.tar.gz
```

Navigate to the OpenCV directory and create a build directory
```bash 
cd opencv-4.7.0
mkdir build
cd build
```

Configure the OpenCV build using CMake:
```bash
cmake 
-D CMAKE_BUILD_TYPE=RELEASE
-D CMAKE_INSTALL_PREFIX=/usr/local


```
```bash







conda install cmake
conda install cudatoolkit=11.8
conda install -c conda-forge cudnn=8.8.0.121
sudo apt-get install libopenjp2-7-dev
sudo apt-get install libblas-dev liblapack-dev
sudo apt-get install ccache
sudo apt-get install build-essential cmake pkg-config unzip yasm git checkinstall  # generic tools
sudo apt-get install git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev  # required
sudo apt-get install python3-dev python3-numpy python3-pip

sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev
```
Go to the environment folder

```bash
cd /path/to/your/envs/safeAR
```
# Explanation
```bash
git fetch origin
git tag
```
```bash
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout <desired_version>
cd ..

git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout <desired_version>
cd ..
```

GeForce RTX 4060 =	8.9 (https://developer.nvidia.com/cuda-gpus)

```bash
mkdir opencv/build
cd opencv/build


cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=~/miniconda3/envs/safeAR/opencv_contrib/modules/ -D PYTHON3_EXECUTABLE=~/miniconda3/envs/safeAR/bin/python3 -D PYTHON3_INCLUDE_DIR=~/miniconda3/envs/safeAR/include/python3.10/ -D PYTHON3_INCLUDE_DIR2=~/miniconda3/envs/safeAR/include/python3.10/ -D PYTHON3_LIBRARY=~/miniconda3/envs/safeAR/lib/libpython3.10.so -D PYTHON3_NUMPY_INCLUDE_DIRS=~/miniconda3/envs/safeAR/lib/python3.10/site-packages/numpy/core/include/ -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_PC_FILE_NAME=opencv.pc -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D CUDA_TOOLKIT_ROOT_DIR=~/usr/local/cuda-11.8 -D CUDA_NVCC_EXECUTABLE=/usr/local/cuda-11.8/bin/nvcc -D CUDA_ARCH_BIN=8.9 -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D WITH_CUFFT=ON -D WITH_CUBLAS=ON -D WITH_V4L=ON -D WITH_OPENCL=ON -D WITH_OPENGL=ON -D WITH_GSTREAMER=ON -D WITH_TBB=ON ..


make -j$(nproc)
sudo make install

```
-- Checking for module 'gtkglext-1.0'
--   No package 'gtkglext-1.0' found
-- Checking for module 'libavresample'
--   No package 'libavresample' found
-- Checking for module 'libdc1394-2'
--   No package 'libdc1394-2' found
 VTK is not found. Please set -DVTK_DIR in CMake to VTK build directory, or to VTK install subdirectory with VTKConfig.cmake file
```bash
# Navigate to the conda environment folder
cd /path/to/your/conda/env

# Clone the OpenCV repository
git clone https://github.com/opencv/opencv.git

# Clone the OpenCV contrib repository
git clone https://github.com/opencv/opencv_contrib.git

# Navigate to the opencv_contrib directory
cd opencv_contrib

# Checkout the specific version (4.8.1)
git checkout 4.8.1

# Navigate back to the parent directory
cd ..

# Navigate to the opencv directory
cd opencv

# Checkout the specific version (4.8.1)
git checkout 4.8.1

# Create a build directory and navigate into it
mkdir build
cd build

make -j$(nproc)
sudo make install
```

