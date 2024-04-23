### cuPy Installation

Define PATH variables for CUDA and cuDNN:

```bash


Before installing cuPy, recommend you to upgrade setuptools and pip:
```bash
pip install -U setuptools pip
```

Part of the CUDA features in CuPy will be activated only when the corresponding libraries are installed.

cuTENSOR: v2.0

The library to accelerate tensor operations. See Environment variables for the details.

NCCL: v2.16 / v2.17

The library to perform collective multi-GPU / multi-node computations.

cuDNN: v8.8

The library to accelerate deep neural network computations.

cuSPARSELt: v0.2.0

The library to accelerate sparse matrix-matrix multiplication



### OpenCV with CUDA and cuDNN on Ubuntu 22.04 LTS, System-wide Installation

### !9 de abril de 2024
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#uninstallation

1) Desinstalar o CUDA 


tiagociiic@tiagociiic:~$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
tiagociiic@tiagociiic:~$ nvidia-smi
Fri Apr 19 08:11:14 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.171.04             Driver Version: 535.171.04   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4060 ...    Off | 00000000:01:00.0  On |                  N/A |
| N/A   34C    P8               2W /  60W |     49MiB /  8188MiB |      4%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1199      G   /usr/lib/xorg/Xorg                           45MiB |
+---------------------------------------------------------------------------------------+
tiagociiic@tiagociiic:~$ cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2 <= NOthing is returned
tiagociiic@tiagociiic:~$ echo $CUDA_HOME
echo $LD_LIBRARY_PATH
Nothing is returnede


......DEpois de muito batalhar... está feito...
 276  dpkg --get-selections | grep hold
  277  nvcc -V
  278  nvidia-smi
  279  cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
  280  echo $CUDA_HOME
  281  echo $LD_LIBRARY_PATH
  282  sudo apt-get --purge remove cuda-* nvidia-* gds-tools-* libcublas-* libcufft-* libcufile-* libcurand-* libcusolver-* libcusparse-* libnpp-* libnvidia-* libnvjitlink-* libnvjpeg-* nsight* nvidia-* libnvidia-* libcudnn8*
  283  sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"
  284  sudo apt-get autoremove
  285  sudo apt-get autoclean
  286  sudo rm -rf /usr/local/cuda*
  287  sudo dpkg -r cuda
  288  sudo dpkg -r $(dpkg -l | grep '^ii cudnn' | awk '{print $2}')
  289  dpkg -l | grep cudnn
  290  sudo dpkg --remove cudnn cudnn-local-repo-ubuntu2204-9.1.0 cudnn9 cudnn9-cuda-12 cudnn9-cuda-12-4 libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples libcudnn9-static-cuda-12
  291  sudo apt-get autoremove
  292  sudo apt-get autoclean
  293  sudo apt-get purge nvidia-*
  294  sudo rm /var/lib/dpkg/info/[package_name].*
  295  sudo dpkg --configure -a
  296  sudo apt-get update
  297  dpkg -l | grep cudnn
  298  sudo dpkg --remove cudnn cudnn-local-repo-ubuntu2204-9.1.0 cudnn9 cudnn9-cuda-12 cudnn9-cuda-12-4 libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples libcudnn9-static-cuda-12
  299  sudo apt-get autoremove
  300  sudo apt-get autoclean
  301  sudo rm /var/lib/dpkg/info/[package_name].*
  302  sudo rm -r /usr/local/cuda-*/lib64/libcudnn*
  303  sudo rm /var/lib/dpkg/info/cudnn.*
  304  nvcc -V
  305  nvidia-smi
  306  cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
  307  echo $CUDA_HOME
  308  echo $LD_LIBRARY_PATH
  309  ls /usr/local/cuda
  310  ls /usr/lib/nvidia-*
  311  ls /usr/lib32/nvidia-*
  312  find /usr/local/cuda -name 'libcudnn*'
  313  env | grep CUDA
  314  env | grep cuDNN
  315  lspci | grep -i nvidia
  316  cd
  317  ls
  318  cd snap/
  319  ls
  320  cd
  321  cd Templates/
  322  cd
  323  ls
  324  cd DO
  325  cd Downloads/
  326  ls
  327  nvcc --version
  328  history
  329  sudo nana /etc/default/grub
  330  sudo nano /etc/default/grub
  331  sudo update-grub
  332  sudo nano /etc/default/grub
  333  sudo update-grub
  334  sudo reboot
  335  lsmod | grep noveuau
  336  lsmod | grep nouveau
  337  sudo apt update
  338  sudo rm /etc/apt/sources.list.d/cudnn.list
  339  nano /etc/apt/sources.list
  340  ls /etc/apt/sources.list.d
  341  sudo rm /etc/apt/sources.list.d/cudnn-local-ubuntu2204-9.1.0.list 
  342  cd /var/
  343  ls
  344  sudo update
  345  sudo apt update
  346  sudo apt upgrade
  347  lsmod | grep nouveau
  348  sudo apt autoremove
  349  sudo apt autoclean
  350  sudo apt update
  351  history

---
Instalação CUDA Toolkit para Ubuntu 22.04 LTS (deb local)
Recomenda-se a instalação de package especificas ao SO, no caso do Ubuntu, .deb, ao invés de .run

Nota: por agora sem GDS packages.


Post installation actions


export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
# verica a versão do driver
cat /proc/driver/nvidia/version

NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  550.54.15  Release Build  (dvs-builder@U16-A24-23-2)  Tue Mar  5 22:15:33 UTC 2024
GCC version:  gcc version 12.3.0 (Ubuntu 12.3.0-1ubuntu1~22.04) 

# verifica a versão do CUDA
nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0


 Install Third-party Libraries
Some CUDA samples use third-party libraries which may not be installed by default on your system. These samples attempt to detect any required libraries when building.

If a library is not detected, it waives itself and warns you which library is missing. To build and run these samples, you must install the missing libraries. In cases where these dependencies are not installed, follow the instructions below.

sudo apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa-dev libfreeimage-dev libglfw3-dev

CUDA e DRIVERS instalados com sucesso!!!
Nota: 
tiagociiic@tiagociiic:~$ gcc --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

tiagociiic@tiagociiic:~$ gcc g++ --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

Vamos ao cuDNN agora (ubuntu local e deb)
-----------------------
(cuDNN, ou CUDA Deep Neural Network library, é uma biblioteca de software desenvolvida pela NVIDIA para acelerar a computação de redes neuronais profundas (Deep Neural Networks) em GPUs NVIDIA.)
https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html

distro: ubuntu2204
arch: x86_64
https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local


wget https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.1.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn

tiagociiic@tiagociiic:/usr/share/keyrings$ sudo apt-get -y install cudnn
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
E: Unable to locate package cudnn


### Install the per-CUDA meta-packages (cuDNN 9.1.0 with CUDA 12.4)

sudo apt-get -y install cudnn9-cuda-12


Verificando a Instalação no Linux

Para verificar se o cuDNN está instalado e está funcionando corretamente, compile o exemplo mnistCUDNN localizado na pasta /usr/src/cudnn_samples_v9 no arquivo Debian.
Instale os exemplos do cuDNN.
Vá para o caminho gravável.
Compile o exemplo mnistCUDNN.
Execute o exemplo mnistCUDNN.

```bash
sudo apt-get -y install libcudnn9-samples
cd $HOME/cudnn_samples_v9/mnistCUDNN
make clean && make
./mnistCUDNN
```

Se o cuDNN estiver instalado e funcionando corretamente em seu sistema Linux, você verá uma mensagem semelhante à seguinte, se tudo estiver funcionando corretamente:

Test passed!


| Architecture | OS Name | OS Version | Distro Information: Kernel | Distro Information: GCC | Distro Information: Glibc |
| --- | --- | --- | --- | --- | --- |
| x86\_64 | Ubuntu | 22.04 | 6.2.0 | 11.4.0 | 2.35 |

This table shows that cuDNN supports Ubuntu 22.04 x86\_64 with kernel version 6.2.0, GCC version 11.4.0, and glibc version 2.35.

| cuDNN Package | CUDA Toolkit Version | Supports static linking? | NVIDIA Driver Version for Linux | NVIDIA Driver Version for Windows | CUDA Compute Capability | Supported NVIDIA Hardware |
| --- | --- | --- | --- | --- | --- | --- |
| cuDNN 9.1.0 for CUDA 12.x | 12.4 | Yes | >=525.60.13 | >=527.41 | 9.0 3, 8.9 3, 8.6, 8.0, 7.5, 7.0, 6.1, 6.0, 5.0 | NVIDIA Hopper 3, NVIDIA Ada Lovelace architecture 3, NVIDIA Ampere architecture, NVIDIA Turing, NVIDIA Volta, NVIDIA Pascal, NVIDIA Maxwell |

This table shows that cuDNN 9.1.0 for CUDA 12.x supports CUDA Toolkit versions 12.4, and it supports static linking. The NVIDIA driver versions required for Linux and Windows are >=525.60.13 and >=527.41, respectively. The supported CUDA compute capabilities are 9.0 3, 8.9 3, 8.6, 8.0, 7.5, 7.0, 6.1, 6.0, and 5.0, and the supported NVIDIA hardware includes NVIDIA Hopper 3, NVIDIA Ada Lovelace architecture 3, NVIDIA Ampere architecture, NVIDIA Turing, NVIDIA Volta, NVIDIA Pascal, and NVIDIA Maxwell.

Source: https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html#support-matrix)

Vamos ao OpenCV agora...
------------------------

cmake -S /home/tiagociiic/OpenCV/opencv-4.9.0 -B /home/tiagociiic/OpenCV/opencv-4.9.0/build -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_CUBLAS=ON -D CMAKE_C_COMPILER=/usr/bin/gcc -D CMAKE_CXX_COMPILER=/usr/bin/g++ -D WITH_TBB=ON -D OPENCV_DNN_CUDA=ON -D OPENCV_ENABLE_NONFREE=ON -D CUDA_ARCH_BIN=8.9 -D OPENCV_EXTRA_MODULES_PATH=$HOME/OpenCV/opencv_contrib-4.9.0/modules -D BUILD_EXAMPLES=OFF -D HAVE_opencv_python3=ON -D ENABLE_FAST_MATH=ON

![alt text](image.png)





```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

```bash



https://gist.github.com/minhhieutruong0705/8f0ec70c400420e0007c15c98510f133

tutorial 2:
https://www.samontab.com/web/2023/02/installing-opencv-4-7-0-in-ubuntu-22-04-lts/ 

tutorial 3:
https://www.youtube.com/watch?v=8zXHSfFyXZs

tutorial 4:
https://www.youtube.com/watch?v=Nn5OfJjBGmg

tutorial 5:
https://www.youtube.com/watch?v=whAFl-izD-4

tutotial 6 (After new Ubuntu install): https://medium.com/@juancrrn/installing-opencv-4-with-cuda-in-ubuntu-20-04-fde6d6a0a367 


(tutotial 6)   Installing CUDA and cuDNN in Ubuntu 22.04 for deep learning (https://medium.com/@juancrrn/installing-cuda-and-cudnn-in-ubuntu-20-04-for-deep-learning-dad8841714d6)

```bash
sudo apt-get update 
sudo apt-get upgrade
sudo apt install linux-headers-$(uname -r)

wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run

https://askubuntu.com/questions/206283/how-can-i-uninstall-a-nvidia-driver-completely
-----
sudo apt-get remove --purge '^nvidia-.*'
sudo apt autoremove
sudo apt autoclean
sudo apt-get install ubuntu-desktop
----
echo -e "blacklist nouveau\noptions nouveau modeset=0" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf

sudo update-initramfs -u


sudo sh cuda_12.4.1_550.54.15_linux.run --toolkit --silent --override

export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:${PATH}

source ~/.bashrc
nvcc -V

cat /proc/driver/nvidia/version


git clone https://github.com/NVIDIA/cuda-samples.git
Then navigate to the directory of the deviceQuery utility and compile it:

$ cd cuda-samples/Samples/1_Utilities/deviceQuery/
$ make
./deviceQuery

# Onde fica isto?
sudo apt install nvidia-cuda-toolkit
Install the cuDNN library
3.1. Install zlib
sudo apt install zlib1g


downloaded installer: https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
wget https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.1.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn

sudo apt-get install libcudnn8=${cudnn_version}-1+cuda12.4
$ sudo apt-get install libcudnn8-dev=${cudnn_version}-1+cuda12.4
$ sudo apt-get install libcudnn8-samples=${cudnn_version}-1+cuda12.4

sudo apt-get install libfreeimage-dev
```

wget https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb.1
sudo cp /var/cudnn-local-*/cudnn-*-keyring.gpg /usr/share/keyrings/


```bash

sudo apt install cmake
sudo apt install gcc g++
sudo apt install python3 python3-dev python3-numpy git
sudo apt install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libeigen3-dev
sudo apt install libpng-dev libjpeg-dev 

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

Check CUDA_ARCH_BIN in https://developer.nvidia.com/cuda-gpus for you GPU. GeForce RTX 4060 = 8.9

cd opencv
mkdir build
cd build


cmake -S /home/tiagociiic/OpenCV/opencv -B /home/tiagociiic/OpenCV/opencv/build -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_CUBLAS=ON -D CMAKE_C_COMPILER=/usr/bin/gcc-10 -D CMAKE_CXX_COMPILER=/usr/bin/g++-10 -D WITH_TBB=ON -D OPENCV_DNN_CUDA=ON -D OPENCV_ENABLE_NONFREE=ON -D CUDA_ARCH_BIN=8.9 -D OPENCV_EXTRA_MODULES_PATH=$HOME/OpenCV/opencv_contrib/modules -D BUILD_EXAMPLES=OFF -D HAVE_opencv_python3=ON -D ENABLE_FAST_MATH=ON


cmake -S /home/tiagociiic/OpenCV/opencv -B /home/tiagociiic/OpenCV/opencv/build -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_CUBLAS=ON -D CMAKE_C_COMPILER=/usr/bin/gcc-10 -D CMAKE_CXX_COMPILER=/usr/bin/g++-10 -D CMAKE_C_COMPILER_AR=/usr/bin/gcc-ar-10 -D CMAKE_C_COMPILER_RANLIB=/usr/bin/gcc-ranlib-10 -D WITH_TBB=ON -D OPENCV_DNN_CUDA=ON -D OPENCV_ENABLE_NONFREE=ON -D CUDA_ARCH_BIN=8.9 -D OPENCV_EXTRA_MODULES_PATH=$HOME/OpenCV/opencv_contrib/modules -D BUILD_EXAMPLES=OFF -D HAVE_opencv_python3=ON -D ENABLE_FAST_MATH=ON


cmake -S /home/tiagociiic/OpenCV/opencv -B /home/tiagociiic/OpenCV/opencv/build -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_CUBLAS=ON -D CMAKE_C_COMPILER=/usr/bin/gcc-10 -D CMAKE_CXX_COMPILER=/usr/bin/g++-10 -D CMAKE_C_COMPILER_AR=/usr/bin/gcc-ar-10 -D CMAKE_C_COMPILER_RANLIB=/usr/bin/gcc-ranlib-10 -D CMAKE_CXX_COMPILER_AR=/usr/bin/gcc-ar-10 -D CMAKE_CXX_COMPILER_RANLIB=/usr/bin/gcc-ranlib-10 -D WITH_TBB=ON -D OPENCV_DNN_CUDA=ON -D OPENCV_ENABLE_NONFREE=ON -D CUDA_ARCH_BIN=8.9 -D OPENCV_EXTRA_MODULES_PATH=$HOME/OpenCV/opencv_contrib/modules -D BUILD_EXAMPLES=OFF -D HAVE_opencv_python3=ON -D ENABLE_FAST_MATH=ON

cmake -D CMAKE_C_COMPILER="/usr/bin/gcc-10" -D CMAKE_CXX_COMPILER "/usr/bin/g++-10" 

cmake 
cmake -S /home/tiagociiic/OpenCV/opencv-4.9.0 -B /home/tiagociiic/OpenCV/opencv-4.9.0/build -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_CUBLAS=ON -D CMAKE_C_COMPILER=/usr/bin/gcc -D CMAKE_CXX_COMPILER=/usr/bin/g++ -D WITH_TBB=ON -D OPENCV_DNN_CUDA=ON -D OPENCV_ENABLE_NONFREE=ON -D CUDA_ARCH_BIN=8.9 -D OPENCV_EXTRA_MODULES_PATH=$HOME/OpenCV/opencv_contrib-4.9.0/modules -D BUILD_EXAMPLES=OFF -D HAVE_opencv_python3=ON -D ENABLE_FAST_MATH=ON


Change Gcc to version 10

```
VTK not found: If you don't need VTK, you can ignore this. If you do, install VTK with sudo apt-get install libvtk7-dev.

No modules found in opencv_contrib: This error is due to an incorrect path to the opencv_contrib modules. Make sure the path in -D OPENCV_EXTRA_MODULES_PATH=$HOME/opencv_contrib/modules is correct.

GTK not found: Install GTK with cd.

Gstreamer and libdc1394-2 not found: Install these with sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libdc1394-22-dev.

CUDA requires 'cudev': This error is due to the cudev module not being found in the opencv_contrib modules. Make sure the opencv_contrib repository is correctly cloned and the path is correctly set.
---
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
wget https://github.com/opencv/opencv/archive/refs/tags/4.9.0.tar.gz
tar -xvzf 4.9.0.tar.gz
rm 4.9.0.tar.gz

wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.9.0.tar.gz
tar -xvzf 4.9.0.tar.gz
rm 4.9.0.tar.gz
```

Navigate to the OpenCV directory and create a build directory

```bash 
cd opencv-4.9.0
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

cmake -S /home/tiagociiic/OpenCV/opencv/build \ 
-B /home/tiagociiic/OpenCV/opencv/build\
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/bin \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D WITH_TBB=ON \
-D OPENCV_DNN_CUDA=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D CUDA_ARCH_BIN=8.9 \
-D OPENCV_EXTRA_MODULES_PATH=$HOME/opencv_contrib/modules \
-D BUILD_EXAMPLES=OFF \
-D HAVE_opencv_python3=ON \
-D ENABLE_FAST_MATH=ON\
..

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

