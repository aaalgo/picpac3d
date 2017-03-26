# picpac3d

This package is for sampling and streaming volumetric samples for
training 3D convolutional neural networks.

Highlights:
- Lossless storing 8-bit volumetric data with H265.
- On-the-fly region sampling with OpenGL.

# Building on Ubuntu 16.04

Install libraries
```
apt-get install libboost-all-dev libopencv-dev libglog-dev
apt-get install libgl1-mesa-dev libglew-dev libglfw3-dev
```
And the following H256 libraries
- https://github.com/strukturag/libde265
- https://bitbucket.org/multicoreware/x265/wiki/Home

Then compile with
```
git submodule update --init --recursive
python setup.py build
sudo python setup.py install
```

# Usage


