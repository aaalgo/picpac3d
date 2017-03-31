# picpac3d

This package is for sampling and streaming volumetric samples for
training 3D convolutional neural networks.  It is designed as
part of my solution for Data Science Bowl 2017.

Highlights:
- Lossless storing 8-bit volumetric data with H265.
- On-the-fly region sampling with OpenGL.


# Building on Ubuntu 16.04

Install libraries
```
apt-get install libboost-all-dev libopencv-dev libglog-dev
apt-get install libgl1-mesa-dev libglew-dev libglfw3-dev libglm-dev
apt-get install libx265-dev libde265-dev
```

Then compile with
```
git submodule update --init --recursive
python setup.py build
sudo python setup.py install
```

# Tips

## Cube Stream
```
threads = 1         # PicPac preload streams, must be 1.
decode_threads = 4  # H265 decoding threads.
preload = 256       #
samples0 = 48       
samples1 = 48
pool = 4096         #
```

For 2600k to sustain 50 sample/second, each volume must generate
about 100 examples (samples0 = samples1= 48).  If network trains
slower than that, samples per volume can be lowered accordingly.
To avoid feeding the network samples from the same volume continuously,
a global pool of samples is maintained.  All samples from a newly
loaded volume are first injected into the pool, and samples are
randomly sampled from the pool for feeding into the network.
With the above configuration, the pool contains samples from
4096/(48+48) = 42 volumes.  That is within the window of each 
4096 samples, the samples are from about 42 volumes (actually more
than that, as samles are randomly drawn from the pool, a particular
lucky (unlucky) sample of a volume might linger in the pool for a 
long time, increasing the volume-variety of the pool.


