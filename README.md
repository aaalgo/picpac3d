# picpac3d

This package is for sampling and streaming volumetric data for
training 3D convolutional neural networks.  It is designed as
part of my solution for Data Science Bowl 2017.

Highlights:
- Lossless storing 8-bit volumetric data with H265.
- On-the-fly region sampling and augmentation with random rotation and
  scaling using OpenGL 3D texture.


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

For a core i7 2600k to sustain 50 sample/second, each volume must generate
about 100 examples (samples0 = samples1= 48).  If network trains
slower than that, samples per volume can be lowered accordingly
to improve variety.
To avoid feeding the network samples from the same volume continuously,
a global pool of samples is maintained. All samples from a newly
loaded volume are first added to the pool, and samples are
randomly sampled from the pool for feeding into the network.
With the above configuration, the pool contains samples from
4096/(48+48) = 42 volumes.  That is within the window of each 
4096 samples, the samples are from about 42 volumes (actually more
than that, as samles are randomly drawn from the pool, a particular
lucky (unlucky) sample of a volume might linger in the pool for a 
long time, increasing the volume-variety of the pool.

Because of hardware limitation, the library currently only supports
the following size configuration:

- Input volume must be about 512x512x512.  It doesn't have to be
  exactly this.  The library will automatically clip and pad the data.
- Each sample is of size 64x64x64.


Usage example with Tensorflow:
```
    import picpac3d
    ...
    picpac_config = dict(seed=2017,
                # most are just PicPac configurations.
                threads=1,
                decode_threads=4,  # decoding threads
                preload=512,
                cache=False,
                shuffle=True,
                reshuffle=True,
                batch=1,
                split=1,
                split_fold=0,
                stratify=True,
                channels=1,
                pert_color1=20,    # randomly +/- 0~20 to color
                # for BOWL data, corresponds to about 2mm/pixel.
                pert_min_scale=0.45,
                pert_max_scale=0.55,
                # are we rotating too much?
                pert_angle = 180,  # in degrees
                samples0 = 32,     # sample 32 negative cubes
                samples1 = 64,     # and 64 positive cubes from each volume
                pool = 4096
                )

    stream = picpac3d.CubeStream(FLAGS.db, perturb=True, loop=True, **picpac_config)
    ...
    with tf.Session() as sess:

        for _ in range(FLAGS.maximal_training_steps):
            images, labels = tr_stream.next()
            feed_dict = {X: images, Y: labels}
            mm, _, summaries = sess.run([metrics, train_op, train_summaries], feed_dict=feed_dict)
```
