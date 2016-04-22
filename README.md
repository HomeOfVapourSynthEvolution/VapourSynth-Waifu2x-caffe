Description
===========

waifu2x filter for VapourSynth, based on [waifu2x-caffe](https://github.com/lltcggie/waifu2x-caffe).


Note
====

The folder `models` must be located in the same folder as `Waifu2x-caffe.dll`.

Due to the license of cuDNN, I can't distribute the required dll file. You have to register and download yourself at https://developer.nvidia.com/cudnn.


Usage
=====

    caffe.Waifu2x(clip clip[, int noise=1, int scale=2, int block=128, bint photo=False, bint cudnn=True, bint tta=False])

* clip: The input clip. Must be 32-bit floating point. Any planar color family is supported.

* noise: Noise reduction level.
  * 0 = none
  * 1 = medium
  * 2 = high
  * 3 = highest

* scale: Upscaling factor. Must be a power of 2. Set to 1 for no upscaling.

* block: The block size for dividing the image during processing. Smaller value results in lower VRAM usage, while larger value may not necessarily give faster speed. The optimal value may vary according to different graphics card and image size.

* photo: When set to true, it uses the photo model. When set to false, the anime-style model will be used instead. There are two sets of anime-style models, one is trained under RGB, the other is trained under Y (luma only). The RGB model will be used when the color family of the input is RGB, and the Y model will be used for all the other color family. Note that the photo model is only available for RGB, so this parameter has no effect for non-RGB input.

* cudnn: When set to true, it uses cuDNN as processor. When set to false, CUDA will be used instead.

* tta: Whether TTA(Test-Time Augmentation) mode is used. It increases PSNR by 0.15 or so, but 8 times slower.


Benchmark
=========

Here is the performance comparison between `Waifu2x-w2xc` and `Waifu2x-caffe`.

Image size: 640x360<br />
GPU: GTX 660

w2xc.Waifu2x(noise=0, scale=2, block=block, photo=False, gpu=1)
```
--------
Testing 16 frames with block=64...
Output 16 frames in 102.69 seconds (0.16 fps)
--------
Testing 16 frames with block=128...
Output 16 frames in 44.51 seconds (0.36 fps)
--------
Testing 16 frames with block=192...
Output 16 frames in 34.57 seconds (0.46 fps)
--------
Testing 16 frames with block=256...
Output 16 frames in 28.47 seconds (0.56 fps)
--------
Testing 16 frames with block=320...
Output 16 frames in 28.01 seconds (0.57 fps)
--------
Testing 16 frames with block=384...
Output 16 frames in 24.50 seconds (0.65 fps)
--------
Testing 16 frames with block=448...
Output 16 frames in 23.19 seconds (0.69 fps)
--------
Testing 16 frames with block=512...
Output 16 frames in 23.22 seconds (0.69 fps)
--------
Testing 16 frames with block=576...
Output 16 frames in 23.38 seconds (0.68 fps)
--------
Testing 16 frames with block=640...
Output 16 frames in 23.23 seconds (0.69 fps)
--------
Testing 16 frames with block=704...
Output 16 frames in 22.88 seconds (0.70 fps)
--------
Testing 16 frames with block=768...
Output 16 frames in 22.13 seconds (0.72 fps)
--------
Testing 16 frames with block=832...
Output 16 frames in 22.32 seconds (0.72 fps)
--------
Testing 16 frames with block=896...
Output 16 frames in 22.29 seconds (0.72 fps)
--------
Testing 16 frames with block=960...
Output 16 frames in 22.33 seconds (0.72 fps)
--------
Testing 16 frames with block=1024...
Output 16 frames in 22.43 seconds (0.71 fps)
--------
Testing 16 frames with block=1088...
Output 16 frames in 22.52 seconds (0.71 fps)
--------
Testing 16 frames with block=1152...
Output 16 frames in 22.52 seconds (0.71 fps)
--------
Testing 16 frames with block=1216...
Output 16 frames in 22.52 seconds (0.71 fps)
--------
Testing 16 frames with block=1280...
Output 16 frames in 22.73 seconds (0.70 fps)
--------
```

caffe.Waifu2x(noise=0, scale=2, block=block, photo=False, cudnn=True, tta=False)
```
--------
Testing 16 frames with block=64...
Output 16 frames in 25.72 seconds (0.62 fps)
--------
Testing 16 frames with block=128...
Output 16 frames in 20.75 seconds (0.77 fps)
--------
Testing 16 frames with block=192...
Output 16 frames in 20.86 seconds (0.77 fps)
--------
Testing 16 frames with block=256...
Output 16 frames in 19.75 seconds (0.81 fps)
--------
Testing 16 frames with block=320...
Output 16 frames in 24.09 seconds (0.66 fps)
--------
Testing 16 frames with block=384...
Output 16 frames in 23.10 seconds (0.69 fps)
--------
Testing 16 frames with block=448...
Output 16 frames in 23.45 seconds (0.68 fps)
--------
Testing 16 frames with block=512...
Output 16 frames in 30.14 seconds (0.53 fps)
--------
Testing 16 frames with block=576...
Output 16 frames in 37.26 seconds (0.43 fps)
--------
Testing 16 frames with block=640...
Output 16 frames in 31.39 seconds (0.51 fps)
--------
Testing 16 frames with block=704...
Output 16 frames in 37.04 seconds (0.43 fps)
--------
Testing 16 frames with block=768...
Output 16 frames in 22.78 seconds (0.70 fps)
--------
Testing 16 frames with block=832...
Output 16 frames in 26.70 seconds (0.60 fps)
--------
Testing 16 frames with block=896...
Output 16 frames in 31.42 seconds (0.51 fps)
--------
Testing 16 frames with block=960...
F0214 14:48:56.695240  8276 syncedmem.cpp:56] Check failed: error == cudaSuccess (2 vs. 0)  out of memory
*** Check failure stack trace: ***
--------
```
